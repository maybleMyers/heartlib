"""
HeartMuLa Gradio Web UI
A simple web interface for generating music with HeartMuLa.
"""

import gradio as gr
import torch
import tempfile
import os
import sys
import io
import threading
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from queue import Queue
from datetime import datetime

# Model layer counts for different versions
MODEL_LAYER_COUNTS = {
    "3B": {"backbone": 28, "decoder": 3},
    "7B": {"backbone": 32, "decoder": 3},
}


def _params_to_device(module, device, non_blocking=True):
    """Move only PARAMETERS (weights) to device, NOT buffers.

    This is critical for torchtune models because buffers like RoPE cache
    must stay on GPU while parameters can be swapped to CPU.
    """
    for submodule in module.modules():
        for param_name, param in list(submodule.named_parameters(recurse=False)):
            if param is not None:
                param.data = param.data.to(device, non_blocking=non_blocking)


def _ensure_buffers_on_device(module, device):
    """Ensure all buffers are on the specified device."""
    for name, buf in module.named_buffers(recurse=True):
        if buf is not None and buf.device.type != device.type:
            # Use setattr on parent module to properly update the buffer
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, buf_name = parts
                parent = module.get_submodule(parent_name)
            else:
                parent = module
                buf_name = name
            parent.register_buffer(buf_name, buf.to(device))


class BlockSwapManager:
    """Manages CPU/GPU block swapping for transformer layers.

    Uses the LTX-style approach: only swap PARAMETERS (weights), keep
    BUFFERS (like RoPE position embedding caches) on GPU at all times.
    """

    def __init__(self, model, num_gpu_blocks: int, device: torch.device):
        self.model = model
        self.num_gpu_blocks = num_gpu_blocks
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.total_blocks = 0
        self.swapped_blocks = []
        self.original_forwards = {}
        self.buffers_initialized = False

    def _ensure_all_buffers_on_gpu(self):
        """Ensure ALL buffers in the backbone stay on GPU."""
        if self.buffers_initialized:
            return
        backbone = self.model.backbone
        # Move all buffers to GPU and keep them there
        for name, buf in backbone.named_buffers(recurse=True):
            if buf is not None and buf.device.type != 'cuda':
                # Find the module that owns this buffer and update it
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, buf_name = parts
                    try:
                        parent = backbone.get_submodule(parent_name)
                        parent.register_buffer(buf_name, buf.to(self.device))
                    except AttributeError:
                        pass
                else:
                    backbone.register_buffer(name, buf.to(self.device))
        self.buffers_initialized = True

    def setup(self):
        """Set up block swapping by wrapping forward methods."""
        backbone = self.model.backbone

        if not hasattr(backbone, 'layers'):
            log("Warning: Cannot find 'layers' attribute for block swapping")
            return False

        self.total_blocks = len(backbone.layers)
        blocks_to_swap = max(0, self.total_blocks - self.num_gpu_blocks)

        if blocks_to_swap == 0:
            log(f"All {self.total_blocks} blocks kept on GPU (no swapping needed)")
            return True

        log(f"Setting up block swapping: {self.num_gpu_blocks} on GPU, {blocks_to_swap} swapped to CPU")

        # First, ensure ALL buffers (including RoPE cache) are on GPU
        self._ensure_all_buffers_on_gpu()

        # Move the LAST N blocks' PARAMETERS to CPU (they're used later in forward pass)
        # Keep buffers on GPU!
        swap_start_idx = self.num_gpu_blocks

        for i in range(swap_start_idx, self.total_blocks):
            layer = backbone.layers[i]

            # Store original forward
            self.original_forwards[i] = layer.forward

            # Create wrapped forward that handles device transfer for PARAMETERS ONLY
            def make_swapping_forward(layer_ref, orig_forward, idx, manager):
                def swapping_forward(*args, **kwargs):
                    # Move layer PARAMETERS to GPU (buffers already on GPU)
                    _params_to_device(layer_ref, manager.device, non_blocking=False)
                    if manager.device.type == 'cuda':
                        torch.cuda.synchronize()

                    # Run forward (all tensors should already be on GPU)
                    result = orig_forward(*args, **kwargs)

                    # Move layer PARAMETERS back to CPU
                    _params_to_device(layer_ref, manager.cpu_device, non_blocking=True)

                    return result
                return swapping_forward

            # Replace forward method
            layer.forward = make_swapping_forward(layer, self.original_forwards[i], i, self)

            # Move layer PARAMETERS to CPU, but keep buffers on GPU
            _params_to_device(layer, self.cpu_device, non_blocking=False)
            self.swapped_blocks.append(i)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"Block swapping ready: blocks {swap_start_idx}-{self.total_blocks-1} will swap from CPU")
        log(f"  (Buffers like RoPE cache remain on GPU)")
        return True

    def cleanup(self):
        """Restore original forwards and move all blocks back to GPU."""
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            for i, orig_forward in self.original_forwards.items():
                self.model.backbone.layers[i].forward = orig_forward
            for layer in self.model.backbone.layers:
                _params_to_device(layer, self.device, non_blocking=False)
        self.original_forwards = {}
        self.swapped_blocks = []
        self.buffers_initialized = False

    def get_memory_stats(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        return "CUDA not available"


class LogCapture:
    """Captures stdout/stderr and stores logs for display."""

    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()

    def write(self, text):
        if text.strip():
            with self.lock:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.logs.append(f"[{timestamp}] {text.rstrip()}")
        # Also print to original stdout
        sys.__stdout__.write(text)
        sys.__stdout__.flush()

    def flush(self):
        sys.__stdout__.flush()

    def get_logs(self):
        with self.lock:
            return "\n".join(self.logs[-50:])  # Last 50 lines

    def clear(self):
        with self.lock:
            self.logs = []


# Global log capture
log_capture = LogCapture()


def log(message: str):
    """Log a message to both console and capture buffer."""
    print(message)


# Default example lyrics
DEFAULT_LYRICS = """[Intro]

[Verse]
The sun creeps in across the floor
My cat is stretching by the door
She yawns and blinks her emerald eyes
Another day begins to rise

[Prechorus]
Soft paws padding all around
The sweetest purring is the sound
My furry friend has got me found

[Chorus]
Little cat with whiskers long
You make my heart sing like a song
Curled up warm upon my knee
You are everything to me
My little cat so wild and free

[Verse]
She chases shadows on the wall
And knocks my things down in the hall
But when she curls up on my chest
I know that I am truly blessed

[Bridge]
Through the good times and the bad
You are the best friend I have had
My little cat you make me glad

[Chorus]
Little cat with whiskers long
You make my heart sing like a song
Curled up warm upon my knee
You are everything to me

[Outro]
My little cat
My furry friend"""

# Default tags
DEFAULT_TAGS = "piano,happy"

# Example tag suggestions
TAG_SUGGESTIONS = [
    "piano", "guitar", "synthesizer", "drums", "violin", "bass",
    "happy", "sad", "romantic", "energetic", "calm", "melancholic",
    "pop", "rock", "jazz", "classical", "electronic", "folk",
    "wedding", "party", "relaxation", "workout", "study", "sleep"
]


def generate_music(
    lyrics: str,
    tags: str,
    max_duration_seconds: float,
    temperature: float,
    topk: int,
    cfg_scale: float,
    model_path: str,
    model_version: str,
    num_gpu_blocks: int,
    progress=gr.Progress()
):
    """Generate music using HeartMuLa (loads model fresh each time to avoid OOM)."""
    log_capture.clear()
    old_stdout = sys.stdout
    sys.stdout = log_capture

    pipe = None
    block_swap_manager = None

    try:
        # Validate inputs
        if not model_path or not os.path.exists(model_path):
            return None, f"Model path does not exist: {model_path}", ""

        if not lyrics.strip():
            return None, "Please enter some lyrics.", ""

        if not tags.strip():
            return None, "Please enter at least one tag (e.g., 'piano,happy').", ""

        log("=" * 50)
        log("Starting music generation...")
        log(f"Tags: {tags}")
        log(f"Max duration: {max_duration_seconds}s")
        log(f"Temperature: {temperature}, Top-K: {topk}, CFG Scale: {cfg_scale}")

        progress(0.1, desc="Loading model...")

        # Load model fresh for this generation
        from heartlib import HeartMuLaGenPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        log(f"Loading HeartMuLa-{model_version} from {model_path}...")
        log(f"Using device: {device}, dtype: {dtype}")

        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            version=model_version,
        )

        # Ensure model is on correct device
        pipe.model = pipe.model.to(device)
        pipe.device = device

        progress(0.15, desc="Model loaded, setting up...")

        # Get total blocks for this model version
        total_blocks = MODEL_LAYER_COUNTS.get(model_version, {}).get("backbone", 28)

        # Set up block swapping if requested
        if num_gpu_blocks > 0 and num_gpu_blocks < total_blocks and torch.cuda.is_available():
            log(f"Configuring block swapping: {num_gpu_blocks} blocks on GPU...")

            block_swap_manager = BlockSwapManager(pipe.model, num_gpu_blocks, device)
            if block_swap_manager.setup():
                blocks_swapped = total_blocks - num_gpu_blocks
                log(f"Block swapping ready: {num_gpu_blocks} on GPU, {blocks_swapped} swap from CPU")
                log(block_swap_manager.get_memory_stats())
            else:
                block_swap_manager = None
                log("Block swapping setup failed, using full GPU mode")
        else:
            log(f"No block swapping: all {total_blocks} blocks on GPU")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory after model load: {allocated:.2f}GB")

        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name

        # Convert duration from seconds to milliseconds
        max_audio_length_ms = int(max_duration_seconds * 1000)

        log(f"Output path: {output_path}")
        log(f"Max audio length: {max_audio_length_ms}ms")

        progress(0.2, desc="Generating music (this may take a while)...")
        log("Starting generation loop...")

        with torch.no_grad():
            pipe(
                {
                    "lyrics": lyrics,
                    "tags": tags,
                },
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )

        progress(1.0, desc="Generation complete!")
        log("Generation complete!")

        # Get memory info after generation
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory after generation: {allocated:.2f}GB")

        # Build info message
        info = f"""Generation Complete!
- Duration: {max_duration_seconds:.1f}s (max)
- Temperature: {temperature}
- Top-K: {topk}
- CFG Scale: {cfg_scale}
- Tags: {tags}
- Output: {output_path}"""

        return output_path, info, log_capture.get_logs()

    except Exception as e:
        import traceback
        error_msg = f"Error during generation: {str(e)}"
        log(error_msg)
        log(traceback.format_exc())
        return None, error_msg, log_capture.get_logs()
    finally:
        sys.stdout = old_stdout
        # Clean up to free GPU memory
        if block_swap_manager is not None:
            block_swap_manager.cleanup()
        if pipe is not None:
            # Explicitly delete model components to release references
            if hasattr(pipe, 'model') and pipe.model is not None:
                del pipe.model
            if hasattr(pipe, 'audio_codec') and pipe.audio_codec is not None:
                del pipe.audio_codec
            del pipe
        # Force garbage collection before clearing CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory after cleanup: {allocated:.2f}GB")


def add_tag(current_tags: str, new_tag: str) -> str:
    """Add a tag to the current tags string."""
    if not new_tag:
        return current_tags

    current_list = [t.strip() for t in current_tags.split(",") if t.strip()]
    new_tag = new_tag.strip().lower()

    if new_tag not in current_list:
        current_list.append(new_tag)

    return ",".join(current_list)


def clear_tags() -> str:
    """Clear all tags."""
    return ""


# Build the Gradio interface
with gr.Blocks(
    title="HeartMuLa Music Generator",
) as demo:
    gr.Markdown(
        """
        # HeartMuLa Music Generator

        Generate music conditioned on lyrics and tags using HeartMuLa - a family of open-sourced music foundation models.

        **Features:**
        - Text-to-music generation with lyrics and style tags
        - Support for multiple languages (English, Chinese, Japanese, Korean, Spanish)
        - Adjustable generation parameters for creative control
        """
    )

    with gr.Row():
        # Left column - Model & Input
        with gr.Column(scale=1):
            gr.Markdown("### Model Settings")

            with gr.Group():
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="./ckpt",
                    value="./ckpt",
                    info="Path to the HeartMuLa checkpoint directory"
                )

                model_version = gr.Dropdown(
                    label="Model Version",
                    choices=["3B", "7B"],
                    value="3B",
                    info="HeartMuLa model size (7B not yet released)"
                )

                num_gpu_blocks = gr.Slider(
                    label="GPU Blocks (Block Swapping)",
                    minimum=0,
                    maximum=28,
                    value=14,
                    step=1,
                    info="Blocks to keep on GPU (0 = all on GPU, no swapping). 3B has 28 blocks."
                )

            gr.Markdown("### Style Tags")
            gr.Markdown("*Comma-separated tags (e.g., piano,happy,romantic)*")

            with gr.Group():
                tags_input = gr.Textbox(
                    label="Tags",
                    placeholder="piano,happy,romantic",
                    value=DEFAULT_TAGS,
                    info="Describe the style, mood, instruments"
                )

                gr.Markdown("**Quick Add Tags:**")
                with gr.Row():
                    tag_dropdown = gr.Dropdown(
                        label="Select Tag",
                        choices=TAG_SUGGESTIONS,
                        scale=3
                    )
                    add_tag_btn = gr.Button("Add", scale=1)
                    clear_tags_btn = gr.Button("Clear", scale=1)

        # Middle column - Lyrics
        with gr.Column(scale=1):
            gr.Markdown("### Lyrics")
            gr.Markdown("*Use section markers like [Verse], [Chorus], [Bridge], etc.*")

            lyrics_input = gr.Textbox(
                label="Lyrics",
                placeholder="[Verse]\nYour lyrics here...\n\n[Chorus]\nChorus lyrics...",
                value=DEFAULT_LYRICS,
                lines=20,
                max_lines=30
            )

            with gr.Row():
                clear_lyrics_btn = gr.Button("Clear Lyrics")
                load_example_btn = gr.Button("Load Example")

        # Right column - Generation Parameters & Output
        with gr.Column(scale=1):
            gr.Markdown("### Generation Parameters")

            with gr.Group():
                max_duration = gr.Slider(
                    label="Max Duration (seconds)",
                    minimum=10,
                    maximum=240,
                    value=120,
                    step=10,
                    info="Maximum length of generated audio"
                )

                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    info="Higher = more random/creative, Lower = more deterministic"
                )

                topk = gr.Slider(
                    label="Top-K",
                    minimum=1,
                    maximum=200,
                    value=50,
                    step=1,
                    info="Number of top tokens to sample from"
                )

                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance)",
                    minimum=1.0,
                    maximum=5.0,
                    value=1.5,
                    step=0.1,
                    info="Higher = stronger adherence to tags/lyrics"
                )

            gr.Markdown("### Generate")

            generate_btn = gr.Button("Generate Music", variant="primary", size="lg")

            gr.Markdown("### Output")

            output_audio = gr.Audio(
                label="Generated Music",
                type="filepath",
                interactive=False
            )

            output_info = gr.Textbox(
                label="Generation Info",
                interactive=False,
                lines=6
            )

    # Console output section
    with gr.Accordion("Console Output", open=True):
        console_output = gr.Textbox(
            label="Logs",
            interactive=False,
            lines=15,
            max_lines=30
        )

    # Advanced Settings (collapsible)
    with gr.Accordion("Parameter Guide", open=False):
        gr.Markdown(
            """
            ### Parameter Descriptions

            | Parameter | Range | Default | Description |
            |-----------|-------|---------|-------------|
            | **Max Duration** | 10-240s | 120s | Maximum length of generated audio in seconds |
            | **Temperature** | 0.1-2.0 | 1.0 | Controls randomness. Higher values produce more diverse but potentially less coherent results |
            | **Top-K** | 1-200 | 50 | Limits sampling to top K most likely tokens. Lower values = more focused, higher = more diverse |
            | **CFG Scale** | 1.0-5.0 | 1.5 | Classifier-Free Guidance. Higher values make the model follow tags/lyrics more strictly |

            ### GPU Block Swapping

            The **GPU Blocks** slider controls memory usage by swapping transformer blocks between GPU and CPU:
            - **0 (default)**: All blocks on GPU (fastest, requires most VRAM ~12GB for 3B)
            - **1-27**: Keep N blocks on GPU, swap remaining blocks from CPU as needed
            - Lower values = less VRAM but slower generation

            **Memory Guidelines for 3B Model:**
            | GPU Blocks | Approx. VRAM | Speed |
            |------------|--------------|-------|
            | 28 (all)   | ~12GB        | Fastest |
            | 20         | ~9GB         | Fast |
            | 14         | ~6GB         | Medium |
            | 7          | ~4GB         | Slower |

            ### Tips for Better Results

            1. **Lyrics Format**: Use section markers like `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`
            2. **Tags**: Use descriptive tags for instruments (piano, guitar), mood (happy, sad), genre (pop, jazz)
            3. **Temperature**: Start with 1.0, increase for more creativity or decrease for consistency
            4. **CFG Scale**: Increase if the output doesn't match your tags/lyrics well

            ### Supported Languages

            HeartMuLa supports multilingual lyrics including:
            - English
            - Chinese (中文)
            - Japanese (日本語)
            - Korean (한국어)
            - Spanish (Español)
            """
        )

    # Event handlers
    # Update slider based on model version
    def update_blocks_slider(version):
        max_blocks = MODEL_LAYER_COUNTS.get(version, {}).get("backbone", 28)
        return gr.update(maximum=max_blocks, info=f"Blocks to keep on GPU (0 = all on GPU). {version} has {max_blocks} blocks.")

    model_version.change(
        fn=update_blocks_slider,
        inputs=[model_version],
        outputs=[num_gpu_blocks]
    )

    generate_btn.click(
        fn=generate_music,
        inputs=[lyrics_input, tags_input, max_duration, temperature, topk, cfg_scale, model_path, model_version, num_gpu_blocks],
        outputs=[output_audio, output_info, console_output]
    )

    add_tag_btn.click(
        fn=add_tag,
        inputs=[tags_input, tag_dropdown],
        outputs=[tags_input]
    )

    clear_tags_btn.click(
        fn=clear_tags,
        outputs=[tags_input]
    )

    clear_lyrics_btn.click(
        fn=lambda: "",
        outputs=[lyrics_input]
    )

    load_example_btn.click(
        fn=lambda: DEFAULT_LYRICS,
        outputs=[lyrics_input]
    )


if __name__ == "__main__":
    print("Starting HeartMuLa Gradio UI...")
    print("Open http://localhost:7860 in your browser")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
