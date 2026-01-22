"""
HeartMuLa Gradio Web UI
A simple web interface for generating music with HeartMuLa.
"""

import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
import torch
import tempfile
import os
import sys
import io
import threading
import random
import time
import json
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from queue import Queue
from datetime import datetime

# Defaults file path
HEARTMULA_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "heartmula_defaults.json")

# Model layer counts for different versions
MODEL_LAYER_COUNTS = {
    "3B": {"backbone": 28, "decoder": 3},
    "7B": {"backbone": 32, "decoder": 3},
}

# Global stop event for batch cancellation
stop_event = threading.Event()

# Global model cache to avoid reloading between generations
_cached_pipe = None
_cached_pipe_config = None  # Stores (model_path, model_version, model_dtype, num_gpu_blocks, compile_model)


def stop_generation():
    """Signal to stop the current generation."""
    stop_event.set()
    return "Stopping..."


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
        """Restore original forwards. Model will be moved to CPU separately during full cleanup."""
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            for i, orig_forward in self.original_forwards.items():
                self.model.backbone.layers[i].forward = orig_forward
        self.original_forwards = {}
        self.swapped_blocks = []
        self.buffers_initialized = False

    def restore_state(self):
        """Restore block swapping state after model was moved to CPU.

        This is needed when the model is offloaded to CPU during generation
        (e.g., for codec detokenization) and needs to be restored for the
        next generation in a batch.
        """
        if not self.swapped_blocks:
            return

        backbone = self.model.backbone

        # Move non-swapped blocks (first N blocks) back to GPU
        for i in range(self.num_gpu_blocks):
            if i < len(backbone.layers):
                backbone.layers[i].to(self.device)

        # Move non-layer components back to GPU
        self.model.text_embeddings.to(self.device)
        self.model.audio_embeddings.to(self.device)
        self.model.unconditional_text_embedding.to(self.device)
        self.model.projection.to(self.device)
        self.model.codebook0_head.to(self.device)
        self.model.audio_head.data = self.model.audio_head.data.to(self.device)
        self.model.muq_linear.to(self.device)
        backbone.norm.to(self.device)
        self.model.decoder.to(self.device)

        # Swapped blocks should stay on CPU - move their params back to CPU
        # (they may have been moved to GPU by model.to(device) in _forward)
        for i in self.swapped_blocks:
            _params_to_device(backbone.layers[i], self.cpu_device, non_blocking=False)

        # Ensure all buffers are on GPU
        self.buffers_initialized = False
        self._ensure_all_buffers_on_gpu()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def get_memory_stats(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        return "CUDA not available"


def log(message: str):
    """Log a message to the console."""
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


# Maximum number of audio outputs in the UI
MAX_AUDIO_OUTPUTS = 12


def create_audio_outputs(file_paths: list, labels: list = None) -> list:
    """Create list of audio outputs for gr.Audio components.

    Returns a list of MAX_AUDIO_OUTPUTS tuples, where each tuple is either
    (filepath, label) or (None, None) for empty slots.
    """
    if labels is None:
        labels = [os.path.basename(p) for p in file_paths] if file_paths else []

    outputs = []
    for i in range(MAX_AUDIO_OUTPUTS):
        if i < len(file_paths):
            outputs.append(gr.update(value=file_paths[i], label=labels[i], visible=True))
        else:
            outputs.append(gr.update(value=None, visible=False))

    return outputs


def generate_music(
    lyrics: str,
    tags: str,
    negative_prompt: str,
    max_duration_seconds: float,
    temperature: float,
    topk: int,
    cfg_scale: float,
    model_path: str,
    model_version: str,
    num_gpu_blocks: int,
    model_dtype: str,
    batch_count: int,
    seed: int,
    output_folder: str,
    compile_model: bool = False,
    ref_audio: str = None,
    ref_strength: float = 0.7,
    num_steps: int = 10,
):
    """Generate music using HeartMuLa with batch support.

    Yields: (file_list, status_text)
    """
    global stop_event
    stop_event.clear()

    pipe = None
    block_swap_manager = None
    all_generated_music = []
    all_seeds = []
    batch_count = int(batch_count)
    seed = int(seed)

    def cleanup():
        """Clean up GPU memory."""
        nonlocal pipe, block_swap_manager
        if block_swap_manager is not None:
            block_swap_manager.cleanup()
        if pipe is not None:
            if hasattr(pipe, 'model') and pipe.model is not None:
                try:
                    pipe.model.reset_caches()
                except Exception:
                    pass
                pipe.model.to("cpu")
                del pipe.model
            if hasattr(pipe, 'audio_codec') and pipe.audio_codec is not None:
                pipe.audio_codec.to("cpu")
                del pipe.audio_codec
            del pipe

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            log(f"GPU Memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    try:
        # Validate inputs
        if not model_path or not os.path.exists(model_path):
            yield (*create_audio_outputs([]), f"Model path does not exist: {model_path}")
            return

        if not lyrics.strip():
            yield (*create_audio_outputs([]), "Please enter some lyrics.")
            return

        if not tags.strip():
            yield (*create_audio_outputs([]), "Please enter at least one tag (e.g., 'piano,happy').")
            return

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        log("=" * 50)
        log(f"Starting batch generation ({batch_count} songs)...")
        log(f"Tags: {tags}")
        if negative_prompt.strip():
            log(f"Negative prompt: {negative_prompt}")
        if ref_audio and os.path.isfile(ref_audio):
            log(f"Reference audio: {ref_audio} (strength={ref_strength}, steps={num_steps})")
        log(f"Max duration: {max_duration_seconds}s")
        log(f"Temperature: {temperature}, Top-K: {topk}, CFG Scale: {cfg_scale}")

        yield (*create_audio_outputs([]), "Loading model...")

        from heartlib import HeartMuLaGenPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Map dtype string to torch dtype
        dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        dtype = dtype_map.get(model_dtype, torch.bfloat16)
        if not torch.cuda.is_available():
            dtype = torch.float32

        log(f"Loading HeartMuLa-{model_version} from {model_path}...")
        log(f"Using device: {device}, dtype: {dtype}")

        # Get total blocks for this model version
        total_blocks = MODEL_LAYER_COUNTS.get(model_version, {}).get("backbone", 28)

        # Clamp num_gpu_blocks to total_blocks (0 means all blocks)
        effective_gpu_blocks = min(num_gpu_blocks, total_blocks) if num_gpu_blocks > 0 else total_blocks

        # Always use selective loading for better memory efficiency when CUDA available
        use_selective_loading = torch.cuda.is_available()

        # Load pipeline - skip automatic model move if using selective loading
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            version=model_version,
            skip_model_move=use_selective_loading,
        )

        # Set up selective loading for better memory management
        if use_selective_loading:
            log(f"Configuring selective loading: {effective_gpu_blocks} blocks on GPU...")

            # Move blocks to GPU one by one (more memory efficient than model.to())
            backbone = pipe.model.backbone
            for i, layer in enumerate(backbone.layers):
                if i < effective_gpu_blocks:
                    layer.to(device)

            # Move non-layer components to GPU
            pipe.model.text_embeddings.to(device)
            pipe.model.audio_embeddings.to(device)
            pipe.model.unconditional_text_embedding.to(device)
            pipe.model.projection.to(device)
            pipe.model.codebook0_head.to(device)
            pipe.model.audio_head.data = pipe.model.audio_head.data.to(device)
            pipe.model.muq_linear.to(device)
            pipe.model.backbone.norm.to(device)
            pipe.model.decoder.to(device)

            # Only set up block swapping if some blocks need to stay on CPU
            if effective_gpu_blocks < total_blocks:
                block_swap_manager = BlockSwapManager(pipe.model, effective_gpu_blocks, device)
                if block_swap_manager.setup():
                    blocks_swapped = total_blocks - effective_gpu_blocks
                    log(f"Block swapping ready: {effective_gpu_blocks} on GPU, {blocks_swapped} swap from CPU")
                    log(block_swap_manager.get_memory_stats())
                    # Prevent pipeline from moving entire model to GPU during generation
                    pipe._skip_auto_move = True
                else:
                    block_swap_manager = None
                    log("Block swapping setup failed, keeping partial GPU load")
            else:
                log(f"All {total_blocks} blocks on GPU (no swapping needed)")
                log(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
        else:
            log(f"No selective loading: all {total_blocks} blocks on GPU")
            pipe.model.to(device)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory after model load: {allocated:.2f}GB")

        # Optionally compile model for faster inference (reduces CPU overhead)
        if compile_model:
            try:
                log("Compiling model with CUDA graphs (first run will be slower)...")
                pipe.model.compile_model(mode="reduce-overhead")
            except (AssertionError, RuntimeError) as e:
                log(f"Warning: CUDA graph compilation failed ({e}), running without compilation")
                compile_model = False  # Disable for this run

        # Convert duration from seconds to milliseconds
        max_audio_length_ms = int(max_duration_seconds * 1000)

        # Batch generation loop
        for i in range(batch_count):
            if stop_event.is_set():
                log("Generation stopped by user.")
                labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
                yield (*create_audio_outputs(all_generated_music, labels), "Stopped by user.")
                return

            # Handle seed
            if seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = seed + i

            all_seeds.append(current_seed)

            # Set random seed for reproducibility
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)
            random.seed(current_seed)

            status_text = f"Processing {i+1}/{batch_count} (Seed: {current_seed})"
            labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
            yield (*create_audio_outputs(all_generated_music, labels), status_text)

            log(f"\n{'='*50}")
            log(f"Generating song {i+1}/{batch_count} (Seed: {current_seed})")

            # Create output path with timestamp and seed
            timestamp = int(time.time())
            output_filename = f"heartmula_{timestamp}_{current_seed}.mp3"
            output_path = os.path.join(output_folder, output_filename)

            log(f"Output path: {output_path}")
            log("Starting generation loop...")

            start_time = time.perf_counter()

            with torch.no_grad():
                inputs = {
                    "lyrics": lyrics,
                    "tags": tags,
                }
                if ref_audio and os.path.isfile(ref_audio):
                    inputs["ref_audio"] = ref_audio

                # Build metadata dictionary with all generation settings
                generation_metadata = {
                    "lyrics": lyrics,
                    "tags": tags,
                    "negative_prompt": negative_prompt if negative_prompt.strip() else "",
                    "max_duration": max_duration_seconds,
                    "temperature": temperature,
                    "topk": topk,
                    "cfg_scale": cfg_scale,
                    "model_version": model_version,
                    "model_path": model_path,
                    "model_dtype": model_dtype,
                    "num_gpu_blocks": num_gpu_blocks,
                    "seed": current_seed,
                    "ref_strength": ref_strength,
                    "num_steps": num_steps,
                    "compile_model": compile_model,
                    "generated_at": datetime.now().isoformat(),
                }

                try:
                    pipe(
                        inputs,
                        max_audio_length_ms=max_audio_length_ms,
                        save_path=output_path,
                        topk=topk,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        negative_prompt=negative_prompt if negative_prompt.strip() else None,
                        stop_check=stop_event.is_set,
                        ref_audio=ref_audio if ref_audio and os.path.isfile(ref_audio) else None,
                        ref_strength=ref_strength,
                        num_steps=num_steps,
                        metadata=generation_metadata,
                    )
                except AssertionError as e:
                    if "is_key_in_tls" in str(e) or "tree_manager" in str(e):
                        raise RuntimeError(
                            "CUDA graph state corrupted. This happens when running multiple generations "
                            "with 'Compile Model' enabled. Please disable 'Compile Model (CUDA Graphs)' "
                            "in the Model Settings, or restart the application."
                        ) from e
                    raise

            elapsed = time.perf_counter() - start_time
            log(f"Song {i+1} complete! ({elapsed:.1f}s)")

            # Add file path to list
            all_generated_music.append(output_path)

            status_text = f"Completed {i+1}/{batch_count}"
            labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
            yield (*create_audio_outputs(all_generated_music, labels), status_text)

            # Get memory info after generation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                log(f"GPU Memory after song {i+1}: {allocated:.2f}GB")

            # Restore block swapping state for next generation in batch
            if block_swap_manager is not None and i < batch_count - 1:
                block_swap_manager.restore_state()

        # Final status
        final_status = f"Completed {batch_count} song(s)!" if batch_count > 1 else "Generation complete!"
        log(f"\n{'='*50}")
        log(final_status)
        labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
        yield (*create_audio_outputs(all_generated_music, labels), final_status)

    except Exception as e:
        import traceback
        error_msg = f"Error during generation: {str(e)}"
        log(error_msg)
        log(traceback.format_exc())
        labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
        yield (*create_audio_outputs(all_generated_music, labels), error_msg)
    finally:
        cleanup()


# Build the Gradio interface
with gr.Blocks(
    theme=themes.Default(
        primary_hue=colors.Color(
            name="heart_purple",
            c50="#F5E6FF",
            c100="#E0BAFF",
            c200="#CA91FF",
            c300="#B469FF",
            c400="#9E40FF",
            c500="#8816FF",
            c600="#6B09D9",
            c700="#5003B3",
            c800="#38008C",
            c900="#220066",
            c950="#110033"
        )
    ),
    css="""
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #219651) !important;
    }
    """,
    title="HeartMuLa Music Generator"
) as demo:
    with gr.Tabs():
        with gr.Tab("Generation"):
            # Row 1: Main Inputs + Status
            with gr.Row():
                # Left column - Lyrics & Tags
                with gr.Column(scale=3):
                    with gr.Accordion("Lyrics", open=True):
                        with gr.Row():
                            lyrics_input = gr.Textbox(
                                label="main lyric prompt",
                                placeholder="[Verse]\nYour lyrics here...\n\n[Chorus]\nChorus lyrics...",
                                value=DEFAULT_LYRICS,
                                lines=12,
                            )
                    tags_input = gr.Textbox(
                        label="Style Tags",
                        placeholder="piano,happy,romantic",
                        value=DEFAULT_TAGS,
                        info="Comma-separated tags (e.g., piano,happy,romantic)"
                    )
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="noise,distortion,low quality",
                        value="",
                        info="Comma-separated tags to avoid (e.g., noise,distortion)"
                    )
                    with gr.Row():
                        batch_count = gr.Number(label="Batch Count", value=1, minimum=1, step=1, scale=1)
                        seed = gr.Number(label="Seed (-1 = random)", value=-1, scale=1)
                        random_seed_btn = gr.Button("🎲", scale=0, min_width=40)

                # Right column - Status
                with gr.Column(scale=1):
                    status_text = gr.Textbox(label="Status", interactive=False, value="Ready", lines=3)
                    with gr.Accordion("Reference Audio (img2img)", open=True):
                        ref_audio_input = gr.Audio(
                            label="Reference Audio (upload to influence generated sound)",
                            type="filepath",
                        )
                        ref_strength_slider = gr.Slider(
                            label="Strength (1.0 = ignore reference, 0.0 = pure reference)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.01,
                        )
                        num_steps_slider = gr.Slider(
                            label="Diffusion Steps (more = higher quality, slower)",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                        )

            # Row 2: Buttons
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Music", variant="primary", elem_classes="green-btn")
                stop_btn = gr.Button("⏹️ Stop", variant="stop")
                save_defaults_btn = gr.Button("💾 Save Defaults")
                load_defaults_btn = gr.Button("📂 Load Defaults")

            # Row 3: Parameters + Output
            with gr.Row():
                # Left column - Parameters
                with gr.Column():
                    with gr.Accordion("Model Settings", open=True):
                        with gr.Row():
                            model_path = gr.Textbox(
                                label="Model Path",
                                value="./ckpt",
                                info="Path to the HeartMuLa checkpoint directory"
                            )
                            model_version = gr.Dropdown(
                                label="Model Version",
                                choices=["3B", "7B"],
                                value="3B",
                                scale=1
                            )
                            model_dtype = gr.Dropdown(
                                label="Precision",
                                choices=["fp32", "bf16", "fp16"],
                                value="bf16",
                                scale=1
                            )
                        num_gpu_blocks = gr.Slider(
                            label="GPU Blocks",
                            minimum=0,
                            maximum=28,
                            value=14,
                            step=1,
                            info="Blocks on GPU (0 = all). Lower = less VRAM."
                        )
                        with gr.Row():
                            output_folder = gr.Textbox(
                                label="Output Folder",
                                value="./output",
                                info="Directory to save generated music"
                            )
                            compile_checkbox = gr.Checkbox(
                                label="Compile Model (CUDA Graphs)",
                                value=False,
                                info="Faster inference on Linux. First run compiles. Not supported on Windows."
                            )

                    with gr.Accordion("Generation Parameters", open=True):
                        max_duration = gr.Slider(
                            label="Max Duration (seconds)",
                            minimum=10,
                            maximum=240,
                            value=120,
                            step=10
                        )
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                scale=1
                            )
                            topk = gr.Slider(
                                label="Top-K",
                                minimum=1,
                                maximum=200,
                                value=50,
                                step=1,
                                scale=1
                            )
                        cfg_scale = gr.Slider(
                            label="CFG Scale (Guidance)",
                            minimum=1.0,
                            maximum=12.0,
                            value=1.5,
                            step=0.1,
                            info="Higher = stronger adherence to tags/lyrics"
                        )

                # Right column - Output
                with gr.Column():
                    audio_outputs = []
                    for i in range(MAX_AUDIO_OUTPUTS):
                        audio_outputs.append(gr.Audio(
                            label=f"Song {i+1}",
                            visible=False,
                            interactive=False
                        ))

        # Audio Info Tab
        with gr.Tab("Audio Info"):
            gr.Markdown("### Load Settings from MP3")
            gr.Markdown("Upload a HeartMuLa-generated MP3 to view and load its generation settings.")

            with gr.Row():
                with gr.Column(scale=1):
                    audio_info_input = gr.Audio(
                        label="Upload MP3 File",
                        type="filepath",
                    )
                    load_settings_btn = gr.Button("📥 Load Settings to Generation Tab", variant="primary")
                    audio_info_status = gr.Textbox(label="Status", interactive=False, value="", lines=2)

                with gr.Column(scale=2):
                    with gr.Accordion("Metadata Preview", open=True):
                        info_lyrics = gr.Textbox(label="Lyrics", lines=8, interactive=False)
                        info_tags = gr.Textbox(label="Tags", interactive=False)
                        info_negative_prompt = gr.Textbox(label="Negative Prompt", interactive=False)

                    with gr.Accordion("Generation Parameters", open=True):
                        with gr.Row():
                            info_seed = gr.Textbox(label="Seed", interactive=False)
                            info_max_duration = gr.Textbox(label="Max Duration (s)", interactive=False)
                            info_temperature = gr.Textbox(label="Temperature", interactive=False)
                        with gr.Row():
                            info_topk = gr.Textbox(label="Top-K", interactive=False)
                            info_cfg_scale = gr.Textbox(label="CFG Scale", interactive=False)
                            info_ref_strength = gr.Textbox(label="Ref Strength", interactive=False)

                    with gr.Accordion("Model Settings", open=True):
                        with gr.Row():
                            info_model_version = gr.Textbox(label="Model Version", interactive=False)
                            info_model_dtype = gr.Textbox(label="Precision", interactive=False)
                            info_num_gpu_blocks = gr.Textbox(label="GPU Blocks", interactive=False)
                        with gr.Row():
                            info_model_path = gr.Textbox(label="Model Path", interactive=False)
                            info_generated_at = gr.Textbox(label="Generated At", interactive=False)

    # Event handlers
    def update_blocks_slider(version):
        max_blocks = MODEL_LAYER_COUNTS.get(version, {}).get("backbone", 28)
        return gr.update(maximum=max_blocks, info=f"Blocks on GPU (0 = all). {version} has {max_blocks} blocks.")

    def randomize_seed():
        return -1

    model_version.change(
        fn=update_blocks_slider,
        inputs=[model_version],
        outputs=[num_gpu_blocks]
    )

    random_seed_btn.click(
        fn=randomize_seed,
        outputs=[seed]
    )

    generate_btn.click(
        fn=generate_music,
        inputs=[lyrics_input, tags_input, negative_prompt_input, max_duration, temperature, topk, cfg_scale,
                model_path, model_version, num_gpu_blocks, model_dtype,
                batch_count, seed, output_folder, compile_checkbox,
                ref_audio_input, ref_strength_slider, num_steps_slider],
        outputs=audio_outputs + [status_text]
    )

    stop_btn.click(
        fn=stop_generation,
        outputs=[status_text]
    )

    # Save/Load Defaults
    # Components to save (in order)
    defaults_components = [
        lyrics_input, tags_input, negative_prompt_input, batch_count, seed,
        model_path, model_version, model_dtype, num_gpu_blocks, output_folder,
        max_duration, temperature, topk, cfg_scale, compile_checkbox,
        ref_strength_slider, num_steps_slider
    ]

    # Keys for the defaults file (must match order of components)
    defaults_keys = [
        "lyrics", "tags", "negative_prompt", "batch_count", "seed",
        "model_path", "model_version", "model_dtype", "num_gpu_blocks", "output_folder",
        "max_duration", "temperature", "topk", "cfg_scale", "compile_model",
        "ref_strength", "num_steps"
    ]

    def save_defaults(*values):
        """Save current settings to defaults file."""
        settings = {}
        for i, key in enumerate(defaults_keys):
            settings[key] = values[i]
        try:
            with open(HEARTMULA_DEFAULTS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
            return "Defaults saved successfully."
        except Exception as e:
            return f"Error saving defaults: {e}"

    def load_defaults(request: gr.Request = None):
        """Load settings from defaults file."""
        if not os.path.exists(HEARTMULA_DEFAULTS_FILE):
            if request:
                return [gr.update()] * len(defaults_keys) + ["No defaults file found."]
            else:
                return [gr.update()] * len(defaults_keys) + [""]

        try:
            with open(HEARTMULA_DEFAULTS_FILE, 'r') as f:
                loaded = json.load(f)
        except Exception as e:
            return [gr.update()] * len(defaults_keys) + [f"Error loading defaults: {e}"]

        updates = []
        for key in defaults_keys:
            if key in loaded:
                updates.append(gr.update(value=loaded[key]))
            else:
                updates.append(gr.update())

        return updates + ["Defaults loaded successfully."]

    save_defaults_btn.click(
        fn=save_defaults,
        inputs=defaults_components,
        outputs=[status_text]
    )

    load_defaults_btn.click(
        fn=load_defaults,
        inputs=None,
        outputs=defaults_components + [status_text]
    )

    # Audio Info Tab event handlers
    # Store loaded metadata in a state variable
    loaded_metadata_state = gr.State(value=None)

    def load_mp3_metadata(audio_path):
        """Load and display metadata from an MP3 file."""
        if audio_path is None:
            return (
                None,  # metadata state
                "",    # status
                "",    # lyrics
                "",    # tags
                "",    # negative_prompt
                "",    # seed
                "",    # max_duration
                "",    # temperature
                "",    # topk
                "",    # cfg_scale
                "",    # ref_strength
                "",    # model_version
                "",    # model_dtype
                "",    # num_gpu_blocks
                "",    # model_path
                "",    # generated_at
            )

        if not audio_path.lower().endswith('.mp3'):
            return (
                None,
                "Please upload an MP3 file.",
                "", "", "", "", "", "", "", "", "", "", "", "", "", ""
            )

        try:
            from heartlib import HeartMuLaGenPipeline
            metadata = HeartMuLaGenPipeline.read_mp3_metadata(audio_path)

            if metadata is None:
                return (
                    None,
                    "No HeartMuLa metadata found in this MP3.",
                    "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                )

            return (
                metadata,  # state
                "Metadata loaded successfully!",
                metadata.get("lyrics", ""),
                metadata.get("tags", ""),
                metadata.get("negative_prompt", ""),
                str(metadata.get("seed", "")),
                str(metadata.get("max_duration", "")),
                str(metadata.get("temperature", "")),
                str(metadata.get("topk", "")),
                str(metadata.get("cfg_scale", "")),
                str(metadata.get("ref_strength", "")),
                metadata.get("model_version", ""),
                metadata.get("model_dtype", ""),
                str(metadata.get("num_gpu_blocks", "")),
                metadata.get("model_path", ""),
                metadata.get("generated_at", ""),
            )
        except Exception as e:
            return (
                None,
                f"Error reading metadata: {e}",
                "", "", "", "", "", "", "", "", "", "", "", "", "", ""
            )

    def send_settings_to_generation(metadata):
        """Send loaded metadata settings to the generation tab."""
        if metadata is None:
            return [gr.update()] * 16 + ["No metadata loaded. Please upload an MP3 first."]

        # Map metadata to generation tab components
        # Order matches defaults_components: lyrics, tags, negative_prompt, batch_count, seed,
        # model_path, model_version, model_dtype, num_gpu_blocks, output_folder,
        # max_duration, temperature, topk, cfg_scale, compile_model, ref_strength
        updates = [
            gr.update(value=metadata.get("lyrics", "")),
            gr.update(value=metadata.get("tags", "")),
            gr.update(value=metadata.get("negative_prompt", "")),
            gr.update(value=1),  # batch_count - reset to 1
            gr.update(value=metadata.get("seed", -1)),
            gr.update(value=metadata.get("model_path", "./ckpt")),
            gr.update(value=metadata.get("model_version", "3B")),
            gr.update(value=metadata.get("model_dtype", "bf16")),
            gr.update(value=metadata.get("num_gpu_blocks", 14)),
            gr.update(),  # output_folder - keep current
            gr.update(value=metadata.get("max_duration", 120)),
            gr.update(value=metadata.get("temperature", 1.0)),
            gr.update(value=metadata.get("topk", 50)),
            gr.update(value=metadata.get("cfg_scale", 1.5)),
            gr.update(value=metadata.get("compile_model", False)),
            gr.update(value=metadata.get("ref_strength", 0.7)),
        ]

        return updates + ["Settings loaded to Generation tab!"]

    # Info display components for Audio Info tab
    info_display_components = [
        info_lyrics, info_tags, info_negative_prompt,
        info_seed, info_max_duration, info_temperature,
        info_topk, info_cfg_scale, info_ref_strength,
        info_model_version, info_model_dtype, info_num_gpu_blocks,
        info_model_path, info_generated_at
    ]

    audio_info_input.change(
        fn=load_mp3_metadata,
        inputs=[audio_info_input],
        outputs=[loaded_metadata_state, audio_info_status] + info_display_components
    )

    load_settings_btn.click(
        fn=send_settings_to_generation,
        inputs=[loaded_metadata_state],
        outputs=defaults_components + [audio_info_status]
    )

    # Auto-load defaults on startup
    def initial_load_defaults():
        results = load_defaults(None)
        return results[:-1]  # Exclude status message

    demo.load(
        fn=initial_load_defaults,
        inputs=None,
        outputs=defaults_components
    )


def find_available_port(start_port: int = 7860, max_attempts: int = 100) -> int:
    """Find the first available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HeartMuLa Music Generation UI")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port. Auto-detects available port if not specified.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    args = parser.parse_args()

    # Determine port (auto-detect if not specified)
    if args.port is not None:
        port = args.port
    else:
        port = find_available_port(7860)

    print(f"Starting HeartMuLa Gradio UI on port {port}...")
    print(f"Open http://localhost:{port} in your browser")

    # Enable queue for generator/streaming support
    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        show_error=True
    )
