"""
HeartMuLa Gradio Web UI
A simple web interface for generating music with HeartMuLa.
"""

import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
import torch
import torchaudio
import tempfile
import os
import sys
import io
import threading
import random
import time
import json
import gc
import numpy as np
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from queue import Queue
from datetime import datetime
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
from mutagen.flac import FLAC
from omegaconf import OmegaConf

# Add SongGeneration to path
SONGGEN_PATH = os.path.join(os.path.dirname(__file__), "SongGeneration")
if SONGGEN_PATH not in sys.path:
    sys.path.insert(0, SONGGEN_PATH)

# Defaults file paths
HEARTMULA_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "heartmula_defaults.json")
SONGGEN_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "songgen_defaults.json")

# Project root and output directory for Gradio allowed_paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Model layer counts for different versions
MODEL_LAYER_COUNTS = {
    "3B": {"backbone": 28, "decoder": 3},
    "7B": {"backbone": 32, "decoder": 3},
}

# SongGeneration (LeVo) style types
SONGGEN_STYLE_TYPES = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

# SongGeneration model options
SONGGEN_MODELS = ['songgeneration_large', 'songgeneration_base', 'songgeneration_base_new', 'songgeneration_base_full']

# SongGeneration Description Builder options
SONGGEN_GENDERS = ['None', 'Male', 'Female']
SONGGEN_TIMBRES = ['None', 'Dark', 'Bright', 'Warm', 'Soft', 'Vocal', 'Varies']
SONGGEN_EMOTIONS = ['None', 'Sad', 'Emotional', 'Angry', 'Happy', 'Uplifting', 'Intense', 'Romantic', 'Melancholic']

# Global stop event for batch cancellation
stop_event = threading.Event()

# Global model cache to avoid reloading between generations
_cached_pipe = None
_cached_pipe_config = None  # Stores (model_path, model_version, model_dtype, num_gpu_blocks, compile_model)


def stop_generation():
    """Signal to stop the current generation."""
    stop_event.set()
    return "Stopping..."


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
    use_rl_models: bool = False,
    ref_audio_semantic: str = None,
    ref_audio_img2img: str = None,
    ref_strength: float = 0.5,
    normalize_loudness: bool = True,
    loudness_boost: float = 1.0,
    num_steps: int = 10,
    ref_audio_sec: float = 10.0,
):
    """Generate music using HeartMuLa with batch support.

    Yields: (file_list, status_text)
    """
    global stop_event
    stop_event.clear()

    pipe = None
    block_swap_enabled = False
    all_generated_music = []
    all_seeds = []
    batch_count = int(batch_count)
    seed = int(seed)

    def cleanup():
        """Clean up GPU memory."""
        nonlocal pipe
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

        # Reset dynamo/CUDA graphs state safely before garbage collection
        # This must be done in a try-except because CUDA graphs TLS may not be
        # initialized in Gradio's worker threads
        try:
            torch._dynamo.reset()
        except (AssertionError, RuntimeError):
            # TLS not initialized in this thread - safe to ignore
            pass

        # Reset CUDA graph trees separately - this is required to clear the
        # thread-local storage that causes AssertionError on subsequent runs
        try:
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees
            reset_cudagraph_trees()
        except (ImportError, AssertionError, RuntimeError):
            pass

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
        if ref_audio_semantic and os.path.isfile(ref_audio_semantic):
            log(f"Semantic reference: {ref_audio_semantic} (analyze {ref_audio_sec}s)")
        if ref_audio_img2img and os.path.isfile(ref_audio_img2img):
            log(f"Audio reference (img2img): {ref_audio_img2img} (strength={ref_strength}, steps={num_steps})")
        log(f"Max duration: {max_duration_seconds}s")
        log(f"Temperature: {temperature}, Top-K: {topk}, CFG Scale: {cfg_scale}")

        yield (*create_audio_outputs([]), "Loading model...")

        # Reset dynamo and CUDA graph trees state at the start to ensure clean TLS
        # in Gradio's worker thread
        try:
            torch._dynamo.reset()
        except (AssertionError, RuntimeError):
            pass
        try:
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees
            reset_cudagraph_trees()
        except (ImportError, AssertionError, RuntimeError):
            pass

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

        model_variant = "RL-20260123" if use_rl_models else "standard"
        log(f"Loading HeartMuLa-{model_version} ({model_variant}) from {model_path}...")
        log(f"Using device: {device}, dtype: {dtype}")

        # Get total blocks for this model version
        total_blocks = MODEL_LAYER_COUNTS.get(model_version, {}).get("backbone", 28)

        # Clamp num_gpu_blocks to total_blocks (0 means all blocks)
        effective_gpu_blocks = min(num_gpu_blocks, total_blocks) if num_gpu_blocks > 0 else total_blocks

        # Always use selective loading for better memory efficiency when CUDA available
        use_selective_loading = torch.cuda.is_available()

        # Determine if we should load MuQ-MuLan for reference audio conditioning
        # This uses semantic embeddings which is how the model was trained
        should_load_muq = ref_audio_semantic is not None and os.path.isfile(ref_audio_semantic)
        if should_load_muq:
            log("Semantic reference audio provided - loading MuQ-MuLan for semantic conditioning...")

        # Load pipeline - skip automatic model move if using selective loading
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            version=model_version,
            skip_model_move=use_selective_loading,
            load_muq_mulan=should_load_muq,
            use_rl_models=use_rl_models,
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
                blocks_to_swap = total_blocks - effective_gpu_blocks
                pipe.model.enable_block_swap(blocks_to_swap, device)
                pipe.model.prepare_block_swap_before_forward()
                block_swap_enabled = True
                log(f"Block swapping ready: {effective_gpu_blocks} on GPU, {blocks_to_swap} swap from CPU")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    log(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                # Prevent pipeline from moving entire model to GPU during generation
                pipe._skip_auto_move = True
            else:
                log(f"All {total_blocks} blocks on GPU (no swapping needed)")
                log(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
        else:
            log(f"No selective loading: all {total_blocks} blocks on GPU")
            pipe.model.to(device)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory after model load: {allocated:.2f}GB")

        # Optionally compile model for faster inference
        # Note: torch.compile now works with block swapping (WAN-style orchestration)
        if compile_model:
            try:
                log("Compiling model (first run will be slower)...")
                pipe.model.compile_model(mode="max-autotune-no-cudagraphs")
            except (AssertionError, RuntimeError) as e:
                log(f"Warning: Compilation failed ({e}), running without compilation")
                compile_model = False  # Disable for this run

        # Convert duration from seconds to milliseconds
        max_audio_length_ms = int(max_duration_seconds * 1000)

        # Batch generation loop
        for i in range(batch_count):
            if stop_event.is_set():
                log("Generation stopped by user.")
                cleanup()  # Explicitly free GPU memory before returning
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
                # Semantic reference for MuQ-MuLan style guidance
                if ref_audio_semantic and os.path.isfile(ref_audio_semantic):
                    inputs["ref_audio"] = ref_audio_semantic
                    inputs["muq_segment_sec"] = ref_audio_sec

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
                    "num_steps": num_steps,
                    "ref_audio_semantic": ref_audio_semantic if ref_audio_semantic and os.path.isfile(ref_audio_semantic) else "",
                    "ref_audio_img2img": ref_audio_img2img if ref_audio_img2img and os.path.isfile(ref_audio_img2img) else "",
                    "ref_strength": ref_strength,
                    "normalize_loudness": normalize_loudness,
                    "loudness_boost": loudness_boost,
                    "ref_audio_sec": ref_audio_sec if ref_audio_semantic and os.path.isfile(ref_audio_semantic) else 0,
                    "compile_model": compile_model,
                    "use_rl_models": use_rl_models,
                    "generated_at": datetime.now().isoformat(),
                }

                # Prepare pipeline kwargs
                pipe_kwargs = {
                    "max_audio_length_ms": max_audio_length_ms,
                    "save_path": output_path,
                    "topk": topk,
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "num_steps": num_steps,
                    "negative_prompt": negative_prompt if negative_prompt.strip() else None,
                    "metadata": generation_metadata,
                    "stop_check": lambda: stop_event.is_set(),
                }

                # Add img2img reference if provided
                if ref_audio_img2img and os.path.isfile(ref_audio_img2img):
                    pipe_kwargs["ref_audio"] = ref_audio_img2img
                    pipe_kwargs["ref_strength"] = ref_strength
                    pipe_kwargs["normalize_loudness"] = normalize_loudness
                    pipe_kwargs["loudness_boost"] = loudness_boost

                try:
                    pipe(
                        inputs,
                        **pipe_kwargs,
                    )
                except AssertionError as e:
                    if "is_key_in_tls" in str(e) or "tree_manager" in str(e):
                        raise RuntimeError(
                            "CUDA graph state corrupted. This happens when running multiple generations "
                            "with 'Compile Model' enabled. Please disable 'Compile Model (CUDA Graphs)' "
                            "in the Model Settings, or restart the application."
                        ) from e
                    raise

                # Check if stop was triggered during generation
                if stop_event.is_set():
                    log("Generation stopped by user.")
                    cleanup()  # Explicitly free GPU memory before returning
                    labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
                    yield (*create_audio_outputs(all_generated_music, labels), "Stopped by user.")
                    return

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
            if block_swap_enabled and i < batch_count - 1:
                pipe.model.prepare_block_swap_before_forward()

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


def generate_music_songgen(
    lyrics: str,
    tags: str,
    max_duration_seconds: float,
    temperature: float,
    topk: int,
    cfg_scale: float,
    songgen_model: str,
    songgen_ckpt_path: str,
    batch_count: int,
    seed: int,
    output_folder: str,
    use_flash_attn: bool = False,
    gpu_blocks: int = 0,
    generate_type: str = "mixed",
    style_type: str = "Auto",
    ref_audio_path: str = None,
    compile_model: bool = False,
    top_p: float = 0.0,
    gender: str = "None",
    timbre: str = "None",
    emotion: str = "None",
    bpm: int = 0,
):
    """Generate music using SongGeneration (LeVo) backend.

    Args:
        gpu_blocks: Number of LM layers to keep on GPU (0=all layers on GPU).
                   Lower values = less VRAM but slower. Large model has 32 layers.

    Yields: (file_list, status_text)
    """
    global stop_event
    stop_event.clear()

    all_generated_music = []
    all_seeds = []
    batch_count = int(batch_count)
    seed = int(seed)

    # Change to SongGeneration directory for imports
    original_cwd = os.getcwd()

    def cleanup():
        """Clean up GPU memory."""
        os.chdir(original_cwd)

        # Reset dynamo/CUDA graphs state
        try:
            torch._dynamo.reset()
        except (AssertionError, RuntimeError):
            pass
        try:
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees
            reset_cudagraph_trees()
        except (ImportError, AssertionError, RuntimeError):
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            log(f"GPU Memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    try:
        # Resolve paths before changing directory
        ckpt_path = os.path.abspath(os.path.join(songgen_ckpt_path, songgen_model))
        output_folder = os.path.abspath(output_folder)
        if ref_audio_path:
            ref_audio_path = os.path.abspath(ref_audio_path)

        # Validate inputs
        if not os.path.exists(ckpt_path):
            yield (*create_audio_outputs([]), f"Checkpoint path does not exist: {ckpt_path}")
            return

        if not lyrics.strip():
            yield (*create_audio_outputs([]), "Please enter some lyrics.")
            return

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Handle dropdown values - convert from index to string if needed
        GENERATE_TYPES = ["mixed", "vocal", "bgm", "separate"]

        if isinstance(style_type, int):
            if 0 <= style_type < len(SONGGEN_STYLE_TYPES):
                style_type = SONGGEN_STYLE_TYPES[style_type]
            else:
                style_type = "Auto"
        elif style_type not in SONGGEN_STYLE_TYPES:
            style_type = "Auto"

        if isinstance(generate_type, int):
            if 0 <= generate_type < len(GENERATE_TYPES):
                generate_type = GENERATE_TYPES[generate_type]
            else:
                generate_type = "mixed"
        elif generate_type not in GENERATE_TYPES:
            generate_type = "mixed"

        if isinstance(songgen_model, int):
            if 0 <= songgen_model < len(SONGGEN_MODELS):
                songgen_model = SONGGEN_MODELS[songgen_model]
            else:
                songgen_model = SONGGEN_MODELS[0]

        # Convert gpu_blocks to int
        gpu_blocks = int(gpu_blocks)

        # Build preview of description for logging
        log_desc_parts = []
        if gender != "None":
            log_desc_parts.append(gender.lower())
        if timbre != "None":
            log_desc_parts.append(timbre.lower())
        if tags.strip():
            log_desc_parts.append(tags.strip())
        if emotion != "None":
            log_desc_parts.append(emotion.lower())
        if bpm > 0:
            log_desc_parts.append(f"the bpm is {bpm}")
        log_description = ", ".join(log_desc_parts) if log_desc_parts else "(none)"

        log("=" * 50)
        log(f"Starting SongGeneration batch ({batch_count} songs)...")
        log(f"Model: {songgen_model}")
        log(f"Style: {style_type}")
        log(f"Description: {log_description}")
        if ref_audio_path and os.path.isfile(ref_audio_path):
            log(f"Reference audio: {ref_audio_path}")
        log(f"Generate type: {generate_type}")
        log(f"Temperature: {temperature}, Top-K: {topk}, Top-P: {top_p}, CFG Scale: {cfg_scale}")

        yield (*create_audio_outputs([]), "Loading SongGeneration model...")

        # Change to SongGeneration directory and ensure it's in sys.path
        os.chdir(SONGGEN_PATH)
        if SONGGEN_PATH not in sys.path:
            sys.path.insert(0, SONGGEN_PATH)
        # Flow1dVAE has its own internal packages (tools, libs, models_gpt, etc.)
        flow1dvae_path = os.path.join(SONGGEN_PATH, "codeclm", "tokenizer", "Flow1dVAE")
        if flow1dvae_path not in sys.path:
            sys.path.insert(0, flow1dvae_path)

        # Register OmegaConf resolvers
        try:
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        except Exception:
            pass
        try:
            OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        except Exception:
            pass
        try:
            OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
        except Exception:
            pass
        try:
            OmegaConf.register_new_resolver("get_fname", lambda: "songgen")
        except Exception:
            pass

        # Import SongGeneration modules
        from codeclm.models import builders
        from codeclm.models import CodecLM

        # Load config
        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        model_path = os.path.join(ckpt_path, 'model.pt')
        cfg = OmegaConf.load(cfg_path)
        cfg.lm.use_flash_attn_2 = use_flash_attn
        cfg.mode = 'inference'
        config_max_dur = cfg.max_dur
        sample_rate = cfg.sample_rate

        # Use user's max_duration, capped at config maximum
        max_dur = min(max_duration_seconds, config_max_dur)

        log(f"use_flash_attn: {use_flash_attn}")
        log(f"Max duration: {max_dur}s (config max: {config_max_dur}s), Sample rate: {sample_rate}")

        # Check if we need separator for reference audio
        use_ref_audio = ref_audio_path is not None and os.path.isfile(ref_audio_path)

        # Load auto prompts
        auto_prompt = torch.load(os.path.join(SONGGEN_PATH, 'tools/new_prompt.pt'), map_location='cpu')

        # Determine if we should use layer-level offloading
        # gpu_blocks is the offload depth: 0 = all on GPU, 4+ = offload at that depth
        use_offload = gpu_blocks > 0
        offload_profiler = None

        if use_offload:
            log(f"Layer offloading enabled: depth={gpu_blocks} (lower=more VRAM, higher=less VRAM)")
        else:
            log(f"All LM layers on GPU (no offloading)")

        # When using offloading, we need to be careful about GPU memory
        # Process ref audio first (if any), delete tokenizers, then load LM
        ref_pmt_wav = None
        ref_vocal_wav = None
        ref_bgm_wav = None
        encoded_pmt = None
        encoded_vocal = None
        encoded_bgm = None

        if use_ref_audio:
            yield (*create_audio_outputs([]), "Processing reference audio...")
            from generate import Separator

            # Load and run separator
            separator = Separator(
                dm_model_path=os.path.join(SONGGEN_PATH, 'third_party/demucs/ckpt/htdemucs.pth'),
                dm_config_path=os.path.join(SONGGEN_PATH, 'third_party/demucs/ckpt/htdemucs.yaml')
            )
            ref_pmt_wav, ref_vocal_wav, ref_bgm_wav = separator.run(ref_audio_path)

            # Load audio tokenizer, encode, then delete
            audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()
            pmt_wav = ref_pmt_wav.cuda()
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            pmt_wav = torch.stack(list(pmt_wav), dim=0)
            with torch.no_grad():
                encoded_pmt, _ = audio_tokenizer.encode(pmt_wav)
            del audio_tokenizer
            del separator
            gc.collect()
            torch.cuda.empty_cache()

            # Load separate tokenizer, encode vocals/bgm, then delete
            seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
            seperate_tokenizer = seperate_tokenizer.eval().cuda()
            vocal_wav = ref_vocal_wav.cuda()
            bgm_wav = ref_bgm_wav.cuda()
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            vocal_wav = torch.stack(list(vocal_wav), dim=0)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            bgm_wav = torch.stack(list(bgm_wav), dim=0)
            with torch.no_grad():
                encoded_vocal, encoded_bgm = seperate_tokenizer.encode(vocal_wav, bgm_wav)
            del seperate_tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                log(f"GPU Memory after ref audio processing: {allocated:.2f}GB")

        # Load LM model
        yield (*create_audio_outputs([]), "Loading language model...")
        audiolm = builders.get_lm_model(cfg)
        checkpoint = torch.load(model_path, map_location='cpu')
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        del checkpoint  # Free memory
        gc.collect()
        audiolm = audiolm.eval()

        # Use layer-level offloading if enabled
        if use_offload:
            from codeclm.utils.offload_profiler import OffloadProfiler
            log(f"Setting up layer offloading with depth={gpu_blocks}...")

            # Create offload profiler
            offload_profiler = OffloadProfiler(
                device_index=0,
                cpu_mem_gb=-1,  # No limit
                pre_copy_step=1,
                clean_cache_after_forward=False,
                debug=False
            )

            # Offload layers - depth controls granularity of offloading
            # Depth 4 is recommended (offloads at sublayer level like self_attn, mlp)
            offload_layer_dict = {
                'transformer': gpu_blocks,   # Main 36-layer transformer
                'transformer2': gpu_blocks   # Sub 12-layer transformer
            }

            offload_profiler.offload_layer(
                module=audiolm,
                offload_layer_dict=offload_layer_dict,
                ignore_layer_list=[],
                dtype=torch.float16
            )

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                log(f"GPU Memory after offload setup: {allocated:.2f}GB")
        else:
            audiolm = audiolm.cuda().to(torch.float16)

        # Optionally compile model for faster inference
        if compile_model and not use_offload:
            try:
                log("Compiling SongGeneration model (first run will be slower)...")
                # Use default mode - safest for autoregressive generation with
                # changing sequence lengths. Avoids constant autotune overhead.
                audiolm.transformer = torch.compile(
                    audiolm.transformer,
                    dynamic=True
                )
                audiolm.transformer2 = torch.compile(
                    audiolm.transformer2,
                    dynamic=True
                )
                log("Model compiled successfully")
            except (AssertionError, RuntimeError) as e:
                log(f"Warning: Compilation failed ({e}), running without compilation")
        elif compile_model and use_offload:
            log("Warning: torch.compile disabled - incompatible with layer offloading")

        # Create model wrapper (without decoder for now - will load later for decoding)
        model = CodecLM(
            name="tmp",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=max_dur,
            seperate_tokenizer=None,  # Will load later for decoding
        )

        # Set generation parameters
        model.set_generation_params(
            duration=max_dur,
            extend_stride=5,
            temperature=temperature,
            cfg_coef=cfg_scale,
            top_k=topk,
            top_p=top_p,
            record_tokens=True,
            record_window=50
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory after model load: {allocated:.2f}GB")

        # Batch generation loop
        for i in range(batch_count):
            if stop_event.is_set():
                log("Generation stopped by user.")
                cleanup()
                labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
                yield (*create_audio_outputs(all_generated_music, labels), "Stopped by user.")
                return

            # Handle seed
            if seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = seed + i

            all_seeds.append(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)

            status_text = f"Processing {i+1}/{batch_count} (Seed: {current_seed})"
            labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
            yield (*create_audio_outputs(all_generated_music, labels), status_text)

            log(f"\n{'='*50}")
            log(f"Generating song {i+1}/{batch_count} (Seed: {current_seed})")

            # Create output path
            timestamp = int(time.time())
            output_filename = f"songgen_{timestamp}_{current_seed}.flac"
            output_path = os.path.join(output_folder, output_filename)

            log(f"Output path: {output_path}")

            start_time = time.perf_counter()

            # Prepare input - use pre-encoded tokens or auto prompts
            if use_ref_audio:
                # Use pre-encoded reference audio tokens (encoded earlier, before LM load)
                pmt_wav = encoded_pmt
                vocal_wav_enc = encoded_vocal
                bgm_wav_enc = encoded_bgm
                melody_is_wav = False
            else:
                # Use auto prompt
                # Get style prompts - structure is dict of lists of tensors
                style_prompts = auto_prompt[style_type]

                # Flatten if it's a dict (each value is a list of tensors)
                if isinstance(style_prompts, dict):
                    # Flatten: collect all tensors from all sub-lists
                    all_prompts = []
                    for v in style_prompts.values():
                        if isinstance(v, (list, tuple)):
                            all_prompts.extend(v)
                        else:
                            all_prompts.append(v)
                    style_prompts = all_prompts

                # Select random prompt from the flattened list
                idx = np.random.randint(0, len(style_prompts))
                prompt_token = style_prompts[idx]

                # Ensure it's a tensor
                if not isinstance(prompt_token, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(prompt_token)}")

                pmt_wav = prompt_token[:, [0], :]
                vocal_wav_enc = prompt_token[:, [1], :]
                bgm_wav_enc = prompt_token[:, [2], :]
                melody_is_wav = False

            # Build description from dropdowns + manual tags
            description_parts = []
            if gender != "None":
                description_parts.append(gender.lower())
            if timbre != "None":
                description_parts.append(timbre.lower())
            if tags.strip():
                description_parts.append(tags.strip())
            if emotion != "None":
                description_parts.append(emotion.lower())
            if bpm > 0:
                description_parts.append(f"the bpm is {bpm}")
            final_description = ", ".join(description_parts) if description_parts else None

            # Generate tokens with LM
            generate_inp = {
                'lyrics': [lyrics.replace("  ", " ")],
                'descriptions': [final_description],
                'melody_wavs': pmt_wav,
                'vocal_wavs': vocal_wav_enc,
                'bgm_wavs': bgm_wav_enc,
                'melody_is_wav': melody_is_wav,
            }

            # Reset offload profiler cache before generation if using offloading
            if use_offload and offload_profiler is not None:
                offload_profiler.reset_empty_cache_mem_line()

            # Generate tokens with LM - model.generate() is now a generator
            # that yields progress updates and final result
            tokens = None
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    for item in model.generate(**generate_inp, return_tokens=True):
                        if item[0] == "progress":
                            _, step, total = item
                            progress_status = f"Generating {i+1}/{batch_count}: step {step}/{total}"
                            yield (*create_audio_outputs(all_generated_music, labels), progress_status)
                        else:
                            # Final result
                            tokens = item[1]
                            break

            mid_time = time.perf_counter()
            log(f"LM generation took {mid_time - start_time:.1f}s")

            # Load tokenizer for decoding if needed (only on first batch item)
            if model.seperate_tokenizer is None:
                yield (*create_audio_outputs(all_generated_music, [f"Song {j+1}" for j in range(len(all_generated_music))]), "Loading decoder...")
                # Clear cache before loading tokenizer
                gc.collect()
                torch.cuda.empty_cache()

                decode_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
                decode_tokenizer = decode_tokenizer.eval().cuda()
                model.seperate_tokenizer = decode_tokenizer

            # Generate audio from tokens
            with torch.no_grad():
                if use_ref_audio:
                    wav_out = model.generate_audio(tokens, ref_pmt_wav, ref_vocal_wav, ref_bgm_wav, chunked=True, gen_type=generate_type)
                else:
                    wav_out = model.generate_audio(tokens, chunked=True, gen_type=generate_type)

            end_time = time.perf_counter()
            log(f"Diffusion/decoding took {end_time - mid_time:.1f}s")

            # Save output
            torchaudio.save(output_path, wav_out[0].cpu().float(), sample_rate)

            # Build and write metadata
            songgen_metadata = {
                "lyrics": lyrics,
                "tags": tags,
                "description": final_description,
                "gender": gender,
                "timbre": timbre,
                "emotion": emotion,
                "bpm": bpm,
                "max_duration": max_duration_seconds,
                "temperature": temperature,
                "topk": topk,
                "top_p": top_p,
                "cfg_scale": cfg_scale,
                "seed": current_seed,
                "songgen_model": songgen_model,
                "songgen_ckpt_path": songgen_ckpt_path,
                "style_type": style_type,
                "generate_type": generate_type,
                "use_flash_attn": use_flash_attn,
                "gpu_blocks": gpu_blocks,
                "compile_model": compile_model,
                "ref_audio_path": ref_audio_path if use_ref_audio else None,
                "generated_at": datetime.now().isoformat(),
                "generation_time_seconds": end_time - start_time,
            }
            write_flac_metadata(output_path, songgen_metadata)

            elapsed = time.perf_counter() - start_time
            log(f"Song {i+1} complete! Total: {elapsed:.1f}s")

            all_generated_music.append(output_path)

            status_text = f"Completed {i+1}/{batch_count}"
            labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
            yield (*create_audio_outputs(all_generated_music, labels), status_text)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                log(f"GPU Memory after song {i+1}: {allocated:.2f}GB")

        # Clean up offload profiler
        if offload_profiler is not None:
            offload_profiler.stop()
            del offload_profiler
            offload_profiler = None

        # Final status
        final_status = f"Completed {batch_count} song(s)!" if batch_count > 1 else "Generation complete!"
        log(f"\n{'='*50}")
        log(final_status)
        labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
        yield (*create_audio_outputs(all_generated_music, labels), final_status)

    except Exception as e:
        import traceback
        error_msg = f"Error during SongGeneration: {str(e)}"
        log(error_msg)
        log(traceback.format_exc())
        # Clean up offload profiler on error
        try:
            if offload_profiler is not None:
                offload_profiler.stop()
        except:
            pass
        labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
        yield (*create_audio_outputs(all_generated_music, labels), error_msg)
    finally:
        cleanup()


def write_flac_metadata(flac_path, metadata):
    """Write metadata to a FLAC file using Vorbis comments."""
    try:
        audio = FLAC(flac_path)
        # Store metadata as JSON in a custom tag
        audio["SONGGEN_METADATA"] = json.dumps(metadata)
        # Also store some key fields as standard tags for easy viewing
        if "lyrics" in metadata:
            audio["LYRICS"] = metadata["lyrics"][:500] if len(metadata.get("lyrics", "")) > 500 else metadata.get("lyrics", "")
        if "tags" in metadata:
            audio["DESCRIPTION"] = metadata.get("tags", "")
        if "seed" in metadata:
            audio["COMMENT"] = f"Seed: {metadata['seed']}"
        audio.save()
    except Exception as e:
        log(f"Warning: Could not write FLAC metadata: {e}")


def read_flac_metadata(flac_path):
    """Read SongGen metadata from a FLAC file."""
    try:
        audio = FLAC(flac_path)
        if "SONGGEN_METADATA" in audio:
            return json.loads(audio["SONGGEN_METADATA"][0])
    except Exception as e:
        log(f"Warning: Could not read FLAC metadata: {e}")
    return None


# Post-processing functions
def save_audio_as_mp3(data, sr, is_stereo, metadata=None):
    """Save audio data as MP3 file with metadata and return the path."""
    # Save as temporary WAV first
    temp_wav = tempfile.mktemp(suffix=".wav")
    if not is_stereo:
        data = data.flatten()
    sf.write(temp_wav, data, sr)

    # Convert to MP3
    output_path = tempfile.mktemp(suffix=".mp3")
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_path, format="mp3", bitrate="192k")

    # Clean up temp WAV
    os.remove(temp_wav)

    # Write metadata if provided
    if metadata:
        try:
            try:
                tags = ID3(output_path)
            except ID3NoHeaderError:
                tags = ID3()

            # Store HeartMuLa metadata as JSON in TXXX frame
            tags.add(TXXX(encoding=3, desc="HEARTMULA_METADATA", text=json.dumps(metadata)))
            tags.save(output_path)
        except Exception as e:
            log(f"Warning: Could not write metadata: {e}")

    return output_path


def read_audio_metadata(audio_path):
    """Read HeartMuLa metadata from an audio file."""
    if not audio_path or not os.path.exists(audio_path):
        return None

    # Try reading from MP3 ID3 tags (no network required)
    if audio_path.lower().endswith('.mp3'):
        try:
            tags = ID3(audio_path)
            for frame in tags.values():
                if isinstance(frame, TXXX) and frame.desc == "HEARTMULA_METADATA":
                    return json.loads(frame.text[0])
        except ID3NoHeaderError:
            pass
        except Exception:
            pass

    return None


def load_audio_for_processing(audio_path):
    """Load audio file and return sample rate, data, and metadata."""
    if audio_path is None:
        return None, None, "No audio file provided."

    try:
        data, sr = sf.read(audio_path)
        # Convert to mono if stereo for simpler processing, then back to stereo
        if len(data.shape) > 1:
            is_stereo = True
        else:
            is_stereo = False
            data = data.reshape(-1, 1)

        # Extract metadata
        metadata = read_audio_metadata(audio_path)

        duration = len(data) / sr
        metadata_status = " (with metadata)" if metadata else ""
        return (sr, data, is_stereo, metadata), None, f"Loaded: {os.path.basename(audio_path)} ({duration:.2f}s, {sr}Hz){metadata_status}"
    except Exception as e:
        return None, None, f"Error loading audio: {e}"


def trim_audio(audio_state, start_time, end_time):
    """Trim audio between two time points."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    duration = len(data) / sr

    # Validate times
    if start_time < 0:
        start_time = 0
    if end_time > duration:
        end_time = duration
    if start_time >= end_time:
        return None, f"Invalid time range: start ({start_time:.2f}s) must be less than end ({end_time:.2f}s)"

    # Convert times to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Trim
    trimmed_data = data[start_sample:end_sample]

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(trimmed_data, sr, is_stereo, metadata)

    new_duration = len(trimmed_data) / sr
    return output_path, f"Trimmed: {start_time:.2f}s to {end_time:.2f}s (new duration: {new_duration:.2f}s)"


def adjust_loudness(audio_state, gain_db):
    """Adjust audio loudness by a gain in dB."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    # Convert dB to linear gain
    linear_gain = 10 ** (gain_db / 20)

    # Apply gain
    adjusted_data = data * linear_gain

    # Clip to prevent distortion
    adjusted_data = np.clip(adjusted_data, -1.0, 1.0)

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(adjusted_data, sr, is_stereo, metadata)

    return output_path, f"Loudness adjusted by {gain_db:+.1f} dB"


def apply_bass_boost(audio_state, boost_db, cutoff_freq):
    """Apply bass boost using a low-shelf filter."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    # Measure original RMS loudness
    original_rms = np.sqrt(np.mean(data ** 2))

    # Design a low-shelf filter for bass boost
    # Using a simple approach: boost frequencies below cutoff
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    # Create a low-pass filter for the bass frequencies
    b_low, a_low = signal.butter(2, normalized_cutoff, btype='low')

    # Convert dB to linear gain
    bass_gain = 10 ** (boost_db / 20)

    # Process each channel
    result_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        # Extract bass frequencies
        bass = signal.filtfilt(b_low, a_low, channel)
        # Boost bass and add back
        result_data[:, ch] = channel + bass * (bass_gain - 1)

    # Restore original loudness level
    new_rms = np.sqrt(np.mean(result_data ** 2))
    if new_rms > 0:
        result_data = result_data * (original_rms / new_rms)

    # Soft clip to prevent harsh distortion while preserving loudness
    result_data = np.tanh(result_data)

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)

    return output_path, f"Bass boost: {boost_db:+.1f} dB below {cutoff_freq} Hz"


def apply_treble_boost(audio_state, boost_db, cutoff_freq):
    """Apply treble boost using a high-shelf filter."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    # Measure original RMS loudness
    original_rms = np.sqrt(np.mean(data ** 2))

    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    # Create a high-pass filter for treble frequencies
    b_high, a_high = signal.butter(2, normalized_cutoff, btype='high')

    # Convert dB to linear gain
    treble_gain = 10 ** (boost_db / 20)

    # Process each channel
    result_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        # Extract treble frequencies
        treble = signal.filtfilt(b_high, a_high, channel)
        # Boost treble and add back
        result_data[:, ch] = channel + treble * (treble_gain - 1)

    # Restore original loudness level
    new_rms = np.sqrt(np.mean(result_data ** 2))
    if new_rms > 0:
        result_data = result_data * (original_rms / new_rms)

    # Soft clip to prevent harsh distortion
    result_data = np.tanh(result_data)

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)

    return output_path, f"Treble boost: {boost_db:+.1f} dB above {cutoff_freq} Hz"


def normalize_audio(audio_state, target_db):
    """Normalize audio to a target peak level in dB."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    # Find current peak
    current_peak = np.max(np.abs(data))
    if current_peak == 0:
        return None, "Audio is silent, cannot normalize."

    current_db = 20 * np.log10(current_peak)

    # Calculate required gain
    gain_db = target_db - current_db
    linear_gain = 10 ** (gain_db / 20)

    # Apply gain
    normalized_data = data * linear_gain

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(normalized_data, sr, is_stereo, metadata)

    return output_path, f"Normalized to {target_db:.1f} dB (gain: {gain_db:+.1f} dB)"


def apply_fade_in(audio_state, fade_duration):
    """Apply fade-in effect."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    fade_samples = int(fade_duration * sr)
    if fade_samples > len(data):
        fade_samples = len(data)

    # Create fade curve
    fade_curve = np.linspace(0, 1, fade_samples)

    # Apply fade to each channel
    result_data = data.copy()
    for ch in range(data.shape[1]):
        result_data[:fade_samples, ch] = data[:fade_samples, ch] * fade_curve

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)

    return output_path, f"Fade-in applied: {fade_duration:.2f}s"


def apply_fade_out(audio_state, fade_duration):
    """Apply fade-out effect."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    fade_samples = int(fade_duration * sr)
    if fade_samples > len(data):
        fade_samples = len(data)

    # Create fade curve
    fade_curve = np.linspace(1, 0, fade_samples)

    # Apply fade to each channel
    result_data = data.copy()
    for ch in range(data.shape[1]):
        result_data[-fade_samples:, ch] = data[-fade_samples:, ch] * fade_curve

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)

    return output_path, f"Fade-out applied: {fade_duration:.2f}s"


def change_speed(audio_state, speed_factor):
    """Change audio playback speed (affects pitch)."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state

    if speed_factor <= 0:
        return None, "Speed factor must be positive."

    # Resample to change speed
    new_length = int(len(data) / speed_factor)

    result_data = np.zeros((new_length, data.shape[1]))
    for ch in range(data.shape[1]):
        result_data[:, ch] = signal.resample(data[:, ch], new_length)

    # Save as MP3 with metadata
    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)

    new_duration = new_length / sr
    return output_path, f"Speed changed to {speed_factor:.2f}x (new duration: {new_duration:.2f}s)"


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
            # Backend selector at the top
            with gr.Row():
                backend_selector = gr.Radio(
                    label="Backend",
                    choices=["HeartMuLa", "SongGeneration (LeVo)"],
                    value="HeartMuLa",
                    info="Select the music generation backend"
                )

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
                        label="Style Tags / Description",
                        placeholder="piano,happy,romantic",
                        value=DEFAULT_TAGS,
                        info="Comma-separated tags (HeartMuLa) or description (SongGeneration)"
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
                    with gr.Accordion("Reference Audio", open=True):
                        gr.Markdown("**Semantic Reference** - *Transfers high-level style (genre, mood) via MuQ-MuLan*")
                        ref_audio_semantic = gr.Audio(
                            label="Semantic Reference (style/genre)",
                            type="filepath",
                        )
                        ref_audio_sec_slider = gr.Slider(
                            label="Semantic analysis length (seconds)",
                            minimum=10,
                            maximum=120,
                            value=30,
                            step=5,
                        )
                        gr.Markdown("**Audio Reference (img2img)** - *Transfers timbre, rhythm, texture directly*")
                        ref_audio_img2img = gr.Audio(
                            label="Audio Reference (timbre/rhythm)",
                            type="filepath",
                        )
                        ref_strength_slider = gr.Slider(
                            label="Reference Strength (0=pure ref, 1=ignore ref)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                        )
                        normalize_loudness_checkbox = gr.Checkbox(
                            label="Normalize Loudness",
                            value=True,
                            info="Match output loudness to reference audio"
                        )
                        loudness_boost_slider = gr.Slider(
                            label="Loudness Boost",
                            minimum=0.1,
                            maximum=10.0,
                            value=1.0,
                            step=0.1,
                            info="Multiply output volume (1.0 = no change)"
                        )
                        num_steps_slider = gr.Slider(
                            label="Flow Matching Steps (more = higher quality, slower)",
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
                    # HeartMuLa Model Settings
                    with gr.Accordion("HeartMuLa Settings", open=True, visible=True) as heartmula_settings:
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
                            compile_checkbox = gr.Checkbox(
                                label="Compile Model (CUDA Graphs)",
                                value=False,
                                info="Faster inference. First run compiles and will be slower."
                            )
                            use_rl_models_checkbox = gr.Checkbox(
                                label="Use 20260123 Models",
                                value=False,
                                info="Use HeartCodec-oss-20260123 and HeartMuLa-RL-oss-3B-20260123"
                            )

                    # SongGeneration (LeVo) Settings
                    with gr.Accordion("SongGeneration (LeVo) Settings", open=True, visible=False) as songgen_settings:
                        with gr.Row():
                            songgen_ckpt_path = gr.Textbox(
                                label="Checkpoint Path",
                                value="./SongGeneration/ckpt",
                                info="Path to SongGeneration ckpt directory"
                            )
                            songgen_model = gr.Dropdown(
                                label="Model",
                                choices=SONGGEN_MODELS,
                                value="songgeneration_large",
                                scale=1
                            )
                        with gr.Row():
                            songgen_style = gr.Dropdown(
                                label="Style Type",
                                choices=SONGGEN_STYLE_TYPES,
                                value="Auto",
                                info="Auto-prompt style (ignored if reference audio provided)"
                            )
                            songgen_generate_type = gr.Dropdown(
                                label="Generate Type",
                                choices=["mixed", "vocal", "bgm", "separate"],
                                value="mixed",
                                info="Output type: mixed, vocal only, bgm only, or separate tracks"
                            )
                        with gr.Row():
                            songgen_flash_attn = gr.Checkbox(
                                label="Flash Attention",
                                value=False,
                                info="Use Flash Attention 2 (requires compatible GPU)"
                            )
                            songgen_gpu_blocks = gr.Slider(
                                label="Offload Depth",
                                minimum=0,
                                maximum=8,
                                value=0,
                                step=1,
                                info="Layer offload depth: 0=all on GPU (~20GB), 4=recommended (~8GB), 6-8=aggressive (~5GB)"
                            )
                        songgen_compile = gr.Checkbox(
                            label="Compile Model (torch.compile)",
                            value=False,
                            info="Faster inference after first run. May not work with layer offloading."
                        )

                        with gr.Accordion("Description Builder", open=True):
                            gr.Markdown("*Build style description from presets (combines with manual tags)*")
                            with gr.Row():
                                songgen_gender = gr.Dropdown(
                                    label="Gender",
                                    choices=SONGGEN_GENDERS,
                                    value="None",
                                    scale=1
                                )
                                songgen_timbre = gr.Dropdown(
                                    label="Timbre",
                                    choices=SONGGEN_TIMBRES,
                                    value="None",
                                    scale=1
                                )
                                songgen_emotion = gr.Dropdown(
                                    label="Emotion",
                                    choices=SONGGEN_EMOTIONS,
                                    value="None",
                                    scale=1
                                )
                            songgen_bpm = gr.Number(
                                label="BPM (0 = auto)",
                                value=0,
                                minimum=0,
                                maximum=200,
                                step=1,
                                info="Target beats per minute (leave 0 for automatic)"
                            )

                        with gr.Accordion("Lyric Structure Tags", open=False):
                            gr.Markdown("""
**Vocal sections (require lyrics):**
- `[verse]` - Lyrical verse
- `[chorus]` - Lyrical chorus
- `[bridge]` - Lyrical bridge

**Instrumental sections (no lyrics):**
- `[intro-short/medium/long]` - Intros
- `[inst-short/medium/long]` - Instrumental breaks
- `[outro-short/medium/long]` - Outros
- `[silence]` - Silence

**Format:** Sections separated by ` ; ` and sentences by `.`
                            """)

                    # Common settings
                    with gr.Accordion("Output Settings", open=True):
                        output_folder = gr.Textbox(
                            label="Output Folder",
                            value="./output",
                            info="Directory to save generated music"
                        )

                    with gr.Accordion("Generation Parameters", open=True):
                        max_duration = gr.Slider(
                            label="Max Duration (seconds)",
                            minimum=10,
                            maximum=270,
                            value=120,
                            step=10,
                            info="Max varies by model: base=150s, large/full=270s"
                        )
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=2.0,
                                value=0.9,
                                step=0.1,
                                scale=1
                            )
                            topk = gr.Slider(
                                label="Top-K",
                                minimum=1,
                                maximum=100,
                                value=50,
                                step=1,
                                scale=1
                            )
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                label="CFG Scale (Guidance)",
                                minimum=0.1,
                                maximum=5.0,
                                value=1.5,
                                step=0.1,
                                info="Higher = stronger adherence to tags/lyrics",
                                scale=1
                            )
                            songgen_top_p = gr.Slider(
                                label="Top-P (Nucleus)",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.05,
                                info="0.0 = use Top-K only",
                                scale=1
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
                            info_num_steps = gr.Textbox(label="Flow Steps", interactive=False)

                    with gr.Accordion("Reference Audio Settings", open=True):
                        with gr.Row():
                            info_ref_audio_semantic = gr.Textbox(label="Semantic Reference", interactive=False)
                            info_ref_audio_sec = gr.Textbox(label="Semantic Length (s)", interactive=False)
                        with gr.Row():
                            info_ref_audio_img2img = gr.Textbox(label="img2img Reference", interactive=False)
                            info_ref_strength = gr.Textbox(label="Ref Strength", interactive=False)

                    with gr.Accordion("Model Settings", open=True):
                        with gr.Row():
                            info_model_version = gr.Textbox(label="Model Version", interactive=False)
                            info_model_dtype = gr.Textbox(label="Precision", interactive=False)
                            info_num_gpu_blocks = gr.Textbox(label="GPU Blocks", interactive=False)
                        with gr.Row():
                            info_model_path = gr.Textbox(label="Model Path", interactive=False)
                            info_generated_at = gr.Textbox(label="Generated At", interactive=False)

        # Post-Processing Tab
        with gr.Tab("Post-Processing"):
            gr.Markdown("### Audio Post-Processing")
            gr.Markdown("Load an audio file and apply various effects. Changes are cumulative - load the output to apply more effects.")

            # State to hold the loaded audio data
            pp_audio_state = gr.State(value=None)

            with gr.Row():
                # Left column - Input and Controls
                with gr.Column(scale=1):
                    with gr.Accordion("Input Audio", open=True):
                        pp_audio_input = gr.Audio(
                            label="Load Audio File",
                            type="filepath",
                        )
                        pp_load_btn = gr.Button("📂 Load Audio", variant="primary")
                        pp_load_status = gr.Textbox(label="Status", interactive=False, value="", lines=2)

                    with gr.Accordion("Trim / Cut", open=True):
                        gr.Markdown("*Cut the audio between two time points*")
                        with gr.Row():
                            pp_trim_start = gr.Number(label="Start Time (s)", value=0, minimum=0)
                            pp_trim_end = gr.Number(label="End Time (s)", value=30, minimum=0)
                        pp_trim_btn = gr.Button("✂️ Trim Audio")

                    with gr.Accordion("Loudness", open=True):
                        pp_gain_db = gr.Slider(
                            label="Gain (dB)",
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=0.5,
                            info="Positive = louder, Negative = quieter"
                        )
                        pp_loudness_btn = gr.Button("🔊 Adjust Loudness")

                        gr.Markdown("---")
                        pp_normalize_db = gr.Slider(
                            label="Target Peak Level (dB)",
                            minimum=-12,
                            maximum=0,
                            value=-1,
                            step=0.5,
                            info="Normalize peak to this level"
                        )
                        pp_normalize_btn = gr.Button("📊 Normalize")

                    with gr.Accordion("EQ / Tone", open=True):
                        gr.Markdown("**Bass Boost**")
                        with gr.Row():
                            pp_bass_boost_db = gr.Slider(
                                label="Boost (dB)",
                                minimum=0,
                                maximum=12,
                                value=6,
                                step=0.5
                            )
                            pp_bass_cutoff = gr.Slider(
                                label="Cutoff (Hz)",
                                minimum=60,
                                maximum=300,
                                value=150,
                                step=10
                            )
                        pp_bass_btn = gr.Button("🔉 Apply Bass Boost")

                        gr.Markdown("**Treble Boost**")
                        with gr.Row():
                            pp_treble_boost_db = gr.Slider(
                                label="Boost (dB)",
                                minimum=0,
                                maximum=12,
                                value=6,
                                step=0.5
                            )
                            pp_treble_cutoff = gr.Slider(
                                label="Cutoff (Hz)",
                                minimum=2000,
                                maximum=10000,
                                value=4000,
                                step=100
                            )
                        pp_treble_btn = gr.Button("🔊 Apply Treble Boost")

                    with gr.Accordion("Fades", open=True):
                        with gr.Row():
                            pp_fade_in_duration = gr.Slider(
                                label="Fade-In Duration (s)",
                                minimum=0,
                                maximum=10,
                                value=2,
                                step=0.1
                            )
                            pp_fade_in_btn = gr.Button("📈 Fade In")
                        with gr.Row():
                            pp_fade_out_duration = gr.Slider(
                                label="Fade-Out Duration (s)",
                                minimum=0,
                                maximum=10,
                                value=2,
                                step=0.1
                            )
                            pp_fade_out_btn = gr.Button("📉 Fade Out")

                    with gr.Accordion("Speed", open=True):
                        pp_speed_factor = gr.Slider(
                            label="Speed Factor",
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            info="<1 = slower, >1 = faster (affects pitch)"
                        )
                        pp_speed_btn = gr.Button("⏩ Change Speed")

                # Right column - Output
                with gr.Column(scale=1):
                    with gr.Accordion("Output", open=True):
                        pp_output_audio = gr.Audio(
                            label="Processed Audio",
                            interactive=False,
                        )
                        pp_output_status = gr.Textbox(label="Processing Status", interactive=False, value="", lines=2)

    # Event handlers
    def update_blocks_slider(version):
        max_blocks = MODEL_LAYER_COUNTS.get(version, {}).get("backbone", 28)
        return gr.update(maximum=max_blocks, info=f"Blocks on GPU (0 = all). {version} has {max_blocks} blocks.")

    def randomize_seed():
        return -1

    def switch_backend(backend):
        """Switch visibility between HeartMuLa and SongGeneration settings."""
        if backend == "HeartMuLa":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    backend_selector.change(
        fn=switch_backend,
        inputs=[backend_selector],
        outputs=[heartmula_settings, songgen_settings]
    )

    model_version.change(
        fn=update_blocks_slider,
        inputs=[model_version],
        outputs=[num_gpu_blocks]
    )

    random_seed_btn.click(
        fn=randomize_seed,
        outputs=[seed]
    )

    def generate_music_wrapper(
        backend,
        # Common params
        lyrics, tags, negative_prompt, max_duration_val, temperature_val, topk_val, cfg_scale_val,
        batch_count_val, seed_val, output_folder_val,
        # HeartMuLa params
        model_path_val, model_version_val, num_gpu_blocks_val, model_dtype_val,
        compile_val, use_rl_val,
        ref_audio_semantic_val, ref_audio_img2img_val, ref_strength_val, normalize_loudness_val, loudness_boost_val,
        num_steps_val, ref_audio_sec_val,
        # SongGeneration params
        songgen_ckpt_path_val, songgen_model_val, songgen_style_val, songgen_generate_type_val,
        songgen_flash_attn_val, songgen_gpu_blocks_val, songgen_compile_val,
        songgen_top_p_val, songgen_gender_val, songgen_timbre_val, songgen_emotion_val, songgen_bpm_val,
    ):
        """Wrapper to call the appropriate generation function based on backend."""
        if backend == "HeartMuLa":
            yield from generate_music(
                lyrics, tags, negative_prompt, max_duration_val, temperature_val, topk_val, cfg_scale_val,
                model_path_val, model_version_val, num_gpu_blocks_val, model_dtype_val,
                batch_count_val, seed_val, output_folder_val, compile_val, use_rl_val,
                ref_audio_semantic_val, ref_audio_img2img_val, ref_strength_val, normalize_loudness_val, loudness_boost_val,
                num_steps_val, ref_audio_sec_val,
            )
        else:
            # Use ref_audio_semantic as the reference audio for SongGeneration
            yield from generate_music_songgen(
                lyrics, tags, max_duration_val, temperature_val, topk_val, cfg_scale_val,
                songgen_model_val, songgen_ckpt_path_val,
                batch_count_val, seed_val, output_folder_val,
                songgen_flash_attn_val, songgen_gpu_blocks_val, songgen_generate_type_val,
                songgen_style_val, ref_audio_semantic_val, songgen_compile_val,
                songgen_top_p_val, songgen_gender_val, songgen_timbre_val, songgen_emotion_val, int(songgen_bpm_val),
            )

    generate_event = generate_btn.click(
        fn=generate_music_wrapper,
        inputs=[
            backend_selector,
            # Common params
            lyrics_input, tags_input, negative_prompt_input, max_duration, temperature, topk, cfg_scale,
            batch_count, seed, output_folder,
            # HeartMuLa params
            model_path, model_version, num_gpu_blocks, model_dtype,
            compile_checkbox, use_rl_models_checkbox,
            ref_audio_semantic, ref_audio_img2img, ref_strength_slider, normalize_loudness_checkbox, loudness_boost_slider,
            num_steps_slider, ref_audio_sec_slider,
            # SongGeneration params
            songgen_ckpt_path, songgen_model, songgen_style, songgen_generate_type,
            songgen_flash_attn, songgen_gpu_blocks, songgen_compile,
            songgen_top_p, songgen_gender, songgen_timbre, songgen_emotion, songgen_bpm,
        ],
        outputs=audio_outputs + [status_text]
    )

    stop_btn.click(
        fn=stop_generation,
        outputs=[status_text],
        cancels=[generate_event]
    )

    # Save/Load Defaults
    # HeartMuLa-specific components (unique to HeartMuLa)
    heartmula_components = [
        lyrics_input, tags_input, negative_prompt_input, batch_count, seed,
        model_path, model_version, model_dtype, num_gpu_blocks, output_folder,
        max_duration, temperature, topk, cfg_scale, compile_checkbox, use_rl_models_checkbox,
        num_steps_slider, ref_strength_slider, normalize_loudness_checkbox, loudness_boost_slider, ref_audio_sec_slider
    ]

    # Keys for the HeartMuLa defaults file
    heartmula_keys = [
        "lyrics", "tags", "negative_prompt", "batch_count", "seed",
        "model_path", "model_version", "model_dtype", "num_gpu_blocks", "output_folder",
        "max_duration", "temperature", "topk", "cfg_scale", "compile_model", "use_rl_models",
        "num_steps", "ref_strength", "normalize_loudness", "loudness_boost", "ref_audio_sec"
    ]

    # SongGen-specific components (unique to SongGen - no overlap with heartmula_components)
    songgen_only_components = [
        songgen_ckpt_path, songgen_model, songgen_style, songgen_generate_type,
        songgen_flash_attn, songgen_gpu_blocks, songgen_compile,
        songgen_top_p, songgen_gender, songgen_timbre, songgen_emotion, songgen_bpm,
    ]

    # Keys for the SongGen-only components
    songgen_only_keys = [
        "songgen_ckpt_path", "songgen_model", "songgen_style", "songgen_generate_type",
        "songgen_flash_attn", "songgen_gpu_blocks", "songgen_compile",
        "songgen_top_p", "songgen_gender", "songgen_timbre", "songgen_emotion", "songgen_bpm",
    ]

    # Shared component indices in heartmula_components that SongGen also uses
    # These are: lyrics(0), tags(1), batch_count(3), seed(4), output_folder(9),
    #            max_duration(10), temperature(11), topk(12), cfg_scale(13)
    shared_indices = [0, 1, 3, 4, 9, 10, 11, 12, 13]
    shared_keys = ["lyrics", "tags", "batch_count", "seed", "output_folder",
                   "max_duration", "temperature", "topk", "cfg_scale"]

    # All unique components for UI updates (HeartMuLa + SongGen-only)
    all_defaults_components = heartmula_components + songgen_only_components

    # For backward compatibility
    defaults_components = heartmula_components
    defaults_keys = heartmula_keys

    def save_defaults(backend, *values):
        """Save current settings to defaults file based on backend."""
        heartmula_values = values[:len(heartmula_keys)]
        songgen_only_values = values[len(heartmula_keys):]

        if backend == "SongGeneration (LeVo)":
            # Save SongGen defaults - shared fields from heartmula + songgen-only
            settings = {}
            # Add shared fields from heartmula_values
            for idx, key in zip(shared_indices, shared_keys):
                settings[key] = heartmula_values[idx]
            # Add songgen-only fields
            for i, key in enumerate(songgen_only_keys):
                settings[key] = songgen_only_values[i]
            try:
                with open(SONGGEN_DEFAULTS_FILE, 'w') as f:
                    json.dump(settings, f, indent=2)
                return "SongGen defaults saved successfully."
            except Exception as e:
                return f"Error saving SongGen defaults: {e}"
        else:
            # Save HeartMuLa defaults
            settings = {}
            for i, key in enumerate(heartmula_keys):
                settings[key] = heartmula_values[i]
            try:
                with open(HEARTMULA_DEFAULTS_FILE, 'w') as f:
                    json.dump(settings, f, indent=2)
                return "HeartMuLa defaults saved successfully."
            except Exception as e:
                return f"Error saving HeartMuLa defaults: {e}"

    def load_defaults(backend, request: gr.Request = None):
        """Load settings from defaults file based on backend."""
        total_components = len(heartmula_keys) + len(songgen_only_keys)

        if backend == "SongGeneration (LeVo)":
            defaults_file = SONGGEN_DEFAULTS_FILE
        else:
            defaults_file = HEARTMULA_DEFAULTS_FILE

        if not os.path.exists(defaults_file):
            if request:
                return [gr.update()] * total_components + [f"No {backend} defaults file found."]
            else:
                return [gr.update()] * total_components + [""]

        try:
            with open(defaults_file, 'r') as f:
                loaded = json.load(f)
        except Exception as e:
            return [gr.update()] * total_components + [f"Error loading defaults: {e}"]

        # Build updates list
        updates = [gr.update()] * total_components

        if backend == "SongGeneration (LeVo)":
            # Load shared fields into heartmula component positions
            for idx, key in zip(shared_indices, shared_keys):
                if key in loaded:
                    updates[idx] = gr.update(value=loaded[key])
            # Load songgen-only fields
            offset = len(heartmula_keys)
            for i, key in enumerate(songgen_only_keys):
                if key in loaded:
                    updates[offset + i] = gr.update(value=loaded[key])
        else:
            # Load HeartMuLa fields
            for i, key in enumerate(heartmula_keys):
                if key in loaded:
                    updates[i] = gr.update(value=loaded[key])

        return updates + [f"{backend} defaults loaded successfully."]

    save_defaults_btn.click(
        fn=save_defaults,
        inputs=[backend_selector] + all_defaults_components,
        outputs=[status_text]
    )

    load_defaults_btn.click(
        fn=load_defaults,
        inputs=[backend_selector],
        outputs=all_defaults_components + [status_text]
    )

    # Audio Info Tab event handlers
    # Store loaded metadata in a state variable
    loaded_metadata_state = gr.State(value=None)

    def load_mp3_metadata(audio_path):
        """Load and display metadata from an MP3 file."""
        # Number of display fields (excluding state and status)
        num_fields = 18
        empty_result = (None, "") + tuple([""] * num_fields)

        if audio_path is None:
            return empty_result

        if not audio_path.lower().endswith('.mp3'):
            return (None, "Please upload an MP3 file.") + tuple([""] * num_fields)

        try:
            from heartlib import HeartMuLaGenPipeline
            metadata = HeartMuLaGenPipeline.read_mp3_metadata(audio_path)

            if metadata is None:
                return (None, "No HeartMuLa metadata found in this MP3.") + tuple([""] * num_fields)

            # Extract ref_audio paths (handle both old and new metadata formats)
            ref_semantic = metadata.get("ref_audio_semantic", metadata.get("ref_audio", ""))
            ref_img2img = metadata.get("ref_audio_img2img", "")

            return (
                metadata,  # state
                "Metadata loaded successfully!",  # status
                # Metadata Preview
                metadata.get("lyrics", ""),
                metadata.get("tags", ""),
                metadata.get("negative_prompt", ""),
                # Generation Parameters
                str(metadata.get("seed", "")),
                str(metadata.get("max_duration", "")),
                str(metadata.get("temperature", "")),
                str(metadata.get("topk", "")),
                str(metadata.get("cfg_scale", "")),
                str(metadata.get("num_steps", "")),
                # Reference Audio Settings
                os.path.basename(ref_semantic) if ref_semantic else "",
                str(metadata.get("ref_audio_sec", "")),
                os.path.basename(ref_img2img) if ref_img2img else "",
                str(metadata.get("ref_strength", "")),
                # Model Settings
                metadata.get("model_version", ""),
                metadata.get("model_dtype", ""),
                str(metadata.get("num_gpu_blocks", "")),
                metadata.get("model_path", ""),
                metadata.get("generated_at", ""),
            )
        except Exception as e:
            return (None, f"Error reading metadata: {e}") + tuple([""] * num_fields)

    def send_settings_to_generation(metadata):
        """Send loaded metadata settings to the generation tab."""
        total_components = len(heartmula_keys) + len(songgen_only_keys)
        if metadata is None:
            return [gr.update()] * total_components + ["No metadata loaded. Please upload an MP3 first."]

        # Map metadata to generation tab components
        # HeartMuLa components (21)
        heartmula_updates = [
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
            gr.update(value=metadata.get("use_rl_models", False)),
            gr.update(value=metadata.get("num_steps", 10)),
            gr.update(value=metadata.get("ref_strength", 0.5)),
            gr.update(value=metadata.get("normalize_loudness", True)),
            gr.update(value=metadata.get("loudness_boost", 1.0)),
            gr.update(value=metadata.get("ref_audio_sec", 30)),
        ]

        # SongGen-only components - no update (keep current values)
        songgen_updates = [gr.update()] * len(songgen_only_keys)

        return heartmula_updates + songgen_updates + ["Settings loaded to Generation tab!"]

    # Info display components for Audio Info tab
    # Order must match load_mp3_metadata return values
    info_display_components = [
        # Metadata Preview
        info_lyrics, info_tags, info_negative_prompt,
        # Generation Parameters
        info_seed, info_max_duration, info_temperature,
        info_topk, info_cfg_scale, info_num_steps,
        # Reference Audio Settings
        info_ref_audio_semantic, info_ref_audio_sec,
        info_ref_audio_img2img, info_ref_strength,
        # Model Settings
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
        outputs=all_defaults_components + [audio_info_status]
    )

    # Post-Processing Tab event handlers
    pp_load_btn.click(
        fn=load_audio_for_processing,
        inputs=[pp_audio_input],
        outputs=[pp_audio_state, pp_output_audio, pp_load_status]
    )

    pp_trim_btn.click(
        fn=trim_audio,
        inputs=[pp_audio_state, pp_trim_start, pp_trim_end],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_loudness_btn.click(
        fn=adjust_loudness,
        inputs=[pp_audio_state, pp_gain_db],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_normalize_btn.click(
        fn=normalize_audio,
        inputs=[pp_audio_state, pp_normalize_db],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_bass_btn.click(
        fn=apply_bass_boost,
        inputs=[pp_audio_state, pp_bass_boost_db, pp_bass_cutoff],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_treble_btn.click(
        fn=apply_treble_boost,
        inputs=[pp_audio_state, pp_treble_boost_db, pp_treble_cutoff],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_fade_in_btn.click(
        fn=apply_fade_in,
        inputs=[pp_audio_state, pp_fade_in_duration],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_fade_out_btn.click(
        fn=apply_fade_out,
        inputs=[pp_audio_state, pp_fade_out_duration],
        outputs=[pp_output_audio, pp_output_status]
    )

    pp_speed_btn.click(
        fn=change_speed,
        inputs=[pp_audio_state, pp_speed_factor],
        outputs=[pp_output_audio, pp_output_status]
    )

    # Auto-load defaults on startup
    def initial_load_defaults():
        """Load both HeartMuLa and SongGen defaults on startup."""
        total_components = len(heartmula_keys) + len(songgen_only_keys)
        updates = [gr.update()] * total_components

        # Load HeartMuLa defaults
        if os.path.exists(HEARTMULA_DEFAULTS_FILE):
            try:
                with open(HEARTMULA_DEFAULTS_FILE, 'r') as f:
                    heartmula_loaded = json.load(f)
                for i, key in enumerate(heartmula_keys):
                    if key in heartmula_loaded:
                        updates[i] = gr.update(value=heartmula_loaded[key])
            except Exception:
                pass

        # Load SongGen-only defaults
        if os.path.exists(SONGGEN_DEFAULTS_FILE):
            try:
                with open(SONGGEN_DEFAULTS_FILE, 'r') as f:
                    songgen_loaded = json.load(f)
                offset = len(heartmula_keys)
                for i, key in enumerate(songgen_only_keys):
                    if key in songgen_loaded:
                        updates[offset + i] = gr.update(value=songgen_loaded[key])
            except Exception:
                pass

        return updates

    demo.load(
        fn=initial_load_defaults,
        inputs=None,
        outputs=all_defaults_components
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
        show_error=True,
        allowed_paths=[DEFAULT_OUTPUT_DIR]
    )
