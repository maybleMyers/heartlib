"""
ACE-Step Gradio Web UI
A web interface for generating music with ACE-Step, with all features from the original implementation.
"""

import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
import torch
import tempfile
import os
import sys
import threading
import random
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
from typing import Optional, List, Dict, Any, Tuple, Union

# Add ACE-Step to path
ACE_STEP_PATH = os.path.join(os.path.dirname(__file__), "ACE-Step-1.5")
if ACE_STEP_PATH not in sys.path:
    sys.path.insert(0, ACE_STEP_PATH)

# Import ACE-Step inference API
from acestep.inference import generate_music as acestep_generate_music, GenerationParams, GenerationConfig

# Defaults file path
ACE_STEP_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "acestep_defaults.json")

# Global stop event for batch cancellation
stop_event = threading.Event()

# Global handlers
_dit_handler = None
_llm_handler = None
_dataset_handler = None

# =============================================================================
# Constants from ACE-Step
# =============================================================================

VALID_LANGUAGES = [
    'ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en',
    'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'ht', 'hu', 'id',
    'is', 'it', 'ja', 'ko', 'la', 'lt', 'ms', 'ne', 'nl', 'no',
    'pa', 'pl', 'pt', 'ro', 'ru', 'sa', 'sk', 'sr', 'sv', 'sw',
    'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yue', 'zh',
    'unknown', 'instrumental'
]

TASK_TYPES_TURBO = ["text2music", "repaint", "cover"]
TASK_TYPES_BASE = ["text2music", "repaint", "cover", "extract", "lego", "complete"]

TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"
]

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

TASK_INSTRUCTIONS = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "repaint": "Repaint the mask area based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "extract": "Extract the {TRACK_NAME} track from the audio:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
}

# Maximum number of audio outputs in the UI
MAX_AUDIO_OUTPUTS = 8


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

# Default caption
DEFAULT_CAPTION = "upbeat pop song, female vocals, happy mood, acoustic guitar, piano"


def get_available_lm_models():
    """Scan checkpoints directory for available LM models."""
    # Always include standard models
    models = ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"]

    # Scan for additional models in checkpoints directory
    checkpoints_dir = os.path.join(ACE_STEP_PATH, "checkpoints")
    if os.path.exists(checkpoints_dir):
        for name in os.listdir(checkpoints_dir):
            if name.startswith("acestep-5Hz-lm-") and name not in models:
                models.append(name)

    # Sort by size (extract number from name)
    models.sort(key=lambda x: float(x.split("-")[-1].replace("B", "")) if x.split("-")[-1].replace("B", "").replace(".", "").isdigit() else 0)
    return models


def get_available_dit_models():
    """Scan checkpoints directory for available DiT models."""
    checkpoints_dir = os.path.join(ACE_STEP_PATH, "checkpoints")
    models = []
    if os.path.exists(checkpoints_dir):
        for name in os.listdir(checkpoints_dir):
            if name.startswith("acestep-v15-"):
                models.append(name)
    return models if models else ["acestep-v15-turbo", "acestep-v15-base"]


def get_available_checkpoints():
    """Get available checkpoint paths."""
    checkpoints_dir = os.path.join(ACE_STEP_PATH, "checkpoints")
    if os.path.exists(checkpoints_dir):
        return [checkpoints_dir]
    return ["./ACE-Step-1.5/checkpoints"]


def get_handlers():
    """Get or initialize handlers."""
    global _dit_handler, _llm_handler, _dataset_handler
    if _dit_handler is None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.dataset_handler import DatasetHandler
        _dit_handler = AceStepHandler()
        _llm_handler = LLMHandler()
        _dataset_handler = DatasetHandler()
    return _dit_handler, _llm_handler, _dataset_handler


def create_audio_outputs(file_paths: list, labels: list = None) -> list:
    """Create list of audio outputs for gr.Audio components."""
    if labels is None:
        labels = [os.path.basename(p) for p in file_paths] if file_paths else []

    outputs = []
    for i in range(MAX_AUDIO_OUTPUTS):
        if i < len(file_paths):
            outputs.append(gr.update(value=file_paths[i], label=labels[i], visible=True))
        else:
            outputs.append(gr.update(value=None, visible=False))

    return outputs


def get_instruction_for_task(task_type: str, track_name: str = None, track_classes: list = None) -> str:
    """Get the appropriate instruction for a task type."""
    if task_type == "extract":
        if track_name:
            return TASK_INSTRUCTIONS["extract"].format(TRACK_NAME=track_name)
        return "Extract the track from the audio:"
    elif task_type == "lego":
        if track_name:
            return TASK_INSTRUCTIONS["lego"].format(TRACK_NAME=track_name)
        return "Generate the track based on the audio context:"
    elif task_type == "complete":
        if track_classes:
            return TASK_INSTRUCTIONS["complete"].format(TRACK_CLASSES=", ".join(track_classes))
        return "Complete the input track:"
    return TASK_INSTRUCTIONS.get(task_type, DEFAULT_DIT_INSTRUCTION)


def initialize_service(
    checkpoint_path: str,
    model_type: str,
    device: str,
    init_llm: bool,
    lm_model: str,
    backend: str,
    use_flash_attention: bool,
    compile_model: bool,
    offload_to_cpu: bool,
    offload_dit_to_cpu: bool,
    quantization: bool,
):
    """Initialize ACE-Step service."""
    dit_handler, llm_handler, _ = get_handlers()

    try:
        # Map model type to config path
        config_map = {
            "turbo": "acestep-v15-turbo",
            "base": "acestep-v15-base",
        }
        config_path = config_map.get(model_type, "acestep-v15-turbo")

        # Quantization value
        quant_value = "int8_weight_only" if quantization else None

        log(f"Initializing ACE-Step with checkpoint: {checkpoint_path}")
        log(f"Config: {config_path}, Device: {device}")

        # Initialize DiT
        status, success = dit_handler.initialize_service(
            project_root=checkpoint_path,
            config_path=config_path,
            device=device,
            use_flash_attention=use_flash_attention,
            compile_model=compile_model,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_dit_to_cpu,
            quantization=quant_value,
        )

        if not success:
            return f"DiT initialization failed: {status}", gr.update(interactive=False)

        # Initialize LLM if requested
        if init_llm:
            checkpoint_dir = os.path.join(ACE_STEP_PATH, "checkpoints")
            log(f"Initializing LLM from: {checkpoint_dir}")
            lm_status, lm_success = llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model,
                backend=backend,
                device=device,
                offload_to_cpu=offload_to_cpu,
                dtype=dit_handler.dtype,
            )
            if lm_success:
                status += f"\nLLM: {lm_status}"
            else:
                status += f"\nLLM failed: {lm_status}"

        log(f"Initialization complete: {status}")
        return status, gr.update(interactive=True)

    except Exception as e:
        import traceback
        error_msg = f"Initialization error: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        return error_msg, gr.update(interactive=False)


def load_lora_adapter(lora_path: str, use_lora: bool, lora_scale: float):
    """Load a LoRA adapter."""
    dit_handler, _, _ = get_handlers()
    try:
        if not lora_path or not lora_path.strip():
            return "Please provide a LoRA path", False, 1.0

        if not os.path.exists(lora_path):
            return f"LoRA path does not exist: {lora_path}", False, 1.0

        # Call handler's load_lora method if it exists
        if hasattr(dit_handler, 'load_lora'):
            success = dit_handler.load_lora(lora_path, scale=lora_scale)
            if success:
                return f"LoRA loaded: {os.path.basename(lora_path)}", True, lora_scale
            else:
                return "Failed to load LoRA", False, 1.0
        else:
            return "LoRA loading not supported in this version", False, 1.0
    except Exception as e:
        return f"Error loading LoRA: {e}", False, 1.0


def unload_lora_adapter():
    """Unload the current LoRA adapter."""
    dit_handler, _, _ = get_handlers()
    try:
        if hasattr(dit_handler, 'unload_lora'):
            dit_handler.unload_lora()
            return "LoRA unloaded", False, 1.0
        else:
            return "LoRA unloading not supported", False, 1.0
    except Exception as e:
        return f"Error unloading LoRA: {e}", False, 1.0


def convert_audio_to_codes(src_audio_path: str):
    """Convert source audio to audio codes."""
    dit_handler, _, _ = get_handlers()
    try:
        if not src_audio_path:
            return "No audio file provided"

        if hasattr(dit_handler, 'audio_to_codes'):
            codes = dit_handler.audio_to_codes(src_audio_path)
            return codes if codes else "Failed to convert audio to codes"
        else:
            return "Audio to codes conversion not supported"
    except Exception as e:
        return f"Error converting audio: {e}"


def transcribe_audio(src_audio_path: str):
    """Transcribe audio to get lyrics and metadata."""
    dit_handler, llm_handler, _ = get_handlers()
    try:
        if not src_audio_path:
            return "", "", "No audio file provided"

        # This would call the transcription functionality
        if hasattr(llm_handler, 'transcribe'):
            result = llm_handler.transcribe(src_audio_path)
            return result.get('lyrics', ''), result.get('caption', ''), "Transcription complete"
        else:
            return "", "", "Transcription not supported"
    except Exception as e:
        return "", "", f"Error transcribing: {e}"


def format_lyrics(lyrics: str):
    """Format lyrics with proper structure tags."""
    dit_handler, llm_handler, _ = get_handlers()
    try:
        if hasattr(llm_handler, 'format_lyrics'):
            formatted = llm_handler.format_lyrics(lyrics)
            return formatted if formatted else lyrics
        return lyrics
    except Exception as e:
        log(f"Error formatting lyrics: {e}")
        return lyrics


def generate_random_caption():
    """Generate a random music caption."""
    dit_handler, llm_handler, _ = get_handlers()
    try:
        if hasattr(llm_handler, 'generate_random_caption'):
            return llm_handler.generate_random_caption()
        # Fallback random captions
        captions = [
            "upbeat pop song, female vocals, happy mood, acoustic guitar, piano",
            "melancholic indie rock, male vocals, rainy day vibes, electric guitar, drums",
            "energetic electronic dance, synthesizers, powerful bass, club atmosphere",
            "gentle acoustic ballad, soft vocals, romantic, fingerpicked guitar",
            "epic cinematic orchestral, dramatic strings, powerful brass, heroic theme",
        ]
        return random.choice(captions)
    except Exception as e:
        return DEFAULT_CAPTION


def generate_music(
    # Basic inputs
    caption: str,
    lyrics: str,
    vocal_language: str,
    instrumental: bool,
    # Metadata
    bpm: float,
    key_scale: str,
    time_signature: str,
    duration: float,
    # Generation settings
    batch_count: int,
    seed: int,
    inference_steps: int,
    guidance_scale: float,
    shift: float,
    infer_method: str,
    custom_timesteps: str,
    # Task settings
    task_type: str,
    instruction: str,
    # Audio inputs
    reference_audio: str = None,
    src_audio: str = None,
    audio_code_string: str = "",
    # Repainting
    repainting_start: float = 0.0,
    repainting_end: float = -1,
    audio_cover_strength: float = 1.0,
    # Advanced
    use_adg: bool = False,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    # LM settings
    think: bool = True,
    allow_lm_batch: bool = True,
    lm_temperature: float = 0.85,
    lm_cfg_scale: float = 2.0,
    lm_top_k: int = 0,
    lm_top_p: float = 0.9,
    lm_negative_prompt: str = "NO USER INPUT",
    use_cot_metas: bool = True,
    use_cot_caption: bool = True,
    use_cot_language: bool = True,
    # LoRA
    use_lora: bool = False,
    lora_scale: float = 1.0,
    # Output
    output_folder: str = "./output",
    audio_format: str = "mp3",
):
    """Generate music using ACE-Step inference API."""
    global stop_event
    stop_event.clear()

    dit_handler, llm_handler, _ = get_handlers()
    all_generated_music = []
    all_codes = []
    all_seeds = []
    batch_count = int(batch_count)
    seed = int(seed)

    try:
        # Validate
        if dit_handler.model is None:
            yield (*create_audio_outputs([]), [], "Please initialize the model first.")
            return

        if not caption.strip():
            yield (*create_audio_outputs([]), [], "Please enter a caption/description.")
            return

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        log("=" * 50)
        log(f"Starting batch generation ({batch_count} songs)...")
        log(f"Task type: {task_type}")
        log(f"Caption: {caption[:100]}...")
        log(f"Duration: {duration}s, Steps: {inference_steps}")

        yield (*create_audio_outputs([]), [], "Starting generation...")

        # Convert parameters
        duration_value = float(duration) if duration and duration > 0 else -1.0
        bpm_value = int(bpm) if bpm and bpm > 0 else None

        # Parse custom timesteps
        timesteps = None
        if custom_timesteps and custom_timesteps.strip():
            try:
                timesteps = [float(x.strip()) for x in custom_timesteps.split(",")]
            except:
                pass

        # Handle lyrics
        lyrics_to_use = lyrics.strip() if lyrics.strip() else ("[Instrumental]" if instrumental else "")

        # Build seed list for batch
        seed_list = []
        for i in range(batch_count):
            if seed == -1:
                seed_list.append(random.randint(0, 2**32 - 1))
            else:
                seed_list.append(seed + i)

        all_seeds = seed_list.copy()

        # Create GenerationParams using ACE-Step inference API
        gen_params = GenerationParams(
            task_type=task_type,
            instruction=instruction if instruction else DEFAULT_DIT_INSTRUCTION,
            reference_audio=reference_audio,
            src_audio=src_audio,
            audio_codes=audio_code_string if not think else "",
            caption=caption,
            lyrics=lyrics_to_use,
            instrumental=instrumental,
            vocal_language=vocal_language,
            bpm=bpm_value,
            keyscale=key_scale if key_scale else "",
            timesignature=time_signature if time_signature else "",
            duration=duration_value,
            inference_steps=int(inference_steps),
            guidance_scale=float(guidance_scale),
            seed=seed,
            use_adg=use_adg,
            cfg_interval_start=float(cfg_interval_start),
            cfg_interval_end=float(cfg_interval_end),
            shift=float(shift),
            infer_method=infer_method,
            timesteps=timesteps,
            repainting_start=float(repainting_start),
            repainting_end=float(repainting_end),
            audio_cover_strength=float(audio_cover_strength),
            thinking=think,
            lm_temperature=float(lm_temperature),
            lm_cfg_scale=float(lm_cfg_scale),
            lm_top_k=int(lm_top_k),
            lm_top_p=float(lm_top_p),
            lm_negative_prompt=lm_negative_prompt,
            use_cot_metas=use_cot_metas,
            use_cot_caption=use_cot_caption,
            use_cot_language=use_cot_language,
        )

        # Create GenerationConfig
        gen_config = GenerationConfig(
            batch_size=batch_count,
            allow_lm_batch=allow_lm_batch,
            use_random_seed=(seed == -1),
            seeds=seed_list,
            audio_format=audio_format,
        )

        status_text = f"Generating {batch_count} song(s)..."
        yield (*create_audio_outputs([]), [], status_text)

        start_time = time.perf_counter()

        # Call ACE-Step inference API
        result = acestep_generate_music(
            dit_handler,
            llm_handler,
            params=gen_params,
            config=gen_config,
        )

        # Check for stop
        if stop_event.is_set():
            log("Generation stopped by user.")
            yield (*create_audio_outputs([]), [], "Stopped by user.")
            return

        # Process result
        if result.success and result.audios:
            from acestep.audio_utils import AudioSaver
            audio_saver = AudioSaver(default_format=audio_format)

            for idx, audio_dict in enumerate(result.audios):
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                codes = audio_dict.get("codes", "")
                audio_seed = seed_list[idx] if idx < len(seed_list) else 0

                if audio_tensor is not None:
                    # Create output path
                    timestamp = int(time.time())
                    output_filename = f"acestep_{timestamp}_{audio_seed}.{audio_format}"
                    output_path = os.path.join(output_folder, output_filename)

                    # Save audio to file
                    saved_path = audio_saver.save_audio(
                        audio_data=audio_tensor,
                        output_path=output_path,
                        sample_rate=sample_rate,
                        format=audio_format,
                    )
                    all_generated_music.append(saved_path)
                    all_codes.append(codes)
                    log(f"Song {idx+1} saved to: {saved_path}")
        else:
            error_msg = result.error or result.status_message or "Unknown error"
            log(f"Generation failed: {error_msg}")

        elapsed = time.perf_counter() - start_time
        log(f"Generation complete! ({elapsed:.1f}s)")

        # Memory info
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log(f"GPU Memory: {allocated:.2f}GB")

        # Final status
        if all_generated_music:
            final_status = f"Completed {len(all_generated_music)} song(s)!"
        else:
            final_status = result.status_message or "Generation failed"

        log(f"\n{'='*50}")
        log(final_status)
        labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
        yield (*create_audio_outputs(all_generated_music, labels), all_codes, final_status)

    except Exception as e:
        import traceback
        error_msg = f"Error during generation: {str(e)}"
        log(error_msg)
        log(traceback.format_exc())
        labels = [f"Song {j+1} (Seed: {all_seeds[j]})" for j in range(len(all_generated_music))]
        yield (*create_audio_outputs(all_generated_music, labels), all_codes, error_msg)


def compute_quality_score(audio_path: str):
    """Compute quality score for generated audio."""
    dit_handler, _, _ = get_handlers()
    try:
        if hasattr(dit_handler, 'compute_quality_score'):
            score = dit_handler.compute_quality_score(audio_path)
            return f"Quality Score: {score:.2f}"
        return "Quality scoring not available"
    except Exception as e:
        return f"Error computing score: {e}"


def generate_lrc(audio_path: str, lyrics: str):
    """Generate synchronized LRC lyrics."""
    dit_handler, llm_handler, _ = get_handlers()
    try:
        if hasattr(llm_handler, 'generate_lrc'):
            lrc = llm_handler.generate_lrc(audio_path, lyrics)
            return lrc if lrc else "Failed to generate LRC"
        return "LRC generation not available"
    except Exception as e:
        return f"Error generating LRC: {e}"


# =============================================================================
# Post-processing functions
# =============================================================================

def save_audio_as_mp3(data, sr, is_stereo, metadata=None):
    """Save audio data as MP3 file with metadata and return the path."""
    temp_wav = tempfile.mktemp(suffix=".wav")
    if not is_stereo:
        data = data.flatten()
    sf.write(temp_wav, data, sr)

    output_path = tempfile.mktemp(suffix=".mp3")
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_path, format="mp3", bitrate="192k")
    os.remove(temp_wav)

    return output_path


def load_audio_for_processing(audio_path):
    """Load audio file and return sample rate, data, and metadata."""
    if audio_path is None:
        return None, None, "No audio file provided."

    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            is_stereo = True
        else:
            is_stereo = False
            data = data.reshape(-1, 1)

        duration = len(data) / sr
        return (sr, data, is_stereo, None), None, f"Loaded: {os.path.basename(audio_path)} ({duration:.2f}s, {sr}Hz)"
    except Exception as e:
        return None, None, f"Error loading audio: {e}"


def trim_audio(audio_state, start_time, end_time):
    """Trim audio between two time points."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    duration = len(data) / sr

    if start_time < 0:
        start_time = 0
    if end_time > duration:
        end_time = duration
    if start_time >= end_time:
        return None, f"Invalid time range: start ({start_time:.2f}s) must be less than end ({end_time:.2f}s)"

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    trimmed_data = data[start_sample:end_sample]

    output_path = save_audio_as_mp3(trimmed_data, sr, is_stereo, metadata)
    new_duration = len(trimmed_data) / sr
    return output_path, f"Trimmed: {start_time:.2f}s to {end_time:.2f}s (new duration: {new_duration:.2f}s)"


def adjust_loudness(audio_state, gain_db):
    """Adjust audio loudness by a gain in dB."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    linear_gain = 10 ** (gain_db / 20)
    adjusted_data = data * linear_gain
    adjusted_data = np.clip(adjusted_data, -1.0, 1.0)

    output_path = save_audio_as_mp3(adjusted_data, sr, is_stereo, metadata)
    return output_path, f"Loudness adjusted by {gain_db:+.1f} dB"


def apply_bass_boost(audio_state, boost_db, cutoff_freq):
    """Apply bass boost using a low-shelf filter."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    original_rms = np.sqrt(np.mean(data ** 2))
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b_low, a_low = signal.butter(2, normalized_cutoff, btype='low')
    bass_gain = 10 ** (boost_db / 20)

    result_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        bass = signal.filtfilt(b_low, a_low, channel)
        result_data[:, ch] = channel + bass * (bass_gain - 1)

    new_rms = np.sqrt(np.mean(result_data ** 2))
    if new_rms > 0:
        result_data = result_data * (original_rms / new_rms)
    result_data = np.tanh(result_data)

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Bass boost: {boost_db:+.1f} dB below {cutoff_freq} Hz"


def apply_treble_boost(audio_state, boost_db, cutoff_freq):
    """Apply treble boost using a high-shelf filter."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    original_rms = np.sqrt(np.mean(data ** 2))
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b_high, a_high = signal.butter(2, normalized_cutoff, btype='high')
    treble_gain = 10 ** (boost_db / 20)

    result_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        treble = signal.filtfilt(b_high, a_high, channel)
        result_data[:, ch] = channel + treble * (treble_gain - 1)

    new_rms = np.sqrt(np.mean(result_data ** 2))
    if new_rms > 0:
        result_data = result_data * (original_rms / new_rms)
    result_data = np.tanh(result_data)

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Treble boost: {boost_db:+.1f} dB above {cutoff_freq} Hz"


def normalize_audio(audio_state, target_db):
    """Normalize audio to a target peak level in dB."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    current_peak = np.max(np.abs(data))
    if current_peak == 0:
        return None, "Audio is silent, cannot normalize."

    current_db = 20 * np.log10(current_peak)
    gain_db = target_db - current_db
    linear_gain = 10 ** (gain_db / 20)
    normalized_data = data * linear_gain

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

    fade_curve = np.linspace(0, 1, fade_samples)
    result_data = data.copy()
    for ch in range(data.shape[1]):
        result_data[:fade_samples, ch] = data[:fade_samples, ch] * fade_curve

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

    fade_curve = np.linspace(1, 0, fade_samples)
    result_data = data.copy()
    for ch in range(data.shape[1]):
        result_data[-fade_samples:, ch] = data[-fade_samples:, ch] * fade_curve

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    return output_path, f"Fade-out applied: {fade_duration:.2f}s"


def change_speed(audio_state, speed_factor):
    """Change audio playback speed (affects pitch)."""
    if audio_state is None:
        return None, "No audio loaded. Please load an audio file first."

    sr, data, is_stereo, metadata = audio_state
    if speed_factor <= 0:
        return None, "Speed factor must be positive."

    new_length = int(len(data) / speed_factor)
    result_data = np.zeros((new_length, data.shape[1]))
    for ch in range(data.shape[1]):
        result_data[:, ch] = signal.resample(data[:, ch], new_length)

    output_path = save_audio_as_mp3(result_data, sr, is_stereo, metadata)
    new_duration = new_length / sr
    return output_path, f"Speed changed to {speed_factor:.2f}x (new duration: {new_duration:.2f}s)"


# =============================================================================
# Dataset functions
# =============================================================================

def scan_audio_directory(directory_path: str):
    """Scan a directory for audio files."""
    _, _, dataset_handler = get_handlers()
    try:
        if not directory_path or not os.path.exists(directory_path):
            return [], "Directory not found"

        if hasattr(dataset_handler, 'scan_directory'):
            files = dataset_handler.scan_directory(directory_path)
            data = []
            for i, f in enumerate(files):
                data.append([i+1, f.get('filename', ''), f.get('duration', ''), f.get('caption', ''), 'No'])
            return data, f"Found {len(files)} audio files"

        # Fallback: simple directory scan
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        files = []
        for f in os.listdir(directory_path):
            if any(f.lower().endswith(ext) for ext in audio_extensions):
                files.append([len(files)+1, f, '', '', 'No'])
        return files, f"Found {len(files)} audio files"
    except Exception as e:
        return [], f"Error scanning directory: {e}"


def load_dataset_json(json_path: str):
    """Load a dataset from JSON file."""
    try:
        if not json_path or not os.path.exists(json_path):
            return [], "File not found"

        with open(json_path, 'r') as f:
            data = json.load(f)

        table_data = []
        for i, item in enumerate(data.get('files', [])):
            table_data.append([
                i+1,
                item.get('filename', ''),
                item.get('duration', ''),
                item.get('caption', ''),
                'Yes' if item.get('labeled', False) else 'No'
            ])
        return table_data, f"Loaded {len(table_data)} files from dataset"
    except Exception as e:
        return [], f"Error loading dataset: {e}"


def save_dataset_json(table_data, dataset_name: str, output_dir: str = "./datasets"):
    """Save dataset to JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_name}.json")

        files = []
        for row in table_data:
            files.append({
                'filename': row[1],
                'duration': row[2],
                'caption': row[3],
                'labeled': row[4] == 'Yes'
            })

        with open(output_path, 'w') as f:
            json.dump({'files': files}, f, indent=2)

        return f"Dataset saved to {output_path}"
    except Exception as e:
        return f"Error saving dataset: {e}"


# =============================================================================
# Build the Gradio interface
# =============================================================================

CUSTOM_CSS = """
.green-btn {
    background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
    color: white !important;
    border: none !important;
}
.green-btn:hover {
    background: linear-gradient(to bottom right, #27ae60, #219651) !important;
}
"""

def build_interface():
    """Build the complete Gradio interface."""
    theme = gr.themes.Default(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.neutral,
    )
    with gr.Blocks(title="ACE-Step Music Generator", theme=theme) as demo:

        # State variables
        codes_state = gr.State(value=[])

        with gr.Tabs():
            # =================================================================
            # Generation Tab
            # =================================================================
            with gr.Tab("Generation"):
                with gr.Row():
                    # Left column - Inputs
                    with gr.Column(scale=2):
                        # Service Configuration
                        with gr.Accordion("Model Settings", open=True):
                            with gr.Row():
                                checkpoint_path = gr.Textbox(
                                    label="Checkpoint Path",
                                    value="./ACE-Step-1.5/checkpoints",
                                    info="Path to ACE-Step checkpoints directory"
                                )
                            with gr.Row():
                                model_type = gr.Dropdown(
                                    label="Model Type",
                                    choices=["turbo", "base"],
                                    value="turbo",
                                    info="Turbo is faster, Base supports more task types"
                                )
                                device = gr.Dropdown(
                                    choices=["auto", "cuda", "cpu"],
                                    value="cuda",
                                    label="Device",
                                )
                            with gr.Row():
                                available_lm_models = get_available_lm_models()
                                lm_model = gr.Dropdown(
                                    label="LM Model",
                                    choices=available_lm_models,
                                    value=available_lm_models[0] if available_lm_models else None,
                                    info="Smaller models use less VRAM"
                                )
                                backend = gr.Dropdown(
                                    label="LM Backend",
                                    choices=["vllm", "pt"],
                                    value="pt",
                                    info="vllm is faster, pt is more compatible"
                                )
                            with gr.Row():
                                init_llm_checkbox = gr.Checkbox(
                                    label="Initialize LLM",
                                    value=True,
                                    info="Enable AI thinking for better results"
                                )
                                use_flash_attention = gr.Checkbox(label="Flash Attention", value=True)
                                compile_checkbox = gr.Checkbox(label="Compile Model", value=True)
                            with gr.Row():
                                offload_cpu = gr.Checkbox(label="Offload to CPU", value=True)
                                offload_dit_cpu = gr.Checkbox(label="Offload DiT to CPU", value=True)
                                quantization = gr.Checkbox(label="Int8 Quantization", value=True)

                            init_btn = gr.Button("Initialize Model", variant="primary", elem_classes="green-btn")
                            init_status = gr.Textbox(label="Init Status", interactive=False, lines=2)

                        # LoRA Configuration
                        with gr.Accordion("LoRA Adapter", open=False):
                            with gr.Row():
                                lora_path = gr.Textbox(
                                    label="LoRA Path",
                                    placeholder="./lora_output/final/adapter",
                                    info="Path to trained LoRA adapter"
                                )
                            with gr.Row():
                                load_lora_btn = gr.Button("Load LoRA", variant="secondary")
                                unload_lora_btn = gr.Button("Unload LoRA", variant="secondary")
                            with gr.Row():
                                use_lora_checkbox = gr.Checkbox(label="Use LoRA", value=False)
                                lora_scale_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                                    label="LoRA Scale"
                                )
                            lora_status = gr.Textbox(label="LoRA Status", value="No LoRA loaded", interactive=False)

                        # Task Configuration
                        with gr.Accordion("Task Settings", open=True):
                            with gr.Row():
                                task_type = gr.Dropdown(
                                    choices=TASK_TYPES_TURBO,
                                    value="text2music",
                                    label="Task Type",
                                    info="Type of generation task",
                                    allow_custom_value=True
                                )
                                track_name = gr.Dropdown(
                                    choices=TRACK_NAMES,
                                    value=None,
                                    label="Track Name",
                                    visible=False,
                                    info="For extract/lego tasks"
                                )

                            complete_track_classes = gr.CheckboxGroup(
                                choices=TRACK_NAMES,
                                label="Track Classes to Complete",
                                visible=False,
                                info="Select which tracks to add to the audio"
                            )

                            instruction_display = gr.Textbox(
                                label="Instruction",
                                value=DEFAULT_DIT_INSTRUCTION,
                                lines=1,
                                info="Task instruction (auto-generated)"
                            )

                        # Audio Inputs
                        with gr.Accordion("Audio Inputs", open=False):
                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label="Reference Audio (style reference)",
                                    type="filepath",
                                )
                                src_audio = gr.Audio(
                                    label="Source Audio (for cover/repaint)",
                                    type="filepath",
                                )
                            with gr.Row():
                                convert_to_codes_btn = gr.Button("Convert to Codes", variant="secondary")

                            audio_code_string = gr.Textbox(
                                label="Audio Codes (LM Hints)",
                                placeholder="Paste audio codes here for guided generation...",
                                lines=4,
                                info="Semantic codes from transcription or previous generation"
                            )
                            transcribe_btn = gr.Button("Transcribe Audio", variant="secondary")

                        # Repainting Controls
                        with gr.Accordion("Repainting Controls", open=False, visible=True) as repainting_accordion:
                            with gr.Row():
                                repainting_start = gr.Number(
                                    label="Repaint Start (s)",
                                    value=0.0,
                                    minimum=0,
                                    step=0.1
                                )
                                repainting_end = gr.Number(
                                    label="Repaint End (s)",
                                    value=-1,
                                    minimum=-1,
                                    step=0.1,
                                    info="-1 means until end"
                                )
                            audio_cover_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                                label="Audio/Codes Strength",
                                info="Influence of reference audio/codes"
                            )

                        # Caption & Lyrics
                        with gr.Accordion("Music Description", open=True):
                            with gr.Row():
                                caption_input = gr.Textbox(
                                    label="Caption",
                                    placeholder="Describe the music style, mood, instruments...",
                                    value=DEFAULT_CAPTION,
                                    lines=3,
                                    scale=4
                                )
                                random_caption_btn = gr.Button("Random", scale=1)

                        with gr.Accordion("Lyrics", open=True):
                            lyrics_input = gr.Textbox(
                                label="Lyrics",
                                placeholder="[Verse]\nYour lyrics here...",
                                value=DEFAULT_LYRICS,
                                lines=10,
                            )
                            with gr.Row():
                                instrumental_checkbox = gr.Checkbox(
                                    label="Instrumental",
                                    value=False,
                                    info="Generate without vocals"
                                )
                                vocal_language = gr.Dropdown(
                                    choices=VALID_LANGUAGES,
                                    value="en",
                                    label="Vocal Language",
                                    allow_custom_value=True,
                                )
                                format_lyrics_btn = gr.Button("Format Lyrics")

                        # Generation Parameters
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                duration = gr.Slider(
                                    label="Duration (s)",
                                    minimum=-1, maximum=300, value=-1, step=10,
                                    info="-1 = auto"
                                )
                                inference_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1, maximum=200, value=50, step=1,
                                )
                            with gr.Row():
                                bpm = gr.Number(label="BPM", value=0, minimum=0, maximum=300, info="0 = auto")
                                key_scale = gr.Textbox(label="Key", placeholder="C Major", value="")
                                time_signature = gr.Dropdown(
                                    choices=["", "2", "3", "4", "6"],
                                    value="",
                                    label="Time Signature"
                                )
                            with gr.Row():
                                batch_count = gr.Number(label="Batch Count", value=1, minimum=1, maximum=8, step=1)
                                seed = gr.Number(label="Seed (-1 = random)", value=-1)
                                random_seed_btn = gr.Button("Random", scale=0, min_width=60)

                        # Advanced Settings
                        with gr.Accordion("Advanced Settings", open=True):
                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=7.0, step=0.1,
                                    label="Guidance Scale",
                                    info="CFG strength (base model only)"
                                )
                                shift = gr.Slider(
                                    minimum=1.0, maximum=5.0, value=3.0, step=0.1,
                                    label="Shift"
                                )
                                infer_method = gr.Dropdown(
                                    choices=["ode", "sde"],
                                    value="ode",
                                    label="Inference Method"
                                )
                            with gr.Row():
                                use_adg = gr.Checkbox(label="Use ADG", value=False, info="Adaptive Dual Guidance")
                                cfg_interval_start = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                                    label="CFG Start"
                                )
                                cfg_interval_end = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                                    label="CFG End"
                                )

                            custom_timesteps = gr.Textbox(
                                label="Custom Timesteps",
                                placeholder="0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0",
                                value="",
                                info="Override inference steps with custom timesteps"
                            )

                        # LM Parameters
                        with gr.Accordion("LM Parameters", open=True):
                            with gr.Row():
                                think_checkbox = gr.Checkbox(label="Enable Thinking", value=False)
                                allow_lm_batch = gr.Checkbox(label="Parallel Thinking", value=False)
                                use_cot_caption = gr.Checkbox(label="Caption Rewrite", value=False)
                            with gr.Row():
                                use_cot_metas = gr.Checkbox(label="CoT Metas", value=False)
                                use_cot_language = gr.Checkbox(label="CoT Language", value=False)
                            with gr.Row():
                                lm_temperature = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=0.85, step=0.1,
                                    label="LM Temperature"
                                )
                                lm_cfg_scale = gr.Slider(
                                    minimum=1.0, maximum=3.0, value=2.0, step=0.1,
                                    label="LM CFG Scale"
                                )
                            with gr.Row():
                                lm_top_k = gr.Slider(
                                    minimum=0, maximum=100, value=0, step=1,
                                    label="LM Top-K"
                                )
                                lm_top_p = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.9, step=0.01,
                                    label="LM Top-P"
                                )
                            lm_negative_prompt = gr.Textbox(
                                label="LM Negative Prompt",
                                value="NO USER INPUT",
                                lines=2
                            )

                        # Output Settings
                        with gr.Accordion("Output Settings", open=True):
                            with gr.Row():
                                audio_format = gr.Dropdown(
                                    choices=["mp3", "flac", "wav"],
                                    value="mp3",
                                    label="Audio Format"
                                )
                                output_folder = gr.Textbox(
                                    label="Output Folder",
                                    value="./output"
                                )

                    # Right column - Controls & Output
                    with gr.Column(scale=2):
                        # Generate buttons
                        with gr.Row():
                            generate_btn = gr.Button(
                                "Generate Music",
                                variant="primary",
                                elem_classes="green-btn",
                                interactive=False
                            )
                            stop_btn = gr.Button("Stop", variant="stop")

                        with gr.Row():
                            save_defaults_btn = gr.Button("Save Defaults")
                            load_defaults_btn = gr.Button("Load Defaults")

                        status_text = gr.Textbox(label="Status", interactive=False, value="Ready", lines=5)

                        # Results Section
                        with gr.Accordion("Generated Music", open=True):
                            audio_outputs = []
                            codes_displays = []

                            for i in range(MAX_AUDIO_OUTPUTS):
                                with gr.Group(visible=(i < 2)) as audio_group:
                                    audio_outputs.append(gr.Audio(
                                        label=f"Song {i+1}",
                                        visible=(i < 2),
                                        interactive=False
                                    ))
                                    with gr.Row():
                                        with gr.Accordion(f"Details {i+1}", open=False):
                                            codes_displays.append(gr.Textbox(
                                                label="Audio Codes",
                                                interactive=False,
                                                lines=3
                                            ))

            # =================================================================
            # Training Tab
            # =================================================================
            with gr.Tab("Training"):
                gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h2>LoRA Training for ACE-Step</h2>
                    <p>Build datasets from your audio files and train custom LoRA adapters</p>
                </div>
                """)

                with gr.Tabs():
                    # Dataset Builder
                    with gr.Tab("Dataset Builder"):
                        with gr.Row():
                            with gr.Column():
                                gr.HTML("<h4>Load Existing Dataset</h4>")
                                with gr.Row():
                                    load_dataset_path = gr.Textbox(
                                        label="Dataset JSON Path",
                                        placeholder="./datasets/my_dataset.json",
                                        scale=3
                                    )
                                    load_dataset_btn = gr.Button("Load", variant="primary", scale=1)
                                load_dataset_status = gr.Textbox(label="Status", interactive=False)

                            with gr.Column():
                                gr.HTML("<h4>Scan New Directory</h4>")
                                with gr.Row():
                                    scan_directory = gr.Textbox(
                                        label="Audio Directory",
                                        placeholder="/path/to/audio/folder",
                                        scale=3
                                    )
                                    scan_btn = gr.Button("Scan", variant="secondary", scale=1)
                                scan_status = gr.Textbox(label="Status", interactive=False)

                        dataset_table = gr.Dataframe(
                            headers=["#", "Filename", "Duration", "Caption", "Labeled"],
                            datatype=["number", "str", "str", "str", "str"],
                            label="Audio Files",
                            interactive=False,
                        )

                        with gr.Row():
                            dataset_name = gr.Textbox(label="Dataset Name", value="my_dataset")
                            custom_tag = gr.Textbox(label="Custom Tag", placeholder="my_style")
                            all_instrumental = gr.Checkbox(label="All Instrumental", value=True)

                        with gr.Row():
                            auto_label_btn = gr.Button("Auto-Label All", variant="primary")
                            save_dataset_btn = gr.Button("Save Dataset", variant="secondary")

                        dataset_status = gr.Textbox(label="Dataset Status", interactive=False, lines=2)

                    # Train LoRA
                    with gr.Tab("Train LoRA"):
                        with gr.Row():
                            with gr.Column():
                                tensor_dir = gr.Textbox(
                                    label="Preprocessed Tensors Directory",
                                    value="./datasets/preprocessed_tensors"
                                )
                                load_tensors_btn = gr.Button("Load Tensors", variant="secondary")
                                tensor_status = gr.Textbox(label="Status", interactive=False)

                            with gr.Column():
                                lora_rank = gr.Slider(minimum=4, maximum=256, step=4, value=64, label="LoRA Rank")
                                lora_alpha = gr.Slider(minimum=4, maximum=512, step=4, value=128, label="LoRA Alpha")
                                lora_dropout = gr.Slider(minimum=0.0, maximum=0.5, step=0.05, value=0.1, label="Dropout")

                        with gr.Row():
                            learning_rate = gr.Number(label="Learning Rate", value=3e-4)
                            train_epochs = gr.Slider(minimum=100, maximum=4000, step=100, value=1000, label="Epochs")
                            train_batch_size = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Batch Size")
                            gradient_accumulation = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Gradient Accum")

                        lora_output_dir = gr.Textbox(label="Output Directory", value="./lora_output")

                        with gr.Row():
                            start_training_btn = gr.Button("Start Training", variant="primary")
                            stop_training_btn = gr.Button("Stop Training", variant="stop")

                        training_progress = gr.Textbox(label="Training Progress", interactive=False, lines=2)
                        training_log = gr.Textbox(label="Training Log", interactive=False, lines=8)

            # =================================================================
            # Post-Processing Tab
            # =================================================================
            with gr.Tab("Post-Processing"):
                gr.Markdown("### Audio Post-Processing")
                gr.Markdown("Load an audio file and apply various effects.")

                pp_audio_state = gr.State(value=None)

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Input Audio", open=True):
                            pp_audio_input = gr.Audio(label="Load Audio File", type="filepath")
                            pp_load_btn = gr.Button("Load Audio", variant="primary")
                            pp_load_status = gr.Textbox(label="Status", interactive=False, lines=2)

                        with gr.Accordion("Trim / Cut", open=True):
                            with gr.Row():
                                pp_trim_start = gr.Number(label="Start Time (s)", value=0, minimum=0)
                                pp_trim_end = gr.Number(label="End Time (s)", value=30, minimum=0)
                            pp_trim_btn = gr.Button("Trim Audio")

                        with gr.Accordion("Loudness", open=True):
                            pp_gain_db = gr.Slider(label="Gain (dB)", minimum=-20, maximum=20, value=0, step=0.5)
                            pp_loudness_btn = gr.Button("Adjust Loudness")
                            gr.Markdown("---")
                            pp_normalize_db = gr.Slider(label="Target Peak (dB)", minimum=-12, maximum=0, value=-1, step=0.5)
                            pp_normalize_btn = gr.Button("Normalize")

                        with gr.Accordion("EQ / Tone", open=True):
                            gr.Markdown("**Bass Boost**")
                            with gr.Row():
                                pp_bass_boost_db = gr.Slider(label="Boost (dB)", minimum=0, maximum=12, value=6, step=0.5)
                                pp_bass_cutoff = gr.Slider(label="Cutoff (Hz)", minimum=60, maximum=300, value=150, step=10)
                            pp_bass_btn = gr.Button("Apply Bass Boost")

                            gr.Markdown("**Treble Boost**")
                            with gr.Row():
                                pp_treble_boost_db = gr.Slider(label="Boost (dB)", minimum=0, maximum=12, value=6, step=0.5)
                                pp_treble_cutoff = gr.Slider(label="Cutoff (Hz)", minimum=2000, maximum=10000, value=4000, step=100)
                            pp_treble_btn = gr.Button("Apply Treble Boost")

                        with gr.Accordion("Fades", open=True):
                            with gr.Row():
                                pp_fade_in_duration = gr.Slider(label="Fade-In (s)", minimum=0, maximum=10, value=2, step=0.1)
                                pp_fade_in_btn = gr.Button("Fade In")
                            with gr.Row():
                                pp_fade_out_duration = gr.Slider(label="Fade-Out (s)", minimum=0, maximum=10, value=2, step=0.1)
                                pp_fade_out_btn = gr.Button("Fade Out")

                        with gr.Accordion("Speed", open=True):
                            pp_speed_factor = gr.Slider(label="Speed Factor", minimum=0.5, maximum=2.0, value=1.0, step=0.05)
                            pp_speed_btn = gr.Button("Change Speed")

                    with gr.Column(scale=1):
                        with gr.Accordion("Output", open=True):
                            pp_output_audio = gr.Audio(label="Processed Audio", interactive=False)
                            pp_output_status = gr.Textbox(label="Processing Status", interactive=False, lines=2)

        # =====================================================================
        # Event Handlers
        # =====================================================================

        def randomize_seed():
            return -1

        random_seed_btn.click(fn=randomize_seed, outputs=[seed])

        # Task type change handler
        def on_task_type_change(task_type_val, track_name_val, complete_classes_val, init_llm_val):
            # Update instruction based on task type and track info
            instruction = get_instruction_for_task(task_type_val, track_name_val, complete_classes_val)
            # Show/hide track name for extract/lego tasks
            show_track = task_type_val in ["extract", "lego"]
            # Show complete_track_classes for complete task
            show_complete = task_type_val == "complete"
            # Show repainting controls for repaint/lego/cover
            show_repaint = task_type_val in ["repaint", "lego", "cover"]
            return (
                instruction,
                gr.update(visible=show_track),
                gr.update(visible=show_complete),
                gr.update(visible=show_repaint)
            )

        task_type.change(
            fn=on_task_type_change,
            inputs=[task_type, track_name, complete_track_classes, init_llm_checkbox],
            outputs=[instruction_display, track_name, complete_track_classes, repainting_accordion]
        )

        # Update instruction when track_name changes
        track_name.change(
            fn=lambda tt, tn, cc, llm: get_instruction_for_task(tt, tn, cc),
            inputs=[task_type, track_name, complete_track_classes, init_llm_checkbox],
            outputs=[instruction_display]
        )

        # Update instruction when complete_track_classes changes
        complete_track_classes.change(
            fn=lambda tt, tn, cc, llm: get_instruction_for_task(tt, tn, cc),
            inputs=[task_type, track_name, complete_track_classes, init_llm_checkbox],
            outputs=[instruction_display]
        )

        # Model type change - update task type choices
        def on_model_type_change(model_type):
            if model_type == "turbo":
                return gr.update(choices=TASK_TYPES_TURBO, value="text2music")
            else:
                return gr.update(choices=TASK_TYPES_BASE, value="text2music")

        model_type.change(
            fn=on_model_type_change,
            inputs=[model_type],
            outputs=[task_type]
        )

        # Initialize model
        init_btn.click(
            fn=initialize_service,
            inputs=[
                checkpoint_path, model_type, device, init_llm_checkbox, lm_model, backend,
                use_flash_attention, compile_checkbox, offload_cpu, offload_dit_cpu, quantization
            ],
            outputs=[init_status, generate_btn]
        )

        # LoRA handlers
        load_lora_btn.click(
            fn=load_lora_adapter,
            inputs=[lora_path, use_lora_checkbox, lora_scale_slider],
            outputs=[lora_status, use_lora_checkbox, lora_scale_slider]
        )

        unload_lora_btn.click(
            fn=unload_lora_adapter,
            outputs=[lora_status, use_lora_checkbox, lora_scale_slider]
        )

        # Convert to codes
        convert_to_codes_btn.click(
            fn=convert_audio_to_codes,
            inputs=[src_audio],
            outputs=[audio_code_string]
        )

        # Format lyrics
        format_lyrics_btn.click(
            fn=format_lyrics,
            inputs=[lyrics_input],
            outputs=[lyrics_input]
        )

        # Random caption
        random_caption_btn.click(
            fn=generate_random_caption,
            outputs=[caption_input]
        )

        # Generate music
        generate_outputs = audio_outputs + [codes_state, status_text]

        generate_event = generate_btn.click(
            fn=generate_music,
            inputs=[
                caption_input, lyrics_input, vocal_language, instrumental_checkbox,
                bpm, key_scale, time_signature, duration, batch_count, seed,
                inference_steps, guidance_scale, shift, infer_method, custom_timesteps,
                task_type, instruction_display,
                reference_audio, src_audio, audio_code_string,
                repainting_start, repainting_end, audio_cover_strength,
                use_adg, cfg_interval_start, cfg_interval_end,
                think_checkbox, allow_lm_batch,
                lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
                use_cot_metas, use_cot_caption, use_cot_language,
                use_lora_checkbox, lora_scale_slider,
                output_folder, audio_format
            ],
            outputs=generate_outputs
        )

        stop_btn.click(
            fn=stop_generation,
            outputs=[status_text],
            cancels=[generate_event]
        )

        # Save/Load Defaults
        defaults_components = [
            caption_input, lyrics_input, vocal_language, instrumental_checkbox,
            batch_count, seed, checkpoint_path, model_type, device, init_llm_checkbox,
            lm_model, backend, use_flash_attention, compile_checkbox, offload_cpu,
            offload_dit_cpu, quantization,
            duration, inference_steps, guidance_scale, bpm, key_scale, time_signature,
            shift, infer_method, custom_timesteps,
            task_type,
            use_adg, cfg_interval_start, cfg_interval_end,
            think_checkbox, allow_lm_batch, use_cot_metas, use_cot_caption, use_cot_language,
            lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
            audio_format, output_folder
        ]

        defaults_keys = [
            "caption", "lyrics", "vocal_language", "instrumental",
            "batch_count", "seed", "checkpoint_path", "model_type", "device", "init_llm",
            "lm_model", "backend", "use_flash_attention", "compile_model", "offload_cpu",
            "offload_dit_cpu", "quantization",
            "duration", "inference_steps", "guidance_scale", "bpm", "key_scale", "time_signature",
            "shift", "infer_method", "custom_timesteps",
            "task_type",
            "use_adg", "cfg_interval_start", "cfg_interval_end",
            "think", "allow_lm_batch", "use_cot_metas", "use_cot_caption", "use_cot_language",
            "lm_temperature", "lm_cfg_scale", "lm_top_k", "lm_top_p", "lm_negative_prompt",
            "audio_format", "output_folder"
        ]

        def save_defaults(*values):
            settings = {}
            for i, key in enumerate(defaults_keys):
                settings[key] = values[i]
            try:
                with open(ACE_STEP_DEFAULTS_FILE, 'w') as f:
                    json.dump(settings, f, indent=2)
                return "Defaults saved successfully."
            except Exception as e:
                return f"Error saving defaults: {e}"

        def load_defaults():
            if not os.path.exists(ACE_STEP_DEFAULTS_FILE):
                return [gr.update()] * len(defaults_keys) + ["No defaults file found."]

            try:
                with open(ACE_STEP_DEFAULTS_FILE, 'r') as f:
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

        # Dataset handlers
        scan_btn.click(
            fn=scan_audio_directory,
            inputs=[scan_directory],
            outputs=[dataset_table, scan_status]
        )

        load_dataset_btn.click(
            fn=load_dataset_json,
            inputs=[load_dataset_path],
            outputs=[dataset_table, load_dataset_status]
        )

        save_dataset_btn.click(
            fn=lambda data, name: save_dataset_json(data, name),
            inputs=[dataset_table, dataset_name],
            outputs=[dataset_status]
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
            results = load_defaults()
            return results[:-1]  # Exclude status message

        demo.load(
            fn=initial_load_defaults,
            inputs=None,
            outputs=defaults_components
        )

    return demo


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


# Build the interface
demo = build_interface()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACE-Step Music Generation UI")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port. Auto-detects available port if not specified.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    args = parser.parse_args()

    # Determine port
    if args.port is not None:
        port = args.port
    else:
        port = find_available_port(7860)

    print(f"Starting ACE-Step Gradio UI on port {port}...")
    print(f"Open http://localhost:{port} in your browser")

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS
    )
