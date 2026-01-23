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

# --- WINDOWS & TORCH 2.8 SYSTEM FIXES ---
if torch.cuda.is_available():
    try:
        import torch._inductor.config
        import torch._dynamo
        
        # 1. Fix "int too large" OverflowError
        for target in [torch._inductor.config, getattr(torch._inductor.config, 'triton', None)]:
            if target and hasattr(target, 'static_cuda_launcher'):
                target.static_cuda_launcher = False
        
        # 2. VRAM Safety: Disable CUDA Graphs to prevent 4-minute slow-downs
        torch._inductor.config.triton.cudagraphs = False
        
        # 3. Stability Tweaks
        torch._inductor.config.fallback_random = True
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 32
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

# Global Constants
HEARTMULA_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), "heartmula_defaults.json")
MODEL_LAYER_COUNTS = {"3B": {"backbone": 28, "decoder": 3}, "7B": {"backbone": 32, "decoder": 3}}
MAX_AUDIO_OUTPUTS = 12

# Global State
stop_event = threading.Event()
_cached_pipe = None
_cached_config = None 

def stop_generation():
    """Signal to stop the current generation."""
    stop_event.set()
    return "Stopping..."

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# --- COMPONENT ENFORCEMENT ---

def _strip_hooks_nuclear(model):
    """
    Removes the 'hooks' that automatically move layers to CPU.
    """
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except: pass

    # Deep cleanup
    for m in model.modules():
        if hasattr(m, "_hf_hook"): del m._hf_hook
        if hasattr(m, "_old_forward"): 
            m.forward = m._old_forward
            del m._old_forward
        if hasattr(m, "hf_device_map"): del m.hf_device_map

def _enforce_gpu_alignment(model, device):
    """
    Specifically targets the layers causing the RuntimeError and forces them to GPU.
    """
    # 1. Force the main model container
    model.to(device)
    
    # 2. Target Critical Components (Embeddings & Heads)
    # These are the specific layers that drift to CPU and crash the compilation
    critical_components = [
        "text_embeddings", 
        "audio_embeddings", 
        "unconditional_text_embedding",
        "projection",
        "codebook0_head",
        "muq_linear",
        "decoder",
        "audio_head"
    ]
    
    for name in critical_components:
        if hasattr(model, name):
            comp = getattr(model, name)
            comp.to(device)
            # Double check weights inside
            if hasattr(comp, "weight"):
                comp.weight.data = comp.weight.data.to(device)

    # 3. Target the Causal Mask Buffer (The other source of crashes)
    if hasattr(model, "backbone_causal_mask") and model.backbone_causal_mask is not None:
        model.backbone_causal_mask = model.backbone_causal_mask.to(device)

    # 4. Lock methods
    model.cpu = lambda: model

def create_audio_outputs(file_paths: list, labels: list = None) -> list:
    if labels is None: labels = [os.path.basename(p) for p in file_paths] if file_paths else []
    outputs = []
    for i in range(MAX_AUDIO_OUTPUTS):
        if i < len(file_paths):
            outputs.append(gr.update(value=file_paths[i], label=labels[i], visible=True))
        else:
            outputs.append(gr.update(value=None, visible=False))
    return outputs

# --- GENERATION LOGIC ---

def generate_music(
    lyrics, tags, negative_prompt, max_duration_seconds,
    temperature, topk, cfg_scale, model_path,
    model_version, num_gpu_blocks, model_dtype,
    batch_count, seed, output_folder,
    compile_model, ref_audio, num_steps, ref_audio_sec
):
    global stop_event, _cached_pipe, _cached_config
    stop_event.clear()
    all_generated_music = []
    
    current_config = (model_path, model_version, model_dtype, num_gpu_blocks, compile_model, ref_audio is not None)

    try:
        # --- 1. LOAD MODEL ---
        if _cached_pipe is None or _cached_config != current_config:
            yield (*create_audio_outputs([]), "Loading Model & Aligning Memory...")
            
            if _cached_pipe is not None:
                del _cached_pipe
                import gc; gc.collect()
                torch.cuda.empty_cache()

            from heartlib import HeartMuLaGenPipeline
            device = torch.device("cuda")
            dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(model_dtype, torch.bfloat16)

            pipe = HeartMuLaGenPipeline.from_pretrained(
                model_path, device=device, dtype=dtype, version=model_version,
                skip_model_move=True, load_muq_mulan=(ref_audio is not None and os.path.isfile(ref_audio)),
            )

            # --- NUCLEAR MEMORY FIXES ---
            pipe._skip_auto_move = True # Tell pipe logic to stop
            _strip_hooks_nuclear(pipe.model)
            _enforce_gpu_alignment(pipe.model, device)
            
            if hasattr(pipe, "audio_codec"): 
                pipe.audio_codec.to(device)
                _strip_hooks_nuclear(pipe.audio_codec)

            # --- 2. COMPILE ---
            if compile_model:
                yield (*create_audio_outputs([]), "Compiling Backbone (First run ~2 mins)...")
                log("Compiling backbone...")
                
                # Only compile the backbone to keep embeddings flexible and avoid crashes
                pipe.model.backbone = torch.compile(pipe.model.backbone, mode="default", dynamic=True)
                
                # Warmup
                warmup_out = os.path.join(tempfile.gettempdir(), f"w_{int(time.time())}.mp3")
                try:
                    with torch.inference_mode():
                        pipe({"lyrics": "test", "tags": "test"}, max_audio_length_ms=500, save_path=warmup_out)
                    log("Compilation successful.")
                except Exception as e:
                    log(f"Warmup notice: {e}")

            _cached_pipe = pipe
            _cached_config = current_config

        pipe = _cached_pipe
        max_ms = int(max_duration_seconds * 1000)

        # --- 3. GENERATION LOOP ---
        for i in range(int(batch_count)):
            if stop_event.is_set(): break
            
            current_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed) + i
            torch.manual_seed(current_seed)
            
            yield (*create_audio_outputs(all_generated_music), f"Generating Song {i+1}/{batch_count} (Seed: {current_seed})...")

            # CRITICAL FIX: Re-verify that embeddings are on CUDA before inference
            # This catches any drift that happened between songs
            _enforce_gpu_alignment(pipe.model, torch.device("cuda"))

            output_path = os.path.join(output_folder, f"heart_{int(time.time())}_{current_seed}.mp3")
            inputs = {"lyrics": lyrics, "tags": tags}
            if ref_audio and os.path.isfile(ref_audio):
                inputs["ref_audio"] = ref_audio
                inputs["muq_segment_sec"] = ref_audio_sec

            # Use inference_mode() to save VRAM
            with torch.inference_mode():
                pipe(
                    inputs, 
                    max_audio_length_ms=max_ms, 
                    save_path=output_path, 
                    topk=topk, 
                    temperature=temperature, 
                    cfg_scale=cfg_scale, 
                    num_steps=num_steps, 
                    negative_prompt=negative_prompt
                )

            all_generated_music.append(output_path)
            labels = [f"Song {j+1} (Seed: {current_seed})" for j in range(len(all_generated_music))]
            yield (*create_audio_outputs(all_generated_music, labels), f"Completed {i+1}/{batch_count}")

        yield (*create_audio_outputs(all_generated_music, labels), "Done!")

    except Exception as e:
        import traceback
        log(f"Error: {traceback.format_exc()}")
        yield (*create_audio_outputs(all_generated_music), f"Error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- UI SETUP ---

DEFAULT_LYRICS = """[Intro]\n\n[Verse]\nThe sun creeps in across the floor\nMy cat is stretching by the door\nShe yawns and blinks her emerald eyes\nAnother day begins to rise\n\n[Chorus]\nLittle cat with whiskers long\nYou make my heart sing like a song\nCurled up warm upon my knee\nYou are everything to me"""
DEFAULT_TAGS = "piano,happy"

with gr.Blocks(theme=themes.Default(primary_hue="purple"), title="HeartMuLa Music Generator") as demo:
    with gr.Tabs():
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Accordion("Lyrics", open=True):
                        lyrics_in = gr.Textbox(label="main lyric prompt", value=DEFAULT_LYRICS, lines=12)
                    tags_in = gr.Textbox(label="Style Tags", value=DEFAULT_TAGS)
                    neg_in = gr.Textbox(label="Negative Prompt")
                    with gr.Row():
                        batch_in = gr.Number(label="Batch Count", value=1, minimum=1)
                        seed_in = gr.Number(label="Seed (-1 = random)", value=-1)
                        rand_btn = gr.Button("🎲")
                with gr.Column(scale=1):
                    status = gr.Textbox(label="Status", interactive=False)
                    with gr.Accordion("Ref Audio", open=True):
                        ref_aud_in = gr.Audio(label="Reference Audio", type="filepath")
                        steps_in = gr.Slider(label="Steps", value=10, minimum=1, maximum=50)
                        ref_sec_in = gr.Slider(label="Ref Length (s)", value=30, minimum=10, maximum=120)
            
            with gr.Row():
                gen_btn = gr.Button("🎵 Generate Music", variant="primary")
                stop_btn = gr.Button("⏹️ Stop", variant="stop")
                save_btn = gr.Button("💾 Save Defaults")
                load_btn = gr.Button("📂 Load Defaults")

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Model Settings", open=True):
                        path_in = gr.Textbox(label="Model Path", value="./ckpt")
                        ver_in = gr.Dropdown(label="Model Version", choices=["3B", "7B"], value="3B")
                        prec_in = gr.Dropdown(label="Precision", choices=["fp32", "bf16", "fp16"], value="bf16")
                        gpu_in = gr.Slider(label="GPU Blocks", value=14, minimum=0, maximum=32)
                        out_in = gr.Textbox(label="Output Folder", value="./output")
                        comp_in = gr.Checkbox(label="Compile Model (Speed Boost)", value=False)
                    with gr.Accordion("Generation Parameters", open=True):
                        dur_in = gr.Slider(label="Max Duration (s)", value=120, minimum=10, maximum=240)
                        temp_in = gr.Slider(label="Temperature", value=1.0, minimum=0.1, maximum=2.0)
                        topk_in = gr.Slider(label="Top-K", value=50, minimum=1, maximum=200)
                        cfg_in = gr.Slider(label="CFG Scale", value=1.5, minimum=1.0, maximum=10.0)
                with gr.Column():
                    outs = [gr.Audio(label=f"Song {i+1}", visible=False) for i in range(MAX_AUDIO_OUTPUTS)]

        with gr.Tab("Audio Info"):
            gr.Markdown("### Load Settings from MP3")
            info_aud = gr.Audio(label="Upload MP3 File", type="filepath")
            load_meta_btn = gr.Button("📥 Load Settings")
            info_stat = gr.Textbox(label="Status", interactive=False)
            info_lyr = gr.Textbox(label="Lyrics", lines=8, interactive=False)
            info_tag = gr.Textbox(label="Tags", interactive=False)

    # --- EVENT BINDINGS ---
    rand_btn.click(lambda: -1, outputs=[seed_in])
    
    # Metadata Logic
    def read_meta(f):
        if not f: return "No file", "", ""
        try:
            from heartlib import HeartMuLaGenPipeline
            m = HeartMuLaGenPipeline.read_mp3_metadata(f)
            return "Loaded", m.get("lyrics",""), m.get("tags","")
        except: return "Error reading metadata", "", ""
    info_aud.change(read_meta, [info_aud], [info_stat, info_lyr, info_tag])

    # Defaults Logic
    defaults_cmps = [lyrics_in, tags_in, neg_in, batch_in, seed_in, path_in, ver_in, prec_in, gpu_in, out_in, dur_in, temp_in, topk_in, cfg_in, comp_in, steps_in]
    defaults_keys = ["lyrics", "tags", "negative_prompt", "batch_count", "seed", "model_path", "model_version", "model_dtype", "num_gpu_blocks", "output_folder", "max_duration", "temperature", "topk", "cfg_scale", "compile_model", "num_steps"]
    
    def save_defaults(*vals):
        d = {k:v for k,v in zip(defaults_keys, vals)}
        try: 
            with open(HEARTMULA_DEFAULTS_FILE, 'w') as f: json.dump(d, f, indent=2)
            return "Defaults saved."
        except Exception as e: return f"Error: {e}"
    
    def load_defaults():
        if not os.path.exists(HEARTMULA_DEFAULTS_FILE): return [gr.update()]*len(defaults_keys) + [""]
        try:
            with open(HEARTMULA_DEFAULTS_FILE, 'r') as f: d = json.load(f)
            return [gr.update(value=d.get(k)) for k in defaults_keys] + ["Defaults loaded."]
        except: return [gr.update()]*len(defaults_keys) + ["Error loading defaults."]

    save_btn.click(save_defaults, defaults_cmps, [status])
    load_btn.click(load_defaults, outputs=defaults_cmps + [status])
    
    demo.load(lambda: load_defaults()[:-1], outputs=defaults_cmps)

    # Generation Trigger
    gen_btn.click(
        generate_music, 
        [lyrics_in, tags_in, neg_in, dur_in, temp_in, topk_in, cfg_in, path_in, ver_in, gpu_in, prec_in, batch_in, seed_in, out_in, comp_in, ref_aud_in, steps_in, ref_sec_in],
        outs + [status]
    )
    stop_btn.click(stop_generation, outputs=[status])

# --- PORT FINDER ---
def find_port(p):
    import socket
    while True:
        try:
            with socket.socket() as s: 
                s.bind(('', p)); return p
        except: p += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    port = find_port(args.port or 7860)
    os.makedirs("./output", exist_ok=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=args.share)