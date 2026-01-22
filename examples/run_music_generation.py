from heartlib import HeartMuLaGenPipeline
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music with HeartMuLa using optional reference audio conditioning"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--lyrics", type=str, default="./assets/lyrics.txt")
    parser.add_argument("--tags", type=str, default="./assets/tags.txt")
    parser.add_argument("--save_path", type=str, default="./assets/output.mp3")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                        help="Model precision: fp32 (best quality), bf16 (balanced), fp16 (smallest VRAM)")

    parser.add_argument("--max_audio_length_ms", type=int, default=240_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)

    # Reference audio arguments
    parser.add_argument("--ref_audio", type=str, default=None,
                        help="Path to reference audio for style conditioning (uses MuQ-MuLan embeddings)")
    parser.add_argument("--muq_segment_sec", type=float, default=10.0,
                        help="Length of segment to sample for MuQ-MuLan embedding (default: 10s)")

    # Optional img2img latent blending (in addition to MuQ-MuLan conditioning)
    parser.add_argument("--enable_img2img", action="store_true",
                        help="Also enable latent blending during flow matching")
    parser.add_argument("--ref_strength", type=float, default=0.7,
                        help="Img2img strength: 1.0=ignore reference, 0.0=pure reference. Default: 0.7")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Flow matching steps for img2img (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[args.dtype]
    print(f"Using dtype: {dtype}")

    # Determine if we should load MuQ-MuLan for reference audio
    # MuQ-MuLan checkpoint expected at: {model_path}/MuQ-MuLan-large/
    load_muq = args.ref_audio is not None
    if load_muq:
        print(f"Reference audio provided: {args.ref_audio}")
        print(f"Loading MuQ-MuLan from: {args.model_path}/MuQ-MuLan-large/")

    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=torch.device("cuda"),
        dtype=dtype,
        version=args.version,
        load_muq_mulan=load_muq,
    )
    with torch.no_grad():
        inputs = {
            "lyrics": args.lyrics,
            "tags": args.tags,
        }
        if args.ref_audio:
            inputs["ref_audio"] = args.ref_audio

        pipe(
            inputs,
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            ref_audio=args.ref_audio,
            muq_segment_sec=args.muq_segment_sec,
            enable_img2img=args.enable_img2img,
            ref_strength=args.ref_strength,
            num_steps=args.num_steps,
        )

    if args.ref_audio:
        mode = "MuQ-MuLan conditioning"
        if args.enable_img2img:
            mode += f" + img2img (strength={args.ref_strength})"
        print(f"Generated music ({mode}) saved to {args.save_path}")
    else:
        print(f"Generated music saved to {args.save_path}")
