from heartlib import HeartMuLaGenPipeline
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
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

    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=torch.device("cuda"),
        dtype=dtype,
        version=args.version,
    )
    with torch.no_grad():
        pipe(
            {
                "lyrics": args.lyrics,
                "tags": args.tags,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    print(f"Generated music saved to {args.save_path}")
