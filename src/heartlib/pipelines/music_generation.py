import json
import os
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torchaudio
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.pipelines.base import Pipeline
from transformers.utils.generic import ModelOutput

from ..heartcodec.modeling_heartcodec import HeartCodec
from ..heartmula.modeling_heartmula import HeartMuLa
from ..accelerators.torchtune_metal import try_enable_torchtune_metal

# Metadata constants for MP3 ID3 tags
HEARTMULA_METADATA_PREFIX = "HEARTMULA_"


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)


class HeartMuLaGenPipeline(Pipeline):
    def __init__(
        self,
        model: HeartMuLa,
        audio_codec: HeartCodec,
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
        skip_model_move: bool = False,
    ):
        # Don't call super().__init__ with model if we want to skip the automatic .to(device)
        # This allows block swapping to be set up before moving model to GPU
        if skip_model_move:
            # Manually set attributes that Pipeline.__init__ would set
            self.model = model
            self.device = device
            self._dtype = dtype
            self.framework = "pt"
            self._num_workers = None
            self._batch_size = None
            self._preprocess_params = {}
            self._forward_params = {}
            self._postprocess_params = {}
            self.call_count = 0
            self.tokenizer = None
            self.feature_extractor = None
            self.image_processor = None
            self.processor = None
        else:
            super().__init__(model, device=device, dtype=dtype)
            self.model = model
            self._dtype = dtype

        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config
        self._device = device

        # Optional, opt-in MPS fast path (custom Metal kernels) for torchtune Llama blocks.
        # Enable with: HEARTLIB_ENABLE_MPS_METAL=1
        try:
            try_enable_torchtune_metal(
                self.model,
                enabled=(os.getenv("HEARTLIB_ENABLE_MPS_METAL", "0") == "1"),
                verbose=(os.getenv("HEARTLIB_MPS_METAL_VERBOSE", "0") == "1"),
            )
        except Exception:
            # Never fail inference if optional kernels are unavailable.
            pass

        self._parallel_number = audio_codec.config.num_quantizers + 1
        self._muq_dim = model.config.muq_dim

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "negative_prompt": kwargs.get("negative_prompt", None),
            "ref_audio": kwargs.get("ref_audio", None),  # For img2img latent conditioning
            "ref_strength": kwargs.get("ref_strength", 0.7),
        }
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "num_steps": kwargs.get("num_steps", 10),
            "ref_strength": kwargs.get("ref_strength", 0.7),
        }
        postprocess_kwargs = {
            "save_path": kwargs.get("save_path", "output.mp3"),
            "codes_path": kwargs.get("codes_path", None),
            "metadata": kwargs.get("metadata", None),
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, input_: Dict[str, Any], **preprocess_parameters: Any):
        cfg_scale: float = preprocess_parameters.get("cfg_scale", 1.5)
        negative_prompt: Optional[str] = preprocess_parameters.get("negative_prompt", None)
        ref_audio_kwarg: Optional[str] = preprocess_parameters.get("ref_audio", None)
        ref_strength: float = preprocess_parameters.get("ref_strength", 0.7)

        tags = input_["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        def _load_ref_audio(ref: Any) -> tuple[torch.Tensor, int]:
            """
            Returns (mono_waveform, sample_rate) where mono_waveform is 1D [T].
            """
            if isinstance(ref, str):
                wav, sr = torchaudio.load(ref)
            elif isinstance(ref, torch.Tensor):
                wav = ref
                sr = int(input_.get("ref_audio_sr", 0) or 0)
                if sr <= 0:
                    raise ValueError(
                        "ref_audio was provided as a Tensor but `ref_audio_sr` was missing/invalid."
                    )
            else:
                raise TypeError(
                    f"ref_audio must be a file path or torch.Tensor, got {type(ref)}"
                )

            # Accept [T], [C,T], or [B,C,T] (take the first batch).
            if wav.ndim == 3:
                wav = wav[0]
            if wav.ndim == 2:
                wav = wav.mean(dim=0)
            elif wav.ndim != 1:
                raise ValueError(f"Unsupported ref_audio tensor shape: {tuple(wav.shape)}")

            wav = wav.to(dtype=torch.float32)
            return wav, int(sr)

        def _prepare_muq_audio(wav: torch.Tensor, sr: int) -> torch.Tensor:
            """
            Resample to MuQ sample rate (default 24k) and take/pad a ~10s segment.
            Returns waveform shaped [1, T] on self._device.
            """
            muq_sr = int(input_.get("muq_sample_rate", 24_000))
            seg_s = float(input_.get("muq_segment_sec", 10.0))
            seg_len = max(1, int(round(muq_sr * seg_s)))

            if sr != muq_sr:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=muq_sr)

            if wav.numel() >= seg_len:
                start = (wav.numel() - seg_len) // 2
                wav = wav[start : start + seg_len]
            else:
                wav = F.pad(wav, (0, seg_len - wav.numel()))

            # Common MuQ-style encoders expect [B, T].
            return wav.unsqueeze(0).to(device=self._device)

        def _run_muq_mulan(audio_bt: torch.Tensor, sample_rate: int) -> torch.Tensor:
            """
            Runs the provided MuQ-MuLan model and returns a 1D [muq_dim] embedding.
            Tries a few common APIs / output layouts.
            """
            if self.muq_mulan is None:
                raise ValueError(
                    "ref_audio was provided but `muq_mulan` is None. "
                    "Pass a pretrained MuQ-MuLan model to HeartMuLaGenPipeline."
                )

            model = self.muq_mulan
            was_training = getattr(model, "training", False)
            if hasattr(model, "eval"):
                model.eval()

            with torch.inference_mode():
                out = None
                # Common: model.encode_audio(audio, sample_rate=...)
                if hasattr(model, "encode_audio") and callable(getattr(model, "encode_audio")):
                    try:
                        out = model.encode_audio(audio_bt, sample_rate=sample_rate)
                    except TypeError:
                        out = model.encode_audio(audio_bt)
                # Fallback: callable model(audio, sample_rate=...)
                if out is None and callable(model):
                    try:
                        out = model(audio_bt, sample_rate=sample_rate)
                    except TypeError:
                        out = model(audio_bt)

            if was_training and hasattr(model, "train"):
                model.train()

            def _to_tensor(x: Any) -> Optional[torch.Tensor]:
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return x
                if isinstance(x, (tuple, list)) and x:
                    return _to_tensor(x[0])
                if isinstance(x, (dict, ModelOutput)):
                    for k in (
                        "joint_embedding",
                        "joint_embeds",
                        "embedding",
                        "embeddings",
                        "audio_embedding",
                        "audio_embeds",
                        "audio_embed",
                        "audio_features",
                        "audio_feature",
                    ):
                        if k in x:
                            return _to_tensor(x[k])
                for attr in (
                    "joint_embedding",
                    "embedding",
                    "embeddings",
                    "audio_embedding",
                    "audio_embeds",
                    "audio_features",
                ):
                    if hasattr(x, attr):
                        return _to_tensor(getattr(x, attr))
                return None

            emb = _to_tensor(out)
            if emb is None:
                raise ValueError(
                    "Could not extract an embedding from `muq_mulan` output. "
                    "Expected a Tensor or a dict/ModelOutput with an embedding field."
                )

            # Accept [D], [1,D], or [B,D] (take first).
            emb = emb.detach()
            if emb.ndim == 2:
                emb = emb[0]
            elif emb.ndim != 1:
                raise ValueError(f"Unsupported muq embedding shape: {tuple(emb.shape)}")

            if emb.numel() != self._muq_dim:
                raise ValueError(
                    f"MuQ-MuLan embedding dim mismatch: expected {self._muq_dim}, got {emb.numel()}."
                )

            # Normalize is common for joint embeddings; safe and improves conditioning stability.
            emb = emb / (emb.norm(p=2) + 1e-12)
            return emb.to(device="cpu", dtype=self.dtype)

        # Semantic reference audio for MuQ-MuLan (from input dict)
        ref_audio_semantic = input_.get("ref_audio", None)
        # img2img reference audio for latent conditioning (from kwargs, separate from semantic)
        ref_audio_img2img = ref_audio_kwarg
        ref_latent = None
        ref_audio_path = None

        # Process semantic reference for MuQ-MuLan embedding
        if ref_audio_semantic is not None and self.muq_mulan is not None:
            wav, sr = _load_ref_audio(ref_audio_semantic)
            muq_sr = int(input_.get("muq_sample_rate", 24_000))
            audio_bt = _prepare_muq_audio(wav, sr)
            muq_embed = _run_muq_mulan(audio_bt, sample_rate=muq_sr)

            # Offload MuQ-MuLan to CPU after use to free GPU memory
            if hasattr(self.muq_mulan, 'to'):
                self.muq_mulan.to("cpu")
            torch.cuda.empty_cache()
        else:
            muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype)

        # Process img2img reference for latent conditioning (separate from semantic)
        if ref_audio_img2img is not None and isinstance(ref_audio_img2img, str) and os.path.isfile(ref_audio_img2img) and ref_strength < 1.0:
            ref_audio_path = ref_audio_img2img

            # Ensure audio_codec is on CPU for tokenization to avoid OOM
            # (HeartMuLa model may be using most GPU memory)
            self.audio_codec.to("cpu")

            # Encode reference audio to latent on CPU
            ref_latent = self.audio_codec.tokenize(ref_audio_path, device="cpu")
            # Keep on CPU - will be moved to GPU in _forward when needed

        # The reserved slot is the blank "+1" token after tags_ids.
        muq_idx = len(tags_ids)

        lyrics = input_["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp:
                lyrics = fp.read()
        assert isinstance(
            lyrics, str
        ), f"lyrics must be a string, but got {type(lyrics)}"
        lyrics = lyrics.lower()

        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        # Process negative prompt for classifier-free guidance
        negative_tokens = None
        if negative_prompt is not None and negative_prompt.strip():
            negative_prompt = negative_prompt.lower()
            if not negative_prompt.startswith("<tag>"):
                negative_prompt = f"<tag>{negative_prompt}"
            if not negative_prompt.endswith("</tag>"):
                negative_prompt = f"{negative_prompt}</tag>"
            negative_ids = self.text_tokenizer.encode(negative_prompt).ids
            if negative_ids[0] != self.config.text_bos_id:
                negative_ids = [self.config.text_bos_id] + negative_ids
            if negative_ids[-1] != self.config.text_eos_id:
                negative_ids = negative_ids + [self.config.text_eos_id]
            negative_tokens = torch.tensor(negative_ids, dtype=torch.long)

        def _cfg_cat(tensor: torch.Tensor, scale: float) -> torch.Tensor:
            tensor = tensor.unsqueeze(0)
            if scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
            "negative_tokens": negative_tokens,
            "ref_latent": ref_latent,
            "ref_audio_path": ref_audio_path,
        }

    def _forward(
        self,
        input_tensors: Dict[str, Any],
        **forward_parameters: Any,
    ) -> ModelOutput:
        max_audio_length_ms: int = forward_parameters.get(
            "max_audio_length_ms", 120_000
        )
        temperature: float = forward_parameters.get("temperature", 1.0)
        topk: int = forward_parameters.get("topk", 50)
        cfg_scale: float = forward_parameters.get("cfg_scale", 1.5)
        num_steps: int = forward_parameters.get("num_steps", 10)
        ref_strength: float = forward_parameters.get("ref_strength", 0.7)

        prompt_tokens = input_tensors["tokens"]
        prompt_tokens_mask = input_tensors["tokens_mask"]
        continuous_segment = input_tensors["muq_embed"]
        starts = input_tensors["muq_idx"]
        prompt_pos = input_tensors["pos"]
        negative_tokens = input_tensors.get("negative_tokens", None)
        ref_latent = input_tensors.get("ref_latent", None)
        ref_audio_path = input_tensors.get("ref_audio_path", None)

        # Ensure model is on the correct device (may have been offloaded after previous generation)
        # Skip this check when block swapping is active, as some parameters are intentionally on CPU
        if not getattr(self, '_skip_auto_move', False):
            model_device = next(self.model.parameters()).device
            if model_device != self._device:
                self.model.to(self._device)

        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1

        # Move inputs to the correct device
        prompt_tokens = prompt_tokens.to(self._device)
        prompt_tokens_mask = prompt_tokens_mask.to(self._device)
        continuous_segment = continuous_segment.to(self._device)
        prompt_pos = prompt_pos.to(self._device)

        # Setup caches with explicit device to handle model being temporarily on CPU
        self.model.setup_caches(bs_size, device=self._device)

        # Compute negative embedding for CFG if provided
        negative_embedding = None
        if negative_tokens is not None:
            negative_tokens = negative_tokens.to(self._device)
            with torch.no_grad():
                neg_embeds = self.model.text_embeddings(negative_tokens)
                # Mean-pool across all tokens to get a single embedding
                negative_embedding = neg_embeds.mean(dim=0, keepdim=False)

        device_type = (
            self._device.type if isinstance(self._device, torch.device) else "cpu"
        )
        # Autocast support varies by PyTorch build/version (not all support "mps").
        # Prefer autocast when available, but never fail if unsupported.
        def _autocast_ctx():
            try:
                return torch.autocast(device_type=device_type, dtype=self.dtype)
            except (RuntimeError, TypeError, ValueError):
                return nullcontext()

        autocast_ctx = _autocast_ctx()

        # Keep a stable view of the base position tensor to avoid re-slicing every step.
        base_pos = prompt_pos[..., -1:]

        with torch.inference_mode(), autocast_ctx:
            curr_token = self.model.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
                negative_embedding=negative_embedding,
            )

            # Preallocate the padded audio token + mask and reuse them every step.
            padded_token = torch.full(
                (curr_token.shape[0], 1, self._parallel_number),
                fill_value=self.config.empty_id,
                device=curr_token.device,
                dtype=torch.long,
            )
            padded_token_mask = torch.ones(
                (curr_token.shape[0], 1, self._parallel_number),
                device=curr_token.device,
                dtype=torch.bool,
            )
            padded_token_mask[..., -1] = False

            max_audio_frames = max_audio_length_ms // 80
            # Preallocate a frame buffer for the *un-padded* audio tokens (first sample only).
            frame_buf = torch.empty(
                (max_audio_frames + 1, curr_token.shape[1]),
                device=curr_token.device,
                dtype=curr_token.dtype,
            )
            frame_buf[0] = curr_token[0]
            frame_len = 1

            for i in tqdm(range(max_audio_frames)):
                padded_token[:, 0, :-1] = curr_token
                curr_token = self.model.generate_frame(
                    tokens=padded_token,
                    tokens_mask=padded_token_mask,
                    input_pos=base_pos + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                    negative_embedding=negative_embedding,
                )
                if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                    break
                frame_buf[frame_len] = curr_token[0]
                frame_len += 1

        frames = frame_buf[:frame_len].transpose(0, 1).contiguous()

        # Offload HeartMuLa model to CPU to free GPU memory for detokenize
        # This prevents OOM during the detokenize step
        self.model.reset_caches()
        self.model.to("cpu")
        torch.cuda.empty_cache()

        # Move audio_codec to GPU for detokenization
        self.audio_codec.to(self._device)

        # Move ref_latent to GPU if provided
        if ref_latent is not None:
            ref_latent = ref_latent.to(self._device)

        wav = self.audio_codec.detokenize(
            frames,
            num_steps=num_steps,
            ref_latent=ref_latent,
            ref_strength=ref_strength,
        )

        # Move audio_codec back to CPU to free GPU memory for next generation
        self.audio_codec.to("cpu")
        torch.cuda.empty_cache()

        # Include tokens in the output so postprocess can optionally persist them.
        # This is opt-in (see postprocess `codes_path`) and does not change default behavior.
        return ModelOutput(wav=wav, codes=frames.detach().cpu(), ref_audio_path=ref_audio_path)

    def postprocess(
        self, model_outputs: ModelOutput, **postprocess_parameters: Any
    ) -> None:
        save_path: str = postprocess_parameters.get("save_path", "output.mp3")
        codes_path: Optional[str] = postprocess_parameters.get("codes_path", None)
        metadata: Optional[Dict[str, Any]] = postprocess_parameters.get("metadata", None)
        wav = model_outputs["wav"]
        ref_audio_path = model_outputs.get("ref_audio_path", None)

        # Normalize img2img output to match reference audio loudness
        if ref_audio_path is not None:
            ref_wav, ref_sr = torchaudio.load(ref_audio_path)
            # Resample if needed
            if ref_sr != 48000:
                ref_wav = torchaudio.transforms.Resample(ref_sr, 48000)(ref_wav)

            # Compute RMS
            ref_rms = ref_wav.pow(2).mean().sqrt()
            out_rms = wav.pow(2).mean().sqrt()

            # Scale output to match reference RMS
            if out_rms > 0:
                wav = wav * (ref_rms / out_rms)

        # Determine format from extension
        if save_path.lower().endswith('.mp3'):
            torchaudio.save(save_path, wav, 48000, format="mp3")
        else:
            torchaudio.save(save_path, wav, 48000)

        # Write metadata to MP3 if provided
        if metadata is not None and save_path.lower().endswith('.mp3'):
            self._write_mp3_metadata(save_path, metadata)

        if codes_path:
            codes = model_outputs.get("codes", None)
            if codes is None:
                raise ValueError(
                    "codes_path was provided but no `codes` were found in model outputs."
                )
            torch.save(codes, codes_path)

    def _write_mp3_metadata(self, save_path: str, metadata: Dict[str, Any]):
        """Write generation metadata to MP3 ID3 tags."""
        try:
            from mutagen.mp3 import MP3
            from mutagen.id3 import TXXX, COMM, TIT2, error as ID3Error

            # Load or create ID3 tags
            try:
                audio = MP3(save_path)
                if audio.tags is None:
                    audio.add_tags()
            except ID3Error:
                audio = MP3(save_path)
                audio.add_tags()

            # Set title if we have tags
            if "tags" in metadata:
                audio.tags.add(TIT2(encoding=3, text=f"HeartMuLa: {metadata['tags'][:50]}"))

            # Store each metadata field as a TXXX frame
            for key, value in metadata.items():
                frame_desc = f"{HEARTMULA_METADATA_PREFIX}{key}"
                # Convert value to string for storage
                if isinstance(value, (dict, list)):
                    str_value = json.dumps(value)
                else:
                    str_value = str(value)
                audio.tags.add(TXXX(encoding=3, desc=frame_desc, text=str_value))

            # Also store complete metadata as JSON in a comment for easy parsing
            audio.tags.add(COMM(
                encoding=3,
                lang='eng',
                desc='HEARTMULA_JSON',
                text=json.dumps(metadata)
            ))

            audio.save()
        except ImportError:
            print("Warning: mutagen not installed. Metadata will not be saved to MP3.")
        except Exception as e:
            print(f"Warning: Failed to write metadata to MP3: {e}")

    @staticmethod
    def read_mp3_metadata(file_path: str) -> Optional[Dict[str, Any]]:
        """Read HeartMuLa generation metadata from an MP3 file.

        Returns:
            Dictionary of metadata if found, None otherwise.
        """
        try:
            from mutagen.mp3 import MP3
            from mutagen.id3 import ID3

            audio = MP3(file_path)
            if audio.tags is None:
                return None

            # First try to get the JSON comment (easiest)
            for key in audio.tags.keys():
                if key.startswith('COMM') and 'HEARTMULA_JSON' in str(audio.tags[key]):
                    frame = audio.tags[key]
                    try:
                        return json.loads(frame.text[0] if isinstance(frame.text, list) else frame.text)
                    except (json.JSONDecodeError, IndexError):
                        pass

            # Fallback: reconstruct from TXXX frames
            metadata = {}
            for key in audio.tags.keys():
                if key.startswith('TXXX'):
                    frame = audio.tags[key]
                    if frame.desc.startswith(HEARTMULA_METADATA_PREFIX):
                        field_name = frame.desc[len(HEARTMULA_METADATA_PREFIX):]
                        value = frame.text[0] if isinstance(frame.text, list) else frame.text
                        # Try to parse as JSON for complex types
                        try:
                            metadata[field_name] = json.loads(value)
                        except json.JSONDecodeError:
                            # Try to convert numeric strings
                            try:
                                if '.' in value:
                                    metadata[field_name] = float(value)
                                else:
                                    metadata[field_name] = int(value)
                            except ValueError:
                                metadata[field_name] = value

            return metadata if metadata else None

        except ImportError:
            print("Warning: mutagen not installed. Cannot read MP3 metadata.")
            return None
        except Exception as e:
            print(f"Warning: Failed to read metadata from MP3: {e}")
            return None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
        version: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        skip_model_move: bool = False,
        *,
        load_muq_mulan: bool = False,
        muq_model_id: Optional[str] = None,
        muq_cache_dir: Optional[str] = None,
        muq_revision: Optional[str] = None,
    ):
        if os.path.exists(
            heartcodec_path := os.path.join(pretrained_path, "HeartCodec-oss")
        ):
            # Load HeartCodec on CPU if skip_model_move (will be moved to GPU for detokenization)
            codec_device = "cpu" if skip_model_move else device
            heartcodec = HeartCodec.from_pretrained(heartcodec_path, device_map=codec_device)
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartCodec at {heartcodec_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.exists(
            heartmula_path := os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        ):
            heartmula = HeartMuLa.from_pretrained(
                heartmula_path, dtype=dtype, quantization_config=bnb_config
            )
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartMuLa at {heartmula_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.isfile(
            vocab_path := os.path.join(pretrained_path, "tokenizer.json")
        ):
            tokenizer = Tokenizer.from_file(vocab_path)
        else:
            raise FileNotFoundError(
                f"Expected to find tokenizer.json for HeartMuLa at {vocab_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.isfile(
            gen_config_path := os.path.join(pretrained_path, "gen_config.json")
        ):
            gen_config = HeartMuLaGenConfig.from_file(gen_config_path)
        else:
            raise FileNotFoundError(
                f"Expected to find gen_config.json for HeartMuLa at {gen_config_path} but not found. Please check your folder {pretrained_path}."
            )

        # Optional: load MuQ-MuLan for reference audio conditioning.
        # First check for local checkpoint, then fall back to model_id if specified.
        if not load_muq_mulan:
            load_muq_mulan = os.getenv("HEARTLIB_LOAD_MUQ_MULAN", "0") == "1"

        muq_mulan = None
        if load_muq_mulan:
            try:
                from muq import MuQMuLan  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "MuQ-MuLan requested, but the `muq` package is not installed. "
                    "Install it with: pip install muq"
                ) from e

            # Check for local checkpoint first
            local_muq_path = os.path.join(pretrained_path, "MuQ-MuLan-large")
            if os.path.exists(local_muq_path):
                print(f"[MuQ-MuLan] Loading from local checkpoint: {local_muq_path}")
                muq_mulan = MuQMuLan.from_pretrained(local_muq_path)
            else:
                # Fall back to model_id (HuggingFace or other path)
                model_id = (
                    muq_model_id
                    or os.getenv("HEARTLIB_MUQ_MULAN_ID", "").strip()
                    or "OpenMuQ/MuQ-MuLan-large"
                )
                print(f"[MuQ-MuLan] Loading from: {model_id}")
                kwargs: Dict[str, Any] = {}
                if muq_cache_dir is not None:
                    kwargs["cache_dir"] = muq_cache_dir
                if muq_revision is not None:
                    kwargs["revision"] = muq_revision
                muq_mulan = MuQMuLan.from_pretrained(model_id, **kwargs)

            if hasattr(muq_mulan, "to"):
                muq_mulan = muq_mulan.to(device)
            if hasattr(muq_mulan, "eval"):
                muq_mulan.eval()
            print(f"[MuQ-MuLan] Model loaded successfully on {device}")

        return cls(heartmula, heartcodec, muq_mulan, tokenizer, gen_config, device, dtype, skip_model_move)
