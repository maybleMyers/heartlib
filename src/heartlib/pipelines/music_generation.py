from transformers.pipelines.base import Pipeline
from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import json
from transformers import BitsAndBytesConfig


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
            self._dtype = dtype  # Use _dtype since dtype is a property
            self.framework = "pt"
            # Set additional attributes that Pipeline base class sets
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
            super().__init__(model, dtype=dtype)
            self.model = model
            self._dtype = dtype

        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config

        self._parallel_number = audio_codec.config.num_quantizers + 1
        self._muq_dim = model.config.muq_dim

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "negative_prompt": kwargs.get("negative_prompt", None),
        }
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "stop_check": kwargs.get("stop_check", None),
        }
        postprocess_kwargs = {
            "save_path": kwargs.get("save_path", "output.mp3"),
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float, negative_prompt: str = None):

        # process tags
        tags = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        # encapsulate with special <tag> and </tag> tokens
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        # process reference audio
        ref_audio = inputs.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = torch.zeros([self._muq_dim], dtype=self._dtype)
        muq_idx = len(tags_ids)

        # process lyrics
        lyrics = inputs["lyrics"]
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

        # cat them together. tags, ref_audio, lyrics
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        # process negative prompt for CFG
        negative_tokens = None
        if negative_prompt is not None and negative_prompt.strip():
            negative_prompt = negative_prompt.lower()
            # encapsulate with special <tag> and </tag> tokens (same format as positive tags)
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

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
            "negative_tokens": negative_tokens,
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        stop_check: callable = None,
    ):
        prompt_tokens = model_inputs["tokens"]
        prompt_tokens_mask = model_inputs["tokens_mask"]
        continuous_segment = model_inputs["muq_embed"]
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"]
        negative_tokens = model_inputs.get("negative_tokens", None)

        # Ensure model is on the correct device (may have been offloaded after previous generation)
        # Skip this check when block swapping is active, as some parameters are intentionally on CPU
        if not getattr(self, '_skip_auto_move', False):
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                self.model.to(self.device)

        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1

        # Move inputs to the correct device
        prompt_tokens = prompt_tokens.to(self.device)
        prompt_tokens_mask = prompt_tokens_mask.to(self.device)
        continuous_segment = continuous_segment.to(self.device)
        prompt_pos = prompt_pos.to(self.device)

        # Compute negative embedding for CFG if provided
        negative_embedding = None
        if negative_tokens is not None:
            negative_tokens = negative_tokens.to(self.device)
            with torch.no_grad():
                neg_embeds = self.model.text_embeddings(negative_tokens)
                # Mean-pool across all tokens to get a single embedding
                negative_embedding = neg_embeds.mean(dim=0, keepdim=False)

        # Setup caches with explicit device to handle model being temporarily on CPU
        self.model.setup_caches(bs_size, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=self._dtype):
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
        frames.append(curr_token[0:1,])

        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], self._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * self.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        max_audio_frames = max_audio_length_ms // 80

        # Async EOS check: check previous frame's result while computing current frame
        # This pipelines the CPU sync with GPU compute
        eos_threshold = self.config.audio_eos_id
        pending_eos_check = None  # Will hold async EOS comparison result

        for i in tqdm(range(max_audio_frames)):
            # Check for stop signal
            if stop_check is not None and stop_check():
                break

            # Check EOS from PREVIOUS iteration (async - computed while GPU was busy)
            if pending_eos_check is not None and torch.any(pending_eos_check):
                break

            curr_token, curr_token_mask = _pad_audio_token(curr_token)
            with torch.autocast(device_type=self.device.type, dtype=self._dtype):
                curr_token = self.model.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                    negative_embedding=negative_embedding,
                )

            # Start async EOS check - GPU computes this while we loop back
            pending_eos_check = curr_token[0:1, :] >= eos_threshold

            frames.append(curr_token[0:1,])

        # Handle case where last frame was EOS (remove it from frames)
        if pending_eos_check is not None and len(frames) > 0 and torch.any(pending_eos_check):
            frames.pop()

        # Offload HeartMuLa model to CPU to free GPU memory for detokenize
        # This prevents OOM during the detokenize step
        self.model.reset_caches()
        self.model.to("cpu")
        torch.cuda.empty_cache()

        # Move audio_codec to GPU for detokenization if it's on CPU
        codec_was_on_cpu = next(self.audio_codec.parameters()).device.type == "cpu"
        if codec_was_on_cpu:
            self.audio_codec.to(self.device)

        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        wav = self.audio_codec.detokenize(frames)

        # Move audio_codec back to CPU if it was originally there
        if codec_was_on_cpu:
            self.audio_codec.to("cpu")
            torch.cuda.empty_cache()

        return {"wav": wav}

    def postprocess(self, model_outputs: Dict[str, Any], save_path: str):
        wav = model_outputs["wav"]
        torchaudio.save(save_path, wav, 48000)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
        version: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        skip_model_move: bool = False,
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

        return cls(heartmula, heartcodec, None, tokenizer, gen_config, device, dtype, skip_model_move)
