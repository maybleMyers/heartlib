import torch
import torch.nn as nn
from .configuration_heartmula import HeartMuLaConfig
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


def llama3_2_3B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_7B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_400M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )  # 减少了num_heads和num_kv_heads之间的倍速，提升了精确度，但降低了效率


FLAVORS = {
    "llama-3B": llama3_2_3B,
    "llama-300M": llama3_2_300M,
    "llama-7B": llama3_2_7B,
    "llama-400M": llama3_2_400M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(
    probs,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


class HeartMuLa(PreTrainedModel):
    config_class = HeartMuLaConfig
    _is_compiled = False

    def __init__(
        self,
        config: HeartMuLaConfig,
    ):
        super(HeartMuLa, self).__init__(config)

        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)
        self.post_init()

    def setup_caches(self, max_batch_size: int, device: torch.device = None):
        dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device

        try:
            self.reset_caches()
        except RuntimeError:
            pass

        with device:
            if not self.backbone.caches_are_enabled():
                self.backbone.setup_caches(max_batch_size, dtype)
            if not self.decoder.caches_are_enabled():
                self.decoder.setup_caches(
                    max_batch_size,
                    dtype,
                    decoder_max_seq_len=self.config.audio_num_codebooks,
                )

        if not hasattr(self, "backbone_causal_mask") or self.backbone_causal_mask is None:
            self.register_buffer(
                "backbone_causal_mask",
                _create_causal_mask(self.backbone.max_seq_len, device),
            )
        if not hasattr(self, "decoder_causal_mask") or self.decoder_causal_mask is None:
            self.register_buffer(
                "decoder_causal_mask",
                _create_causal_mask(self.config.audio_num_codebooks, device),
            )

        # Pre-allocate reusable buffers to avoid per-frame allocations
        self._cached_batch_size = max_batch_size
        self._cached_device = device
        actual_B = max_batch_size // 2 if max_batch_size > 1 else 1

        # For uncond_mask in generate_frame - pre-concatenated
        self.register_buffer(
            "_uncond_mask",
            torch.cat([
                torch.zeros(actual_B, dtype=torch.bool, device=device),
                torch.ones(actual_B, dtype=torch.bool, device=device),
            ]),
            persistent=False,
        )

        # For batch indices
        self.register_buffer(
            "_batch_indices",
            torch.arange(max_batch_size, device=device),
            persistent=False,
        )

        # For decoder positions - shape [batch_size, num_codebooks]
        self.register_buffer(
            "_decoder_pos_base",
            torch.arange(self.config.audio_num_codebooks, device=device).unsqueeze(0).expand(max_batch_size, -1).contiguous(),
            persistent=False,
        )

        # For unconditional embedding index
        self.register_buffer(
            "_zero_idx",
            torch.zeros(1, device=device, dtype=torch.long),
            persistent=False,
        )

        # Pre-allocate buffer for curr_sample output to avoid torch.cat in inner loop
        self.register_buffer(
            "_sample_buffer",
            torch.zeros(max_batch_size, self.config.audio_num_codebooks, dtype=torch.long, device=device),
            persistent=False,
        )

        # Pre-allocate buffer for decoder input (last_h + c0_embed concatenated)
        backbone_dim = self.text_embeddings.embedding_dim
        self.register_buffer(
            "_decoder_input_buffer",
            torch.zeros(max_batch_size, 2, backbone_dim, dtype=dtype, device=device),
            persistent=False,
        )

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: torch.Tensor = None,
        starts=None,
        negative_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)

        # Use pre-allocated uncond_mask instead of creating new tensors
        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            uncond_mask = self._uncond_mask

        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask, negative_embedding=negative_embedding)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2, dtype=embeds.dtype)  # merge
        if continuous_segments is not None:
            continuous_segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                if negative_embedding is not None:
                    uncond_embed = negative_embedding
                else:
                    # Use pre-allocated zero index
                    uncond_embed = self.unconditional_text_embedding(self._zero_idx)
                mask_expanded = uncond_mask.view(b, 1).expand_as(continuous_segments)
                continuous_segments = torch.where(
                    mask_expanded, uncond_embed, continuous_segments
                )
            # Use pre-allocated batch indices
            h[self._batch_indices[:b], starts] = continuous_segments
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        last_h = h[:, -1, :]  # the last frame
        c0_logits = self.codebook0_head(last_h)  # only predict the audio part

        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_logits = c0_logits[:actual_B, :]
            uncond_logits = c0_logits[actual_B:, :]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            c0_sample = sample_topk(guided_logits, topk, temperature)
            c0_sample = c0_sample.repeat(
                2, 1
            )  # repeat to both branches to keep alignment
        else:
            c0_sample = sample_topk(c0_logits, topk, temperature)

        c0_embed = self._embed_audio(0, c0_sample)

        # Reset decoder caches once per frame
        self.decoder.reset_caches()
        # Use pre-allocated buffer instead of torch.cat
        curr_h = self._decoder_input_buffer[:b]
        curr_h[:, 0, :] = last_h
        curr_h[:, 1, :] = c0_embed.squeeze(1)

        # Use pre-allocated sample buffer instead of repeated torch.cat
        curr_sample = self._sample_buffer[:b]
        curr_sample[:, 0:1] = c0_sample

        # Use pre-allocated decoder positions
        curr_pos = self._decoder_pos_base[:b, :2]
        curr_h = curr_h.to(embeds.dtype)
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_B = b // 2
                cond_ci = ci_logits[:actual_B, :]
                uncond_ci = ci_logits[actual_B:, :]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale

                ci_sample = sample_topk(guided_ci, topk, temperature)
                ci_sample = ci_sample.repeat(2, 1)
            else:
                ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            # Write directly to pre-allocated buffer instead of torch.cat
            curr_sample[:, i:i+1] = ci_sample
            # Use pre-allocated position for next iteration (positions 2,3,4,5,6,7)
            curr_pos = self._decoder_pos_base[:b, i+1:i+2]

        # Return a copy to avoid buffer being overwritten in next call
        return curr_sample.clone()

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def compile_model(self, mode: str = "reduce-overhead"):
        """Compile the model with torch.compile for faster inference.

        Args:
            mode: Compilation mode. "reduce-overhead" uses CUDA graphs for
                  minimum CPU overhead (best for slow CPUs with fast GPUs).
                  "default" for general optimization.

        Note: Only supported on Linux with CUDA. Windows users should skip this.
        """
        if self._is_compiled:
            return

        import sys
        if sys.platform == "win32":
            print("Warning: torch.compile not fully supported on Windows, skipping")
            return

        try:
            self.backbone = torch.compile(self.backbone, mode=mode)
            self.decoder = torch.compile(self.decoder, mode=mode)
            self._is_compiled = True
            print(f"Model compiled with mode='{mode}'")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    def _embed_local_audio(self, tokens):
        """the token from 0-30"""
        audio_tokens = tokens + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks - 1, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks - 1, -1
        )
        return audio_embeds

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(
        self, tokens: torch.Tensor, uncond_mask: torch.Tensor | None, negative_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, _ = tokens.size()
        text_embeds = self.text_embeddings(tokens[:, :, -1])

        if uncond_mask is not None:
            if negative_embedding is not None:
                uncond_text_embed = negative_embedding
            else:
                uncond_text_embed = self.unconditional_text_embedding(
                    torch.zeros(1, device=tokens.device, dtype=torch.long)
                )
            mask_expanded = uncond_mask.view(B, 1, 1).expand_as(text_embeds)
            text_embeds = torch.where(
                mask_expanded,
                uncond_text_embed,
                text_embeds,
            )

        text_embeds = text_embeds.unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)
