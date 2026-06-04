"""Frozen RF-DETR-Large @ 704 teacher for HR->LR query-aligned distillation.

Loads the fine-tuned 2D checkpoint, exposes the shared decoder queries
(refpoint_embed_weight, query_feat_weight) and a forward() that returns
pred_logits/pred_boxes plus a per-query foreground weight. Frozen, eval-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..config import Config


class FrozenRFDETRTeacher(nn.Module):
    """Frozen 2D RF-DETR-Large teacher exposing shared decoder queries."""

    def __init__(self, cfg: Config):
        super().__init__()
        # Local imports to avoid pulling rfdetr unless distillation is on.
        from rfdetr.config import RFDETRLargeConfig
        from rfdetr.models.lwdetr import build_model_from_config
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        self._nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
        self.cfg = cfg

        Q = int(cfg.distill_num_queries)
        model_cfg = RFDETRLargeConfig(
            num_classes=cfg.distill_teacher_num_classes,
            num_queries=Q,
            num_select=Q,
        )
        # checkpoint was trained with group_detr=13; the eval path slices to
        # the first num_queries rows, so no trimming needed here.
        lwdetr = build_model_from_config(model_cfg)

        ckpt_path = Path(cfg.distill_teacher_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Distillation teacher checkpoint not found: {ckpt_path}"
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # shape-filter so incidental head/projector mismatches are tolerated.
        model_sd = lwdetr.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        msg = lwdetr.load_state_dict(filtered, strict=False)
        print(
            f"[Teacher] Loaded '{ckpt_path.name}'  "
            f"loaded={len(filtered)}  missing={len(msg.missing_keys)}  "
            f"skipped(shape)={len(skipped)}"
        )

        self.lwdetr = lwdetr
        # force inference behaviour: no aux loss, exactly num_queries preds.
        self.lwdetr.aux_loss = False
        self.lwdetr.num_queries = Q

        # first Q rows = canonical inference queries (group 0); shared with student.
        with torch.no_grad():
            shared_refpoint = self.lwdetr.refpoint_embed.weight[:Q].detach().clone()
            shared_query_feat = self.lwdetr.query_feat.weight[:Q].detach().clone()
        self.register_buffer(
            "refpoint_embed_weight", shared_refpoint, persistent=False,
        )
        self.register_buffer(
            "query_feat_weight", shared_query_feat, persistent=False,
        )

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # Capture the decoder's per-slot inputs (tgt + refpoints) so the
        # student's decoder can be fed the same tensors for KD-DETR alignment.
        self._captured_decoder_inputs: Dict[str, torch.Tensor] = {}

        def _decoder_pre_hook(_module, args, kwargs):
            # args = (tgt, memory, ...);  refpoints_unsigmoid is kwarg.
            if len(args) >= 1:
                self._captured_decoder_inputs["tgt"] = args[0].detach()
            if "refpoints_unsigmoid" in kwargs:
                self._captured_decoder_inputs["refpoints"] = (
                    kwargs["refpoints_unsigmoid"].detach()
                )
            return None

        self.lwdetr.transformer.decoder.register_forward_pre_hook(
            _decoder_pre_hook, with_kwargs=True,
        )

        # Capture last-layer decoder output hs[-1] (B, Q, D) for CRRCD.
        def _decoder_post_hook(_module, _args, _kwargs, output):
            # decoder.forward returns [intermediate, refpoints] when
            # return_intermediate=True, else a single tensor (export path).
            if isinstance(output, (list, tuple)) and len(output) >= 1:
                hs = output[0]
            else:
                hs = output
            if hs is not None:
                self._captured_decoder_inputs["hs"] = hs[-1].detach()
            return None

        self.lwdetr.transformer.decoder.register_forward_hook(
            _decoder_post_hook, with_kwargs=True,
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.lwdetr.transformer.d_model)

    def train(self, mode: bool = True):  # type: ignore[override]
        # stay in eval regardless of parent .train(True).
        return super().train(False)

    @torch.no_grad()
    def forward(self, centre_clean: torch.Tensor) -> Dict[str, torch.Tensor]:
        """centre_clean: (B, 3, H, W) ImageNet-normalised HR frames.

        Returns pred_logits (B, Q, K_t), pred_boxes (B, Q, 4), foreground_weight (B, Q).
        """
        return self._forward_impl(centre_clean, refpoint_w=None, query_feat_w=None)

    @torch.no_grad()
    def forward_general(
        self,
        centre_clean: torch.Tensor,
        refpoint_w: torch.Tensor,
        query_feat_w: torch.Tensor,
        min_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run the teacher with externally-supplied queries (KD-DETR general
        sampling): refpoint_w/query_feat_w are random tensors shared with the
        student so the distillation losses stay slot-aligned.
        """
        return self._forward_impl(
            centre_clean,
            refpoint_w=refpoint_w,
            query_feat_w=query_feat_w,
            min_weight_override=min_weight,
        )

    @torch.no_grad()
    def _forward_impl(
        self,
        centre_clean: torch.Tensor,
        refpoint_w,
        query_feat_w,
        min_weight_override: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        assert centre_clean.dim() == 4, (
            f"teacher expects (B, 3, H, W), got {tuple(centre_clean.shape)}"
        )
        nested = self._nested_tensor_from_tensor_list(centre_clean)

        if refpoint_w is None:
            out = self.lwdetr(nested)
        else:
            # swap in custom queries by replacing the Embedding objects
            # (not their .weight) so LWDETR.forward picks them up by attribute.
            Q = int(refpoint_w.shape[0])
            orig_rp = self.lwdetr.refpoint_embed
            orig_qf = self.lwdetr.query_feat
            orig_nq_l = self.lwdetr.num_queries
            orig_nq_t = self.lwdetr.transformer.num_queries

            new_rp = nn.Embedding(Q, refpoint_w.shape[1]).to(refpoint_w.device)
            new_rp.weight = nn.Parameter(
                refpoint_w.detach().to(refpoint_w.device).clone(),
                requires_grad=False,
            )
            new_qf = nn.Embedding(Q, query_feat_w.shape[1]).to(query_feat_w.device)
            new_qf.weight = nn.Parameter(
                query_feat_w.detach().to(query_feat_w.device).clone(),
                requires_grad=False,
            )
            self.lwdetr.refpoint_embed = new_rp
            self.lwdetr.query_feat = new_qf
            self.lwdetr.num_queries = Q
            self.lwdetr.transformer.num_queries = Q
            try:
                out = self.lwdetr(nested)
            finally:
                self.lwdetr.refpoint_embed = orig_rp
                self.lwdetr.query_feat = orig_qf
                self.lwdetr.num_queries = orig_nq_l
                self.lwdetr.transformer.num_queries = orig_nq_t

        logits = out["pred_logits"]                       # (B, Q, K_real + 1)
        boxes = out["pred_boxes"]                         # (B, Q, 4)
        # drop the trailing no-object slot (build_model adds num_classes+1).
        K_real = int(self.cfg.distill_teacher_num_classes)
        if logits.shape[-1] > K_real:
            logits = logits[..., :K_real]
        # per-query max foreground prob (sigmoid focal, no explicit bg slot).
        w = logits.sigmoid().amax(dim=-1)                 # (B, Q)
        floor = (
            float(min_weight_override)
            if min_weight_override is not None
            else float(self.cfg.distill_min_weight)
        )
        if floor > 0.0:
            w = w.clamp(min=floor)

        result = {
            "pred_logits": logits.detach(),
            "pred_boxes": boxes.detach(),
            "foreground_weight": w.detach(),
        }
        # expose the captured per-slot decoder inputs for KD-DETR alignment.
        if "tgt" in self._captured_decoder_inputs:
            result["decoder_tgt"] = self._captured_decoder_inputs["tgt"]
        if "refpoints" in self._captured_decoder_inputs:
            result["decoder_refpoints"] = self._captured_decoder_inputs["refpoints"]
        if "hs" in self._captured_decoder_inputs:
            result["decoder_hs"] = self._captured_decoder_inputs["hs"]
        # reset so we don't leak stale tensors into the next call.
        self._captured_decoder_inputs = {}
        return result
