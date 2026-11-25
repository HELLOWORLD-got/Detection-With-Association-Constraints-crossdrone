"""Global joint loss, Eq. (21)-(24).

::

    L = L_cls + lam_reg * L_reg + lam_conf * L_conf

* ``L_cls``  (Eq. 22) mean NLL over the N matched pairs, padded pairs
  contributing ``-log p_hat(c_empty)``.
* ``L_reg``  (Eq. 23) L1 + GIoU averaged over visible (instance, view) pairs
  within each scene.
* ``L_conf`` (Eq. 24) BCE over all N x K slots; a slot is positive iff its
  matched instance is real and visible in that view, exactly the
  ``I(m) * v_k,m`` product of the paper.

All three terms go through the same optimal assignment sigma_hat of Eq. (20).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_loss
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_cls: float = 1.0,
        weight_reg: float = 1.0,
        weight_conf: float = 1.0,
        weight_l1: float = 5.0,
        weight_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_conf = weight_conf
        self.weight_l1 = weight_l1
        self.weight_giou = weight_giou

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        class_logits = outputs["class_logits"]  # (B, N, C+1)
        boxes = outputs["boxes"]  # (B, K, N, 4)
        confidences = outputs["confidences"]  # (B, K, N)

        batch, num_queries, _ = class_logits.shape
        num_views = boxes.shape[1]
        device = class_logits.device

        assignments = self.matcher(class_logits, boxes, targets)
        log_probs = F.log_softmax(class_logits, dim=-1)

        cls_terms = []
        reg_terms = []
        conf_terms = []

        for b in range(batch):
            count = int(targets["num_instances"][b])
            target_index = assignments[b][:, 1]  # (N,) matched ground-truth slot
            is_real = (target_index < count).to(boxes.dtype)  # I(m), Eq. (18)
            slot = torch.arange(num_queries, device=device)

            # --- Eq. (22): classification ---------------------------------
            matched_labels = targets["labels"][b][target_index]
            cls_terms.append(-log_probs[b][slot, matched_labels].mean())

            # --- Eq. (23): box regression ---------------------------------
            matched_boxes = targets["boxes"][b][:, target_index]  # (K, N, 4)
            matched_visibility = targets["visibility"][b][:, target_index]  # (K, N)
            valid = matched_visibility * is_real.unsqueeze(0)  # I(m) * v_k,m
            per_pair = box_loss(
                boxes[b].float(), matched_boxes, self.weight_l1, self.weight_giou
            )
            reg_terms.append((per_pair * valid).sum() / valid.sum().clamp(min=1.0))

            # --- Eq. (24): confidence -------------------------------------
            target_conf = valid.clamp(0.0, 1.0)
            if "confidence_logits" in outputs:
                bce = F.binary_cross_entropy_with_logits(
                    outputs["confidence_logits"][b].float(), target_conf, reduction="none"
                )
            else:
                # Compatibility for external callers supplying probabilities only.
                bce = F.binary_cross_entropy(
                    confidences[b].float(), target_conf, reduction="none"
                )
            conf_terms.append(bce.sum() / (num_queries * num_views))

        loss_cls = torch.stack(cls_terms).mean()
        loss_reg = torch.stack(reg_terms).mean()
        loss_conf = torch.stack(conf_terms).mean()

        loss_cls = loss_cls * self.weight_cls
        loss_reg = loss_reg * self.weight_reg
        loss_conf = loss_conf * self.weight_conf
        total = loss_cls + loss_reg + loss_conf
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_conf": loss_conf,
        }

    def build_targets(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Convenience re-export of :func:`model.targets.pad_targets`."""
        from .targets import pad_targets

        return pad_targets(*args, **kwargs)


# Name used by the reference implementation's __init__.
HungarianLoss = SetCriterion
