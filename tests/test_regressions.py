import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from crossdrone.diagnostics import Diagnostics, evaluation_scenes
from model import build_model, ModelConfig, pad_targets
from model.loss import SetCriterion
from model.matcher import HungarianMatcher, _hungarian_numpy
from model.box_ops import box_loss


class RegressionTests(unittest.TestCase):
    def test_wrong_saturated_confidence_keeps_gradient(self):
        logits = torch.tensor([[[-25., 25.]]], requires_grad=True)
        boxes = torch.tensor([[[[.5, .5, .2, .2], [0., 0., 0., 0.]]]])
        outputs = {"class_logits": torch.tensor([[[10., -10.], [-10., 10.]]]),
                   "boxes": boxes, "confidences": logits.sigmoid(), "confidence_logits": logits}
        target = pad_targets([torch.tensor([0])], [boxes[0, :, :1]],
                             [torch.ones(1, 1)], 2, 1)
        loss = SetCriterion(1, HungarianMatcher())(outputs, target)["loss_conf"]
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[[-.5, .5]]]))

    def test_frozen_backbone_preserves_batchnorm(self):
        backbone = nn.Sequential(nn.Conv2d(3, 32, 1), nn.BatchNorm2d(32))
        with patch("model.association.build_backbone", return_value=backbone):
            model = build_model(ModelConfig(pretrained=False, freeze_backbone=True,
                                           hidden_dim=32, num_heads=4))
        model.train()
        before = backbone[1].running_mean.clone()
        backbone(torch.randn(2, 3, 4, 4) + 10)
        torch.testing.assert_close(backbone[1].running_mean, before)
        self.assertFalse(backbone.training)
        self.assertTrue(model.mvoq.training)
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))

    def test_tiny_perfect_box_has_zero_regression_loss(self):
        boxes = torch.tensor([[.5, .5, .0001, .0001]])
        torch.testing.assert_close(box_loss(boxes, boxes), torch.zeros(1))

    def test_evaluation_selection_matches_training_prefix(self):
        large = SimpleNamespace(ids=range(26))
        small = [SimpleNamespace(ids=range(2)) for _ in range(5)]
        data = {"train": [large, *small], "val": small, "test": small}
        settings = {"limit_train": 2, "limit_val": 3}
        self.assertEqual(evaluation_scenes(data, settings, "train"), small[:2])
        self.assertEqual(evaluation_scenes(data, settings, "val"), small[:3])
        self.assertEqual(evaluation_scenes(data, settings, "val", full=True), small)
        self.assertEqual(evaluation_scenes(data, settings, "val", limit=1), small[:1])
        self.assertEqual(evaluation_scenes(data, settings, "test"), small)
        settings["overfit"] = True
        self.assertEqual(evaluation_scenes(data, settings, "val"), small[:2])

    def test_diagnostic_separates_localization_from_filtering(self):
        truth = torch.tensor([[[.5, .5, .1, .1]]]*2)
        outputs = {"boxes": truth[None], "class_logits": torch.tensor([[[20., -20.]]]),
                   "confidences": torch.zeros(1, 2, 1)}
        target = {"boxes": truth, "visibility": torch.ones(2, 1, dtype=torch.bool),
                  "sequence": "1", "timestamp": "0", "sizes": [(100, 100)]*2}
        metric = Diagnostics()
        metric.update(outputs, [target])
        result = metric.compute()["summary"]
        self.assertEqual(result["raw_oracle_coverage_iou50"], 1.)
        self.assertEqual(result["kept_oracle_coverage_iou50"], 0.)
        self.assertEqual(result["center_error_px"]["mean"], 0.)

    def test_fallback_assignment_rejects_nonfinite_and_matches_scipy(self):
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(1)
        for shape in ((5, 5), (3, 5), (5, 3)):
            cost = rng.normal(size=shape)
            a, b = _hungarian_numpy(cost)
            c, d = linear_sum_assignment(cost)
            self.assertAlmostEqual(cost[a, b].sum(), cost[c, d].sum())
        with self.assertRaises(ValueError):
            _hungarian_numpy(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
