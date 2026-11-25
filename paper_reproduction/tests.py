import unittest

import torch

from .network import Architecture, TripletBlock, position_grid
from .objective import LossWeights, objective
from .scoring import Evaluation


class PaperEquations(unittest.TestCase):
    def test_equation_11_matches_explicit_keys(self):
        torch.manual_seed(7)
        cfg = Architecture(width=16, heads=4)
        block = TripletBlock(cfg).double()
        slots, instances = torch.randn(2, 3, 16, dtype=torch.double), torch.randn(2, 3, 16, dtype=torch.double)
        tokens = torch.randn(2, 6, 16, dtype=torch.double)
        positions = position_grid(2, 3, 16, tokens)
        q = block.query(slots).unflatten(-1, (4, 4))
        keys = block.key(instances[:, :, None] * positions).unflatten(-1, (4, 4))
        values = block.value(tokens).unflatten(-1, (4, 4))
        scores = torch.einsum("bnhd,bnthd->bnht", q, keys) / 2
        context = torch.einsum("bnht,bthd->bnhd", scores.softmax(-1), values).flatten(-2)
        hidden = block.norm_attention(slots + block.readout(context))
        expected = block.norm_ff(hidden + block.ff(hidden))
        torch.testing.assert_close(block(slots, instances, tokens, positions), expected)

    def test_equations_20_to_28_on_permuted_slots_with_invisible_view(self):
        boxes = torch.tensor([
            [[.3, .45, .1, .2], [.6, .55, .1, .2]],
            [[.35, .45, .1, .2], [0., 0., 0., 0.]],
        ])
        visibility = torch.tensor([[True, True], [True, False]])
        target = {"boxes": boxes, "labels": torch.zeros(2, dtype=torch.long),
                  "visible": visibility, "key": "example", "sizes": [(100, 100)] * 2}
        predicted = torch.cat((boxes[:, [1, 0]], torch.full((2, 1, 4), .25)), dim=1)
        category = torch.tensor([[30., -30.], [30., -30.], [-30., 30.]])
        confidence = torch.where(torch.cat((visibility[:, [1, 0]], torch.zeros(2, 1, dtype=torch.bool)), 1),
                                 30., -30.)
        result = objective({"category": category[None], "box": predicted[None],
                            "visibility": confidence[None]}, [target], LossWeights())
        self.assertLess(result["loss"].item(), 1e-6)
        metric = Evaluation()
        metric.add(category, predicted, confidence, target)
        scores = metric.result()
        for key in ("mAP", "mAP50", "MDA", "UCR"):
            self.assertAlmostEqual(scores[key], 1.)


if __name__ == "__main__":
    unittest.main()
