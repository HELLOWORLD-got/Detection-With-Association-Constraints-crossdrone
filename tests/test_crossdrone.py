from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from crossdrone.data import CrossDroneDataset, Scene
from crossdrone.metrics import AssociationMetrics, DetectionMetrics
from model import TripletAttentionBlock, box_xywh_to_xyxy


class CrossDroneTests(unittest.TestCase):
    def test_triplet_attention_matches_explicit_key_projection(self):
        torch.manual_seed(7)
        layer = TripletAttentionBlock(32, 4, dropout=0).double()
        q = torch.randn(2, 3, 32, dtype=torch.float64)
        instance = torch.randn_like(q).requires_grad_()
        view = torch.randn(2, 15, 32, dtype=q.dtype)
        positions = layer.position_encoding(3, 5, q.device, q.dtype)

        queries = layer.query_projection(q).reshape(2, 3, 4, 8)
        keys = layer.key_projection(instance[:, :, None] * positions).reshape(2, 3, 15, 4, 8)
        values = layer.value_projection(view).reshape(2, 15, 4, 8)
        weights = (torch.einsum("bnhd,bnthd->bhnt", queries, keys) / 8 ** 0.5).softmax(-1)
        explicit = torch.einsum("bhnt,bthd->bnhd", weights, values).reshape_as(q)
        explicit = layer.norm1(q + layer.output_projection(explicit))
        explicit = layer.norm2(explicit + layer.ffn(explicit))

        result = layer(q, instance, view, (3, 5))
        torch.testing.assert_close(result, explicit)
        grad, = torch.autograd.grad(result[0, 1, 0], instance)
        self.assertGreater(grad[0, 1].norm().item(), 0)
        self.assertFalse(grad[0, [0, 2]].any())
        self.assertFalse(grad[1].any())

    def test_crop_flip_preserves_ids_and_updates_visibility(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (100, 100)).save(image)
            scene = Scene("0", "1.23", (image, image), ((100, 100), (100, 100)),
                          ("CAR_1",), (((0.08, 0.16, 0.32, 0.48),), ((0, 0, 0, 0),)))
            dataset = CrossDroneDataset([scene], (64, 64), training=True)
            with patch("crossdrone.data.random.uniform", return_value=0.8), \
                 patch("crossdrone.data.random.random", return_value=0.0):
                images, target = dataset[0]
            self.assertEqual(images.shape, (2, 3, 64, 64))
            self.assertEqual(target["ids"], ["CAR_1"])
            self.assertEqual(target["visibility"].tolist(), [[True], [False]])
            torch.testing.assert_close(target["boxes"][:, 0],
                                       torch.tensor([[0.75, 0.4, 0.3, 0.4], [0, 0, 0, 0]]))

    def test_detection_and_association_metrics(self):
        truth = torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.7, 0.6, 0.2, 0.2]]] * 2)
        visibility = torch.ones(2, 2, dtype=torch.bool)
        target = {"boxes": truth, "visibility": visibility, "sizes": [(1920, 1080)] * 2}
        boxes = box_xywh_to_xyxy(truth)
        metric = AssociationMetrics()
        metric.update(boxes, visibility, target)
        self.assertEqual(metric.compute()["MDA"], 1.0)
        self.assertEqual(metric.compute()["UCR"], 1.0)

        swapped = boxes.clone()
        swapped[1] = swapped[1].flip(0)
        metric = AssociationMetrics()
        metric.update(swapped, visibility, target)
        self.assertEqual(metric.compute()["MDA"], 0.0)
        self.assertEqual(metric.compute()["UCR"], 0.0)

        detection = DetectionMetrics()
        detection.update(swapped, torch.ones(2, 2), target)
        self.assertAlmostEqual(detection.compute()["mAP"], 1.0)


if __name__ == "__main__":
    unittest.main()
