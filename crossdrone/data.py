"""Read synchronized Cross-Drone annotations without losing cross-view IDs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
import random
import re
import struct
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


NUM_VIEWS = 2
NUM_QUERIES = 25
SPLITS = {"train": ("1", "3", "4", "5", "6", "7", "8"), "val": ("2",), "test": ("0",)}
VEHICLE = re.compile(r"^(?:CAR_|BP_(?:Car_|Sedan|SUV|Van_|Campervan|BoxTruck|Trailer))")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
Box = tuple[float, float, float, float]


@dataclass
class Scene:
    sequence: str
    timestamp: str
    paths: tuple[Path, Path]
    sizes: tuple[tuple[int, int], tuple[int, int]]
    ids: tuple[str, ...]
    boxes: tuple[tuple[Box, ...], ...]  # normalized corners, zeros when invisible


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG header in {path}")
    return struct.unpack(">II", header[16:])


def _view_boxes(node, width, height, counts, excluded):
    result = {}
    for box in node.findall("box"):
        instance = box.attrib["id"]
        if not VEHICLE.match(instance):
            excluded[instance] += 1
            continue
        if box.attrib["outside"] != "0":
            continue
        left, top, right, bottom = (
            float(box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr")
        )
        if not all(map(isfinite, (left, top, right, bottom))):
            raise ValueError(f"non-finite annotation for {instance}")
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(float(width), right), min(float(height), bottom)
        if left >= right or top >= bottom:
            counts["degenerate_visible_boxes"] += 1
            continue
        if instance in result:
            raise ValueError(f"duplicate visible ID {instance} in one view")
        result[instance] = (left / width, top / height, right / width, bottom / height)
    return result


def index_dataset(root: Path, splits: dict) -> tuple[dict[str, list[Scene]], dict]:
    """Return all complete frames and an audit before query-capacity filtering."""
    names = [name for sequences in splits.values() for name in sequences]
    if len(set(names)) != len(names):
        raise ValueError("sequence assignments must be unique across splits")
    if any(not name.isdecimal() for name in names):
        raise ValueError("sequence names must be decimal directory names")
    scenes_by_sequence = {}
    audit = {}
    fingerprint = sha256()
    for name in sorted(names, key=int):
        directory = root / name
        xml = (directory / f"{name}.xml").read_bytes()
        fingerprint.update(xml)
        counts = Counter(dict.fromkeys(
            ("xml_pairs", "missing_first_view", "missing_second_view", "incomplete_pairs",
             "complete_pairs", "over_capacity", "degenerate_visible_boxes", "max_instances"), 0
        ))
        excluded = Counter()
        seen = set()
        scenes = []
        for _, element in ET.iterparse(BytesIO(xml), events=("end",)):
            if element.tag != "pair":
                continue
            timestamp = element.attrib["timestamp"]
            if timestamp in seen or not re.fullmatch(r"\d+(?:\.\d+)?", timestamp):
                raise ValueError(f"invalid or repeated timestamp {name}:{timestamp}")
            seen.add(timestamp)
            paths = tuple(directory / str(view) / f"{timestamp}.png" for view in (1, 2))
            nodes = tuple(element.find(f"img{view}") for view in (1, 2))
            exists = tuple(path.is_file() and node is not None for path, node in zip(paths, nodes))
            counts["xml_pairs"] += 1
            counts["missing_first_view"] += not exists[0]
            counts["missing_second_view"] += not exists[1]
            if not all(exists):
                counts["incomplete_pairs"] += 1
                element.clear()
                continue
            sizes = tuple(_png_size(path) for path in paths)
            fingerprint.update(f"{name}:{timestamp}:{sizes}".encode())
            by_view = [_view_boxes(node, *size, counts, excluded) for node, size in zip(nodes, sizes)]
            ids = tuple(sorted(by_view[0].keys() | by_view[1].keys()))
            boxes = tuple(tuple(view.get(instance, (0.0, 0.0, 0.0, 0.0)) for instance in ids)
                          for view in by_view)
            scenes.append(Scene(name, timestamp, paths, sizes, ids, boxes))
            counts["complete_pairs"] += 1
            counts["over_capacity"] += len(ids) > NUM_QUERIES
            counts["max_instances"] = max(counts["max_instances"], len(ids))
            element.clear()
        scenes_by_sequence[name] = scenes
        audit[name] = {**counts, "excluded_id_occurrences": dict(sorted(excluded.items()))}
    grouped = {
        split: [scene for name in sequences for scene in scenes_by_sequence[name]]
        for split, sequences in splits.items()
    }
    return grouped, {
        "sequences": audit,
        "splits": {
            split: {"sequences": list(splits[split]), "complete": len(scenes),
                    "over_capacity": sum(len(scene.ids) > NUM_QUERIES for scene in scenes)}
            for split, scenes in grouped.items()
        },
        "annotation_and_availability_sha256": fingerprint.hexdigest(),
    }


class CrossDroneDataset(Dataset):
    def __init__(self, scenes, size, *, training=False, limit=None):
        if training:
            scenes = [scene for scene in scenes if len(scene.ids) <= NUM_QUERIES]
        self.scenes = scenes[:limit]
        self.size = size
        self.training = training
        self.color_jitter = ColorJitter(0.2, 0.2, 0.2, 0.05)
        if not self.scenes:
            raise ValueError("no eligible paired images in split")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        boxes = torch.tensor(scene.boxes, dtype=torch.float32).view(NUM_VIEWS, len(scene.ids), 4)
        images = []
        crop_width = random.uniform(0.8, 1.0) if self.training else 1.0
        crop_height = random.uniform(0.8, 1.0) if self.training else 1.0
        offset_x = random.random() if self.training else 0.0
        offset_y = random.random() if self.training else 0.0
        flip = self.training and random.random() < 0.5
        for v, (path, (width, height)) in enumerate(zip(scene.paths, scene.sizes)):
            with Image.open(path) as source:
                image = source.convert("RGB")
            if self.training:
                cw, ch = max(1, round(width * crop_width)), max(1, round(height * crop_height))
                x, y = round((width - cw) * offset_x), round((height - ch) * offset_y)
                image = TF.crop(image, y, x, ch, cw)
                boxes[v] = ((boxes[v] * boxes.new_tensor([width, height, width, height])
                             - boxes.new_tensor([x, y, x, y]))
                            / boxes.new_tensor([cw, ch, cw, ch])).clamp_(0, 1)
                if flip:
                    image = TF.hflip(image)
                    boxes[v, :, [0, 2]] = 1 - boxes[v, :, [2, 0]]
                image = self.color_jitter(image)
            image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
            images.append(TF.normalize(TF.to_tensor(image), MEAN, STD))
        size = boxes[..., 2:] - boxes[..., :2]
        visible = (size > 0).all(dim=-1)
        keep = visible.any(dim=0)
        boxes = torch.cat((boxes[..., :2] + size / 2, size), dim=-1)
        boxes = boxes.masked_fill(~visible[..., None], 0)
        target = {
            "sequence": scene.sequence,
            "timestamp": scene.timestamp,
            "ids": [instance for instance, retained in zip(scene.ids, keep.tolist()) if retained],
            "sizes": scene.sizes,
            "labels": torch.zeros(int(keep.sum()), dtype=torch.long),
            "boxes": boxes[:, keep],
            "visibility": visible[:, keep],
        }
        return torch.stack(images), target


def collate(samples):
    images, targets = zip(*samples)
    return torch.stack(images), targets
