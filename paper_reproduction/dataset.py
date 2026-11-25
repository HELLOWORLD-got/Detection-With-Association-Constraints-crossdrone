"""Cross-Drone XML adapter. Each target row belongs to one physical vehicle."""
from collections import Counter
from dataclasses import dataclass
import hashlib
from math import isfinite
import random
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


@dataclass
class Pair:
    sequence: str
    time: str
    paths: tuple
    sizes: tuple
    ids: tuple
    boxes: torch.Tensor


ROAD_PREFIXES = ("CAR_", "BP_Car_", "BP_Sedan", "BP_SUV", "BP_Van_", "BP_Campervan", "BP_BoxTruck", "BP_Trailer")


def read_sequence(root, sequence, audit, digest):
    directory = root / sequence
    annotation = directory / f"{sequence}.xml"
    xml = annotation.read_bytes()
    digest.update(xml)
    for frame in ET.fromstring(xml).findall("pair"):
        audit["annotated"] += 1
        time = frame.attrib["timestamp"]
        views = tuple(frame.find(f"img{view}") for view in (1, 2))
        paths = tuple(directory / str(view) / f"{time}.png" for view in (1, 2))
        present = [path.is_file() and node is not None for path, node in zip(paths, views)]
        if not all(present):
            audit["incomplete"] += 1
            continue
        sizes = []
        mappings = []
        for path, node in zip(paths, views):
            with Image.open(path) as im:
                width, height = im.size
            sizes.append((width, height))
            mapping = {}
            for box in node.findall("box"):
                attrs = box.attrib
                name = attrs["id"]
                if not name.startswith(ROAD_PREFIXES) or attrs["outside"] != "0":
                    continue
                values = [float(attrs[key]) for key in ("xtl", "ytl", "xbr", "ybr")]
                if not all(map(isfinite, values)):
                    raise ValueError(f"Non-finite coordinates: {sequence}/{time}/{name}")
                x0, y0, x1, y1 = [max(0., min(1., value / limit)) for value, limit in
                                  zip(values, (width, height, width, height))]
                if x0 >= x1 or y0 >= y1:
                    audit["degenerate_boxes"] += 1
                    continue
                if name in mapping:
                    raise ValueError(f"Duplicate visible object {name}: {path}")
                mapping[name] = x0, y0, x1, y1
            mappings.append(mapping)
        ids = tuple(sorted(set().union(*mappings)))
        absent = (0., 0., 0., 0.)
        boxes = torch.tensor([[view.get(name, absent) for name in ids] for view in mappings],
                             dtype=torch.float32).reshape(2, len(ids), 4)
        audit["complete"] += 1
        digest.update(f"{sequence}/{time}/{sizes}".encode())
        yield Pair(sequence, time, paths, tuple(sizes), ids, boxes)


def index_data(root, splits, capacity=25):
    flat = [sequence for sequences in splits.values() for sequence in sequences]
    if len(set(flat)) != len(flat):
        raise ValueError("Each sequence must occur once across train, val and test")
    audit, data = {}, {}
    digest = hashlib.sha256()
    for split, names in splits.items():
        tally = Counter(annotated=0, incomplete=0, complete=0, degenerate_boxes=0, over_capacity=0)
        data[split] = [pair for name in names for pair in read_sequence(root, name, tally, digest)]
        tally["over_capacity"] = sum(len(pair.ids) > capacity for pair in data[split])
        audit[split] = dict(tally)
    return data, audit, digest.hexdigest()


class CrossDrone(Dataset):
    def __init__(self, pairs, resolution, augment=False):
        if not pairs:
            raise ValueError("Empty split after selection")
        self.pairs = pairs
        self.resolution = resolution
        self.augment = augment
        self.jitter = ColorJitter(.2, .2, .2, .05)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        corners = pair.boxes.clone()
        images = []
        if self.augment:
            fx, fy = random.uniform(.8, 1.), random.uniform(.8, 1.)
            ox, oy = random.random(), random.random()
            flip = random.random() < .5
        for view, (path, (w, h)) in enumerate(zip(pair.paths, pair.sizes)):
            with Image.open(path) as source:
                image = source.convert("RGB")
            if self.augment:
                cw, ch = int(round(w * fx)), int(round(h * fy))
                x, y = int(round((w - cw) * ox)), int(round((h - ch) * oy))
                image = TF.crop(image, y, x, ch, cw)
                shift = torch.tensor([x / w, y / h, x / w, y / h])
                scale = torch.tensor([cw / w, ch / h, cw / w, ch / h])
                corners[view] = ((corners[view] - shift) / scale).clamp(0, 1)
                if flip:
                    image = TF.hflip(image)
                    corners[view] = corners[view][:, [2, 1, 0, 3]]
                    corners[view, :, [0, 2]] = 1 - corners[view, :, [0, 2]]
                image = self.jitter(image)
            image = TF.resize(image, self.resolution, antialias=True)
            images.append(TF.normalize(TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225]))
        size = corners[..., 2:] - corners[..., :2]
        visible = (size > 0).all(-1)
        keep = visible.any(0)
        boxes = torch.cat((corners[..., :2] + size / 2, size), -1)
        boxes = boxes.masked_fill(~visible[..., None], 0)
        return torch.stack(images), {
            "labels": torch.zeros(int(keep.sum()), dtype=torch.long),
            "boxes": boxes[:, keep], "visible": visible[:, keep], "sizes": pair.sizes,
            "key": f"{pair.sequence}/{pair.time}",
            "ids": [name for name, retained in zip(pair.ids, keep.tolist()) if retained],
        }


def collate(rows):
    images, targets = zip(*rows)
    return torch.stack(images), targets
