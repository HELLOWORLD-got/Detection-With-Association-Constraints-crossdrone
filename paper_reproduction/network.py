"""Independent implementation of Fig. 3 and equations (1)–(14)."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


@dataclass
class Architecture:
    views: int = 2
    classes: int = 1
    queries: int = 25
    width: int = 256
    heads: int = 8
    fusion_depth: int = 3
    instance_depth: int = 3
    object_depth: int = 2
    backbone: str = "resnet50"


def position_grid(height, width, channels, reference):
    axis = [torch.linspace(0, 1, n, device=reference.device) for n in (height, width)]
    frequencies = 10000 ** (-torch.arange(channels // 4, device=reference.device) / (channels // 4))
    coords = torch.stack(torch.meshgrid(*axis, indexing="ij"), dim=-1)
    phases = 2 * torch.pi * coords[..., None] * frequencies
    return torch.stack((phases.sin(), phases.cos()), -1).reshape(height * width, channels).to(reference.dtype)


class ImageEncoder(nn.Module):
    def __init__(self, cfg, pretrained=False, weights_file=None):
        super().__init__()
        resnet = cfg.backbone == "resnet50"
        if not resnet and cfg.backbone != "vit_b_16":
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")
        factory = models.resnet50 if resnet else models.vit_b_16
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if resnet else models.ViT_B_16_Weights.IMAGENET1K_V1
        net = factory(weights=weights if pretrained and not weights_file else None)
        if weights_file:
            net.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        self.resnet = resnet
        if resnet:
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.project = nn.Conv2d(2048, cfg.width, 1)
        else:
            self.patchify = net.conv_proj
            self.class_token = net.class_token
            self.transformer = net.encoder
            self.project = nn.Conv2d(net.hidden_dim, cfg.width, 1)

    def forward(self, images):
        if self.resnet:
            return self.project(self.features(images))
        patches = self.patchify(images)
        batch, channels, height, width = patches.shape
        cls, pos = self.transformer.pos_embedding.split([1, self.transformer.pos_embedding.shape[1] - 1], dim=1)
        side = int(pos.shape[1] ** .5)
        pos = F.interpolate(pos.transpose(1, 2).reshape(1, channels, side, side),
                            size=(height, width), mode="bicubic", align_corners=False)
        tokens = torch.cat((self.class_token.expand(batch, -1, -1),
                            patches.flatten(2).transpose(1, 2)), dim=1)
        tokens = tokens + torch.cat((cls, pos.flatten(2).transpose(1, 2)), dim=1)
        encoded = self.transformer.ln(self.transformer.layers(self.transformer.dropout(tokens)))
        return self.project(encoded[:, 1:].transpose(1, 2).reshape(batch, channels, height, width))


def mlp(width, hidden, output):
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, output))


class AttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.ff = mlp(cfg.width, 4 * cfg.width, cfg.width)
        self.norm_attention = nn.LayerNorm(cfg.width)
        self.norm_ff = nn.LayerNorm(cfg.width)

    def forward(self, query, memory):
        update = self.attention(query, memory, memory, need_weights=False)[0]
        hidden = self.norm_attention(query + update)
        return self.norm_ff(hidden + self.ff(hidden))


class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cross = nn.ModuleList(AttentionBlock(cfg) for _ in range(cfg.fusion_depth))
        self.merge = nn.Linear(cfg.views * cfg.width, cfg.width)
        self.global_attention = nn.ModuleList(AttentionBlock(cfg) for _ in range(cfg.fusion_depth))

    def forward(self, views):
        updated = []
        for index, tokens in enumerate(views):
            other = torch.cat([v for j, v in enumerate(views) if j != index], dim=1)
            for block in self.cross:
                tokens = block(tokens, other)
            updated.append(tokens)
        scene = self.merge(torch.cat(updated, dim=-1))
        for block in self.global_attention:
            scene = block(scene, scene)
        return scene


class InstanceQueries(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.slots = nn.Embedding(cfg.queries, cfg.width)
        nn.init.normal_(self.slots.weight, std=.02)
        self.blocks = nn.ModuleList(AttentionBlock(cfg) for _ in range(cfg.instance_depth))
        self.classifier = mlp(cfg.width, cfg.width, cfg.classes + 1)

    def forward(self, scene, positions):
        memory = scene + positions
        instances = self.slots.weight[None].expand(scene.shape[0], -1, -1)
        for block in self.blocks:
            instances = block(instances, memory)
        return instances, self.classifier(instances)


class TripletBlock(nn.Module):
    """Eq. (11) with the previously selected position-modulated instance Key."""
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.query = nn.Linear(cfg.width, cfg.width)
        self.key = nn.Linear(cfg.width, cfg.width, bias=False)
        self.value = nn.Linear(cfg.width, cfg.width)
        self.readout = nn.Linear(cfg.width, cfg.width)
        self.ff = mlp(cfg.width, 4 * cfg.width, cfg.width)
        self.norm_attention = nn.LayerNorm(cfg.width)
        self.norm_ff = nn.LayerNorm(cfg.width)

    def forward(self, slots, instances, tokens, positions):
        query = self.query(slots).unflatten(-1, (self.heads, -1))
        value = self.value(tokens).unflatten(-1, (self.heads, -1))
        # Reassociate q·W_k(f_i ⊙ p_t) to avoid allocating B×N×T×D keys.
        coefficients = torch.einsum("bnhd,hdc->bnhc", query, self.key.weight.unflatten(0, (self.heads, -1)))
        weights = torch.einsum("bnhc,tc->bnht", coefficients * instances[:, :, None], positions)
        weights = (weights / query.shape[-1] ** .5).softmax(dim=-1)
        context = torch.einsum("bnht,bthd->bnhd", weights, value).flatten(-2)
        hidden = self.norm_attention(slots + self.readout(context))
        return self.norm_ff(hidden + self.ff(hidden))


class ViewQueries(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.slots = nn.Embedding(cfg.queries, cfg.width)
        nn.init.normal_(self.slots.weight, std=.02)
        self.blocks = nn.ModuleList(TripletBlock(cfg) for _ in range(cfg.object_depth))
        self.box = mlp(cfg.width, cfg.width, 4)
        self.visibility = mlp(cfg.width, cfg.width, 1)

    def forward(self, instances, tokens, positions):
        slots = self.slots.weight[None].expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            slots = block(slots, instances, tokens, positions)
        return self.box(slots).sigmoid(), self.visibility(slots).squeeze(-1)


class AssociationNetwork(nn.Module):
    def __init__(self, cfg, pretrained=False, weights_file=None):
        super().__init__()
        if cfg.views < 2 or cfg.queries < 1 or cfg.classes < 1 or cfg.width < 4 or cfg.heads < 1:
            raise ValueError("Invalid number of views, queries, classes or channels")
        if cfg.width % cfg.heads or cfg.width % 4 or min(cfg.fusion_depth, cfg.instance_depth, cfg.object_depth) < 1:
            raise ValueError("Invalid attention dimensions or depth")
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg, pretrained, weights_file)
        self.fusion = Fusion(cfg)
        self.instances = InstanceQueries(cfg)
        self.views = nn.ModuleList(ViewQueries(cfg) for _ in range(cfg.views))

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:3] != (self.cfg.views, 3):
            raise ValueError("Expected images with shape B,K,3,H,W")
        b, k = images.shape[:2]
        maps = self.encoder(images.flatten(0, 1))
        _, d, h, w = maps.shape
        tokens = maps.reshape(b, k, d, h * w).transpose(-2, -1).unbind(1)
        positions = position_grid(h, w, d, maps)
        instances, category = self.instances(self.fusion(tokens), positions)
        box, visibility = zip(*(branch(instances, view, positions) for branch, view in zip(self.views, tokens)))
        return {"category": category, "box": torch.stack(box, dim=1),
                "visibility": torch.stack(visibility, dim=1)}
