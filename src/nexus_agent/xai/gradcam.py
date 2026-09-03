"""Visual attribution maps (Constitution Art. VI §1(1), Art. XII §5).

"Grad-CAM++ is hooked at the final convolutional block of the vision
backbone (or computed via attention rollout for a ViT backbone), with the
resulting heatmap upsampled to source-tile resolution by bilinear
interpolation."

`grad_cam_plusplus` targets `Vgg16Embedder` (`vision/embedding.py`)'s final
conv layer (conv5_3). `attention_rollout` targets `Dinov2Embedder`'s
transformer blocks. Both return an (H, W) float32 heatmap normalized to
[0, 1] at the input patch's own resolution.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus_agent.vision.embedding import Dinov2Embedder, Vgg16Embedder


def _last_conv2d(module: torch.nn.Module) -> torch.nn.Conv2d:
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise RuntimeError(
            "no nn.Conv2d found inside the embedder's backbone -- Grad-CAM++ "
            "needs a convolutional target layer to hook."
        )
    return convs[-1]


def grad_cam_plusplus(
    embedder: "Vgg16Embedder",
    patch: np.ndarray,
    target_scalar_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray:
    """Grad-CAM++ (Chattopadhay et al. 2018) over the last `nn.Conv2d` inside
    `embedder._backbone` -- conv5_3 for `Vgg16Embedder`, matching Art. XII
    §5's "final convolutional block."

    `patch`: (H, W, 3) uint8 RGB, the same shape `crop_patch()` produces.
    `target_scalar_fn(pooled_features) -> scalar` picks what to explain;
    it defaults to `.sum()` of the global-average-pooled conv output --
    the exact scalar reduction `Vgg16Embedder.embed_patch` itself performs
    (before zero-padding to 1024-dim) -- so the default heatmap explains
    "what drove this cell's own embedding," not an arbitrary auxiliary head.

    Returns an (H, W) float32 array normalized to [0, 1]. Runs a real
    forward+backward pass (not `torch.inference_mode()`, unlike
    `embed_patch` -- Grad-CAM++ needs gradients), so it's slower than a
    plain embedding call but still CPU-fast for a single patch.
    """
    target_layer = _last_conv2d(embedder._backbone)

    activations: dict[str, torch.Tensor] = {}

    def _forward_hook(_module: torch.nn.Module, _input: tuple, output: torch.Tensor) -> torch.Tensor:
        # `retain_grad()` on the conv output itself (a non-leaf tensor), plus
        # handing the *next* layer -- VGG's `nn.ReLU(inplace=True)` -- a clone
        # to mutate instead of this tensor, rather than `register_full_backward_hook`:
        # a full-backward-hook wraps the module's output in an autograd
        # `BackwardHookFunction`, and the very next in-place ReLU mutating that
        # wrapped view raises "view ... modified inplace" -- a known PyTorch
        # gotcha for Grad-CAM on VGG-style backbones with in-place ReLU.
        output.retain_grad()
        activations["value"] = output
        return output.clone()

    handle_fwd = target_layer.register_forward_hook(_forward_hook)
    try:
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)
        tensor = embedder._preprocess(tensor).unsqueeze(0)
        features = embedder._backbone(tensor)  # (1, C, H', W')
        pooled = features.mean(dim=[2, 3])  # (1, C) -- same reduction embed_patch performs
        target = target_scalar_fn(pooled) if target_scalar_fn is not None else pooled.sum()
        embedder._backbone.zero_grad(set_to_none=True)
        target.backward()
    finally:
        handle_fwd.remove()

    activation = activations["value"]  # (1, C, H', W')
    grad = activation.grad  # (1, C, H', W'), populated by retain_grad() above

    grad_sq = grad**2
    grad_cube = grad**3
    alpha_denom = 2 * grad_sq + (activation * grad_cube).sum(dim=(2, 3), keepdim=True)
    alpha = grad_sq / (alpha_denom + 1e-8)
    weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))  # (1, 1, H', W')

    cam_min = cam.amin(dim=(2, 3), keepdim=True)
    cam_max = cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    cam = F.interpolate(cam, size=patch.shape[:2], mode="bilinear", align_corners=False)
    return cam.squeeze(0).squeeze(0).detach().numpy().astype(np.float32)


def attention_rollout(embedder: "Dinov2Embedder", patch: np.ndarray) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) over `Dinov2Embedder`'s
    `torch.hub` ViT-L/14 blocks, Art. XII §5's named alternative to
    Grad-CAM++ for a ViT backbone.

    Registers a forward hook on each block's `attn.attn_drop`, assuming it's
    an `nn.Dropout` submodule that receives the post-softmax attention-
    probability tensor as its *input*, immediately after the softmax and
    before any further projection -- true of many eager-mode ViT/timm
    implementations. If a given checkpoint's implementation doesn't expose
    that as a hookable `nn.Module`, this raises `RuntimeError` naming the
    assumption rather than silently rolling out a tensor that isn't actually
    an attention map.

    DISCLOSURE: the real `facebookresearch/dinov2` `torch.hub` checkpoint
    this repo actually loads (`Dinov2Embedder`) does NOT satisfy this
    assumption. Its `Attention.forward` (see the hub-cached
    `dinov2/layers/attention.py`) calls the fused
    `torch.nn.functional.scaled_dot_product_attention` kernel directly and
    never materializes a separate post-softmax attention-probability tensor;
    `attn_drop` there is a plain `float` dropout-rate attribute, not an
    `nn.Dropout` module, so there is nothing to hook. Calling this function
    against a real `Dinov2Embedder` therefore always raises `RuntimeError`
    below, as designed -- this is disclosed rather than silently rolling out
    nonsense. Making rollout actually work against that checkpoint would
    require monkey-patching its attention forward pass to compute
    `softmax(q @ k^T)` eagerly instead of via the fused kernel, which is out
    of scope here.

    For each layer: average the captured (1, num_heads, N, N) tensor over
    heads, mix in the identity (`0.5*A + 0.5*I`) to account for the residual
    connection, row-normalize, then chain-multiply across layers in order
    (`A_hat_1 @ A_hat_2 @ ... @ A_hat_L`). The CLS row over patch tokens
    (`rollout[0, 1:]`) is reshaped to the square patch grid, upsampled by
    bilinear interpolation to `patch.shape[:2]`, and min-max normalized to
    [0, 1].
    """
    captured: list[torch.Tensor] = []

    def _hook(_module: torch.nn.Module, hook_input: tuple, _output: torch.Tensor) -> None:
        captured.append(hook_input[0].detach())

    blocks = embedder._model.blocks
    handles = []
    for block in blocks:
        attn_drop = getattr(getattr(block, "attn", None), "attn_drop", None)
        if not isinstance(attn_drop, torch.nn.Module):
            for h in handles:
                h.remove()
            raise RuntimeError(
                "attention_rollout assumes every DINOv2 transformer block exposes "
                "`.attn.attn_drop` as a hookable `nn.Module` (the post-softmax "
                "attention-probability dropout) -- this checkpoint's implementation "
                f"has `attn_drop = {attn_drop!r}` instead (likely a fused "
                "scaled_dot_product_attention implementation that never materializes "
                "post-softmax attention weights), so rollout can't be computed without "
                "risking a silently-wrong heatmap."
            )
        handles.append(attn_drop.register_forward_hook(_hook))

    try:
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="bilinear")
        mean = torch.tensor(embedder._MEAN).view(1, 3, 1, 1)
        std = torch.tensor(embedder._STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            embedder._model(tensor)
    finally:
        for h in handles:
            h.remove()

    num_tokens = captured[0].shape[-1]
    identity = torch.eye(num_tokens)
    attn_hats = []
    for attn in captured:
        attn_avg = attn.mean(dim=1).squeeze(0)  # (num_heads, N, N) -> (N, N)
        attn_hat = 0.5 * attn_avg + 0.5 * identity
        attn_hat = attn_hat / attn_hat.sum(dim=-1, keepdim=True)
        attn_hats.append(attn_hat)

    rollout = attn_hats[0]
    for attn_hat in attn_hats[1:]:
        rollout = rollout @ attn_hat

    cls_row = rollout[0, 1:]  # CLS attention over patch tokens
    num_patch_tokens = cls_row.shape[0]
    grid_size = int(math.sqrt(num_patch_tokens))
    grid = cls_row.reshape(1, 1, grid_size, grid_size)
    upsampled = F.interpolate(grid, size=patch.shape[:2], mode="bilinear", align_corners=False)
    upsampled = upsampled.squeeze(0).squeeze(0)

    cam_min = upsampled.min()
    cam_max = upsampled.max()
    normalized = (upsampled - cam_min) / (cam_max - cam_min + 1e-8)
    return normalized.detach().numpy().astype(np.float32)
