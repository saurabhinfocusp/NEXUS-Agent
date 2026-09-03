"""Grad-CAM++ (Vgg16Embedder) is a fast test -- VGG16 weights are already
cached under `~/.cache/torch/hub/checkpoints/` on this dev box, so
instantiating `Vgg16Embedder` doesn't trigger a network fetch. Attention
rollout (Dinov2Embedder) is marked `integration`: DINOv2's `torch.hub`
checkpoint (~1.2GB) is a lazy download on first instantiation, so this test
must not run as part of the default fast suite even if the weights happen
to already be warm on a given machine.

NOTE on `attention_rollout`: the real `facebookresearch/dinov2` hub
checkpoint's `Attention.forward` uses the fused
`scaled_dot_product_attention` kernel and never materializes a hookable
post-softmax attention tensor (`attn_drop` there is a `float`, not an
`nn.Dropout`) -- see `gradcam.py`'s docstring for the full disclosure. So
against the real model this function is expected to raise `RuntimeError`,
which is what this test actually verifies, rather than a valid heatmap.
"""

import numpy as np
import pytest

from nexus_agent.xai.gradcam import attention_rollout, grad_cam_plusplus


def test_grad_cam_plusplus_produces_a_valid_heatmap():
    from nexus_agent.vision.embedding import Vgg16Embedder

    embedder = Vgg16Embedder()
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    heatmap = grad_cam_plusplus(embedder, patch)

    assert heatmap.shape == (64, 64)
    assert heatmap.dtype == np.float32
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0)
    assert not np.allclose(heatmap, 0.0)


@pytest.fixture(scope="module")
def dinov2_embedder():
    from nexus_agent.vision.embedding import Dinov2Embedder

    try:
        return Dinov2Embedder()
    except Exception as exc:  # e.g. no network to fetch the hub repo/weights
        pytest.skip(f"Dinov2Embedder() could not be instantiated ({exc}); needs network access.")


@pytest.mark.integration
def test_attention_rollout_raises_on_the_real_fused_attention_checkpoint(dinov2_embedder):
    """The real `facebookresearch/dinov2` hub checkpoint uses a fused
    `scaled_dot_product_attention` kernel with no hookable post-softmax
    attention tensor -- `attention_rollout` is documented to raise
    `RuntimeError` naming that assumption rather than silently produce a
    heatmap from the wrong tensor. This is the honest, currently-correct
    behavior against the real model, not a placeholder we forgot to fix.
    """
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="attn_drop"):
        attention_rollout(dinov2_embedder, patch)
