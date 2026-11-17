"""OmiCLIP image and gene-sentence encoders (Constitution Art. XII §2, as
amended in v1.3).

OmiCLIP (Chen et al., Nature Methods 2025, "A visual-omics foundation model
to bridge histopathology with spatial transcriptomics") is a CoCa ViT-L/14
CLIP model trained on 2.2 million paired H&E patches and transcriptomic
profiles. Each profile is written as a "gene sentence" -- the names of its
50 most-expressed genes, highest first -- so an image patch and an
expression profile land in one shared 768-dim space. Weights come from the
Loki release (`WangGuangyuLab/Loki` on Hugging Face, BSD-3-Clause), pinned
to one revision so `model_version` always names the exact weights used.

Used here two ways:

- **Morphology encoder** (`OmiclipEmbedder.embed_patches`): the default
  Vision embedder. The 768-dim L2-normalized image latent is zero-padded to
  Art. XII §2's fixed 1024-dim contract, as VGG16's 512-dim output is.
- **Cross-modal agreement** (`analyst/agreement.py`): because image and
  expression share one space, each cell's image can be checked against its
  own expression profile -- an evidence signal no single-modality encoder
  can give.

Disclosures:
- OmiCLIP was trained on Visium *spot* patches (~55 µm, many cells). This
  pipeline applies it to single-cell CellPose crops, which is outside its
  training distribution. Nothing here has measured how well it transfers.
- Loki removes housekeeping genes from gene sentences using a list it does
  not publish with the model; `gene_sentences` applies Loki's other filters
  but not that one, so sentences can differ from the ones OmiCLIP saw.
- First use downloads a 7.7 GB checkpoint (with training state; ~2.5 GB of
  it is model weights). It is loaded with `torch.load(weights_only=True)`,
  allowing only the numpy scalar/dtype types its training metadata pickles
  (`_load_state_dict`) -- never an unrestricted pickle load.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from nexus_agent.vision.embedding import EMBEDDING_DIM, EmbeddingBackend

OMICLIP_REPO = "WangGuangyuLab/Loki"
OMICLIP_REVISION = "8e08d3a131428feed2341167084d7b15c143a2a7"
OMICLIP_FILENAME = "checkpoint.pt"
OMICLIP_ARCH = "coca_ViT-L-14"
OMICLIP_DIM = 768
GENES_PER_SENTENCE = 50
BATCH_SIZE = 32


def omiclip_checkpoint_path() -> str:
    """Local path to the pinned checkpoint: `settings.omiclip_checkpoint_path`
    if set, else the Hugging Face cache (downloading on first use).
    """
    from nexus_agent.shared.config import settings

    if settings.omiclip_checkpoint_path:
        return settings.omiclip_checkpoint_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(OMICLIP_REPO, OMICLIP_FILENAME, revision=OMICLIP_REVISION)


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """The model weights from the Loki training checkpoint, which also holds
    optimizer and grad-scaler state. `weights_only=True` rejects it as-is
    because its metadata pickles numpy scalars and dtypes, so exactly those
    types are allowed and nothing else -- the file came from the network,
    and an unrestricted pickle load would execute arbitrary code in it.
    """
    allowed = [
        np.dtype,
        # numpy 2 moved numpy.core to numpy._core; the pickle names the old path
        ((getattr(np, "_core", None) or np.core).multiarray.scalar, "numpy.core.multiarray.scalar"),
        *(getattr(np.dtypes, name) for name in dir(np.dtypes) if name.endswith("DType")),
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def gene_sentences(var_names, expression: np.ndarray, n_genes: int = GENES_PER_SENTENCE) -> list[str]:
    """OmiCLIP's text input for each row of `expression` ((n_cells,
    n_genes), raw or normalized -- the per-cell ranking is the same): the
    names of its `n_genes` most-expressed genes, highest first, separated
    by spaces. Gene names containing '.' or '-' are dropped first, as Loki
    does. Genes with zero expression are never included, so a sparse cell
    gets a shorter sentence rather than arbitrary zero-count genes.
    """
    names = np.array([str(g) for g in var_names])
    keep = np.array(["." not in g and "-" not in g for g in names])
    names = names[keep]
    matrix = np.asarray(expression, dtype=np.float64)[:, keep]

    sentences = []
    for row in matrix:
        order = np.argsort(-row, kind="stable")[:n_genes]
        sentences.append(" ".join(names[i] for i in order if row[i] > 0))
    return sentences


class OmiclipEmbedder(EmbeddingBackend):
    model_version = f"omiclip-coca-vitl14-loki@{OMICLIP_REVISION[:7]}-zeropad1024-v1"

    def __init__(self, device: str | None = None, checkpoint_path: str | None = None) -> None:
        import open_clip

        from nexus_agent.shared.config import settings

        self._device = device or settings.nexus_device
        model, _train_preprocess, self._preprocess = open_clip.create_model_and_transforms(OMICLIP_ARCH, pretrained=None)
        # strict: every OmiCLIP weight must land, or this isn't OmiCLIP
        model.load_state_dict(_load_state_dict(checkpoint_path or omiclip_checkpoint_path()), strict=True)
        self._model = model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(OMICLIP_ARCH)

    def image_tensor(self, patches: list[np.ndarray]) -> torch.Tensor:
        """(n, 3, 224, 224) model input, using OmiCLIP's own preprocessing."""
        from PIL import Image

        return torch.stack([self._preprocess(Image.fromarray(np.ascontiguousarray(p))) for p in patches]).to(self._device)

    @torch.inference_mode()
    def image_latents(self, patches: list[np.ndarray]) -> np.ndarray:
        """(n, 768) L2-normalized image latents."""
        out = []
        for start in range(0, len(patches), BATCH_SIZE):
            latents = self._model.encode_image(self.image_tensor(patches[start : start + BATCH_SIZE]))
            out.append(torch.nn.functional.normalize(latents, dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float64) if out else np.zeros((0, OMICLIP_DIM))

    @torch.inference_mode()
    def text_latents(self, sentences: list[str]) -> np.ndarray:
        """(n, 768) L2-normalized gene-sentence latents."""
        out = []
        for start in range(0, len(sentences), BATCH_SIZE):
            tokens = self._tokenizer(sentences[start : start + BATCH_SIZE]).to(self._device)
            out.append(torch.nn.functional.normalize(self._model.encode_text(tokens), dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float64) if out else np.zeros((0, OMICLIP_DIM))

    def embed_patches(self, patches: list[np.ndarray]) -> np.ndarray:
        latents = self.image_latents(patches)
        return np.concatenate([latents, np.zeros((len(latents), EMBEDDING_DIM - OMICLIP_DIM))], axis=1)

    def embed_patch(self, patch: np.ndarray) -> np.ndarray:
        return self.embed_patches([patch])[0]


def image_latent_from_embedding(embedding_vector) -> np.ndarray:
    """Recover the 768-dim OmiCLIP image latent from a 1024-dim Vision
    embedding (`OmiclipEmbedder.embed_patches` zero-pads it).
    """
    return np.asarray(embedding_vector, dtype=np.float64)[:OMICLIP_DIM]


def is_omiclip_version(model_version: str | None) -> bool:
    return bool(model_version) and model_version.startswith("omiclip-")


@lru_cache(maxsize=1)
def get_omiclip_embedder() -> OmiclipEmbedder:
    """One OmiCLIP instance per process (the model is ~2.5 GB in memory)."""
    return OmiclipEmbedder()
