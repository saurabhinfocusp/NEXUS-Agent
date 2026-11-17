"""Environment-based settings for local infra (Postgres, Redis, S3-compatible
object storage).

Mirrors the variables in .env.example / docker-compose.yml.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "nexus_agent"
    postgres_user: str = "nexus"
    postgres_password: str = "nexus_dev_password"

    redis_host: str = "localhost"
    redis_port: int = 6379

    # S3-compatible object storage (Art. XII §8): images, masks, embeddings,
    # heatmaps. No client wrapper yet -- Phase 3+ is the first consumer, once
    # there's real per-cell data to store; this is just the connection
    # convention so later phases don't have to invent it.
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "nexus"
    s3_secret_key: str = "nexus_dev_password"
    s3_bucket: str = "nexus-agent"

    # Torch/CellPose device for Vision's real inference path (Art. XII §2).
    # Defaults to "cpu" so a plain `environment.yml` install (CPU-only torch
    # wheels, see that file's comment) always works; set NEXUS_DEVICE=cuda
    # once a CUDA-enabled torch/torchvision is installed and a GPU is
    # present. Not validated against `torch.cuda.is_available()` here --
    # that check happens where the device is actually used, so the error if
    # it's wrong (no CUDA torch installed, or no GPU) is a real torch/
    # CellPose error at model-load time, not a silently-ignored setting.
    nexus_device: str = "cpu"

    # Object-store URI of a pretrained fusion base (`analyst/pretrain.py::
    # save_fusion_base` bytes). Unset (the default): Analyst uses the
    # fixed-seed untrained base. Changing this changes the coordinate space
    # of every fused embedding, so cell-type checkpoints trained under
    # another base stop being applied (see learning/celltyping.py).
    fusion_checkpoint_uri: str | None = None

    # Optional LLM layer (`nexus_agent/llm/`), off by default. Neither
    # switch lets a model create or alter a claim: intake only picks the
    # opt-in analyses for a free-text question, and the narrative is prose
    # over already-reviewed claims, verified sentence by sentence.
    llm_intake_enabled: bool = False
    llm_narrative_enabled: bool = False
    llm_model: str = "claude-opus-5-5"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
