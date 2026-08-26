"""Environment-based settings for local infra (Postgres, Redis).

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

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
