from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(RuntimeError):
    pass


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    devin_api_key: str | None = None
    devin_org_id: str | None = None
    devin_api_base: str = "https://api.devin.ai"
    github_token: str | None = None
    github_api_base: str = "https://api.github.com"
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            openai_api_key=_env("OPENAI_API_KEY"),
            llm_provider=_env("PAPERWEIGHT_LLM_PROVIDER"),
            llm_model=_env("PAPERWEIGHT_LLM_MODEL"),
            devin_api_key=_env("DEVIN_API_KEY"),
            devin_org_id=_env("DEVIN_ORG_ID"),
            devin_api_base=_env("DEVIN_API_BASE") or "https://api.devin.ai",
            github_token=_env("GITHUB_TOKEN") or _env("GH_TOKEN"),
            github_api_base=_env("GITHUB_API_BASE") or "https://api.github.com",
        )

    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env_names = ", ".join(n.upper() for n in missing)
            raise ConfigError(f"missing required configuration: {env_names}")
