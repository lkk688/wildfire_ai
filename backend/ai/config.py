"""
AI Configuration Module
Generic OpenAI-Compatible Multi-Provider Architecture.
Supports MiniMax, Local Ollama, OpenAI, DeepSeek, vLLM, and custom endpoints.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional


class AIConfig:
    """
    Manages LLM providers, API endpoints, credentials, and model selections.
    Any OpenAI-compatible API can be configured and hot-swapped.
    """

    # Known Provider Presets
    PRESETS: Dict[str, Dict[str, Any]] = {
        "minimax": {
            "base_url": "https://api.minimaxi.com/v1",
            "default_model": "MiniMax-M3",
            "env_key": "MINIMAX_API_KEY",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "default_model": "qwen2.5vl:7b-q8_0",
            "env_key": "OLLAMA_API_KEY",
            "default_key": "ollama",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
            "env_key": "OPENAI_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "vllm": {
            "base_url": "http://localhost:8000/v1",
            "default_model": "default",
            "env_key": "VLLM_API_KEY",
            "default_key": "vllm",
        },
    }

    @classmethod
    def _read_config_file(cls) -> Dict[str, Any]:
        """Read local provider_config.json if it exists."""
        config_path = Path(__file__).parent / "provider_config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @classmethod
    def get_provider(cls) -> str:
        """Active LLM provider (default: minimax, or from provider_config.json / LLM_PROVIDER)."""
        file_cfg = cls._read_config_file()
        file_provider = file_cfg.get("active_provider")
        env_provider = os.environ.get("LLM_PROVIDER")
        provider = (env_provider or file_provider or "minimax").lower().strip()
        return provider

    @classmethod
    def get_base_url(cls) -> str:
        """Active API base URL."""
        file_cfg = cls._read_config_file()
        custom_url = os.environ.get("LLM_BASE_URL") or file_cfg.get("base_url")
        if custom_url:
            return custom_url.rstrip("/")

        provider = cls.get_provider()
        preset = cls.PRESETS.get(provider, cls.PRESETS["minimax"])
        return preset["base_url"].rstrip("/")

    @classmethod
    def get_model(cls) -> str:
        """Active model identifier."""
        file_cfg = cls._read_config_file()
        custom_model = os.environ.get("LLM_MODEL") or file_cfg.get("model")
        if custom_model:
            return custom_model.strip()

        provider = cls.get_provider()
        preset = cls.PRESETS.get(provider, cls.PRESETS["minimax"])
        return preset["default_model"]

    @classmethod
    def get_api_key(cls) -> str:
        """
        Retrieve API key with provider-aware fallback and ~/.bashrc discovery.
        """
        file_cfg = cls._read_config_file()
        if file_cfg.get("api_key"):
            return file_cfg["api_key"].strip()

        # 1. Direct LLM_API_KEY override
        direct_key = os.environ.get("LLM_API_KEY")
        if direct_key:
            return direct_key.strip()

        provider = cls.get_provider()
        preset = cls.PRESETS.get(provider, cls.PRESETS["minimax"])
        env_var_name = preset.get("env_key", "MINIMAX_API_KEY")

        # 2. Provider-specific env var (e.g. MINIMAX_API_KEY, OPENAI_API_KEY)
        prov_key = os.environ.get(env_var_name)
        if prov_key:
            return prov_key.strip()

        # 3. If provider has a dummy default key (e.g. Ollama / vLLM)
        if "default_key" in preset:
            return preset["default_key"]

        # 4. Check ~/.bashrc for export statements
        bashrc_path = Path.home() / ".bashrc"
        if bashrc_path.exists():
            try:
                with open(bashrc_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.match(rf"^\s*export\s+{env_var_name}=['\"]?([^'\"\s]+)['\"]?", line)
                        if match:
                            return match.group(1).strip()
            except Exception:
                pass

        # 5. Last fallback: try MINIMAX_API_KEY in ~/.bashrc if provider is minimax
        if provider == "minimax" and bashrc_path.exists():
            try:
                with open(bashrc_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.match(r"^\s*export\s+MINIMAX_API_KEY=['\"]?([^'\"\s]+)['\"]?", line)
                        if match:
                            return match.group(1).strip()
            except Exception:
                pass

        return ""

    @classmethod
    def get_provider_status(cls) -> Dict[str, Any]:
        """Return provider configuration summary for UI / diagnostics."""
        provider = cls.get_provider()
        base_url = cls.get_base_url()
        model = cls.get_model()
        key = cls.get_api_key()
        has_key = bool(key)
        masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else ("Configured" if has_key else "Missing")

        return {
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "has_key": has_key,
            "masked_key": masked_key,
            "available_presets": list(cls.PRESETS.keys()),
        }

    @classmethod
    def update_provider_config(
        cls,
        provider: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save provider configuration to provider_config.json."""
        import json
        config_path = Path(__file__).parent / "provider_config.json"
        current = cls._read_config_file()
        provider_key = provider.lower().strip()
        current["active_provider"] = provider_key
        if model is not None:
            current["model"] = model.strip() if model else None
        else:
            preset = cls.PRESETS.get(provider_key, {})
            current["model"] = preset.get("default_model")
        if base_url is not None:
            current["base_url"] = base_url.strip() if base_url else None
        else:
            current.pop("base_url", None)
        if api_key is not None:
            current["api_key"] = api_key.strip() if api_key else None
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        return cls.get_provider_status()

