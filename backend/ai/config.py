"""
AI Configuration Module
Manages API keys, model parameters, and base URLs for MiniMax China Region.
"""

import os
import re
from pathlib import Path


class AIConfig:
    DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
    DEFAULT_MODEL = "MiniMax-M3"
    FALLBACK_MODEL = "MiniMax-M2.5"

    @classmethod
    def get_api_key(cls) -> str:
        """
        Retrieve MINIMAX_API_KEY from environment or parse directly from ~/.bashrc.
        """
        key = os.environ.get("MINIMAX_API_KEY")
        if key:
            return key.strip()

        bashrc_path = Path.home() / ".bashrc"
        if bashrc_path.exists():
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
    def get_base_url(cls) -> str:
        return (os.environ.get("MINIMAX_BASE_URL") or cls.DEFAULT_BASE_URL).rstrip("/")

    @classmethod
    def get_model(cls) -> str:
        return os.environ.get("MINIMAX_MODEL") or cls.DEFAULT_MODEL
