"""
OpenAI-Compatible Generic LLM Client
Standard interface for OpenAI, MiniMax, DeepSeek, Local Ollama, and vLLM endpoints.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from backend.ai.config import AIConfig

logger = logging.getLogger("wildfire.ai_client")


class OpenAICompatibleClient:
    """
    Client for any OpenAI-compatible Chat Completions API endpoint.
    Supports tool/function calling and thinking tag extraction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.provider = provider or AIConfig.get_provider()
        self.api_key = api_key or AIConfig.get_api_key()
        self.base_url = (base_url or AIConfig.get_base_url()).rstrip("/")
        self.model = model or AIConfig.get_model()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 40,
    ) -> Dict[str, Any]:
        """
        Execute chat completion request against active OpenAI-compatible endpoint.
        """
        if not self.api_key:
            return {
                "success": False,
                "error": f"API Key is missing for provider '{self.provider}'. Please set LLM_API_KEY or configure ~/.bashrc.",
                "content": f"⚠️ API Key for provider '{self.provider}' is not configured.",
            }

        endpoint = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Include tools if specified
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "WildfireAI-Platform/1.5",
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.status
                raw_body = response.read().decode("utf-8")
                res_json = json.loads(raw_body)

                if status_code != 200:
                    return {
                        "success": False,
                        "status_code": status_code,
                        "error": f"API returned HTTP {status_code}",
                        "raw": res_json,
                    }

                choices = res_json.get("choices", [])
                if not choices:
                    return {
                        "success": False,
                        "error": "No completion choices returned by model",
                        "raw": res_json,
                    }

                message = choices[0].get("message", {})
                content = message.get("content", "") or ""
                tool_calls = message.get("tool_calls", [])

                # Parse and separate <think>...</think> reasoning tags if present
                reasoning = ""
                clean_content = content
                if "<think>" in content and "</think>" in content:
                    parts = content.split("</think>", 1)
                    think_part = parts[0].replace("<think>", "").strip()
                    reasoning = think_part
                    clean_content = parts[1].strip()

                return {
                    "success": True,
                    "provider": self.provider,
                    "model": res_json.get("model", self.model),
                    "content": clean_content,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls,
                    "finish_reason": choices[0].get("finish_reason", "stop"),
                    "usage": res_json.get("usage", {}),
                }

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            logger.error("LLM HTTPError %s: %s", e.code, error_body)
            return {
                "success": False,
                "status_code": e.code,
                "error": f"LLM HTTP {e.code}: {e.reason}",
                "content": f"⚠️ LLM API Error ({e.code}) from {self.provider}: {error_body[:200]}",
            }
        except Exception as e:
            logger.error("LLM Request Exception: %s", e)
            return {
                "success": False,
                "error": str(e),
                "content": f"⚠️ Failed to connect to {self.provider} endpoint ({self.base_url}): {str(e)}",
            }
