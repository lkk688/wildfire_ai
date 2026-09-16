"""
Wildfire AI Agent Module
Provides LLM integration with MiniMax China API (https://api.minimaxi.com/v1),
context-aware domain reasoning, and real-time GIS tool execution.
"""

from backend.ai.agent import WildfireAIAgent
from backend.ai.config import AIConfig

__all__ = ["WildfireAIAgent", "AIConfig"]
