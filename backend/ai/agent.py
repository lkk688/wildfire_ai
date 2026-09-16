"""
Wildfire AI Agent Service
Orchestrates persona system prompts, autonomous tool execution loops, and domain reasoning.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from backend.ai.minimax_client import MiniMaxClient
from backend.ai.tools import WildfireToolRegistry
from backend.services import GISDataService

logger = logging.getLogger("wildfire.agent")


class WildfireAIAgent:
    def __init__(
        self,
        client: Optional[MiniMaxClient] = None,
        tool_registry: Optional[WildfireToolRegistry] = None,
        gis_service: Optional[GISDataService] = None,
    ):
        self.gis = gis_service or GISDataService()
        self.client = client or MiniMaxClient()
        self.tools = tool_registry or WildfireToolRegistry(gis_service=self.gis)

    def _build_system_prompt(self, persona: str, context: Optional[Dict[str, Any]] = None) -> str:
        region = (context or {}).get("region", "san_bruno")
        region_labels = {
            "san_bruno": "San Bruno WUI (Peninsula / San Francisco Watershed)",
            "san_jose": "San Jose Foothills & Alum Rock WUI Interface",
            "santa_cruz": "Santa Cruz Mountains (CZU Fire Interface)",
        }
        active_region_name = region_labels.get(region, "Northern California WUI")

        base_instructions = f"""You are the **Wildfire AI Copilot**, an intelligent, authoritative decision-support assistant embedded within the Wildfire AI GIS Platform.
Active Target Region: **{active_region_name}**.

### Core Platform Capabilities & Tools Available:
1. **Google X Project Bellwether**: Next-gen AI/ML landscape hazard forecasts (100m resolution, quarterly cadence, 1-year and 5-year absolute burn probabilities).
2. **CAL FIRE Statutory FHSZ**: Regulatory Fire Hazard Severity Zones (Moderate, High, Very High) determining building codes (CBC Chapter 7A) and real estate disclosure.
3. **Commercial Insurance Cat-Models (Verisk / Zesty.ai)**: Actuarial underwriting scores influencing policy non-renewals and California FAIR Plan enrollment.
4. **Quantification of the Negative & Contagion**: Eric Saylors' 130ft radiant heat contagion model, measuring threatened structural value, saved value, and municipal mitigation ROI (S-Ratio).
5. **Real-Time Environmental Telemetry**: NOAA HRRR wind plumes, CDEC/RAWS dead/live fuel moisture (DFM/LFMC), and NASA FIRMS VIIRS satellite thermal hotspots.
6. **Building Hardening & Regulations**: California Department of Insurance (CDI) "Safer from Wildfires" (10 CCR § 2644.9) and IBHS Wildfire Prepared Home standards.
"""

        if persona == "firefighter":
            persona_prompt = """
### Active Mode: 🚒 Incident Commander & Tactical Fire Operations
- Target Audience: Fire Chiefs, Fire Marshals, Engine Captains, Emergency Operations Center (EOC).
- Tone: Professional, tactical, concise, data-driven, prioritizing firefighter safety, structural protection, and asset defense.
- Key Focus Areas:
  * Real-time wind speed, gusts, and directional vector shifts (NOAA HRRR).
  * 1-hr, 10-hr, 100-hr Dead Fuel Moisture (DFM) and Live Fuel Moisture (LFMC) Red Flag thresholds.
  * 130-foot radiant heat propagation corridors: calculate threatened asset value vs saved property value (Quantification of the Negative / Saylors S-Ratio) to justify deployment and resource allocation.
  * Topographic rate of spread (ROS) acceleration along steep canyon slopes (>20 degrees).
  * Discrepancies where dynamic AI probability exceeds static CAL FIRE Moderate zones (tactical under-warning alert).
"""
        else:
            persona_prompt = """
### Active Mode: 🏡 Resident & Homeowner Wildfire Safety
- Target Audience: WUI property owners, residents, neighborhood associations (Firewise USA).
- Tone: Empathetic, empowering, clear, actionable, demystifying insurance confusion and fire danger.
- Key Focus Areas:
  * Demystifying why home insurance rates increased or policies were non-renewed, and how to appeal.
  * Clear, prioritized home hardening steps: Class A roof retrofits, 1/16" ember-resistant vents, double-pane tempered glass.
  * The 3 Defensible Space Zones: Zone 0 (0–5ft non-combustible gravel/paver buffer), Zone 1 (5–30ft lean/clean/green), Zone 2 (30–100ft reduced fuel).
  * Step-by-step guidance to qualify for legally mandated California Department of Insurance (CDI) premium discounts.
  * Evacuation preparedness and situational awareness during Red Flag warnings.
"""

        formatting_rules = """
### Formatting Guidelines:
- Use clean Markdown with bullet points, bold highlights, and structured tables when comparing data.
- Whenever citing data (winds, fuel moisture, probability, dollars saved), call the relevant tools or cite accurate numbers.
- Do not mention internal corporate partnership jargon like 'SBFire_SJSU-2026...'; focus squarely on the practical benefits of the web platform.
- Always conclude with concrete, actionable recommendations for the user.
"""
        return base_instructions + persona_prompt + formatting_rules

    def chat(
        self,
        messages: List[Dict[str, str]],
        persona: str = "resident",
        context: Optional[Dict[str, Any]] = None,
        max_tool_iterations: int = 2,
    ) -> Dict[str, Any]:
        """
        Conduct a multi-turn chat turn with autonomous tool calling.
        """
        system_prompt = self._build_system_prompt(persona, context)
        full_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for m in messages:
            full_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        tool_defs = self.tools.get_tool_definitions()
        executed_tools: List[Dict[str, Any]] = []

        # Tool execution loop
        for iteration in range(max_tool_iterations):
            response = self.client.chat_completion(
                messages=full_messages,
                tools=tool_defs,
                temperature=0.3,
            )

            if not response.get("success"):
                return {
                    "success": False,
                    "content": response.get("content", "Failed to communicate with AI model."),
                    "error": response.get("error"),
                    "executed_tools": executed_tools,
                }

            tool_calls = response.get("tool_calls")
            if not tool_calls:
                # No more tools needed, return final content
                return {
                    "success": True,
                    "content": response.get("content", ""),
                    "reasoning": response.get("reasoning", ""),
                    "model": response.get("model"),
                    "executed_tools": executed_tools,
                }

            # Execute tool calls requested by model
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    fn_args = {}

                logger.info("Executing tool: %s with args %s", fn_name, fn_args)
                tool_result = self.tools.execute_tool(fn_name, fn_args)
                executed_tools.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result_summary": "Success" if tool_result.get("success") else "Error"
                })

                # Append assistant tool call and tool response to conversation
                full_messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": [tc]
                })
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", "call_default"),
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

        # Final call after tool resolution
        final_resp = self.client.chat_completion(messages=full_messages, temperature=0.3)
        return {
            "success": final_resp.get("success", False),
            "content": final_resp.get("content", "Completed analysis."),
            "reasoning": final_resp.get("reasoning", ""),
            "model": final_resp.get("model"),
            "executed_tools": executed_tools,
        }

    def get_suggested_prompts(self, persona: str = "resident") -> List[str]:
        """Return high-value pre-built prompt pills for instant query."""
        if persona == "firefighter":
            return [
                "What is the 130ft radiant heat contagion risk for Crestmoor Canyon?",
                "Fetch live NOAA wind vectors and CDEC fuel moisture for San Bruno.",
                "Compare Bellwether 100m dynamic risk against CAL FIRE Moderate FHSZ.",
                "Calculate threatened building value vs saved property ROI (Saylors S-Ratio).",
                "Where are active thermal anomalies / hotspots detected by NASA VIIRS?"
            ]
        else:
            return [
                "How can I qualify for mandatory California insurance discounts?",
                "What is the difference between Zone 0 (0-5ft) and Zone 1 defensible space?",
                "Why did my insurance company non-renew my policy in the WUI?",
                "How does Bellwether AI predict wildfire risk compared to CAL FIRE?",
                "What home hardening retrofits give the highest protection against embers?"
            ]
