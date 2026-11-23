"""
Wildfire AI Agent Service
Orchestrates persona system prompts, autonomous tool execution loops, and domain reasoning.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from backend.ai.openai_client import OpenAICompatibleClient
from backend.ai.tools import WildfireToolRegistry
from backend.services import GISDataService

logger = logging.getLogger("wildfire.agent")


class WildfireAIAgent:
    def __init__(
        self,
        client: Optional[OpenAICompatibleClient] = None,
        tool_registry: Optional[WildfireToolRegistry] = None,
        gis_service: Optional[GISDataService] = None,
    ):
        self.gis = gis_service or GISDataService()
        self._custom_client = client
        self.tools = tool_registry or WildfireToolRegistry(gis_service=self.gis)

    @property
    def client(self) -> OpenAICompatibleClient:
        return self._custom_client if self._custom_client is not None else OpenAICompatibleClient()



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

### Data available through your tools (all real; cite source and time):
1. **Bellwether** 1-year and 5-year probability of a 100 m cell burning (partner GeoTIFFs, San Mateo / Santa Cruz pilot area only).
2. **USDA Wildfire Risk to Communities** annual burn probability (FSim, 30 m, landscape as of end-2020; national coverage). Older and coarser than Bellwether, but public and available everywhere.
3. **CAL FIRE Fire Hazard Severity Zones** (SRA 2023, LRA 2025) from the official services.
4. **Live conditions**: NOAA HRRR weather via Open-Meteo, National Weather Service alerts, CDEC RAWS fuel moisture.
5. **Active fires**: CAL FIRE incidents, NIFC current perimeters, NASA FIRMS VIIRS detections (last 24 h).
6. **Location context**: ESA WorldCover land cover, CAL FIRE historical perimeters, Microsoft building footprints, USGS 3DEP slope.
No insurance scores, parcel values or property records are available; never invent them.
If a tool fails or a source does not cover a place, say so instead of guessing.
"""

        if persona == "firefighter":
            persona_prompt = """
### Active Mode: 🚒 Incident Commander & Tactical Fire Operations
- Target Audience: Fire Chiefs, Fire Marshals, Engine Captains, Emergency Operations Center (EOC).
- Tone: Professional, tactical, concise, data-driven, prioritizing firefighter safety, structural protection, and asset defense.
- Key Focus Areas:
  * Current wind, gusts, humidity and Red Flag Warnings (HRRR, NWS).
  * 10-h fuel moisture at nearby RAWS stations.
  * Active incidents, perimeters and satellite detections.
  * Structures near a location (building footprints), slope and land cover.
  * Where Bellwether's near-term probability and the FHSZ map disagree.
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
- Whenever citing data (wind, fuel moisture, probabilities, fires), call the relevant tool and give its source and time.
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
        self.tools.default_region = (context or {}).get("region", "san_bruno")
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
                "What are the current wind, humidity and fuel moisture in this region?",
                "Are there active fires or satellite hotspots in California right now?",
                "Where do Bellwether and the CAL FIRE FHSZ map disagree in this region?",
                "How many structures are within 130 ft of 37.62, -122.43, and what is the FHSZ zone there?",
                "Is there a Red Flag Warning in effect here?"
            ]
        else:
            return [
                "How can I qualify for mandatory California insurance discounts?",
                "What is the difference between Zone 0 (0-5ft) and Zone 1 defensible space?",
                "What fire hazard zone is my area in, and has it burned before?",
                "How does Bellwether's forecast differ from the CAL FIRE hazard map?",
                "What home hardening retrofits give the highest protection against embers?"
            ]
