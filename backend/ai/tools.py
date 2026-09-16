"""
Wildfire AI Tool Registry
Provides callable data retrieval and calculation tools for LLM agent function calling.
"""

from typing import Dict, Any, List, Optional
from backend.services import GISDataService


class WildfireToolRegistry:
    def __init__(self, gis_service: Optional[GISDataService] = None):
        self.gis = gis_service or GISDataService()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Return OpenAPI-compliant function definitions for MiniMax / OpenAI function calling.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_live_weather",
                    "description": "Fetch live meteorological telemetry from NOAA HRRR model including wind speed, wind gust, wind direction (degrees), temperature, and relative humidity for the target region.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_live_fuel_moisture",
                    "description": "Retrieve live ground station telemetry from CDEC / RAWS network for Dead Fuel Moisture (10-hr, 100-hr DFM) and Live Fuel Moisture Content (LFMC).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_corridor_risk",
                    "description": "Calculate the 130ft (or wind-adjusted) radiant heat contagion corridor around a coordinate. Evaluates threatened structures, assessed property value, saved negative value, and municipal ROI.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "number",
                                "description": "Latitude of the target center point (e.g., 37.625 for San Bruno, 37.338 for San Jose)"
                            },
                            "lng": {
                                "type": "number",
                                "description": "Longitude of the target center point (e.g., -122.425 for San Bruno, -121.886 for San Jose)"
                            },
                            "radius_feet": {
                                "type": "number",
                                "description": "Radiant heat propagation radius in feet (default 130.0 ft, based on Eric Saylors contagion model)"
                            }
                        },
                        "required": ["lat", "lng"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_model_evaluation",
                    "description": "Retrieve cross-model evaluation and gap analysis data comparing Google X Bellwether ML, CAL FIRE statutory FHSZ, and commercial insurance risk scores.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region": {
                                "type": "string",
                                "description": "Target region identifier: 'san_bruno', 'san_jose', or 'santa_cruz'",
                                "enum": ["san_bruno", "san_jose", "santa_cruz"]
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_home_hardening_guidelines",
                    "description": "Fetch California Department of Insurance (CDI) 'Safer from Wildfires' mandatory discount requirements and IBHS 'Wildfire Prepared Home' standards.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_active_hotspots",
                    "description": "Query active satellite thermal anomalies and hotspots detected by NASA VIIRS sensors in Northern California.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered tool by name with provided arguments.
        """
        try:
            if name == "get_live_weather":
                return {"success": True, "weather": self.gis.get_live_weather()}

            elif name == "get_live_fuel_moisture":
                return {"success": True, "stations": self.gis.get_fuel_moisture_layer()}

            elif name == "query_corridor_risk":
                lat = float(arguments.get("lat", 37.625))
                lng = float(arguments.get("lng", -122.425))
                radius_feet = float(arguments.get("radius_feet", 130.0))
                return {"success": True, "corridor": self.gis.query_corridor(lat, lng, radius_feet)}

            elif name == "get_model_evaluation":
                region = arguments.get("region", "san_bruno")
                return {"success": True, "evaluation": self.gis.get_model_evaluation_data(region)}

            elif name == "get_home_hardening_guidelines":
                return {
                    "success": True,
                    "regulations": {
                        "cdi_framework": "California Department of Insurance 'Safer from Wildfires' (10 CCR § 2644.9)",
                        "structure_hardening": [
                            "Class-A fire-rated roof (metal, composite, or tile)",
                            "6 inches of noncombustible vertical clearance at base of exterior walls",
                            "Ember-resistant, corrosion-resistant mesh vents (1/16 to 1/8 inch)",
                            "Multi-pane or tempered glass windows",
                            "Enclosed eaves and fire-resistant soffits"
                        ],
                        "defensible_space_zones": [
                            "Zone 0 (0-5 ft): Ember-resistant zone, non-combustible gravel/pavers, no vegetation or mulch",
                            "Zone 1 (5-30 ft): Lean, clean, and green zone; trees spaced 18ft apart, dead wood cleared",
                            "Zone 2 (30-100 ft): Reduced fuel zone; grass mowed to max 4 inches, lower branches pruned"
                        ],
                        "community_protection": "Firewise USA designated community status",
                        "insurance_benefit": "Mandatory premium discounts required by California law from admitted carriers"
                    }
                }

            elif name == "get_active_hotspots":
                return {"success": True, "hotspots": self.gis.get_firms_hotspots_layer()}

            else:
                return {"success": False, "error": f"Unknown tool: {name}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
