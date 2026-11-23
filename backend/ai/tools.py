"""
Wildfire AI tool registry. Every tool calls the same real sources as the map (backend/live_sources.py);
results include their source and time so the model can cite them.
"""

from typing import Any, Dict, List, Optional

from backend import live_sources as live
from backend.services import REGION_BOUNDS, GISDataService, region_bbox

REGIONS = ["san_bruno", "san_jose", "santa_cruz"]


def _fn(name: str, description: str, properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required or []}}}


LATLON = {"lat": {"type": "number", "description": "Latitude"}, "lng": {"type": "number", "description": "Longitude"}}
REGION = {"region": {"type": "string", "enum": REGIONS, "description": "Map region (default: the active region)"}}


class WildfireToolRegistry:
    def __init__(self, gis_service: Optional[GISDataService] = None):
        self.gis = gis_service or GISDataService()
        self.default_region = "san_bruno"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            _fn("get_live_weather", "Current NOAA HRRR temperature, humidity, wind and gusts (via Open-Meteo) plus active "
                "National Weather Service alerts such as Red Flag Warnings, at a point or the region centre.",
                {**LATLON, **REGION}),
            _fn("get_live_fuel_moisture", "Latest hourly readings from CDEC RAWS stations near the region: 10-h fuel stick "
                "moisture (%), air temperature, humidity and wind.", REGION),
            _fn("query_location", "Everything the real data say about a point: HRRR weather and alerts, Bellwether 1-/5-year "
                "probability (pilot area only), USDA Wildfire Risk to Communities annual burn probability (national), CAL FIRE FHSZ zone, ESA WorldCover land cover, past fire perimeters, "
                "Microsoft building footprints within the radius, slope and the nearest RAWS fuel moisture.",
                {**LATLON, "radius_feet": {"type": "number", "description": "Radius in feet (default 130)"}}, ["lat", "lng"]),
            _fn("get_model_evaluation", "Bellwether 1-year probability vs CAL FIRE FHSZ, computed on the 100 m grid of the "
                "region from the real layers (crosstab and largest disagreement areas).", REGION),
            _fn("get_active_fires", "Active fires now: CAL FIRE incidents, NIFC current perimeters and NASA FIRMS VIIRS "
                "detections from the last 24 hours, within the region or statewide.",
                {**REGION, "statewide": {"type": "boolean", "description": "Search all of California"}}),
            _fn("get_home_hardening_guidelines", "Reference list of California home-hardening and defensible-space measures "
                "(CDI 'Safer from Wildfires', CAL FIRE Zones 0-2).", {}),
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            region = arguments.get("region") or self.default_region
            if name == "get_live_weather":
                lat, lng = arguments.get("lat"), arguments.get("lng")
                if lat is None or lng is None:
                    lat, lng = REGION_BOUNDS.get(region, REGION_BOUNDS["san_bruno"])["center"]
                return {"success": True, "weather": self.gis.get_weather(float(lat), float(lng))}
            if name == "get_live_fuel_moisture":
                b = region_bbox(region)
                fc = live.fuel_moisture((b[0] - 0.3, b[1] - 0.3, b[2] + 0.3, b[3] + 0.3))
                return {"success": True, "stations": [f["properties"] for f in fc["features"]], "note": fc.get("note")}
            if name == "query_location":
                return {"success": True, "location": self.gis.query_point(
                    float(arguments["lat"]), float(arguments["lng"]), float(arguments.get("radius_feet", 130.0)))}
            if name == "get_model_evaluation":
                d = self.gis.get_model_evaluation_data(region)
                return {"success": True, "analysis": d["analysis"], "models": d["models"]}
            if name == "get_active_fires":
                b = live.CA_BBOX if arguments.get("statewide") else region_bbox(region)
                per = self.gis.get_perimeters(b)
                hs = live.firms_hotspots(b)
                return {"success": True,
                        "calfire_incidents": [f["properties"] for f in per["incidents"]["features"]],
                        "current_perimeters": [f["properties"] for f in per["current"]["features"]],
                        "firms_detections_24h": len(hs["features"]),
                        "firms_sample": [f["properties"] | {"lon": f["geometry"]["coordinates"][0],
                                                            "lat": f["geometry"]["coordinates"][1]}
                                         for f in hs["features"][:20]]}
            if name == "get_home_hardening_guidelines":
                return {"success": True, "regulations": {
                    "cdi_framework": "California Department of Insurance 'Safer from Wildfires' (10 CCR 2644.9): admitted "
                                     "insurers must recognise listed mitigation measures in pricing.",
                    "structure_hardening": ["Class A fire-rated roof", "6 inches of noncombustible vertical clearance at the "
                                            "base of exterior walls", "Ember- and flame-resistant vents",
                                            "Multi-pane or tempered windows", "Enclosed eaves"],
                    "defensible_space_zones": ["Zone 0 (0-5 ft): ember-resistant zone", "Zone 1 (5-30 ft): lean, clean and green",
                                               "Zone 2 (30-100 ft): reduced fuel"],
                    "community": "Firewise USA recognition is one of the CDI-listed community measures",
                    "note": "Reference guidance; confirm current requirements with CAL FIRE and CDI."}}
            return {"success": False, "error": f"Unknown tool: {name}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
