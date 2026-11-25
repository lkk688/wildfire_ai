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
            _fn("get_live_weather", "Current NOAA HRRR temperature, humidity, wind and gusts (NOAA open data; Open-Meteo outside California) plus active "
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
            _fn("list_data_layers", "List the California data-cache layers (id, title, units, source, date). Call this "
                "first to choose layer ids for query_data_layers or get_data_features.",
                {"group": {"type": "string", "description": "Optional group name filter, e.g. 'Fuel treatments'"}}),
            _fn("query_data_layers", "Values of the California data-cache layers at a point (no radius) or summaries "
                "within radius_m of it: fuel model, canopy, slope, burn probability, flame length, land cover, NDVI, "
                "fire-weather climatology, plus FHSZ zone, past fires, ignitions, fuel treatments and buildings nearby.",
                {**LATLON, "radius_m": {"type": "number", "description": "Radius in metres; omit for a single point"},
                 "layers": {"type": "array", "items": {"type": "string"},
                            "description": "Layer ids (default: all except imagery)"}}, ["lat", "lng"]),
            _fn("get_data_features", "Vector features near a point from the data cache: 'ignitions' (1992-present "
                "wildfire origins with cause and size), 'fire_history' perimeters, 'fhsz', 'treatments_calfire', "
                "'treatments_rx', 'treatments_calvtp', 'treatments_federal', 'buildings'.",
                {**LATLON, "layer": {"type": "string"}, "radius_m": {"type": "number", "description": "Default 2000"},
                 "limit": {"type": "integer", "description": "Default 50"}}, ["lat", "lng", "layer"]),
            _fn("get_fire_weather_history", "Fire-weather climatology 2000-2025 of the 4 km gridMET cell at a point: "
                "fire-weather days per month, wind direction on those days, and the 10 worst days with wind and "
                "humidity. Use it to pick wind scenarios for spread or to explain seasonal danger.", LATLON,
                ["lat", "lng"]),
            _fn("simulate_fire_spread", "Simulate one fire from an ignition point for a few hours on 30 m LANDFIRE fuels "
                "(Rothermel spread, crown fire, terrain, uniform wind; no spotting or suppression). Returns burned area, "
                "hourly perimeters, flame length, crown-fire share and buildings exposed. Weather not given defaults to "
                "the cell's fire-weather-day conditions (gridMET 2000-2025). A planning scenario, not a forecast.",
                {**LATLON, "hours": {"type": "number", "description": "Burning time, default 6"},
                 "wind_ms": {"type": "number", "description": "10-m wind speed in m/s"},
                 "wind_from_deg": {"type": "number", "description": "Direction the wind blows from, degrees"},
                 "rh_min": {"type": "number", "description": "Minimum relative humidity, %"},
                 "fm100": {"type": "number", "description": "100-h dead fuel moisture, %"}}, ["lat", "lng"]),
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
            if name == "list_data_layers":
                from backend import geodata
                g = arguments.get("group")
                keep = ("id", "title", "group", "units", "source", "date", "res", "available")
                return {"success": True, "layers": [{k: l.get(k) for k in keep} for l in geodata.catalog()
                                                    if not g or g.lower() in l["group"].lower()]}
            if name == "query_data_layers":
                from backend import geodata
                lat, lng = float(arguments["lat"]), float(arguments["lng"])
                ls = arguments.get("layers") or None
                if arguments.get("radius_m"):
                    return {"success": True, **geodata.area(lng, lat, radius_m=float(arguments["radius_m"]), layers=ls)}
                return {"success": True, **geodata.point(lng, lat, layers=ls)}
            if name == "get_data_features":
                from backend import geodata
                return {"success": True, **geodata.features(arguments["layer"], lon=float(arguments["lng"]),
                                                             lat=float(arguments["lat"]),
                                                             radius_m=float(arguments.get("radius_m") or 2000),
                                                             limit=int(arguments.get("limit") or 50))}
            if name == "get_fire_weather_history":
                from backend.geodata.query import fire_weather
                d = fire_weather(float(arguments["lng"]), float(arguments["lat"]))
                return {"success": d is not None, **(d or {"error": "no gridMET data at this point"})}
            if name == "simulate_fire_spread":
                from backend.firesim.api import simulate
                kw = {k: float(arguments[k]) for k in ("hours", "wind_ms", "wind_from_deg", "rh_min", "fm100")
                      if arguments.get(k) is not None}
                r = simulate(float(arguments["lat"]), float(arguments["lng"]), **kw)
                if r.get("success"):  # perimeters are for the map; the model gets their areas
                    r["isochrones"] = [f["properties"] for f in r["isochrones"]["features"]]
                return r
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
