"""
FastAPI Backend Application for Wildfire Risk Mitigation Dashboard
Serves Bellwether GeoTIFF layers, Sentinel-2 STAC layers, CAL FIRE FHSZ insurance maps,
CAL FIRE Active Incidents & Perimeters, NASA FIRMS hotspots, CDEC Fuel Moisture,
USGS 3DEP Terrain Slope, NOAA HRRR live weather, Microsoft Building Footprints,
San Mateo & San Jose multi-region support, and 130ft radiant heat corridor queries.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.services import GISDataService, STATIC_DIR
from backend.ai import WildfireAIAgent, AIConfig
from src.academic_models import (
    RothermelLevelSetSimulator,
    CellularAutomataSimulator,
    AcademicBenchmarkRunner,
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Wildfire Risk Intelligence & Mitigation Platform API",
    description="AI-Powered Wildfire Risk Analytics, Decision Support, and Home Hardening Platform",
    version="1.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

gis_service = GISDataService()
ai_agent = WildfireAIAgent(gis_service=gis_service)


class QueryCorridorRequest(BaseModel):
    lat: float
    lng: float
    radius_feet: float = 130.0


class AIChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    persona: str = "resident"  # "firefighter" or "resident"
    context: Optional[Dict[str, Any]] = None


class SwitchProviderRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class AcademicSimulationRequest(BaseModel):
    model_type: str = "rothermel_level_set"  # "rothermel_level_set" or "cellular_automata"
    wind_speed_ms: float = 8.0
    wind_dir_deg: float = 45.0
    fuel_moisture: float = 0.08
    duration_minutes: float = 60.0
    grid_size: int = 80
    spotting: bool = True



@app.get("/")
def root():
    return {
        "status": "online",
        "platform": "Wildfire Risk Intelligence & Mitigation Platform",
        "capabilities": [
            "Predictive AI Hazard Mapping (1-Year & 5-Year)",
            "Multi-Source GIS Overlays (NASA FIRMS, CAL FIRE, RAWS, USGS 3DEP)",
            "130ft Radiant Heat Contagion Scoping & Negative Value Quantification",
            "Home Hardening & Insurance Discount Compliance (CDI / IBHS)",
            "Intelligent Wildfire AI Copilot with OpenAI-Compatible Multi-Model Reasoning & Real-Time GIS Tool Calling"
        ],
    }



@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "ref_data_available": gis_service.ref_data_dir.exists(),
    }


@app.get("/api/layers/bellwether")
def get_bellwether_layer(
    is_5_year: bool = Query(False, description="Set True for 5-year model, False for 1-year model"),
    region: str = Query("san_bruno", description="Target region (san_bruno, san_jose, santa_cruz)"),
    min_lat: Optional[float] = Query(None),
    min_lng: Optional[float] = Query(None),
    max_lat: Optional[float] = Query(None),
    max_lng: Optional[float] = Query(None),
):
    """Get Bellwether wildfire probability map overlay dynamically cropped for region or lat/lng bbox."""
    try:
        data = gis_service.get_bellwether_overlay(
            is_5_year=is_5_year,
            region=region,
            min_lat=min_lat,
            min_lng=min_lng,
            max_lat=max_lat,
            max_lng=max_lng
        )
        return {"success": True, "layer": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/buildings")
def get_building_footprints():
    """Get GeoJSON feature collection of Microsoft Building Footprints, OSM & FEMA Structures."""
    try:
        data = gis_service.get_building_footprints_geojson()
        return {"success": True, "buildings": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/calfire-perimeters")
def get_calfire_perimeters():
    """Get CAL FIRE active incidents & NIFC fire perimeters GeoJSON layer."""
    try:
        data = gis_service.get_calfire_perimeters_layer()
        return {"success": True, "perimeters": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/firms-hotspots")
def get_firms_hotspots():
    """Get NASA FIRMS active thermal hotspots GeoJSON layer."""
    try:
        data = gis_service.get_firms_hotspots_layer()
        return {"success": True, "hotspots": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/fuel-moisture")
def get_fuel_moisture_stations():
    """Get CDEC / RAWS station fuel moisture (DFM/LFMC) GeoJSON layer."""
    try:
        data = gis_service.get_fuel_moisture_layer()
        return {"success": True, "stations": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/terrain-slope")
def get_terrain_slope():
    """Get USGS 3DEP LiDAR terrain slope overlay."""
    try:
        data = gis_service.get_terrain_slope_layer()
        return {"success": True, "layer": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bellwether-regions")
def get_bellwether_regions():
    """Get Bellwether region availability guide for San Jose, Santa Clara, and CONUS."""
    try:
        data = gis_service.get_bellwether_regions_guide()
        return {"success": True, "guide": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/calfire")
def get_calfire_layer():
    """Get CAL FIRE FHSZ (Fire Hazard Severity Zones) overlay metadata."""
    try:
        data = gis_service.get_calfire_fhsz_overlay()
        return {"success": True, "layer": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/live")
def get_live_weather():
    """Get live NOAA HRRR meteorological and wind vector parameters."""
    try:
        data = gis_service.get_live_weather()
        return {"success": True, "weather": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-catalog")
def get_data_catalog():
    """Get comprehensive documentation catalog for all multi-source datasets."""
    try:
        catalog = gis_service.get_data_catalog()
        return {"success": True, "catalog": catalog}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk-factors")
def get_risk_factors():
    """Get top 10 aggregated risk factor weights from Bellwether COG."""
    try:
        factors = gis_service.get_risk_factors()
        return {"success": True, "factors": factors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-evaluation")
def get_model_evaluation(
    region: str = Query("san_bruno", description="Target region (san_bruno, san_jose, santa_cruz)")
):
    """
    Get comprehensive Off-the-Shelf Model Evaluation & Gap Analysis data
    comparing Google X Bellwether ML, CAL FIRE FHSZ, and Insurance Cat-Models.
    """
    try:
        data = gis_service.get_model_evaluation_data(region=region)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/chat")
def chat_with_ai(req: AIChatRequest):
    """
    Conversational AI Copilot powered by MiniMax API.
    Supports dual personas ('firefighter' vs 'resident') and real-time GIS tool execution.
    """
    try:
        res = ai_agent.chat(
            messages=req.messages,
            persona=req.persona,
            context=req.context
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/suggested-prompts")
def get_suggested_prompts(persona: str = Query("resident", description="'resident' or 'firefighter'")):
    """Get high-impact recommended prompt pills for the AI Copilot."""
    try:
        prompts = ai_agent.get_suggested_prompts(persona=persona)
        return {"success": True, "prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/provider-status")
def get_ai_provider_status():
    """
    Get active LLM provider status, model selection, and available presets
    (MiniMax, Local Ollama, OpenAI, DeepSeek, vLLM).
    """
    try:
        status = AIConfig.get_provider_status()
        return {"success": True, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/switch-provider")
def switch_ai_provider(req: SwitchProviderRequest):
    """
    Dynamically hot-swap the active LLM provider and model configuration
    (e.g., minimax, ollama, openai, deepseek, vllm, or custom endpoint).
    """
    try:
        updated_status = AIConfig.update_provider_config(
            provider=req.provider,
            model=req.model,
            base_url=req.base_url,
            api_key=req.api_key,
        )
        return {"success": True, "status": updated_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






@app.get("/api/academic/models")
def get_academic_models():
    """
    Returns available academic frontier wildfire models, theoretical paradigms,
    and computational profiles.
    """
    return {
        "success": True,
        "models": [
            {
                "id": "rothermel_level_set",
                "name": "Rothermel Level-Set PDE Solver",
                "paradigm": "Physics-Based Continuous PDE",
                "governing_equations": "∂ϕ/∂t + R(x, ∇ϕ) ‖∇ϕ‖ = 0 (Godunov Upwind Scheme)",
                "strengths": "Strict energy conservation, smooth perimeter tracking, exact normal vectors",
                "limitations": "Computational scaling on ultra-fine grids; requires coupled ember spotting operator",
                "typical_latency_ms": 45.0,
                "open_source": True,
                "reference": "Rothermel (1972) / Osher & Sethian (1988) Level-Set Methods",
            },
            {
                "id": "cellular_automata",
                "name": "Alexandridis Stochastic Cellular Automata",
                "paradigm": "Discrete Spatial Automata (8-Neighbor)",
                "governing_equations": "p_burn = p0 * (1 + p_veg) * pw(V, θ) * ps(Δz)",
                "strengths": "Ultra-fast execution, native stochastic firebrand ember jump modeling",
                "limitations": "Grid-orientation anisotropy; lacks continuous thermodynamic coupling",
                "typical_latency_ms": 95.0,
                "open_source": True,
                "reference": "Alexandridis et al. (2008) Applied Mathematical Modelling",
            },
            {
                "id": "google_ndws",
                "name": "Google Research NDWS Surrogate",
                "paradigm": "Deep Learning Spatial-Temporal ConvNet",
                "governing_equations": "12-channel multimodal satellite/weather to 24-hr fire mask mapping",
                "strengths": "Millisecond inference, scales to regional multi-tile predictions",
                "limitations": "Black-box non-physical spread leakage; OOD failure on extreme wind gusts",
                "typical_latency_ms": 15.0,
                "open_source": True,
                "reference": "Huot et al. (Google Research, NeurIPS 2021 Datasets & Benchmarks)",
            },
            {
                "id": "wsts_transformer",
                "name": "WildfireSpreadTS Spatio-Temporal Transformer",
                "paradigm": "Attention-Based Time Series Forecaster",
                "governing_equations": "Multi-head spatial self-attention on multi-day VIIRS thermal footprints",
                "strengths": "Captures multi-day sequence dynamics and seasonal regime shifts",
                "limitations": "High GPU VRAM footprint; requires dense sequential satellite imagery",
                "typical_latency_ms": 35.0,
                "open_source": True,
                "reference": "Gerhard et al. (NeurIPS 2023 / 2024)",
            },
        ],
    }


@app.get("/api/academic/benchmark-results")
def get_academic_benchmark_results():
    """
    Executes and returns the standardized academic benchmark evaluation comparing
    Level-Set PDE, Stochastic CA, Google NDWS, and Commercial Cat-Models.
    """
    try:
        results = AcademicBenchmarkRunner.run_comprehensive_benchmark()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/academic/simulate")
def run_academic_simulation(req: AcademicSimulationRequest):
    """
    Run an interactive on-demand physical or discrete wildfire propagation simulation.
    """
    try:
        grid_dim = min(max(req.grid_size, 40), 120)
        center = (grid_dim // 2, grid_dim // 2)

        if req.model_type == "cellular_automata":
            ca = CellularAutomataSimulator(grid_shape=(grid_dim, grid_dim), cell_size_m=10.0)
            steps = min(max(int(req.duration_minutes), 10), 100)
            res = ca.simulate(
                ignition_coords=[center],
                total_steps=steps,
                wind_speed_ms=req.wind_speed_ms,
                wind_dir_deg=req.wind_dir_deg,
                fuel_moisture=req.fuel_moisture,
                spotting_enabled=req.spotting,
            )
            return {"success": True, "simulation": res}
        else:
            rls = RothermelLevelSetSimulator(grid_shape=(grid_dim, grid_dim), dx=10.0, dy=10.0, fuel_model_id=4)
            res = rls.solve_level_set(
                ignition_coords=[center],
                total_minutes=min(max(req.duration_minutes, 10.0), 120.0),
                dt_minutes=0.25,
                wind_speed_ms=req.wind_speed_ms,
                wind_dir_deg=req.wind_dir_deg,
                fuel_moisture=req.fuel_moisture,
            )
            return {"success": True, "simulation": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query-corridor")
def query_corridor(req: QueryCorridorRequest):
    """Query wind-adjusted 130ft radiant heat corridor, cropland classification, and parcel ROI metrics."""
    try:
        result = gis_service.query_corridor(lat=req.lat, lng=req.lng, radius_feet=req.radius_feet)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
