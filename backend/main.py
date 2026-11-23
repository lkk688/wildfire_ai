"""
FastAPI Backend Application for Wildfire Risk Mitigation Dashboard
Map layers come from real sources only: Bellwether GeoTIFFs (partner data), NIFC / CAL FIRE perimeters and incidents,
NASA FIRMS, CDEC RAWS, NOAA HRRR (Open-Meteo), NWS alerts, USGS 3DEP, Sentinel-2, CAL FIRE FHSZ, ESA WorldCover and
Microsoft building footprints. See backend/live_sources.py.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import live_sources as live
from backend import heldout
from backend.services import GISDataService, STATIC_DIR, REGION_BOUNDS, parse_bbox
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
            "Bellwether 1-year / 5-year hazard overlays (partner GeoTIFFs, pilot area)",
            "Live layers: NIFC/CAL FIRE perimeters and incidents, NASA FIRMS, CDEC RAWS fuel moisture, NOAA HRRR wind, "
            "NWS alerts, USGS 3DEP slope, Sentinel-2 NDVI, CAL FIRE FHSZ, USDA Wildfire Risk to Communities burn probability, Microsoft building footprints",
            "Point query combining the layers above",
            "AI copilot whose tools call the same live sources"
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


def _bbox(bbox: Optional[str], region: str):
    try:
        return parse_bbox(bbox, region)
    except Exception:
        raise HTTPException(status_code=400, detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'")


def _live(fn):
    """Run a live-source call; upstream failures become 502 with the source's error, never fabricated data."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream data source failed: {type(e).__name__}: {str(e)[:300]}")


@app.get("/api/layers/buildings")
def get_building_footprints(bbox: Optional[str] = Query(None), region: str = Query("san_bruno"),
                            limit: int = Query(4000, le=8000)):
    """Microsoft US Building Footprints inside a bbox (min_lon,min_lat,max_lon,max_lat)."""
    return {"success": True, "buildings": _live(lambda: live.buildings(_bbox(bbox, region), limit=limit))}


@app.get("/api/layers/perimeters")
def get_fire_perimeters(bbox: Optional[str] = Query(None), region: str = Query("san_bruno")):
    """NIFC WFIGS current perimeters, CAL FIRE historical perimeters (1950+) and active CAL FIRE incidents."""
    return {"success": True, **_live(lambda: gis_service.get_perimeters(_bbox(bbox, region)))}


@app.get("/api/layers/firms-hotspots")
def get_firms_hotspots(bbox: Optional[str] = Query(None)):
    """NASA FIRMS VIIRS active-fire detections from the last 24 hours (default: California)."""
    bb = _bbox(bbox, "san_bruno") if bbox else live.CA_BBOX
    return {"success": True, "hotspots": _live(lambda: live.firms_hotspots(bb))}


@app.get("/api/layers/fuel-moisture")
def get_fuel_moisture_stations(bbox: Optional[str] = Query(None), region: str = Query("san_bruno")):
    """CDEC RAWS stations with 10-h fuel stick moisture, temperature, humidity and wind (latest hourly values)."""
    bb = _bbox(bbox, region)
    pad = 0.3  # include nearby stations around small views
    return {"success": True, "stations": _live(lambda: live.fuel_moisture((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)))}


@app.get("/api/layers/terrain-slope")
def get_terrain_slope(bbox: Optional[str] = Query(None), region: str = Query("san_bruno")):
    """Slope (%) computed from the USGS 3DEP DEM, returned as a PNG overlay."""
    return {"success": True, "layer": _live(lambda: live.slope_overlay(_bbox(bbox, region)))}


@app.get("/api/layers/wrc-burn-probability")
def get_wrc_burn_probability(bbox: Optional[str] = Query(None), region: str = Query("san_bruno")):
    """USDA Wildfire Risk to Communities annual burn probability (2024 release), classed PNG overlay."""
    return {"success": True, "layer": _live(lambda: live.wrc_bp_overlay(_bbox(bbox, region)))}


@app.get("/api/layers/ndvi")
def get_ndvi(bbox: Optional[str] = Query(None), region: str = Query("san_bruno")):
    """NDVI from the latest Sentinel-2 scene with <10% cloud over the bbox (acquisition date included)."""
    return {"success": True, "layer": _live(lambda: live.ndvi_overlay(_bbox(bbox, region)))}


@app.get("/api/layers/fhsz")
def get_fhsz(bbox: Optional[str] = Query(None), region: str = Query("san_bruno")):
    """CAL FIRE Fire Hazard Severity Zones (SRA 2023, LRA 2025) as GeoJSON."""
    return {"success": True, "zones": _live(lambda: live.fhsz(_bbox(bbox, region)))}


@app.get("/api/bellwether-regions")
def get_bellwether_regions():
    """Regions offered in the UI and the actual coverage of the Bellwether rasters on this server."""
    cov = gis_service.bellwether_coverage()
    return {"success": True, "guide": {"regions": REGION_BOUNDS,
            "bellwether_coverage": None if cov is None else {"min_lon": cov[0], "min_lat": cov[1], "max_lon": cov[2], "max_lat": cov[3]}}}


@app.get("/api/weather/live")
def get_live_weather(lat: Optional[float] = Query(None), lon: Optional[float] = Query(None), region: str = Query("san_bruno")):
    """Current HRRR conditions (via Open-Meteo) and active NWS alerts at a point (default: region centre)."""
    if lat is None or lon is None:
        lat, lon = REGION_BOUNDS.get(region, REGION_BOUNDS["san_bruno"])["center"]
    return {"success": True, "weather": _live(lambda: gis_service.get_weather(lat, lon))}


@app.get("/api/weather/wind-grid")
def get_wind_grid(bbox: Optional[str] = Query(None), region: str = Query("san_bruno"), n: int = Query(6, ge=2, le=10)):
    """HRRR 10 m wind sampled on an n x n grid over the bbox."""
    return {"success": True, "wind": _live(lambda: live.wind_grid(_bbox(bbox, region), n))}


@app.get("/api/data-catalog")
def get_data_catalog():
    """Description of every data source used by the map."""
    return {"success": True, "catalog": gis_service.get_data_catalog()}


@app.get("/api/data-sources/status")
def get_data_source_status(region: str = Query("san_bruno")):
    """Fetch each source once and report whether it is reachable right now."""
    return {"success": True, "sources": gis_service.check_sources(region)}


@app.get("/api/risk-factors")
def get_risk_factors(region: Optional[str] = Query(None), is_5_year: bool = Query(False)):
    """Bellwether risk factors aggregated over a region: contributing and mitigating features (partner data)."""
    try:
        return {"success": True, "factors": gis_service.get_risk_factors(region, is_5_year)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-evaluation")
def get_model_evaluation(region: str = Query("san_bruno")):
    """Bellwether vs CAL FIRE FHSZ computed on the 100 m grid from the real layers, plus model descriptions."""
    return {"success": True, "data": _live(lambda: gis_service.get_model_evaluation_data(region=region))}


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
    # Security (audit 2026-10-22): this endpoint is unauthenticated and CORS is open, so a network caller could
    # redirect the server's API key to a host of their choice. It is disabled unless explicitly enabled for local
    # use, and it never accepts a base_url or api_key from the request (only preset providers and model names).
    if os.environ.get("WILDFIRE_DEMO_ALLOW_PROVIDER_SWITCH") != "1":
        raise HTTPException(status_code=403, detail="Provider switching is disabled "
                            "(set WILDFIRE_DEMO_ALLOW_PROVIDER_SWITCH=1 for local use).")
    if req.base_url or req.api_key:
        raise HTTPException(status_code=400, detail="base_url and api_key cannot be set over the API; "
                            "configure them via environment variables.")
    if req.provider.lower().strip() not in AIConfig.PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown provider preset: {req.provider}")
    try:
        updated_status = AIConfig.update_provider_config(
            provider=req.provider,
            model=req.model,
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
                "strengths": "Smooth perimeter tracking (simplified surrogate rate of spread, not full Rothermel)",
                "limitations": "Computational scaling on ultra-fine grids; requires coupled ember spotting operator",
                "typical_latency_ms": None,
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
                "typical_latency_ms": None,
                "open_source": True,
                "reference": "Alexandridis et al. (2008) Applied Mathematical Modelling",
            },
            {
                "id": "google_ndws",
                "name": "Google Research NDWS Surrogate",
                "paradigm": "Deep Learning Spatial-Temporal ConvNet",
                "governing_equations": "1 km input features (weather, terrain, vegetation, drought, population, previous fire mask) to next-day fire mask",
                "strengths": "Public dataset and baselines for next-day spread",
                "limitations": "1 km resolution; not run in this app",
                "typical_latency_ms": None,
                "open_source": True,
                "reference": "Huot et al., Next Day Wildfire Spread, IEEE TGRS 2022 (arXiv 2112.02447)",
            },
            {
                "id": "wsts_transformer",
                "name": "WildfireSpreadTS",
                "paradigm": "Multi-temporal spread-prediction dataset (U-Net, ConvLSTM, UTAE baselines)",
                "governing_equations": "23-channel daily inputs to next-day active-fire mask",
                "strengths": "Multi-day sequences across 607 US fire events",
                "limitations": "Daily resolution; not run in this app",
                "typical_latency_ms": None,
                "open_source": True,
                "reference": "Gerard, Zhao & Sullivan, NeurIPS 2023 Datasets & Benchmarks",
            },
        ],
    }


@app.get("/api/academic/benchmark-results")
def get_academic_benchmark_results():
    """
    Executes and returns the standardized academic benchmark evaluation comparing
    Level-Set PDE, Stochastic CA, Google NDWS, and Commercial Cat-Models.
    """
    # Audit 2026-10-22: Removed: the commercial and NDWS rows were hard-coded constants and a hand-drawn ellipse; the rest scored a solver against its own output on synthetic terrain.
    raise HTTPException(status_code=410, detail="Removed: the commercial and NDWS rows were hard-coded constants and a hand-drawn ellipse; the rest scored a solver against its own output on synthetic terrain.")


@app.get("/api/academic/vlm-coupling-benchmark")
def get_vlm_physics_coupling_benchmark():
    """
    Executes and returns the empirical benchmark comparing:
    1. Default Physics Baseline (Unassimilated)
    2. Vanilla Frontier VLM (Zero-Shot Direct Output)
    3. Pure Data-Driven AI (Bellwether / NDWS Proxy)
    4. Proposed VLM-Assimilated Physics (Rothermel PDE + CA)
    across the 50 CAL FIRE DINS ground-truth parcels.
    """
    # Audit 2026-10-22: Removed: the coupled rows used ground-truth defensive actions and label-generated Bellwether/Zone 0 inputs.
    raise HTTPException(status_code=410, detail="Removed: the coupled rows used ground-truth defensive actions and label-generated Bellwether/Zone 0 inputs.")


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


from starlette.responses import FileResponse
from src.vlm_benchmark.service import VLMBenchmarkService

vlm_service = VLMBenchmarkService()
LEGACY_DEVSET_RUNS = 'Removed 2026-11-22: the stored 50-case dev-set runs used a prompt that stated the same made-up weather for every case (Red Flag Warning, gusts 25-35 mph, DFM < 6%), post-fire DINS attributes and the fire name. Use /api/heldout/* (held-out fires, research prompt, verified imagery).'


class VLMPredictRequest(BaseModel):
    model: str = "MiniMax-M3"
    case_id: Optional[str] = None
    custom_attributes: Optional[Dict[str, Any]] = None


class EvaluateRewardRequest(BaseModel):
    candidate_text: str
    case_id: Optional[str] = None
    custom_attributes: Optional[Dict[str, Any]] = None


@app.post("/api/query-corridor")
def query_corridor(req: QueryCorridorRequest):
    """Real data about a clicked point: HRRR weather and alerts, Bellwether, USDA burn probability, FHSZ, land cover, past fires,
    buildings within the radius, slope and the nearest RAWS fuel moisture."""
    try:
        result = gis_service.query_point(lat=req.lat, lng=req.lng, radius_feet=req.radius_feet)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WildfireReason-VLM Frontier Benchmark & Physical Reasoning Endpoints
# ============================================================================

@app.get("/api/vlm/benchmark/cases")
def get_vlm_benchmark_cases():
    """The 50 dev-set cases: DINS attributes, verified imagery epochs and ERA5 fire weather."""
    try:
        return vlm_service.get_catalog()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vlm/benchmark/results")
def get_vlm_benchmark_results():
    """Retired (see detail)."""
    raise HTTPException(status_code=410, detail=LEGACY_DEVSET_RUNS)


@app.get("/api/vlm/benchmark/metrics")
def get_vlm_benchmark_metrics():
    """Retired (see detail)."""
    raise HTTPException(status_code=410, detail=LEGACY_DEVSET_RUNS)


@app.get("/api/vlm/benchmark/image/{case_id}")
def get_vlm_benchmark_image(
    case_id: str,
    epoch: str = Query("prefire", description="prefire | postfire | current (verified Wayback captures; see multi_epoch_images "
                                              "in /api/vlm/benchmark/cases for capture dates) | streetview")
):
    """Serve a case image. Aerial epochs are verified Esri Wayback crops; a missing epoch is a 404, never a substitute."""
    img_path = vlm_service.get_image_path(case_id, epoch=epoch)
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image for case {case_id} (epoch={epoch}) not found")
    return FileResponse(
        str(img_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )


@app.get("/api/vlm/research-results")
def get_vlm_research_results():
    """This project's held-out evaluation numbers (parsed from paper/numbers.tex)."""
    try:
        return gis_service.research_evidence()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vlm/predict")
def run_vlm_live_predict(req: VLMPredictRequest):
    """Retired (see detail); use POST /api/heldout/predict."""
    raise HTTPException(status_code=410, detail=LEGACY_DEVSET_RUNS)


# ============================================================================
# Held-out evaluation browser (VLM view)
# ============================================================================

class HeldoutPredictRequest(BaseModel):
    task: str  # t3 (post-fire) | t2 (pre-fire pairs)
    case_id: str  # global_id for t3, pair_id for t2
    model: str = "claude-sonnet-5-5"
    condition: str = "postfire_image"  # t3: postfire_image | prepost_image; t2: image | meta


@app.get("/api/heldout/summary")
def heldout_summary():
    """Models, per-model metrics computed from the stored per-case outputs, events and live-model availability."""
    try:
        return heldout.summary()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/heldout/cases")
def heldout_cases(task: str = Query("t3", description="t3 (post-fire) | t2 (pre-fire pairs)")):
    """All held-out cases of a task with stored per-model outputs (no addresses or APNs)."""
    try:
        return heldout.cases(task)
    except KeyError:
        raise HTTPException(status_code=400, detail="task must be t3 or t2")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/heldout/image/{global_id}")
def heldout_image(global_id: str, epoch: str = Query("pre", description="pre | post")):
    """Verified Esri Wayback crop of a held-out building (capture dates in /api/heldout/cases)."""
    p = heldout.image_path(global_id, epoch)
    if p is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(p), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/heldout/predict")
def heldout_predict(req: HeldoutPredictRequest):
    """Re-run one held-out case now: same research prompt and same verified images as the stored run."""
    try:
        return heldout.live_predict(req.task, req.case_id, req.model, req.condition)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")
    except (ValueError, SystemExit) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vlm/rlvr-reasoning-benchmark")
def get_rlvr_reasoning_benchmark():
    """Retrieve RLVR / GRPO post-training ablation benchmark across 50 DINS cases."""
    # Audit 2026-10-22: Removed: every prediction in this benchmark was generated from the true outcome label (simulation, not results).
    raise HTTPException(status_code=410, detail="Removed: every prediction in this benchmark was generated from the true outcome label (simulation, not results).")


@app.post("/api/vlm/evaluate-reasoning-reward")
def evaluate_reasoning_reward(req: EvaluateRewardRequest):
    """Evaluate candidate reasoning trace using verifiable physical reward engine."""
    try:
        from src.vlm_benchmark.verifiable_reward_engine import VerifiablePhysicalRewardEngine
        engine = VerifiablePhysicalRewardEngine()
        gt = {}
        attrs = req.custom_attributes or {}
        if req.case_id:
            catalog = vlm_service.get_catalog()
            matching = [it for it in catalog.get("items", []) if str(it["id"]) == str(req.case_id)]
            if matching:
                gt = matching[0]
                attrs = dict(matching[0])
                attrs.update(req.custom_attributes or {})
        return engine.compute_reward(req.candidate_text, gt, attrs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

