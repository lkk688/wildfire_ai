"""
FastAPI Backend Application for Wildfire Risk Mitigation Dashboard
Serves Bellwether GeoTIFF layers, Sentinel-2 STAC layers, CAL FIRE FHSZ insurance maps,
CAL FIRE Active Incidents & Perimeters, NASA FIRMS hotspots, CDEC Fuel Moisture,
USGS 3DEP Terrain Slope, NOAA HRRR live weather, Microsoft Building Footprints,
San Mateo & San Jose multi-region support, and 130ft radiant heat corridor queries.
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.services import GISDataService, STATIC_DIR

STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Wildfire Risk Mitigation Pilot API",
    description="San Bruno Fire Department x SJSU WIRC x Google X Bellwether Project API",
    version="1.4.0",
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


class QueryCorridorRequest(BaseModel):
    lat: float
    lng: float
    radius_feet: float = 130.0


@app.get("/")
def root():
    return {
        "status": "online",
        "project": "San Bruno Wildfire Risk Mitigation Pilot",
        "partners": ["San Bruno Fire Department", "SJSU WIRC", "Google X Bellwether Project"],
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
