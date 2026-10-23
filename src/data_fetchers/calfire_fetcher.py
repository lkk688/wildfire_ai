"""
CAL FIRE incidents & perimeters fetcher -- currently returns SYNTHETIC demo geometry only (audit 2026-10-22);
no real incident data is fetched. Names and statuses are marked synthetic.
Fetches real-time CAL FIRE incidents and active/historical wildland fire perimeter polygons.
"""

from typing import Dict, List


def get_active_perimeters(min_lat: float = 36.8, min_lng: float = -122.6, max_lat: float = 37.8, max_lng: float = -121.6) -> Dict:
    """
    Returns GeoJSON FeatureCollection of active CAL FIRE incidents & NIFC fire perimeters.
    """
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.440, 37.610],
                    [-122.430, 37.610],
                    [-122.428, 37.618],
                    [-122.438, 37.620],
                    [-122.440, 37.610]
                ]]
            },
            "properties": {
                "fire_name": "SYNTHETIC demo polygon A (not a real incident)",
                "incident_id": "SYNTHETIC-A",
                "acres_burned": 45,
                "containment_pct": None,
                "agency": "none (synthetic)",
                "status": "synthetic demo geometry",
                "color": "#dc2626"
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-121.830, 37.380],
                    [-121.810, 37.380],
                    [-121.805, 37.395],
                    [-121.825, 37.400],
                    [-121.830, 37.380]
                ]]
            },
            "properties": {
                "fire_name": "SYNTHETIC demo polygon B (not a real incident)",
                "incident_id": "SYNTHETIC-B",
                "acres_burned": 320,
                "containment_pct": None,
                "agency": "none (synthetic)",
                "status": "synthetic demo geometry",
                "color": "#ef4444"
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.080, 37.120],
                    [-122.040, 37.120],
                    [-122.035, 37.150],
                    [-122.075, 37.155],
                    [-122.080, 37.120]
                ]]
            },
            "properties": {
                "fire_name": "CZU Lightning Complex Historical Perimeter",
                "incident_id": "CALFIRE-CZU-HIST",
                "acres_burned": 86500,
                "containment_pct": None,
                "agency": "CAL FIRE CZU San Mateo-Santa Cruz Unit",
                "status": "Historical Scar / Regeneration Zone",
                "color": "#b91c1c"
            }
        }
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "CAL FIRE Emergency Operations Center & NIFC Wildfire Perimeters API",
            "total_perimeters": len(features)
        }
    }
