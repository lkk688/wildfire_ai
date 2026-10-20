"""
Time-verified pre-fire imagery with per-location provenance (Review 1019, section 2).

Two facts about Esri World Imagery Wayback make naive "pre-fire" crops unsafe:
  1. A Wayback *release date* is not the *capture date* of the pixels at a given
     location; a 2020 release can serve a 2016 image.  The capture date must be
     read from the release's metadata layer (SRC_DATE) for each location.
  2. If a release lacks a tile, silently falling back to the current World
     Imagery service returns *post-fire* pixels.  This module never falls back:
     a missing tile is a rejected sample.

For each structure we
  * pick the latest release published strictly before the fire start date,
  * stitch a centred crop from that release only,
  * query that release's metadata layer at the structure point for the capture
    date, source sensor and resolution,
  * accept the crop only if capture_date < fire_start (and optionally not older
    than `max_staleness_days` / not older than the building's YEARBUILT),
  * return a provenance record (release, capture date, staleness, sha256 of the
    saved JPEG) that is stored next to every sample.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

WAYBACK_CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
    "default028mm/MapServer/tile/{release}/{z}/{y}/{x}"
)
UA = {"User-Agent": "WildfireAI-Provenance/1.0"}
DEFAULT_CACHE = Path("/data/wildfire_benchmark_data/wayback/waybackconfig.json")


@dataclass
class Release:
    release_id: int
    release_date: dt.date
    identifier: str
    metadata_url: str


@dataclass
class ImageryProvenance:
    accepted: bool
    reject_reason: Optional[str]
    fire_start: str
    release_id: Optional[int] = None
    release_identifier: Optional[str] = None
    release_date: Optional[str] = None
    capture_date: Optional[str] = None
    capture_source: Optional[str] = None
    capture_provider: Optional[str] = None
    capture_resolution_m: Optional[float] = None
    staleness_days: Optional[int] = None
    zoom: int = 18
    size_px: int = 512
    image_path: Optional[str] = None
    image_sha256: Optional[str] = None
    tile_sources: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _http_json(url: str, timeout: int = 30, retries: int = 3) -> Dict:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def _http_bytes(url: str, timeout: int = 20, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def load_releases(cache_path: Path = DEFAULT_CACHE, refresh: bool = False) -> List[Release]:
    """Load the Wayback release catalogue (cached on disk), sorted by release date."""
    if cache_path.exists() and not refresh:
        cfg = json.loads(cache_path.read_text())
    else:
        cfg = _http_json(WAYBACK_CONFIG_URL)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cfg))
    out = []
    for rid, v in cfg.items():
        m = re.search(r"(\d{4}-\d{2}-\d{2})", v.get("itemTitle", ""))
        if not m:
            continue
        out.append(Release(int(rid), dt.date.fromisoformat(m.group(1)),
                           v.get("layerIdentifier", ""), v["metadataLayerUrl"]))
    return sorted(out, key=lambda r: r.release_date)


def latest_release_before(releases: List[Release], fire_start: dt.date, margin_days: int = 1) -> Optional[Release]:
    cutoff = fire_start - dt.timedelta(days=margin_days)
    cands = [r for r in releases if r.release_date <= cutoff]
    return cands[-1] if cands else None


def latlon_to_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n * 256.0
    return x, y


def query_capture_metadata(release: Release, lat: float, lon: float, zoom: int = 18) -> Optional[Dict]:
    """Capture date / sensor / resolution of the imagery this release serves at (lat, lon, zoom)."""
    d = 0.0005
    params = {
        "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "sr": "4326",
        "layers": "all", "tolerance": "0", "returnGeometry": "false", "f": "json",
        "mapExtent": f"{lon - d},{lat - d},{lon + d},{lat + d}", "imageDisplay": "512,512,96",
    }
    res = _http_json(release.metadata_url + "/identify?" + urllib.parse.urlencode(params))
    hits = []
    for r in res.get("results", []):
        a = r.get("attributes", {})
        try:
            lo, hi = int(a.get("MinMapLevel", 0)), int(a.get("MaxMapLevel", 99))
        except ValueError:
            continue
        if lo <= zoom <= hi and a.get("SRC_DATE"):
            hits.append(a)
    if not hits:
        return None
    # Highest draw order is what is rendered on top at this zoom.
    a = max(hits, key=lambda x: int(x.get("DrawOrder") or 0))
    try:
        cap = dt.datetime.strptime(str(a["SRC_DATE"]), "%Y%m%d").date()
    except ValueError:
        return None
    res_m = a.get("SRC_RES")
    return {
        "capture_date": cap,
        "source": a.get("SRC_DESC"),
        "provider": a.get("NICE_DESC"),
        "resolution_m": float(res_m) if res_m not in (None, "", "Null") else None,
    }


def _fetch_tile(release_id: int, z: int, x: int, y: int) -> Image.Image:
    raw = _http_bytes(TILE_URL.format(release=release_id, z=z, x=x, y=y))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _tile_source_release(release_id: int, z: int, x: int, y: int) -> Optional[int]:
    """Wayback tilemap: which release actually stores this tile (None if no tile)."""
    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/{release_id}/{z}/{y}/{x}"
    try:
        j = _http_json(url, timeout=15)
    except RuntimeError:
        return None
    if not j.get("valid") or not j.get("data") or j["data"][0] != 1:
        return None
    sel = j.get("select") or [release_id]
    return int(sel[0])


def fetch_prefire_crop(
    lat: float,
    lon: float,
    fire_start: dt.date,
    out_path: Path,
    releases: List[Release],
    zoom: int = 18,
    size: int = 512,
    max_staleness_days: Optional[int] = 5 * 365,
    year_built: Optional[int] = None,
) -> ImageryProvenance:
    prov = ImageryProvenance(accepted=False, reject_reason=None, fire_start=fire_start.isoformat(),
                             zoom=zoom, size_px=size)
    rel = latest_release_before(releases, fire_start)
    if rel is None:
        prov.reject_reason = "no_release_before_fire"
        return prov
    prov.release_id, prov.release_identifier = rel.release_id, rel.identifier
    prov.release_date = rel.release_date.isoformat()

    try:
        meta = query_capture_metadata(rel, lat, lon, zoom)
    except RuntimeError as e:
        prov.reject_reason = f"metadata_error:{e}"[:200]
        return prov
    if meta is None:
        prov.reject_reason = "no_capture_metadata"
        return prov
    cap = meta["capture_date"]
    prov.capture_date = cap.isoformat()
    prov.capture_source, prov.capture_provider = meta["source"], meta["provider"]
    prov.capture_resolution_m = meta["resolution_m"]
    prov.staleness_days = (fire_start - cap).days
    if cap >= fire_start:
        prov.reject_reason = "capture_not_before_fire"
        return prov
    if max_staleness_days is not None and prov.staleness_days > max_staleness_days:
        prov.reject_reason = "capture_too_stale"
        return prov
    if year_built and year_built > cap.year:
        prov.reject_reason = "built_after_capture"
        return prov

    px, py = latlon_to_pixel(lat, lon, zoom)
    h = size / 2.0
    tx0, tx1 = int((px - h) // 256), int((px + h) // 256)
    ty0, ty1 = int((py - h) // 256), int((py + h) // 256)
    coords = [(x, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1)]
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            sources = list(ex.map(lambda c: _tile_source_release(rel.release_id, zoom, *c), coords))
        if any(s is None for s in sources):
            prov.reject_reason = "missing_tile_in_release"
            return prov
        with ThreadPoolExecutor(max_workers=4) as ex:
            tiles = list(ex.map(lambda c: _fetch_tile(rel.release_id, zoom, *c), coords))
    except RuntimeError as e:
        prov.reject_reason = f"tile_error:{e}"[:200]
        return prov
    prov.tile_sources = sorted(set(int(s) for s in sources))

    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for (x, y), t in zip(coords, tiles):
        canvas.paste(t, ((x - tx0) * 256, (y - ty0) * 256))
    cx, cy = int(round(px - tx0 * 256 - h)), int(round(py - ty0 * 256 - h))
    crop = canvas.crop((cx, cy, cx + size, cy + size))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=92)
    data = buf.getvalue()
    out_path.write_bytes(data)
    prov.image_path = str(out_path)
    prov.image_sha256 = hashlib.sha256(data).hexdigest()
    prov.accepted = True
    return prov
