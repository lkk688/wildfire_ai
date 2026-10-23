"""
(Pilot B) Sample distance-matched evacuation items and render the map a decision-maker would have had at time t.

Sampling (test fires only, communities within 5 km of the fire edge): up to MAX_POS communities per fire that the fire
later reached, each paired with a not-reached community from the SAME decision point with the closest distance to the
fire edge (so distance alone cannot separate the pair).

Map: pre-fire Esri World Imagery from the latest Wayback release before the fire started (release-level date only; at
these zooms per-pixel capture dates are not checked), the fire state at t (red), the community cell (yellow box),
a wind arrow (ERA5 at t) and a scale bar. No place names, dates or labels are drawn.

Output: output/evac/vlm_items.jsonl, output/evac/maps/<item_id>.jpg
"""
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vlm_benchmark.imagery_provenance import _fetch_tile, latest_release_before, load_releases  # noqa: E402

EVAC = ROOT / "output" / "evac"
MAX_POS, BAND_M, SIZE, G = 40, 5000, 768, 0.005


def sample():
    d = pd.DataFrame([json.loads(x) for x in (EVAC / "items.jsonl").read_text().splitlines()])
    d = d[d["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(1019)
    out = []
    for ev, g in d[d["dist_m"] <= BAND_M].groupby("event"):
        pos = g[g["reached"] == 1]
        if len(pos) > MAX_POS:
            pos = pos.iloc[rng.choice(len(pos), MAX_POS, replace=False)]
        used = set()
        for _, p in pos.iterrows():
            cand = d[(d["state_id"] == p["state_id"]) & (d["reached"] == 0)]
            cand = cand[~cand.index.isin(used)]
            if cand.empty:
                continue
            j = (cand["dist_m"] - p["dist_m"]).abs().idxmin()
            used.add(j)
            pid = f"{p['state_id']}_{p['cell']}"
            for role, r in (("reached", p), ("not_reached", d.loc[j])):
                rec = r.to_dict()
                rec.update({"pair_id": pid, "role": role, "item_id": f"{r['state_id']}_{r['cell']}"})
                out.append(rec)
    return out


def lonlat_to_px(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n * 256
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n * 256
    return x, y


def render(item, releases, fire_start):
    p = EVAC / "maps" / f"{item['item_id']}.jpg"
    meta = EVAC / "maps" / f"{item['item_id']}.json"
    if p.exists() and meta.exists():  # resumable
        return json.loads(meta.read_text())
    st = json.loads((EVAC / "fire_states" / f"{item['state_id']}.geojson").read_text())
    geom = shape(st["geometry"])
    # centre between the community and the nearest fire edge, zoom so both fit with margin
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    near = nearest_points(geom, Point(item["lon"], item["lat"]))[0]
    clon, clat = (near.x + item["lon"]) / 2, (near.y + item["lat"]) / 2
    span_m = max(item["dist_m"] * 2.2, 3000)
    z = 15
    while z > 10 and 156543.03 * math.cos(math.radians(clat)) / 2 ** z * SIZE < span_m:
        z -= 1
    rel = latest_release_before(releases, fire_start)
    cx, cy = lonlat_to_px(clon, clat, z)
    x0, y0 = cx - SIZE / 2, cy - SIZE / 2
    canvas = Image.new("RGB", (SIZE, SIZE))
    for tx in range(int(x0 // 256), int((x0 + SIZE) // 256) + 1):
        for ty in range(int(y0 // 256), int((y0 + SIZE) // 256) + 1):
            try:
                t = _fetch_tile(rel.release_id, z, tx, ty)
            except Exception:  # noqa: BLE001
                continue
            canvas.paste(t, (int(tx * 256 - x0), int(ty * 256 - y0)))
    over = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for pg in polys:
        pts = [(lonlat_to_px(x, y, z)[0] - x0, lonlat_to_px(x, y, z)[1] - y0) for x, y in pg.exterior.coords]
        dr.polygon(pts, fill=(255, 40, 0, 70), outline=(255, 40, 0, 255), width=3)
    iy, ix = map(int, item["cell"].split("_"))
    (ax, ay), (bx, by) = lonlat_to_px(ix * G, (iy + 1) * G, z), lonlat_to_px((ix + 1) * G, iy * G, z)
    dr.rectangle([ax - x0, ay - y0, bx - x0, by - y0], outline=(255, 230, 0, 255), width=3)
    im = Image.alpha_composite(canvas.convert("RGBA"), over).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    # wind arrow (points where the wind blows TO), top-right
    to = math.radians((item["wind_from_deg"] + 180) % 360)
    ox, oy, L = SIZE - 70, 70, 45
    ex, ey = ox + L * math.sin(to), oy - L * math.cos(to)
    sx, sy = ox - L * math.sin(to), oy + L * math.cos(to)
    dr.ellipse([ox - 58, oy - 58, ox + 58, oy + 58], fill=(255, 255, 255))
    dr.line([sx, sy, ex, ey], fill=(0, 90, 255), width=6)
    for da in (-150, 150):
        a = to + math.radians(da)
        dr.line([ex, ey, ex + 16 * math.sin(a), ey - 16 * math.cos(a)], fill=(0, 90, 255), width=6)
    dr.text((ox - 50, oy + 60), f"wind {item['wind_ms']:.0f} m/s", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    # scale bar, bottom-left, 1 or 2 km
    mpp = 156543.03 * math.cos(math.radians(clat)) / 2 ** z
    km = 1 if 1000 / mpp < SIZE / 3 else 0.5
    px = km * 1000 / mpp
    dr.rectangle([20, SIZE - 40, 20 + px, SIZE - 30], fill=(255, 255, 255), outline=(0, 0, 0))
    dr.text((20, SIZE - 66), f"{km:g} km", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    dr.text((SIZE / 2 - 8, 6), "N", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(p, quality=90)
    info = {"map_path": str(p), "zoom": z, "basemap_release": str(getattr(rel, "release_date", "")), "m_per_px": mpp}
    meta.write_text(json.dumps(info))
    return info


def main():
    items = sample()
    (EVAC / "maps").mkdir(parents=True, exist_ok=True)
    releases = load_releases()
    starts = {}
    for e in {i["event"] for i in items}:
        starts[e] = pd.Timestamp(e.split(" | ")[1]).date()
    rendered = {}
    with (EVAC / "vlm_items.jsonl").open("w") as f:
        for k, it in enumerate(items):
            if it["item_id"] not in rendered:
                rendered[it["item_id"]] = render(it, releases, starts[it["event"]])
            it.update(rendered[it["item_id"]])
            f.write(json.dumps(it, default=str) + "\n")
            if k % 50 == 0:
                print(k, len(items), it["item_id"], flush=True)
    print(pd.DataFrame(items).groupby(["event", "role"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
