"""
Screenshots of the running web demo for the README (docs/assets/ui_*.jpg).

Needs the backend on :8000 and the frontend on :3000 (production build), and Playwright with Chrome:
    python scripts/demo/readme_screenshots.py [--base http://localhost:3000] [--out docs/assets]
No Bellwether (partner) layer appears in any screenshot.
"""
import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

T3_CASE = "b4d4e235-342a-4f54-8f0e-5368e6399dbd"  # French Fire 2024, post-fire capture 2 days after the start
T2_CASE = "p0042"  # Eaton Fire 2025
# Scroll the case browser (it scrolls inside its own container) so the filter row sits just below the top bar
SCROLL_TO_CASE = """() => {
  const el = document.querySelector('#event-filter');
  el.scrollIntoView({block: 'start'});
  let p = el.parentElement;
  while (p && p.scrollHeight <= p.clientHeight + 1) p = p.parentElement;
  (p || document.scrollingElement).scrollBy(0, -84);
}"""


async def main(base: str, out: Path, only: set):
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        ctx = await b.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1.5)
        pg = await ctx.new_page()
        errors = []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        async def shot(name, el=None):
            path = out / f"ui_{name}.jpg"
            if el is None:
                await pg.screenshot(path=str(path), type="jpeg", quality=84)
            else:
                await el.screenshot(path=str(path), type="jpeg", quality=88)
            print("wrote", path, round(path.stat().st_size / 1024), "KB")

        async def settle(ms=2500):
            try:
                await pg.wait_for_load_state("networkidle", timeout=30000)
            except Exception:  # noqa: BLE001
                pass
            await pg.wait_for_timeout(ms)

        if not only or "map" in only:
            # 1. Layers: CAL FIRE hazard zones, current and past fire perimeters, hotspots, RAWS fuel moisture
            await pg.goto(f"{base}/?region=santa_cruz&layers=perimeters,hotspots,fhsz,fuel&bw=none&guide=0")
            await settle(6000)
            await shot("map_layers")
            # 2. Point summary (click a spot inside the region)
            await pg.mouse.click(1120, 470)
            await pg.wait_for_selector("text=Location summary", timeout=20000)
            try:
                await pg.wait_for_selector("text=Querying live sources", state="detached", timeout=60000)
            except Exception:  # noqa: BLE001
                pass
            await pg.wait_for_timeout(1500)
            await shot("map_point")

        if not only or "weather" in only:
            # 3. Weather: fire-weather field, animated wind, NWS alerts, time slider 3 days ahead (NBM part)
            await pg.goto(f"{base}/?region=san_bruno&layers=wind,alerts&bw=none&guide=0")
            await settle(3000)
            await pg.locator("#weather-panel button", has_text="Fire wx").click()
            await pg.wait_for_selector("#time-slider input[type=range]", timeout=60000)
            await pg.mouse.move(1100, 450)
            for _ in range(4):
                await pg.mouse.wheel(0, 60)
                await pg.wait_for_timeout(800)
            await settle(4000)
            rng = pg.locator("#time-slider input[type=range]")
            now = int(await rng.input_value())
            await rng.fill(str(now + 75))
            await pg.mouse.move(1180, 420)
            await pg.wait_for_timeout(3000)
            await shot("weather")

        if not only or "sources" in only:
            # 4. Data sources & live status
            await pg.goto(f"{base}/?panel=sources&guide=0")
            await settle(12000)
            await shot("sources")

        if not only or "geo" in only:
            # California data cache: fuel model + ignition points with the legend (before any click, so no
            # location card with partner values is on screen), then the cache section of the point card alone
            await pg.goto(f"{base}/?lat=37.60&lng=-122.45&z=13&geo=fuel_model,ignitions&layers=&bw=none&guide=0")
            await settle(12000)
            await pg.locator("#map-hint button").click()
            await pg.wait_for_timeout(500)
            await shot("data_cache")
            await pg.locator("#data-layers-panel").scroll_into_view_if_needed()
            await pg.wait_for_timeout(500)
            await shot("data_cache_panel", pg.locator("#data-layers-panel"))
            await pg.mouse.click(1050, 450)
            await pg.wait_for_selector("#local-data-point", timeout=60000)
            try:
                await pg.wait_for_selector("text=Reading cache", state="detached", timeout=120000)
            except Exception:  # noqa: BLE001
                pass
            await pg.wait_for_timeout(1500)
            await shot("data_cache_point", pg.locator("#local-data-point"))

        if not only or "vlm" in only:
            # 5. VLM view: task explanation and scoreboard
            await pg.goto(f"{base}/?mode=vision&task=t3&case={T3_CASE}&guide=0")
            await pg.wait_for_selector("#vlm-intro", timeout=30000)
            await settle(4000)
            await shot("vlm_scoreboard")
            # 6. One post-fire case: images, DINS label, every model's stored answer
            await pg.set_viewport_size({"width": 1600, "height": 1080})
            await pg.evaluate(SCROLL_TO_CASE)
            await pg.wait_for_timeout(1500)
            await shot("vlm_t3_case")
            # 7. Street-level panel
            await shot("street_view", pg.locator("#street-level"))
            # 8. Pre-fire pair, outcome revealed
            await pg.goto(f"{base}/?mode=vision&task=t2&case={T2_CASE}&guide=0")
            await pg.wait_for_selector("#reveal-btn", timeout=30000)
            await settle(3000)
            await pg.locator("#reveal-btn").click()
            await pg.evaluate(SCROLL_TO_CASE)
            await pg.wait_for_timeout(1500)
            await shot("vlm_t2_pair")

        if errors:
            print("console errors:", errors[:5])
        await b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "docs" / "assets"))
    ap.add_argument("--only", default="", help="comma list: map,weather,sources,geo,vlm")
    a = ap.parse_args()
    asyncio.run(main(a.base, Path(a.out), {x for x in a.only.split(",") if x}))
