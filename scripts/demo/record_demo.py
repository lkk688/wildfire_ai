"""
Scripted walkthrough of the web demo, recorded with Playwright and the system Chrome (docs/assets/demo_*.mp4).

  pip install playwright && playwright install ffmpeg      # in any venv; uses channel="chrome"
  # backend on :8000 and a production frontend on :3000 (see README), caches warmed by opening the map once
  python scripts/demo/record_demo.py /tmp/demo_video
  ffmpeg -i /tmp/demo_video/*.webm -c:v libx264 -preset slow -crf 30 -pix_fmt yuv420p -movflags +faststart -an docs/assets/demo_<date>.mp4

The captions state only what the recorded data shows; check them against the current outputs before re-recording
(T3_CASE / T2_CASE and the outro numbers come from output/demo/heldout_cases.json).
"""
import asyncio
import sys

from playwright.async_api import async_playwright

BASE = "http://localhost:3000"
OUT = sys.argv[1]
W, H = 1600, 900
T3_CASE = "b4d4e235-342a-4f54-8f0e-5368e6399dbd"
T2_CASE = "p0042"

CAPTION_JS = """
([text, sub]) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:2147483647;' +
      'max-width:1100px;background:rgba(2,6,23,0.88);border:1px solid rgba(251,146,60,0.6);border-radius:14px;' +
      'padding:12px 22px;color:#f8fafc;font:600 20px/1.35 Inter,system-ui,sans-serif;text-align:center;' +
      'box-shadow:0 10px 40px rgba(0,0,0,0.5);pointer-events:none;transition:opacity .3s';
    document.body.appendChild(el);
  }
  if (!text) { el.style.opacity = '0'; return; }
  el.style.opacity = '1';
  el.innerHTML = text + (sub ? '<div style="font-weight:400;font-size:15px;color:#cbd5e1;margin-top:4px">' + sub + '</div>' : '');
}
"""

CARD_JS = """
([title, lines]) => {
  let el = document.getElementById('demo-card');
  if (!title) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement('div'); el.id = 'demo-card'; document.body.appendChild(el); }
  el.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:#020617;display:flex;flex-direction:column;' +
    'align-items:center;justify-content:center;gap:18px;color:#f8fafc;font-family:Inter,system-ui,sans-serif;text-align:center;padding:60px';
  el.innerHTML = '<div style="font-size:44px;font-weight:800;letter-spacing:.3px">' + title + '</div>' +
    lines.map(l => '<div style="font-size:21px;color:#cbd5e1;max-width:1150px;line-height:1.45">' + l + '</div>').join('');
}
"""


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        ctx = await b.new_context(viewport={"width": W, "height": H}, record_video_dir=OUT, record_video_size={"width": W, "height": H})
        pg = await ctx.new_page()

        async def cap(text="", sub=""):
            await pg.evaluate(CAPTION_JS, [text, sub])

        async def card(title="", lines=()):
            await pg.evaluate(CARD_JS, [title, list(lines)])

        # ---------------- intro
        await pg.goto(BASE + "/", wait_until="domcontentloaded")
        await card("Wildfire AI · demo (22 Nov 2026)", [
            "Two views: a <b>risk map</b> on live public data and a <b>VLM evaluation</b> browser on held-out fires.",
            "Nothing on screen is mocked: map layers are fetched live from public services, labels come from CAL FIRE damage inspections, and model outputs are the stored runs behind the paper.",
        ])
        await pg.wait_for_timeout(6500)
        await card()

        # ---------------- map: overview
        await cap("Risk map · San Bruno WUI", "Bellwether hazard (partner raster) · HRRR wind · RAWS fuel moisture · NWS alerts, all fetched live")
        await pg.wait_for_timeout(7000)

        await cap("Data sources & live status", "Each source is checked when the panel opens: 11 public web services, the partner raster and building footprints; no API keys")
        await pg.get_by_role("button", name="Data sources & live status").click()
        await pg.wait_for_timeout(9000)
        await pg.locator("button:has(svg.lucide-x)").first.click()
        await pg.wait_for_timeout(800)

        await cap("Click anywhere for a point summary", "Weather, alerts, hazard zone, land cover, past fires, slope, buildings and nearest RAWS station")
        await pg.mouse.click(1180, 610)
        await pg.wait_for_timeout(15000)
        await pg.locator("button:has(svg.lucide-x)").first.click()  # close the location card
        await pg.wait_for_timeout(600)

        # ---------------- Santa Cruz
        await cap("Santa Cruz Mountains", "CAL FIRE historic perimeters (1950+) and Fire Hazard Severity Zones (SRA 2023, LRA 2025)")
        await pg.locator("select").first.select_option("santa_cruz")
        await pg.wait_for_timeout(2500)
        await pg.locator("label", has_text="Fire Hazard Severity Zones").locator("input").check()
        await pg.wait_for_timeout(6500)
        await cap("Terrain slope from USGS 3DEP", "Computed from the elevation model on request")
        await pg.locator("label", has_text="Fire Hazard Severity Zones").locator("input").uncheck()
        await pg.locator("label", has_text="Terrain slope").locator("input").check()
        await pg.wait_for_timeout(6500)
        await pg.locator("label", has_text="Terrain slope").locator("input").uncheck()

        await cap("Bellwether vs CAL FIRE hazard zones", "Where the partner model and the state map agree and disagree (only where Bellwether has values)")
        await pg.locator("select").first.select_option("san_bruno")
        await pg.wait_for_timeout(2000)
        await pg.get_by_role("button", name="Bellwether vs FHSZ analysis").click()
        await pg.wait_for_timeout(9000)
        await pg.get_by_role("button", name="Close").first.click()
        await pg.wait_for_timeout(800)

        # ---------------- copilot
        await cap("AI copilot answers from the same live tools", "It is told not to invent values; this answer is generated now")
        await pg.get_by_role("button", name="AI Wildfire Copilot").first.click()
        await pg.wait_for_timeout(1500)
        inp = pg.locator("form input").last
        await inp.fill("What are the current weather and fuel moisture near San Bruno, and is anything burning nearby?")
        await pg.wait_for_timeout(800)
        await inp.press("Enter")
        try:
            await pg.wait_for_function("() => !document.body.innerText.includes('is querying live sources')", timeout=90000)
        except Exception:  # noqa: BLE001
            pass
        await pg.wait_for_timeout(800)
        scroll_js = """(dy) => {
          const marks = [...document.querySelectorAll('span')].filter(e => e.textContent.trim() === 'GIS Tools Executed:');
          const last = marks[marks.length - 1];
          if (!last) return;
          if (dy === 0) { last.scrollIntoView({block: 'start'}); return; }
          let el = last.parentElement;
          while (el && el.scrollHeight <= el.clientHeight + 4) el = el.parentElement;
          if (el) el.scrollBy({top: dy, behavior: 'smooth'});
        }"""
        await pg.evaluate(scroll_js, 0)
        await pg.wait_for_timeout(7000)
        await pg.evaluate(scroll_js, 380)
        await pg.wait_for_timeout(6000)
        await pg.evaluate(scroll_js, 380)
        await pg.wait_for_timeout(5000)

        # ---------------- VLM evaluation: T3
        await pg.goto(BASE + f"/?mode=vision&task=t3&case={T3_CASE}", wait_until="domcontentloaded")
        await pg.wait_for_selector("text=Stored model outputs for this case", timeout=30000)
        await cap("VLM evaluation on held-out fires (2024–25)", "Post-fire task: scores computed from the stored per-building outputs of the paper's runs")
        await pg.wait_for_timeout(8000)
        await cap("One test building: French Fire, July 2024", "Verified Wayback captures before and 2 days after the fire; label from the CAL FIRE inspection")
        await pg.mouse.wheel(0, 330)
        await pg.wait_for_timeout(8000)
        await cap("Re-run the case now", "Same research prompt, same images, sent live to Claude Sonnet 5.5")
        await pg.click("#live-run")
        try:
            await pg.wait_for_selector("text=Live answer", timeout=120000)
        except Exception:  # noqa: BLE001
            pass
        await pg.mouse.wheel(0, 400)
        await pg.wait_for_timeout(7000)

        # ---------------- T2
        await pg.goto(BASE + f"/?mode=vision&task=t2&case={T2_CASE}", wait_until="domcontentloaded")
        await pg.wait_for_selector("text=Stored model outputs for this case", timeout=30000)
        await cap("Pre-fire task: two neighbours, same fire, one destroyed", "Same weather and timing, so only the building and yard differ. Every model is close to chance.")
        await pg.wait_for_timeout(8000)
        await pg.mouse.wheel(0, 300)
        await cap("Eaton Fire, January 2025: which one burned?", "Each model's pick from the stored runs, then the CAL FIRE outcome")
        await pg.wait_for_timeout(6000)
        await pg.click("#reveal-btn")
        await cap("Revealed: Building B was destroyed", "Claude Sonnet picked A. From pre-fire aerial images, the difference is rarely visible.")
        await pg.wait_for_timeout(8000)
        await cap()

        # ---------------- outro
        await card("What the evidence shows", [
            "After a fire, damage is visible: AUROC .89–.94 for the best zero-shot and fine-tuned models.",
            "Before a fire, pre-fire aerial images carry little building-level signal: .57–.62 pair accuracy, close to chance.",
            "Code: backend/ (FastAPI), frontend/ (Next.js), scripts/demo/export_heldout_cases.py",
        ])
        await pg.wait_for_timeout(7000)
        await ctx.close()
        await b.close()


asyncio.run(main())
