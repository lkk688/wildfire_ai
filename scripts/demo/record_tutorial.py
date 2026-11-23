"""
Step-by-step usage tutorial of the web demo, recorded with Playwright and the system Chrome.

  pip install playwright && playwright install ffmpeg      # in any venv; uses channel="chrome"
  # backend on :8000 and a production frontend on :3000 (see README); open the map once to warm the caches
  python scripts/demo/record_tutorial.py /tmp/tutorial            # -> /tmp/tutorial/*.webm + steps.json
  python scripts/demo/make_readme_media.py /tmp/tutorial          # -> docs/assets/tutorial.mp4, tutorial.gif

Everything on screen is the running app with live data; the script only moves a visible pointer, outlines what
it clicks and adds captions. Numbers in the captions are read from the backend (/api/heldout/summary), not typed.
"""
import asyncio
import json
import sys
import time
import urllib.request

from playwright.async_api import async_playwright

BASE = "http://localhost:3000"
API = "http://localhost:8000"
OUT = sys.argv[1]
W, H = 1600, 900
T3_CASE = "b4d4e235-342a-4f54-8f0e-5368e6399dbd"  # French Fire 2024, captured 2 days after the start
T2_CASE = "p0042"  # Eaton Fire 2025

# Visible pointer + click outline (re-installed on every navigation)
INIT_JS = """
(() => {
  const css = `#tut-cursor{position:fixed;z-index:2147483647;width:22px;height:22px;margin:-3px 0 0 -3px;pointer-events:none;
    transition:transform .08s;filter:drop-shadow(0 2px 4px rgba(0,0,0,.6))}
   #tut-cursor.down{transform:scale(.8)}
   .tut-ripple{position:fixed;z-index:2147483646;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;
    border:3px solid #fb923c;pointer-events:none;animation:tutr .6s ease-out forwards}
   @keyframes tutr{from{transform:scale(.3);opacity:1}to{transform:scale(1.4);opacity:0}}
   #tut-spot{position:fixed;z-index:2147483645;border:3px solid #fb923c;border-radius:12px;pointer-events:none;
    box-shadow:0 0 0 9999px rgba(2,6,23,.35),0 0 24px rgba(251,146,60,.8);transition:all .35s}`;
  const add = () => {
    if (document.getElementById('tut-cursor')) return;
    const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
    const c = document.createElement('div'); c.id = 'tut-cursor';
    c.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M3 2l7 19 2.5-7.5L20 11z" fill="#fff" stroke="#0f172a" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    c.style.left = '800px'; c.style.top = '450px';
    document.body.appendChild(c);
    window.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    window.addEventListener('mousedown', e => { c.classList.add('down');
      const r = document.createElement('div'); r.className = 'tut-ripple'; r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
      document.body.appendChild(r); setTimeout(() => r.remove(), 700); }, true);
    window.addEventListener('mouseup', () => c.classList.remove('down'), true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', add); else add();
})();
"""

CAPTION_JS = """
([chapter, text, sub]) => {
  let el = document.getElementById('tut-caption');
  if (!el) {
    el = document.createElement('div'); el.id = 'tut-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:2147483647;max-width:1150px;' +
      'background:rgba(2,6,23,0.92);border:1px solid rgba(251,146,60,0.65);border-radius:14px;padding:12px 24px;color:#f8fafc;' +
      'font:600 21px/1.35 Inter,system-ui,sans-serif;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.55);pointer-events:none';
    document.body.appendChild(el);
  }
  if (!text) { el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.innerHTML = (chapter ? '<div style="font:800 12px/1 Inter,system-ui,sans-serif;letter-spacing:.12em;color:#fdba74;margin-bottom:6px">' + chapter + '</div>' : '') +
    text + (sub ? '<div style="font-weight:400;font-size:16px;color:#cbd5e1;margin-top:4px">' + sub + '</div>' : '');
}
"""

CARD_JS = """
([title, lines]) => {
  let el = document.getElementById('tut-card');
  if (!title) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement('div'); el.id = 'tut-card'; document.body.appendChild(el); }
  el.style.cssText = 'position:fixed;inset:0;z-index:2147483646;background:#020617;display:flex;flex-direction:column;' +
    'align-items:center;justify-content:center;gap:18px;color:#f8fafc;font-family:Inter,system-ui,sans-serif;text-align:center;padding:60px';
  el.innerHTML = '<div style="font-size:46px;font-weight:800">' + title + '</div>' +
    lines.map(l => '<div style="font-size:22px;color:#cbd5e1;max-width:1150px;line-height:1.5">' + l + '</div>').join('');
}
"""

SPOT_JS = """
(sel) => {
  let s = document.getElementById('tut-spot');
  if (!sel) { if (s) s.remove(); return; }
  const e = document.querySelector(sel); if (!e) return;
  const r = e.getBoundingClientRect();
  if (!s) { s = document.createElement('div'); s.id = 'tut-spot'; document.body.appendChild(s); }
  s.style.left = (r.left - 6) + 'px'; s.style.top = (r.top - 6) + 'px'; s.style.width = (r.width + 12) + 'px'; s.style.height = (r.height + 12) + 'px';
}
"""


def summary():
    with urllib.request.urlopen(f"{API}/api/heldout/summary", timeout=60) as r:
        return json.loads(r.read())


async def main():
    s = summary()
    t3 = [m for m in s["t3"]["models"] if m.get("auroc")]
    t2 = [m for m in s["t2"]["models"] if m.get("pair_accuracy") and m["input"] != "metadata only (no image)"]
    t3_best = max(m["auroc"] for m in t3)
    t2_lo, t2_hi = min(m["pair_accuracy"] for m in t2), max(m["pair_accuracy"] for m in t2)
    lora = [m["auroc"] for m in t3 if m["kind"] == "fine-tuned VLM"]

    steps = []
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        ctx = await b.new_context(viewport={"width": W, "height": H}, record_video_dir=OUT,
                                  record_video_size={"width": W, "height": H})
        await ctx.add_init_script(INIT_JS)
        pg = await ctx.new_page()
        t0 = time.monotonic()

        async def mark(name):
            steps.append({"step": name, "t": round(time.monotonic() - t0, 2)})

        async def cap(chapter="", text="", sub=""):
            await pg.evaluate(CAPTION_JS, [chapter, text, sub])

        async def card(title="", lines=()):
            await pg.evaluate(CARD_JS, [title, list(lines)])

        async def spot(sel=None):
            await pg.evaluate(SPOT_JS, sel)

        async def move_to(loc, steps_=18):
            bb = await loc.bounding_box()
            x, y = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
            await pg.mouse.move(x, y, steps=steps_)
            return x, y

        async def click(loc, pause=350):
            await loc.scroll_into_view_if_needed()
            await move_to(loc)
            await pg.wait_for_timeout(pause)
            await loc.click()

        # ================================================================ intro
        await pg.goto(BASE + "/?guide=1", wait_until="domcontentloaded")
        await card("How to use Wildfire AI", [
            "A 3-minute guide to the two views: the <b>risk map</b> (live public data) and the <b>VLM evaluation</b> (AI models on real fires).",
            "Everything you will see is real data from the running app. Nothing is simulated.",
        ])
        await mark("intro")
        await pg.wait_for_timeout(5500)
        await card()
        await pg.wait_for_selector("#guide-modal", timeout=20000)
        await cap("START HERE", "The guide opens on your first visit", "It explains both views; reopen it any time with the Guide button at the top")
        await mark("guide")
        await pg.mouse.move(700, 400, steps=10)
        await pg.wait_for_timeout(6000)
        await click(pg.locator("#guide-go-map"))
        await pg.wait_for_timeout(2500)

        # ================================================================ risk map
        await cap("RISK MAP · STEP 1 OF 5", "Pick a region", "Three Bay Area communities at the edge of wildland")
        await mark("map_region")
        await spot("select")
        await move_to(pg.locator("select").first)
        await pg.wait_for_timeout(2500)
        await pg.locator("select").first.select_option("santa_cruz")
        await spot()
        await pg.wait_for_timeout(4500)

        await cap("RISK MAP · STEP 2 OF 5", "Switch layers on and off", "The legend on the map explains every colour. Here: USDA burn probability and CAL FIRE hazard zones")
        await mark("map_layers")
        await click(pg.get_by_role("button", name="Off"))
        await pg.wait_for_timeout(600)
        for label in ("Satellite hotspots (24 h)", "Wind (HRRR)", "Fuel moisture (RAWS)"):
            await pg.locator("label", has_text=label).locator("input").uncheck()
        await click(pg.locator("label", has_text="Annual burn probability").locator("input"))
        await pg.wait_for_timeout(4000)
        await spot("#map-legend")
        await pg.mouse.move(520, 220, steps=15)
        await pg.wait_for_timeout(4500)
        await spot()
        await click(pg.locator("label", has_text="Fire Hazard Severity Zones").locator("input"))
        await pg.wait_for_timeout(5500)

        await cap("RISK MAP · STEP 3 OF 5", "Click any spot on the map", "You get everything the live sources know about it: weather, hazard, land cover, past fires, slope, buildings")
        await mark("map_click")
        await pg.mouse.move(1000, 560, steps=20)
        await pg.wait_for_timeout(500)
        await pg.mouse.click(1000, 560)
        try:
            await pg.wait_for_selector("text=Querying live sources", state="detached", timeout=30000)
        except Exception:  # noqa: BLE001
            pass
        await pg.wait_for_timeout(5500)
        card_loc = pg.locator("text=Location summary").locator("xpath=ancestor::div[contains(@class,'overflow-y-auto')][1]")
        try:
            await card_loc.evaluate("(e) => e.scrollBy({top: 330, behavior: 'smooth'})")
        except Exception:  # noqa: BLE001
            pass
        await pg.wait_for_timeout(5000)
        await click(pg.locator("button:has(svg.lucide-x)").first)
        await pg.wait_for_timeout(800)

        await cap("RISK MAP · STEP 4 OF 5", "Check where each number comes from", "Data sources & live status tests every source when it opens")
        await mark("map_sources")
        await click(pg.get_by_role("button", name="Data sources & live status"))
        await pg.wait_for_timeout(8000)
        await click(pg.locator("button:has(svg.lucide-x)").first)
        await pg.wait_for_timeout(800)

        await cap("RISK MAP · STEP 5 OF 5", "Ask the AI copilot", "It answers from the same live data and is told not to invent numbers")
        await mark("map_copilot")
        await click(pg.get_by_role("button", name="AI Wildfire Copilot").first)
        await pg.wait_for_timeout(1500)
        inp = pg.locator("form input").last
        await move_to(inp)
        await inp.type("Is anything burning near Santa Cruz right now, and how dry are the fuels?", delay=28)
        await pg.wait_for_timeout(500)
        await inp.press("Enter")
        try:
            await pg.wait_for_function("() => !document.body.innerText.includes('is querying live sources')", timeout=90000)
        except Exception:  # noqa: BLE001
            pass
        await pg.wait_for_timeout(1000)
        scroll_js = """(dy) => {
          const marks = [...document.querySelectorAll('span')].filter(e => e.textContent.trim() === 'GIS Tools Executed:');
          const last = marks[marks.length - 1]; if (!last) return;
          if (dy === 0) { last.scrollIntoView({block: 'start'}); return; }
          let el = last.parentElement; while (el && el.scrollHeight <= el.clientHeight + 4) el = el.parentElement;
          if (el) el.scrollBy({top: dy, behavior: 'smooth'});
        }"""
        await pg.evaluate(scroll_js, 0)
        await pg.wait_for_timeout(6000)
        await pg.evaluate(scroll_js, 380)
        await pg.wait_for_timeout(6000)

        # ================================================================ VLM evaluation
        await cap()
        await pg.goto(BASE + f"/?mode=vision&task=t3&case={T3_CASE}", wait_until="domcontentloaded")
        await pg.wait_for_selector("#vlm-intro", timeout=30000)
        await pg.wait_for_selector("text=Stored model outputs for this case", timeout=30000)
        await cap("VLM EVALUATION · STEP 1 OF 5", "Choose a question", "After the fire: was this home destroyed?  Before the fire: which of two neighbours burned?")
        await mark("vlm_question")
        await spot("#vlm-intro")
        await pg.mouse.move(400, 220, steps=15)
        await pg.wait_for_timeout(7000)
        await spot()

        await cap("VLM EVALUATION · STEP 2 OF 5", "The scoreboard: every model on every test home",
                  f"After the fire the best model reaches AUROC {t3_best:.2f} (1 = perfect, 0.5 = coin flip)")
        await mark("vlm_scoreboard")
        await pg.mouse.wheel(0, 260)
        await pg.wait_for_timeout(7000)

        await cap("VLM EVALUATION · STEP 3 OF 5", "Look at one home, then at each model's answer",
                  "French Fire, July 2024: photos before and 2 days after. Green = the model was right, red = wrong")
        await mark("vlm_case")
        await pg.mouse.wheel(0, 380)
        await pg.wait_for_timeout(8500)

        await cap("VLM EVALUATION · STEP 4 OF 5", "Ask a model again, live", "Same question, same photos, sent to Claude Sonnet now")
        await mark("vlm_live")
        await click(pg.locator("#live-run"))
        try:
            await pg.wait_for_selector("text=Live answer", timeout=120000)
        except Exception:  # noqa: BLE001
            pass
        await pg.mouse.wheel(0, 420)
        await pg.wait_for_timeout(7000)

        await pg.goto(BASE + f"/?mode=vision&task=t2&case={T2_CASE}", wait_until="domcontentloaded")
        await pg.wait_for_selector("text=Stored model outputs for this case", timeout=30000)
        await cap("VLM EVALUATION · STEP 5 OF 5", "Before the fire: which of two neighbours burned?",
                  f"Same fire, same weather; only the building and yard differ. The best model is right in {t2_hi:.0%} of pairs")
        await mark("vlm_pairs")
        await pg.wait_for_timeout(5500)
        await pg.mouse.wheel(0, 640)
        await cap("VLM EVALUATION · STEP 5 OF 5", "Eaton Fire, January 2025: make your own guess", "Each model's pick is on the right; press Reveal for the CAL FIRE outcome")
        await pg.wait_for_timeout(6000)
        await click(pg.locator("#reveal-btn"))
        await cap("VLM EVALUATION · STEP 5 OF 5", "Revealed: Building B was destroyed", "Claude Sonnet picked A. From pre-fire photos the difference is rarely visible")
        await pg.wait_for_timeout(7000)
        await cap()

        # ================================================================ outro
        await card("What you have seen", [
            f"After a fire, AI sees the damage: AUROC up to {t3_best:.2f}; small models fine-tuned on past fires reach "
            f"{min(lora):.2f}–{max(lora):.2f}.",
            f"Before a fire, aerial photos say little about which home will burn: {t2_lo:.0%}–{t2_hi:.0%} of pairs right, against 50% for a coin flip.",
            "Run it yourself: see “Running the demo” in the README.",
        ])
        await mark("outro")
        await pg.wait_for_timeout(7500)
        await mark("end")
        await ctx.close()
        await b.close()
    (open(f"{OUT}/steps.json", "w")).write(json.dumps(steps, indent=1))
    print(json.dumps(steps))


asyncio.run(main())
