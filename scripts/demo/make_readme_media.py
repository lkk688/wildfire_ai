"""
Turn a tutorial recording (scripts/demo/record_tutorial.py) into the README media:
  docs/assets/tutorial.mp4   full tutorial, H.264, no audio
  docs/assets/tutorial.gif   ~45 s highlight reel that autoplays on the GitHub page (cut from the same recording)
  docs/assets/tutorial_map.jpg, tutorial_vlm.jpg   still frames for the README sections

  python scripts/demo/make_readme_media.py <recording_dir>     # needs ffmpeg on PATH
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
REC = Path(sys.argv[1])
WEBM = next(REC.glob("*.webm"))
STEPS = {s["step"]: s["t"] for s in json.loads((REC / "steps.json").read_text())}

# (step, seconds after the step starts, duration): each clip keeps its caption on screen
CLIPS = [
    ("guide", 1.0, 3.0),
    ("map_region", 2.0, 3.5),
    ("map_layers", 5.0, 5.0),
    ("map_click", 4.5, 5.0),
    ("map_weather", 9.0, 4.5),    # drag the time slider
    ("map_weather", 21.0, 6.0),   # play: wind and temperature hour by hour
    ("map_weather", 37.5, 4.5),   # fire-weather index and NWS alerts
    ("map_copilot", 21.0, 4.5),
    ("vlm_question", 1.0, 3.5),
    ("vlm_scoreboard", 1.0, 3.0),
    ("vlm_case", 1.5, 4.0),
    ("vlm_live", 6.5, 3.5),
    ("vlm_pairs", 12.5, 4.5),
    ("outro", 0.6, 4.0),
]
STILLS = {"tutorial_map.jpg": ("map_click", 9.0), "tutorial_weather.jpg": ("map_weather", 26.0), "tutorial_vlm.jpg": ("vlm_case", 4.0)}


GIF_FPS, GIF_W, GIF_COLORS = (int(v) for v in (sys.argv[2:5] if len(sys.argv) >= 5 else (5, 800, 96)))  # GitHub shows GIFs up to 10 MB; fewer colours band the weather fields


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    # 1280 px wide: the animated wind layer is costly to encode; captions stay sharp
    run(["-i", str(WEBM), "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "31", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(ASSETS / "tutorial.mp4")])
    for name, (step, off) in STILLS.items():
        run(["-ss", f"{STEPS[step] + off:.2f}", "-i", str(WEBM), "-frames:v", "1", "-q:v", "3", str(ASSETS / name)])

    parts, labels = [], []
    for i, (step, off, dur) in enumerate(CLIPS):
        a = STEPS[step] + off
        parts.append(f"[0:v]trim=start={a:.2f}:duration={dur:.2f},setpts=PTS-STARTPTS,fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos[c{i}]")
        labels.append(f"[c{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(CLIPS)}:v=1:a=0,split[x][y];" \
            f"[x]palettegen=max_colors={GIF_COLORS}:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    run(["-i", str(WEBM), "-filter_complex", graph, "-loop", "0", str(ASSETS / "tutorial.gif")])
    for f in ("tutorial.mp4", "tutorial.gif", *STILLS):
        print(f, round((ASSETS / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
