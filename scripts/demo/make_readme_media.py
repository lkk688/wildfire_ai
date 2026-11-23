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
    ("guide", 1.0, 4.0),
    ("map_region", 2.0, 4.0),
    ("map_layers", 5.0, 6.0),
    ("map_click", 4.5, 5.0),
    ("map_copilot", 21.0, 4.5),
    ("vlm_question", 1.0, 4.0),
    ("vlm_scoreboard", 1.0, 3.5),
    ("vlm_case", 1.5, 4.5),
    ("vlm_live", 6.5, 4.0),
    ("vlm_pairs", 12.5, 4.5),
    ("outro", 0.6, 4.5),
]
STILLS = {"tutorial_map.jpg": ("map_click", 9.0), "tutorial_vlm.jpg": ("vlm_case", 4.0)}


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    run(["-i", str(WEBM), "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(ASSETS / "tutorial.mp4")])
    for name, (step, off) in STILLS.items():
        run(["-ss", f"{STEPS[step] + off:.2f}", "-i", str(WEBM), "-frames:v", "1", "-q:v", "3", str(ASSETS / name)])

    parts, labels = [], []
    for i, (step, off, dur) in enumerate(CLIPS):
        a = STEPS[step] + off
        parts.append(f"[0:v]trim=start={a:.2f}:duration={dur:.2f},setpts=PTS-STARTPTS,fps=6,scale=880:-1:flags=lanczos[c{i}]")
        labels.append(f"[c{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(CLIPS)}:v=1:a=0,split[x][y];" \
            "[x]palettegen=max_colors=96:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    run(["-i", str(WEBM), "-filter_complex", graph, "-loop", "0", str(ASSETS / "tutorial.gif")])
    for f in ("tutorial.mp4", "tutorial.gif", *STILLS):
        print(f, round((ASSETS / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
