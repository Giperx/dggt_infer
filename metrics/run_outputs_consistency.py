"""CBSR and PD for existing DGGT renders under outputs_nuscenes_pt.

The renders were produced with pretrained/model_latest_nuscenes.pt. These
metrics do not run the model again. rgb and before_rgb are scored separately.
Lyft 1920 and 1224 are pooled by frame, and that report is written under the
1920 directory.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_consistency import run_measurement

CKPT = "pretrained/model_latest_nuscenes.pt"
NUSCENES_VAL = "data/datasets/nuscenes/processed_10Hz_v2/nuScenes_Val.txt"
DDAD_VAL = "data/datasets/ddad_process/valid/valid.txt"
LYFT1920_VAL = "data/datasets/lyft/lyft_val1920_3cams/lyft_val1920.txt"
LYFT1224_VAL = "data/datasets/lyft/lyft_val1224_3cams/lyft_val1224.txt"
IMAGE_DIRS = ["rgb", "before_rgb"]

JOBS = {
    "nuscenes_single": (
        [("nuScenes", "outputs_nuscenes_pt/nuscenes_inference", NUSCENES_VAL)],
        "outputs_nuscenes_pt/nuscenes_inference",
        "DGGT nuScenes CBSR and PD",
    ),
    "nuscenes_multi": (
        [("nuScenes", "outputs_nuscenes_pt/nuscenes_multiframes_inference", NUSCENES_VAL)],
        "outputs_nuscenes_pt/nuscenes_multiframes_inference",
        "DGGT nuScenes multiframe CBSR and PD",
    ),
    "ddad_single": (
        [("DDAD", "outputs_nuscenes_pt/ddad_inference", DDAD_VAL)],
        "outputs_nuscenes_pt/ddad_inference",
        "DGGT DDAD CBSR and PD",
    ),
    "ddad_multi": (
        [("DDAD", "outputs_nuscenes_pt/ddad_multiframes_inference", DDAD_VAL)],
        "outputs_nuscenes_pt/ddad_multiframes_inference",
        "DGGT DDAD multiframe CBSR and PD",
    ),
    "lyft_single": (
        [
            ("lyft1920", "outputs_nuscenes_pt/lyft/lyft1920_inference", LYFT1920_VAL),
            ("lyft1224", "outputs_nuscenes_pt/lyft/lyft1224_inference", LYFT1224_VAL),
        ],
        "outputs_nuscenes_pt/lyft/lyft1920_inference",
        "DGGT Lyft CBSR and PD",
    ),
    "lyft_multi": (
        [
            ("lyft1920", "outputs_nuscenes_pt/lyft_multiframes/lyft1920_multiframes_inference", LYFT1920_VAL),
            ("lyft1224", "outputs_nuscenes_pt/lyft_multiframes/lyft1224_multiframes_inference", LYFT1224_VAL),
        ],
        "outputs_nuscenes_pt/lyft_multiframes/lyft1920_multiframes_inference",
        "DGGT Lyft multiframe CBSR and PD",
    ),
}

LANES = {
    "gpu0": ["nuscenes_single", "ddad_single"],
    "gpu1": ["nuscenes_multi", "ddad_multi", "lyft_single", "lyft_multi"],
}


def run_job(name, workers):
    groups, out_dir, title = JOBS[name]
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"consistency_{timestamp}.txt")
    meta = [
        f"Checkpoint used for these renders: {CKPT}",
        "CBSR and PD do not rerun the model and do not use ground truth.",
        "rgb and before_rgb are scored separately.",
    ]
    if name.startswith("lyft"):
        meta.append("Lyft 1920 and 1224 are pooled by frame, not averaged as two subset means.")
    run_measurement(groups, IMAGE_DIRS, out_path, title, meta, workers=workers)


def main():
    # CBSR and PD scoring is paused. Uncomment the block below to resume.
    print("CBSR and PD scoring is paused.", flush=True)
    return
    # if len(sys.argv) < 2 or sys.argv[1] not in LANES and sys.argv[1] not in JOBS:
    #     names = ", ".join(list(LANES) + list(JOBS))
    #     raise SystemExit(f"usage: python metrics/run_outputs_consistency.py JOB [--workers N]\njobs: {names}")
    # workers = 8
    # if "--workers" in sys.argv:
    #     workers = int(sys.argv[sys.argv.index("--workers") + 1])
    # names = LANES.get(sys.argv[1], [sys.argv[1]])
    # for name in names:
    #     print(f"=== {name} workers={workers} ===", flush=True)
    #     run_job(name, workers)


if __name__ == "__main__":
    main()
