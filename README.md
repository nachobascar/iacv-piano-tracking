# 🎹 Silent‑Video Piano Key Detection

> **Final short project – Image Analysis & Computer Vision (Spring 2025)**

---

## 📖 Overview

This repository contains the Python 3.10 code, sample data, and report for my university project.
Given **only** a silent top‑down (or slightly angled) recording of a pianist, the program:

1. *Rectifies* the keyboard with homography to undo perspective distortion.
2. Uses **MediaPipe Hands** to track finger contact candidates.
3. Decides which **white & black keys** are actually depressed frame‑by‑frame.
4. **Post‑processes / "softens"** the detections to remove flicker.
5. Exports the result to a standards‑compliant **MIDI** file you can instantly play back in tools like [Pianotify](https://pianotify.com/import-midi-file).

The full classical (non‑DL) pipeline, including many pitfalls and future work, is detailed in `project_report.pdf`.

---

## 🗂️ Folder Structure

```text
project_root/
├── main.py                 # Entry‑point script (run everything from here)
├── project_report.pdf      # Written report (methodology & discussion)
├── requirements.txt        # Precise Python dependencies
├── inputs/
│   └── <video_name>/       # One sub‑folder per test video
│       ├── params.json     # Fine‑tuning thresholds for this take
│       ├── reference.png   # Clean keyboard image (no hands)
│       └── video.mp4       # The raw performance
└── outputs/
    └── <video_name>/       # Auto‑generated results live here
        ├── output.json     # Raw per‑frame key states
        ├── softened.json   # Temporal smoothing of `output.json`
        ├── output.mid      # The extracted performance
        ├── frames/         # Debug visuals (hands + key overlay)
        ├── homography/     # Debug: rectification steps
        ├── keys_touched/   # Debug: finger‑key proximity
        └── is_toggled/     # Debug: final "pressed?" decision
```

*Tip:* Supply your own `inputs/<video_name>` folder with the three required files to test a new piece.

---

## ⚙️ Installation

> **Prerequisite:** Python 3.10

1. **Create** and activate a virtual environment *(recommended)*:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate     # Windows PowerShell
   ```
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   | Package         | Version   | OS‑specific note                                                                                                                                                  |
   | --------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `opencv-python` | 4.11.0.86 | On some Linux distros you may need extra system libs (e.g., `apt install libgl1`).  If you require non‑free codecs or SIFT, grab `opencv-contrib-python` instead. |
   | `mediapipe`     | 0.10.21   | Requires a modern pip (≥ 23) to pull prebuilt wheels.                                                                                                             |

If you hit OpenCV build errors on Apple Silicon, try:

```bash
pip install opencv-python-headless
```

then **brew install libjpeg libpng openexr** and re‑run.

---

## 🚀 Usage

Run the pipeline from the project root:

```bash
python main.py -i <video_name> [--calibrate] [--show]
```

| Flag          | Short | Required | Description                                                                                      |
| ------------- | ----- | -------- | ------------------------------------------------------------------------------------------------ |
| `--input`     | `-i`  | **Yes**  | Name of the sub‑folder inside `inputs/` (and destination under `outputs/`).                      |
| `--calibrate` | `-c`  | No       | *Re‑process* the video end‑to‑end.  Omit to re‑use existing `output.json` to shorten dev cycles. |
| `--show`      | `-s`  | No       | Write all debug images & visualisations.  Skipping this makes processing \~3× faster.            |

### Example

```bash
python main.py -i midna -c -s
```

1. Processes `inputs/midna/video.mp4`
2. Saves results in `outputs/midna/`
3. Opens `output.mid` in [Pianotify](https://pianotify.com/import-midi-file) to check accuracy.

---

## ✨ Expected Output

After a successful run with `--show`, you’ll get something like:

* **output.mid** → audible reconstruction of the piece.
* **frames/** → GIF‑like sequence with *red* possible keys & *green* confirmed presses.
* **homography/**, **keys\_touched/**, **is\_toggled/** → step‑by‑step visuals for academic write‑ups.

*(For a deep dive on evaluation metrics, and black‑key woes, see `project_report.pdf`.)*

---

## 🙏 Acknowledgements

* **Professor Vicenzo Caglioti** for guidance on the project solution. 
* **MediaPipe** for real‑time hand tracking.
* **OpenCV** & **NumPy** for the heavy lifting.
