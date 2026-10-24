# CartSense

CartSense is a local Python prototype for processing product images and video. Its pipeline is:

`image/video → YOLOX ONNX detection → ByteTrack → barcode association → identification → cart events`

The included detector recognizes generic COCO objects, not grocery SKUs. A catalogue barcode can identify a SKU; other evidence may need operator review. Catalogue prices are sample data, and the app does not process payments.

## Requirements

- macOS 12+ (Apple Silicon or Intel) or Windows 10/11, 64-bit
- Python 3.10–3.12
- Internet access for the first dependency installation
- About 2 GB of free disk space recommended for Python packages and generated videos

The ONNX model, sample image, and sample catalogue are included. Model weights are not downloaded at runtime. Detection runs on CPU by default when an accelerator provider is unavailable; macOS CoreML is used when ONNX Runtime offers it.

## Quick start

### macOS

From Terminal, enter the project folder and run:

```bash
chmod +x setup-macos.sh
./setup-macos.sh
source .venv/bin/activate
python cli.py image data/samples/bus.jpg
```

The setup script creates `.venv`, upgrades pip, and installs dependencies. If macOS does not have Python 3.10–3.12, install Python 3.12 from [python.org](https://www.python.org/downloads/) or with Homebrew (`brew install python@3.12`), then rerun the script.

### Windows

Open PowerShell in the project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup-windows.ps1
.\.venv\Scripts\Activate.ps1
python cli.py image data/samples/bus.jpg
```

If Python is not installed, install Python 3.12 from [python.org](https://www.python.org/downloads/windows/) and enable **Add python.exe to PATH**. The setup script also accepts the Python launcher (`py -3.12`) when available.

## Run CartSense

Activate the virtual environment first (see above), then run any of these from the project folder:

```bash
python cli.py image data/samples/bus.jpg                 # save outputs/annotated.jpg
python cli.py demo                                        # synthetic end-to-end walkthrough
python cli.py video path/to/clip.mp4 --out outputs/out.mp4
python cli.py live 0                                      # default camera; Q/Esc closes the video window
python -m streamlit run app.py                            # browser dashboard
```

On Windows, use `py` instead of `python` if that is the command registered by your installation. Camera access may need to be enabled for Terminal/PowerShell in the operating system privacy settings. To use a different camera, pass its index, such as `python cli.py live 1`.

For Streamlit, open the local URL printed in the terminal (usually `http://localhost:8501`). Stop a running command with `Ctrl+C`.

## Project map

- `app.py`: Streamlit dashboard for uploaded images/videos, cart review, and manual events.
- `cli.py`: image, video, camera, and synthetic demo commands.
- `src/`: detection, tracking, barcode decoding, identification, event processing, cart, and billing modules.
- `config/config.yaml`: model, event zone, tracker, and catalogue settings.
- `models/`: included YOLOX Tiny ONNX model.
- `data/catalog.json`: editable sample catalogue. `data/local_learned_rules.json` stores local operator rules when created.
- `outputs/`: generated annotations and demo videos.

Resource paths in the config are resolved from the project/config location, so the app can also be launched while another directory is the current working directory. Input and output paths you provide on the command line are interpreted relative to the current directory.

## Notes

- The configured camera zone is in `config/config.yaml`. Items already inside the zone when first seen do not produce an ADD event; the pipeline needs evidence of movement into it.
- Unknown or conflicting barcode evidence is sent for review. The in-memory tracking IDs are not durable product identities.
- Barcode decoding uses zxing-cpp with OpenCV fallback. OCR is optional, disabled by default, and not called from the frame pipeline.
- `index.html` is a separate legacy browser demo and does not share the Python cart/event ledger.
- Detector and event thresholds are engineering defaults, not calibrated performance claims. The sample data is not a real grocery evaluation set.

## Troubleshooting

- **`python` not found on Windows:** try `py -3.12`, or reinstall Python with PATH enabled.
- **Package install fails:** confirm Python is 64-bit and between 3.10 and 3.12, activate `.venv`, then rerun the platform setup script.
- **Camera cannot open:** close other camera apps, check OS camera privacy permissions, and try another camera index.
- **Dashboard command not found:** activate `.venv` and run `python -m streamlit run app.py`.
- **Video output is missing:** check the terminal error and confirm the output folder is writable; outputs are written under `outputs/` by default.
