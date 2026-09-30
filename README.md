# frontal_click_fit

Build a kymograph from a video of a moving reaction front, calibrate pixels to mm from the
tube diameter, manually click the front on the kymograph to fit its speed, and write the
results to `output_data/results_<video>.txt`.

## Setup

Uses [uv](https://docs.astral.sh/uv/) to manage the Python environment (installs Python 3.12
and the dependencies automatically on first run). Runs on macOS, Linux, and Windows.

Install uv if you don't have it:

```bash
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell) -- then close and reopen PowerShell so uv is on the PATH:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Run

```bash
# macOS / Linux:
uv run python frontal_click_fit.py input_videos_mp4/AIBN_300_1_S2.MP4
```

```powershell
# Windows (PowerShell) -- note the backslash:
uv run python frontal_click_fit.py input_videos_mp4\AIBN_300_1_S2.MP4
```

Any video path OpenCV can decode works (`.mp4`, `.mov`, ...). Drop your videos in
`input_videos_mp4/` (or anywhere) and pass the path.

### What happens, in order

1. **Kymograph** — the video is decoded into a space-time image `L`. This is slow the first
   time and is cached to `output_kymographs_raw/`, so reruns of the same video are instant.
   Delete the cache `.npz` to force a rebuild.
2. **Calibration** — done in two overlay steps so a tilted tube doesn't skew the measured
   diameter. Both steps show the first video frame with a bright-cyan line you position, then
   press **Enter** to confirm:
   1. **Axis** — a near-vertical line. Put it on the **left** tube edge, matching the tube's
      lean. Drag the line to move it, or use **WASD**; rotate with **Q/E** (0.25° steps). This
      captures the tube's tilt.
   2. **Calibration arm** — a line locked **perpendicular to the axis**, extending to the
      **right**. Grow it until its tip reaches the **right** tube edge: drag the tip (or **D/A**
      to grow/shrink), and drag the body (or **W/S**) to slide it up/down the axis. A small
      right-angle marker shows the 90° corner. The arm's length is the tube diameter, measured
      straight across the tube regardless of tilt.

   The arm length is taken as the 10 mm tube diameter (change `TUBE_DIAMETER_MM` at the top of
   the script if yours differs). The tube tilt and line geometry are recorded in the results
   file.
3. **Click fit** — the kymograph appears (contrast-stretched so the front stands out).
   **Left-click** several points along the front; each click drops a red **+** and the window
   title keeps a running count. **Backspace** undoes the last point, **Enter** fits once you
   have at least 2 points, and closing the window skips the fit. Only a plain left-click adds a
   point — ctrl/two-finger (right) clicks, option-clicks, other keys, and clicks made while the
   toolbar's zoom or pan tool is on are all ignored, so you can zoom in to place points
   precisely (turn the tool off again before clicking). A least-squares line is fit through the
   clicks and drawn in red — every click influences it in proportion to how far it sits from
   the line, so undo any misclick with Backspace instead of leaving it in for the fit to average
   over.
4. **Results** — written to `output_data/results_<video>.txt` (line formula, slope in
   px/frame and mm/frame, front velocity in mm/s, calibration, tube tilt, line geometry) plus a
   `output_data/results_<video>.png` of the fit. Close the fit window (or press **Enter**) and
   the video replays at ~100× with the fitted front line overlaid (`q` or `Esc` to quit, space
   to pause).

## Notes

- Needs a graphical display — calibration and click-fit open interactive windows, so run it
  locally, not over a plain SSH session.
- Calibration uses keyboard **and** mouse: drag the lines, or nudge with WASD / Q-E / D-A. All
  the interactive controls are shown live in each window's title bar.
