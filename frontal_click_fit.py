import sys
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Generate the kymograph, calibrate px -> mm, click-fit the front, and report to
# output_data/results_<stem>.txt.

# Drop matplotlib's default bindings for the keys we drive the overlays with (s=save, q=quit,
# etc.), so WASD / Q-E reach our handlers cleanly. Backspace goes too: it's the click-fit undo,
# and as the toolbar's "back" key it would also jump the zoom back on every undo.
for _param, _keys in matplotlib.rcParams.items():
    if _param.startswith('keymap.') and isinstance(_keys, list):
        _keys[:] = [k for k in _keys if k not in ('a', 'd', 'w', 's', 'q', 'e', 'backspace')]

BLUR_KERNEL = 5
BAND_TOP_FRACTION = 0.25
BAND_BOTTOM_FRACTION = 0.75
BAND_LEFT_FRACTION = 0.25
BAND_RIGHT_FRACTION = 0.75
LINE_KSIZE = 3
KYMO_SMOOTH = 5
TUBE_DIAMETER_MM = 10.0
# Kymograph display contrast: map this percentile range of L to full black->white. A lower HI
# is more aggressive (faint front lines pop harder, but bright features may saturate).
KYMO_PCT_LO = 2
KYMO_PCT_HI = 95

video_path = sys.argv[1]
stem = os.path.splitext(os.path.basename(video_path))[0]
# Key the cache by source dir too, so same-named .mov / .MP4 sources don't collide.
src = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
cache = os.path.join('output_kymographs_raw', f'{src}__{stem}_L.npz')
os.makedirs('output_data', exist_ok=True)

# Open the video once, read its properties and first frame, keep the handle for later reuse.
cap = cv2.VideoCapture(video_path)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, frame0 = cap.read()
band_top = int(H * BAND_TOP_FRACTION)
band_bottom = int(H * BAND_BOTTOM_FRACTION)
band_left = int(W * BAND_LEFT_FRACTION)
band_right = int(W * BAND_RIGHT_FRACTION)
os.makedirs('output_kymographs_raw', exist_ok=True)

# --- 1. Kymograph ---
# The full-video decode is slow; cache the finished kymograph L so reruns load instantly.
# Delete the .npz to force a rebuild.
if os.path.exists(cache):
    print(f'Loading cached kymograph: {cache}')
    L = np.load(cache)['L']
else:
    print('Decoding video to build kymograph (cached after first run)...')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind past the first frame we already read
    columns = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (BLUR_KERNEL, BLUR_KERNEL), 0)
        band = gray[band_top:band_bottom, band_left:band_right].astype(np.float32)
        d2y = cv2.Sobel(band, cv2.CV_32F, 0, 2, ksize=LINE_KSIZE)
        columns.append(np.abs(d2y).mean(axis=1))
    L = np.array(columns, dtype=np.float32).T
    L = cv2.GaussianBlur(L, (KYMO_SMOOTH, KYMO_SMOOTH), 0)
    L = np.clip(L - np.median(L, axis=1, keepdims=True), 0, None)
    np.savez_compressed(cache, L=L)

n_frames = L.shape[1]

# Contrast stretch: map the KYMO_PCT_LO..KYMO_PCT_HI percentile of L to full black->white so the
# faint front lines stand out instead of being washed out by a few bright pixels. A tighter
# high percentile = more aggressive (faint lines pop harder). Reused by every kymo view.
kymo_vmin, kymo_vmax = np.percentile(L, [KYMO_PCT_LO, KYMO_PCT_HI])
if kymo_vmax <= kymo_vmin:                      # degenerate (flat) kymograph guard
    kymo_vmin, kymo_vmax = float(L.min()), float(L.max()) or 1.0
kymo_extent = [0, n_frames, band_bottom, band_top]


def show_kymo(axis_obj):
    """imshow L with the shared contrast + full-extent view, and pin the limits so the window
    always shows the whole kymograph (no stray zoom)."""
    axis_obj.imshow(L, cmap='gray_r', aspect='auto', origin='upper',
                    extent=kymo_extent, vmin=kymo_vmin, vmax=kymo_vmax)
    axis_obj.set_xlim(kymo_extent[0], kymo_extent[1])
    axis_obj.set_ylim(kymo_extent[2], kymo_extent[3])   # (band_bottom, band_top) -> origin upper


# Save the raw kymograph (no fit) as a PNG.
figk, axk = plt.subplots(figsize=(8, 5))
show_kymo(axk)
axk.set_xlabel('Frame')
axk.set_ylabel('Front position (px)')
axk.set_title(f'{stem} kymograph')
figk.tight_layout()
figk.savefig(os.path.join('output_kymographs_raw', f'{stem}_kymograph_raw.png'), dpi=150)
plt.close(figk)

# --- 2. Calibration: pixels -> mm from the tube diameter (10 mm) ---
# Overlay on the first frame so a tilted tube doesn't skew the diameter:
#   Step 1: a near-vertical AXIS line is driven onto the LEFT tube edge (drag or WASD to move,
#           Q/E to rotate) to capture the tube's true tilt. Its position + angle are saved.
#   Step 2: a CALIBRATION arm, locked perpendicular to the axis and extending right, is grown
#           (drag its tip, or D/A) until it touches the RIGHT tube edge. The arm's length is the
#           tube diameter in px, measured perpendicular to the tube, so it's tilt-correct.
frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
fh, fw = frame0.shape[:2]

MOVE_STEP = 4.0      # px per move/length press
ROT_STEP = 0.25      # deg per Q/E press
DRAG_GRAB_PX = 20.0  # how close (px) a click must be to a line to start dragging it


def _line_endpoints(cx, cy, angle_deg, length):
    """Two endpoints of a line centered at (cx, cy) at angle_deg from vertical (down = +y)."""
    a = np.radians(angle_deg)
    d = np.array([np.sin(a), np.cos(a)])          # direction along the line
    return (cx - d[0] * length / 2, cy - d[1] * length / 2,
            cx + d[0] * length / 2, cy + d[1] * length / 2)


def run_modal(fig, close=True):
    """Show fig and block until its handler calls fig._done() (bound below) or it's closed.
    Uses a private flag rather than close_event->stop cross-wiring, so the Enter that closes
    one window can't leak into the next window's loop and dismiss it instantly. close=False
    leaves the figure open afterwards (the click fit draws its result on the same window)."""
    fig._is_done = False
    fig.canvas.mpl_connect('close_event', lambda e: fig.canvas.stop_event_loop())
    fig.show()
    fig.canvas.flush_events()          # settle pending events before we start blocking
    while not fig._is_done:
        fig.canvas.start_event_loop()  # returns when a handler calls fig.canvas.stop_event_loop()
        if not plt.fignum_exists(fig.number):
            break                      # window was closed with the mouse
    if close:
        plt.close(fig)


def finish_modal(fig):
    """Mark a modal figure done and break its event loop (call from the Enter handler)."""
    fig._is_done = True
    fig.canvas.stop_event_loop()


# --- Step 1: drive the AXIS line onto the LEFT edge (WASD move, Q/E rotate). ---
axis = {'cx': fw * 0.5, 'cy': fh * 0.5, 'angle': 0.0}   # angle 0 = vertical
line_len = fh * 0.9

figa, axa = plt.subplots(figsize=(11, 6))
axa.imshow(frame_rgb)
(x0, y0, x1, y1) = _line_endpoints(axis['cx'], axis['cy'], axis['angle'], line_len)
(axis_line,) = axa.plot([x0, x1], [y0, y1], color='#00e5ff', lw=1.5)
axa.set_title('AXIS: WASD move, Q/E rotate onto the LEFT edge -- Enter when done')
axa.set_xlim(0, fw); axa.set_ylim(fh, 0)
print('Step 1 (AXIS): WASD to move, Q/E to rotate onto the LEFT tube edge. '
      'Enter when done.')


def _redraw_axis():
    ex = _line_endpoints(axis['cx'], axis['cy'], axis['angle'], line_len)
    axis_line.set_data([ex[0], ex[2]], [ex[1], ex[3]])
    axa.set_title(f"AXIS: cx={axis['cx']:.0f} cy={axis['cy']:.0f} "
                  f"angle={axis['angle']:.2f} deg -- drag to move, Q/E rotate, Enter when done")
    figa.canvas.draw_idle()


def _on_key_axis(event):
    base = event.key or ''
    if base in ('a', 'left'):
        axis['cx'] -= MOVE_STEP
    elif base in ('d', 'right'):
        axis['cx'] += MOVE_STEP
    elif base in ('w', 'up'):
        axis['cy'] -= MOVE_STEP
    elif base in ('s', 'down'):
        axis['cy'] += MOVE_STEP
    elif base == 'q':
        axis['angle'] -= ROT_STEP
    elif base == 'e':
        axis['angle'] += ROT_STEP
    elif base in ('enter', 'return'):
        finish_modal(figa)
        return
    else:
        return
    _redraw_axis()


# Drag the line BODY to translate the whole axis (rotation stays on Q/E).
_axis_drag = {'active': False, 'x': 0.0, 'y': 0.0}


def _dist_to_axis(px_, py_):
    """Perpendicular distance from (px_, py_) to the current axis line."""
    c = np.array([axis['cx'], axis['cy']])
    ad = np.array([np.sin(np.radians(axis['angle'])), np.cos(np.radians(axis['angle']))])
    v = np.array([px_, py_]) - c
    return abs(v[0] * ad[1] - v[1] * ad[0])   # |v x axis_dir|


def _on_press_axis(event):
    if event.inaxes is not axa or event.xdata is None:
        return
    if _dist_to_axis(event.xdata, event.ydata) <= DRAG_GRAB_PX:
        _axis_drag.update(active=True, x=event.xdata, y=event.ydata)


def _on_motion_axis(event):
    if not _axis_drag['active'] or event.inaxes is not axa or event.xdata is None:
        return
    axis['cx'] += event.xdata - _axis_drag['x']
    axis['cy'] += event.ydata - _axis_drag['y']
    _axis_drag['x'], _axis_drag['y'] = event.xdata, event.ydata
    _redraw_axis()


def _on_release_axis(event):
    _axis_drag['active'] = False


figa.canvas.mpl_connect('key_press_event', _on_key_axis)
figa.canvas.mpl_connect('button_press_event', _on_press_axis)
figa.canvas.mpl_connect('motion_notify_event', _on_motion_axis)
figa.canvas.mpl_connect('button_release_event', _on_release_axis)
run_modal(figa)

tilt_deg = axis['angle']
a = np.radians(axis['angle'])
axis_dir = np.array([np.sin(a), np.cos(a)])       # unit vector along the axis
n_hat = np.array([axis_dir[1], -axis_dir[0]])     # unit perpendicular pointing RIGHT (+x)
print(f'Tube tilt from vertical: {tilt_deg:.2f} deg  '
      f"(axis at cx={axis['cx']:.1f}, cy={axis['cy']:.1f})")

# --- Step 2: the CALIBRATION arm -- perpendicular to the axis, anchored on it, extending
# right. D/A grow/shrink the arm, W/S slide its anchor up/down the axis. No rotation: it stays
# locked perpendicular. Grow it until the tip touches the RIGHT edge; the arm LENGTH = diameter.
calib = {'along': 0.0, 'length': 40.0}   # along = signed offset of anchor from axis center


RIGHT_ANGLE_PX = 18.0   # size of the little square marking the 90 deg corner


def _calib_arm():
    """Anchor point on the axis and the arm tip (anchor + length along +n_hat)."""
    anchor = np.array([axis['cx'], axis['cy']]) + axis_dir * calib['along']
    tip = anchor + n_hat * calib['length']
    return anchor, tip


def _right_angle_square(anchor):
    """L-shaped path marking the 90 deg corner: one leg down the axis, one along the arm."""
    s = min(RIGHT_ANGLE_PX, calib['length']) if calib['length'] > 0 else RIGHT_ANGLE_PX
    p_axis = anchor + axis_dir * s     # step along the axis
    p_arm = anchor + n_hat * s         # step along the arm
    corner = p_axis + n_hat * s        # far corner of the square
    xs = [p_axis[0], corner[0], p_arm[0]]
    ys = [p_axis[1], corner[1], p_arm[1]]
    return xs, ys


figc, axc = plt.subplots(figsize=(11, 6))
axc.imshow(frame_rgb)
(ax0, ay0, ax1, ay1) = _line_endpoints(axis['cx'], axis['cy'], axis['angle'], line_len)
axc.plot([ax0, ax1], [ay0, ay1], color='#00e5ff', lw=1.0, ls='--')   # the fixed axis (left edge)
anchor, tip = _calib_arm()
(calib_line,) = axc.plot([anchor[0], tip[0]], [anchor[1], tip[1]], color='#00e5ff', lw=1.5)
rsx, rsy = _right_angle_square(anchor)
(right_angle,) = axc.plot(rsx, rsy, color='lime', lw=1.2)   # 90 deg marker at the corner
axc.set_title('CALIB: D/A length, W/S up-down -- extend tip to the RIGHT edge, Enter when done')
axc.set_xlim(0, fw); axc.set_ylim(fh, 0)
print('Step 2 (CALIB): D/A grow-shrink the arm, W/S slide it up-down the axis. '
      'Extend the tip to the RIGHT tube edge. Enter when done.')


def _redraw_calib():
    anchor, tip = _calib_arm()
    calib_line.set_data([anchor[0], tip[0]], [anchor[1], tip[1]])
    rsx, rsy = _right_angle_square(anchor)
    right_angle.set_data(rsx, rsy)
    axc.set_title(f"CALIB: length={calib['length']:.1f} px "
                  f"= {calib['length'] / TUBE_DIAMETER_MM:.2f} px/mm -- "
                  "drag tip=length, drag body=slide, Enter when done")
    figc.canvas.draw_idle()


def _on_key_calib(event):
    base = event.key or ''
    if base in ('d', 'right'):
        calib['length'] += MOVE_STEP
    elif base in ('a', 'left'):
        calib['length'] = max(0.0, calib['length'] - MOVE_STEP)
    elif base in ('w', 'up'):
        calib['along'] -= MOVE_STEP
    elif base in ('s', 'down'):
        calib['along'] += MOVE_STEP
    elif base in ('enter', 'return'):
        finish_modal(figc)
        return
    else:
        return
    _redraw_calib()


# Drag the TIP to set the arm length (diameter), or the BODY to slide the anchor along the axis.
_calib_drag = {'mode': None}   # 'tip' | 'body' | None


def _set_calib_from_point(px_, py_):
    """Project a cursor point onto the axis frame: 'along' = component along the axis from the
    axis center, 'length' = component along n_hat (perpendicular). Used by both drag modes."""
    v = np.array([px_, py_]) - np.array([axis['cx'], axis['cy']])
    return float(np.dot(v, axis_dir)), float(np.dot(v, n_hat))


def _on_press_calib(event):
    if event.inaxes is not axc or event.xdata is None:
        return
    anchor, tip = _calib_arm()
    p = np.array([event.xdata, event.ydata])
    if np.linalg.norm(p - tip) <= DRAG_GRAB_PX:
        _calib_drag['mode'] = 'tip'
    elif np.linalg.norm(p - (anchor + tip) / 2) <= DRAG_GRAB_PX or \
            np.linalg.norm(p - anchor) <= DRAG_GRAB_PX:
        _calib_drag['mode'] = 'body'


def _on_motion_calib(event):
    if _calib_drag['mode'] is None or event.inaxes is not axc or event.xdata is None:
        return
    along, length = _set_calib_from_point(event.xdata, event.ydata)
    if _calib_drag['mode'] == 'tip':
        calib['length'] = max(0.0, length)   # tip drag sets length; anchor (along) unchanged
    else:
        calib['along'] = along               # body drag slides anchor; length unchanged
    _redraw_calib()


def _on_release_calib(event):
    _calib_drag['mode'] = None


figc.canvas.mpl_connect('key_press_event', _on_key_calib)
figc.canvas.mpl_connect('button_press_event', _on_press_calib)
figc.canvas.mpl_connect('motion_notify_event', _on_motion_calib)
figc.canvas.mpl_connect('button_release_event', _on_release_calib)
run_modal(figc)

# Diameter = length of the calibration arm (axis/left edge -> tip on the right edge).
width_px = abs(calib['length'])
px_per_mm = width_px / TUBE_DIAMETER_MM
print(f'Tube width = {width_px:.1f} px over {TUBE_DIAMETER_MM} mm  ->  '
      f'{px_per_mm:.3f} px/mm  ({1 / px_per_mm:.4f} mm/px)\n')

# --- 3. Click fit ---
fig, ax = plt.subplots(figsize=(11, 6))
show_kymo(ax)
ax.set_xlabel('Frame')
ax.set_ylabel('Front position (px)')
print('Left-click points along the front (the window title counts them). Backspace undoes, '
      'Enter fits (2+ points), closing the window skips.')
# Collect the clicks ourselves rather than with plt.ginput: on the MacOSX backend ginput takes a
# ctrl-click / two-finger click as "delete the previous point", an option-click or Esc as
# "done", and any other key (even a bare Shift) or a zoom/pan press as a new point -- so points
# vanished or appeared without you noticing. Here only a plain left-click adds a point, only
# backspace/delete undoes, and only Enter finishes.
clicks = []   # (frame, position_px) per accepted click
(click_marks,) = ax.plot([], [], 'r+', ms=10, mew=1.5, zorder=5)


def _redraw_clicks():
    click_marks.set_data([c[0] for c in clicks], [c[1] for c in clicks])
    ax.set_title(f'{len(clicks)} point(s) -- left-click the front, Backspace undo, '
                 'Enter to fit (2+ points)')
    fig.canvas.draw_idle()


def _on_press_click(event):
    # Left button only (ctrl/two-finger clicks arrive as right, option-clicks as middle); skip
    # the repeat press of a double-click and presses while zoom/pan holds the canvas lock.
    if (event.inaxes is not ax or event.xdata is None or event.button != 1 or event.dblclick
            or fig.canvas.widgetlock.locked()):
        return
    clicks.append((event.xdata, event.ydata))
    _redraw_clicks()


def _on_key_click(event):
    if event.key in ('backspace', 'delete'):
        if clicks:
            clicks.pop()
            _redraw_clicks()
    elif event.key in ('enter', 'return'):
        # A line needs points at 2+ frames -- which also means a stale Enter left over from the
        # CALIB window can't end this one before you've clicked anything.
        if len({c[0] for c in clicks}) >= 2:
            finish_modal(fig)


click_cids = [fig.canvas.mpl_connect('button_press_event', _on_press_click),
              fig.canvas.mpl_connect('key_press_event', _on_key_click)]
_redraw_clicks()
run_modal(fig, close=False)          # stays open: the fit is drawn on this same window below
for cid in click_cids:
    fig.canvas.mpl_disconnect(cid)   # the result view must not take clicks / undo
click_marks.remove()                 # replaced by the fit's red dots
pts = clicks

if not plt.fignum_exists(fig.number) or len({x for x, _ in pts}) < 2:
    print(f'\nClick window closed with {len(pts)} front point(s) -- need at least 2 (at '
          'different frames) and Enter to fit a line. Skipping fit and video replay.')
    sys.exit(0)

px = np.array([p[0] for p in pts])
py = np.array([p[1] for p in pts])   # already in position-px (axis) coords
# Least-squares fit: every click pulls on the line in proportion to its residual.
m, b = np.polyfit(px, py, 1)
print(f'\nMANUAL (your clicks): slope = {m:.4f} px/frame  '
      f'= {m / px_per_mm:.5f} mm/frame  '
      f'= {m / px_per_mm * fps:.5f} mm/s  '
      f'(fps={fps:.3f}, {len(pts)} points)\n')

xf = np.array([0, n_frames])
ax.plot(xf, m * xf + b, 'r-', lw=2.5, label=f'front: {m:.4f} px/frame')
ax.scatter(px, py, c='r', s=40, zorder=5)
ax.legend(fontsize=8)
ax.set_title(f'{stem}: click fit  --  close this window (or press Enter) to play the demo video')
# Re-pin to the full kymograph so the saved PNG shows everything, even if you zoomed while
# clicking (the fit line's endpoints could otherwise re-expand the view).
ax.set_xlim(kymo_extent[0], kymo_extent[1])
ax.set_ylim(kymo_extent[2], kymo_extent[3])

# --- 4. Report ---
img_path = os.path.join('output_data', f'results_{stem}.png')
txt_path = os.path.join('output_data', f'results_{stem}.txt')
fig.savefig(img_path, dpi=150, bbox_inches='tight')
with open(txt_path, 'w') as f:
    f.write(f'video: {stem}\n')
    f.write(f'line: position_px = {m:.6f} * frame + {b:.6f}\n')
    f.write(f'slope: {m:.6f} px/frame = {m / px_per_mm:.6f} mm/frame\n')
    f.write(f'front velocity: {m / px_per_mm * fps:.6f} mm/s\n')
    f.write(f'calibration: {px_per_mm:.6f} px/mm   fps: {fps:.6f}\n')
    f.write(f'tube tilt from vertical: {tilt_deg:.3f} deg\n')
    ax0, ay0, ax1, ay1 = _line_endpoints(axis['cx'], axis['cy'], axis['angle'], line_len)
    f.write(f"axis line: center=({axis['cx']:.3f}, {axis['cy']:.3f}) "
            f"angle={axis['angle']:.3f} deg "
            f"endpoints=({ax0:.3f}, {ay0:.3f})->({ax1:.3f}, {ay1:.3f})\n")
    f.write(f"calib arm: length (perpendicular to axis) = {width_px:.3f} px "
            f"anchored at along={calib['along']:.3f} px "
            f"(= tube diameter {TUBE_DIAMETER_MM} mm)\n")
    f.write(f'\npoints (frame, position_px):\n')
    for x, y in zip(px, py):
        f.write(f'{x:.3f}\t{y:.3f}\n')
print(f'Saved {img_path} and {txt_path}')
# Let Enter close the fit figure too (not just the window's X), then the demo video plays.
fig.canvas.mpl_connect('key_press_event',
                       lambda e: plt.close(fig) if e.key in ('enter', 'return') else None)
plt.show()

# --- Video replay with the fitted front line overlaid ---
# Front position (px) = m * frame + b, already in full-frame row coords (clicks were in
# position-px). SEEKS directly to every PLAYBACK_SPEED-th frame instead of decoding all of them
# -- decoding the 99 skipped frames was what made the old replay slow. Press q or Esc to quit,
# space to pause.
PLAYBACK_SPEED = 100
paused = False
frame_idx = 0
while frame_idx < (total_frames or n_frames):
    if not paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)   # seek -- skips decoding the frames between
        ret, frame = cap.read()
        if not ret:
            break
        y = int(round(m * frame_idx + b))
        if 0 <= y < frame.shape[0]:
            cv2.line(frame, (0, y), (frame.shape[1], y), (0, 0, 255), 2)
        cv2.putText(frame, f'frame {frame_idx}  front y={y}  ({PLAYBACK_SPEED}x)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('front click fit overlay (q quit, space pause)', frame)
        frame_idx += PLAYBACK_SPEED
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if key == ord(' '):
        paused = not paused
cap.release()
cv2.destroyAllWindows()
