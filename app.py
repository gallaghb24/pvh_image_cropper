import streamlit as st
from PIL import Image
import json, pandas as pd
from io import BytesIO
import zipfile
from typing import List, Tuple
import cv2
import numpy as np
import math

# ——————————————————————————————————————————————————————————
# Helper functions
# ——————————————————————————————————————————————————————————
def compute_crop(rec: dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Compute guideline crop rectangle in pixel coordinates."""
    off, fr = rec["imageOffset"], rec["frame"]
    fx, fy, fw, fh = abs(off["x"]), abs(off["y"]), off["w"], off["h"]
    l = int((fx / fw) * img_w)
    t = int((fy / fh) * img_h)
    w = int((fr["w"] / fw) * img_w)
    h = int((fr["h"] / fh) * img_h)
    tgt_ar = fr["w"] / fr["h"]
    cur_ar = w / h if h else tgt_ar
    if abs(cur_ar - tgt_ar) > 1e-3:
        if cur_ar > tgt_ar:  # too wide
            new_w = int(h * tgt_ar)
            l += (w - new_w) // 2
            w = new_w
        else:  # too tall
            new_h = int(w / tgt_ar)
            t += (h - new_h) // 2
            h = new_h
    return l, t, w, h

def auto_custom_start(rec: dict, img_w: int, img_h: int, cw: int, ch: int) -> Tuple[int, int, int, int]:
    """Start from guideline crop, then adapt to target aspect by trimming height only (keep width)."""
    l, t, w, h = compute_crop(rec, img_w, img_h)
    new_h = int(w / (cw / ch))
    t += (h - new_h) // 2
    return l, t, w, new_h

def boxes_intersect(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def face_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def choose_face_for_window(faces: List[Tuple[int,int,int,int]], l:int, t:int, w:int, h:int):
    """Prefer a face that intersects the current window; otherwise pick the face nearest the window center."""
    if not faces:
        return None
    window = (l, t, l + w, t + h)
    cx, cy = l + w / 2.0, t + h / 2.0

    intersecting = [f for f in faces if boxes_intersect(window, f)]
    pool = intersecting if intersecting else faces

    def score(box):
        fx, fy = face_center(box)
        return (fx - cx) ** 2 + (fy - cy) ** 2  # smaller is better

    def area(box):
        x1,y1,x2,y2 = box
        return (x2-x1)*(y2-y1)

    pool.sort(key=lambda b: (score(b), -area(b)))
    return pool[0]

def adjust_crop_to_include_face(l, t, w, h, face_box, iw, ih):
    """Shift crop window minimally so the face box fits inside (if possible). Only shifts; never resizes/zooms."""
    if face_box is None:
        return l, t, w, h
    fx1, fy1, fx2, fy2 = face_box
    crop_x1, crop_y1, crop_x2, crop_y2 = l, t, l + w, t + h
    if (fx1 >= crop_x1 and fx2 <= crop_x2 and fy1 >= crop_y1 and fy2 <= crop_y2):
        return l, t, w, h
    shift_x = 0
    shift_y = 0
    if fx1 < crop_x1:
        shift_x = fx1 - crop_x1
    elif fx2 > crop_x2:
        shift_x = fx2 - crop_x2
    if fy1 < crop_y1:
        shift_y = fy1 - crop_y1
    elif fy2 > crop_y2:
        shift_y = fy2 - crop_y2
    new_l = min(max(l + shift_x, 0), iw - w)
    new_t = min(max(t + shift_y, 0), ih - h)
    return new_l, new_t, w, h

def clamp_crop_from_center(cx, cy, wz, hz, iw, ih, sx, sy):
    """Compute top-left from center + offsets and clamp to image bounds."""
    l2 = max(0, min(cx - wz // 2 + sx, iw - wz))
    t2 = max(0, min(cy - hz // 2 + sy, ih - hz))
    return l2, t2

# ——————————————————————————————————————————————————————————
# Streamlit page config
# ——————————————————————————————————————————————————————————
st.set_page_config(page_title="Smart Crop Automation Prototype", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>[data-testid='stSidebar']{min-width:450px!important;max-width:450px!important;}</style>", unsafe_allow_html=True)
st.title("Smart Crop Automation Prototype")

# ——————————————————————————————————————————————————————————
# Sidebar inputs
# ——————————————————————————————————————————————————————————
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader("Master Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

with st.sidebar.expander("⚙️ Custom Crops", expanded=True):
    df_editor = st.data_editor(pd.DataFrame([{"Width_px": None, "Height_px": None}]), hide_index=True, num_rows="dynamic")
    custom_sizes: List[Tuple[int, int]] = [(int(r.Width_px), int(r.Height_px)) for r in df_editor.itertuples() if pd.notna(r.Width_px) and pd.notna(r.Height_px)]

# ——————————————————————————————————————————————————————————
# Load matching guideline records
# ——————————————————————————————————————————————————————————
records: List[dict] = []
if json_file and image_file:
    data = json.load(json_file)
    fname = image_file.name
    for rec in data:
        spec = rec.get("filename") or rec.get("fileName") or rec.get("asset")
        if spec and (fname == spec or fname.startswith(spec)):
            records.append(rec)
else:
    st.info("Upload JSON and image to begin.")

if not records:
    st.stop()

guidelines = [r for r in records if not (abs(r["imageOffset"]["x"]) < 1e-6 and abs(r["imageOffset"]["y"]) < 1e-6)]

# ——————————————————————————————————————————————————————————
# Display guideline + custom tables
# ——————————————————————————————————————————————————————————
st.subheader("Guideline Crops")
gtable = []
for rec in guidelines:
    w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
    ex, ey = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
    w_px, h_px = int(w_pt * ex / 72), int(h_pt * ey / 72)
    gtable.append({"Template": rec["template"], "Width_px": w_px, "Height_px": h_px, "AR": round(w_px / h_px, 2)})
st.dataframe(pd.DataFrame(gtable), use_container_width=True)

st.subheader("Custom Crops")
ctable = []
for cw, ch in custom_sizes:
    ref = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
    ctable.append({"Template": f"{ref['template']} {cw}×{ch}", "Width_px": cw, "Height_px": ch, "AR": round(cw / ch, 2)})
st.dataframe(pd.DataFrame(ctable), use_container_width=True)

# ——————————————————————————————————————————————————————————
# Load master image + FACE DETECTION
# ——————————————————————————————————————————————————————————
img = Image.open(image_file)
icc_profile = img.info.get("icc_profile")
iw, ih = img.size

img_cv = np.array(img.convert("RGB"))
gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces_np = casc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
faces: List[Tuple[int,int,int,int]] = []
if len(faces_np) > 0:
    for (x, y, w, h) in faces_np:
        faces.append((int(x), int(y), int(x + w), int(y + h)))  # x1,y1,x2,y2

# ——————————————————————————————————————————————————————————
# Custom adjustment UI (FaceAware + WYSIWYG + robust sliders)
# ——————————————————————————————————————————————————————————
shifts = {}
if custom_sizes:
    st.subheader("Adjust Custom Crops")
    tabs = st.tabs([f"{w}×{h}" for w, h in custom_sizes])

    for ((cw, ch), tab) in zip(custom_sizes, tabs):
        # ---------- use size-based keys (stable no matter the tab order) ----------
        key_root = f"{cw}x{ch}"
        base_key = f"base_{key_root}"
        z_key   = f"zoom_{key_root}"
        sx_key  = f"sx_{key_root}"
        sy_key  = f"sy_{key_root}"

        with tab:
            # Persist the post–FaceAware base window per size (WYSIWYG)
            if base_key not in st.session_state:
                base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
                l0, t0, wb, hb = auto_custom_start(base, iw, ih, cw, ch)
                chosen_face = choose_face_for_window(faces, l0, t0, wb, hb)
                l0, t0, wb, hb = adjust_crop_to_include_face(l0, t0, wb, hb, chosen_face, iw, ih)
                st.session_state[base_key] = (l0, t0, wb, hb)
            else:
                l0, t0, wb, hb = st.session_state[base_key]

            # UI state defaults (per size)
            st.session_state.setdefault(z_key, 0)   # zoom delta in %
            st.session_state.setdefault(sx_key, 0)  # width offset
            st.session_state.setdefault(sy_key, 0)  # height offset

            # Read current UI state
            zd = int(st.session_state[z_key])
            sx = int(st.session_state[sx_key])
            sy = int(st.session_state[sy_key])

            # Effective zoom: never allow window larger than image
            zoom_floor = max(wb / iw, hb / ih, 1.0)  # >=1
            zoom_user = 1 + zd / 100.0               # user intent [-10%, +10%]
            zoom = max(zoom_floor, zoom_user)        # enforce floor silently

            # Crop window derived from base + zoom
            wz, hz = int(wb / zoom), int(hb / zoom)
            cx, cy = l0 + wb // 2, t0 + hb // 2

            # Compute legal movement ranges
            min_x = -cx + wz // 2
            max_x =  iw - (cx + wz // 2)
            min_y = -cy + hz // 2
            max_y =  ih - (cy + hz // 2)

            # Clamp current offsets into legal ranges BEFORE rendering
            sx = max(min_x, min(sx, max_x))
            sy = max(min_y, min(sy, max_y))

            # Preview (clamped exactly like export)
            l2, t2 = clamp_crop_from_center(cx, cy, wz, hz, iw, ih, sx, sy)
            prev = img.crop((l2, t2, l2 + wz, t2 + hz)).resize((cw, ch))
            st.image(prev, caption=f"Preview {cw}×{ch}", use_container_width=True)

            # Height offset controls
            colh1, colh2 = st.columns([3, 1])
            if min_y < max_y:
                with colh1:
                    sy = st.slider("Height Offset", min_y, max_y, sy, 1, key=f"syslider_{key_root}")
                with colh2:
                    sy = st.number_input("Height", min_y, max_y, sy, 1, key=f"synum_{key_root}", label_visibility="collapsed")
            else:
                sy = 0  # no movement possible
            st.session_state[sy_key] = int(sy)

            # Width offset controls
            colw1, colw2 = st.columns([3, 1])
            if min_x < max_x:
                with colw1:
                    sx = st.slider("Width Offset", min_x, max_x, sx, 1, key=f"sxslider_{key_root}")
                with colw2:
                    sx = st.number_input("Width", min_x, max_x, sx, 1, key=f"sxnum_{key_root}", label_visibility="collapsed")
            else:
                sx = 0  # no movement possible
            st.session_state[sx_key] = int(sx)

            # Zoom (±10%), applied with floor during computation
            colz1, colz2 = st.columns([3, 1])
            with colz1:
                zd = st.slider("Zoom ±10%", -10, 10, zd, 1, key=f"zslider_{key_root}")
            with colz2:
                zd = st.number_input("Zoom%", -10, 10, zd, 1, key=f"znum_{key_root}", label_visibility="collapsed")
            st.session_state[z_key] = int(zd)

            # Persist for export (use the same base window + current UI)
            shifts[(cw, ch)] = (
                int(st.session_state[sx_key]),
                int(st.session_state[sy_key]),
                max(zoom_floor, 1 + int(st.session_state[z_key]) / 100.0)
            )

# ——————————————————————————————————————————————————————————
# Generate ZIP (JPEG with ICC); uses the SAME persisted base windows
# ——————————————————————————————————————————————————————————
zip_buf = BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    # Guideline crops (no face-aware by design)
    for rec in guidelines:
        ow = int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72)
        oh = int(rec["frame"]["h"] * rec["effectivePpi"]["y"] / 72)
        l, t, wb, hb = compute_crop(rec, iw, ih)
        gcrop = img.crop((l, t, l + wb, t + hb)).resize((ow, oh))
        tmp = BytesIO()
        save_kwargs = {"quality": 95, "subsampling": 0}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        gcrop.save(tmp, format="JPEG", **save_kwargs)
        tmp.seek(0)
        zf.writestr(f"Guidelines/{rec['template']}_{ow}x{oh}.jpg", tmp.getvalue())

    # Custom crops (FaceAware + WYSIWYG)
    for cw, ch in custom_sizes:
        key_root = f"{cw}x{ch}"
        base_key = f"base_{key_root}"
        if base_key in st.session_state:
            l0, t0, wb, hb = st.session_state[base_key]
        else:
            base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
            l0, t0, wb, hb = auto_custom_start(base, iw, ih, cw, ch)
            chosen_face = choose_face_for_window(faces, l0, t0, wb, hb)
            l0, t0, wb, hb = adjust_crop_to_include_face(l0, t0, wb, hb, chosen_face, iw, ih)

        sx, sy, zoom = shifts.get((cw, ch), (0, 0, 1.0))
        wz, hz = int(wb / zoom), int(hb / zoom)
        cx, cy = l0 + wb // 2, t0 + hb // 2
        l2, t2 = clamp_crop_from_center(cx, cy, wz, hz, iw, ih, sx, sy)
        ccrop = img.crop((l2, t2, l2 + wz, t2 + hz)).resize((cw, ch))

        tmp = BytesIO()
        save_kwargs = {"quality": 95, "subsampling": 0}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        ccrop.save(tmp, format="JPEG", **save_kwargs)
        tmp.seek(0)
        zf.writestr(f"Custom/{cw}x{ch}.jpg", tmp.getvalue())

zip_buf.seek(0)
st.download_button("Download Crops", zip_buf.getvalue(), file_name=f"crops_{image_file.name}.zip", mime="application/zip")
