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
    # Keep guideline aspect as best as possible within rounding
    tgt_ar = fr["w"] / fr["h"]
    cur_ar = w / h if h else tgt_ar
    if abs(cur_ar - tgt_ar) > 1e-3:
        if cur_ar > tgt_ar:  # too wide
            new_w = int(round(h * tgt_ar))
            l += (w - new_w) // 2
            w = new_w
        else:  # too tall
            new_h = int(round(w / tgt_ar))
            t += (h - new_h) // 2
            h = new_h
    return l, t, w, h

def boxes_intersect(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def face_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def choose_face_for_window(faces: List[Tuple[int,int,int,int]], l:int, t:int, w:int, h:int):
    """Prefer a face that intersects the window; otherwise pick nearest to window center."""
    if not faces:
        return None
    window = (l, t, l + w, t + h)
    cx, cy = l + w / 2.0, t + h / 2.0
    intersecting = [f for f in faces if boxes_intersect(window, f)]
    pool = intersecting if intersecting else faces

    def score(box):
        fx, fy = face_center(box)
        return (fx - cx) ** 2 + (fy - cy) ** 2

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
    x1, y1, x2, y2 = l, t, l + w, t + h
    if (fx1 >= x1 and fx2 <= x2 and fy1 >= y1 and fy2 <= y2):
        return l, t, w, h
    shift_x = 0
    shift_y = 0
    if fx1 < x1: shift_x = fx1 - x1
    elif fx2 > x2: shift_x = fx2 - x2
    if fy1 < y1: shift_y = fy1 - y1
    elif fy2 > y2: shift_y = fy2 - y2
    new_l = min(max(l + shift_x, 0), iw - w)
    new_t = min(max(t + shift_y, 0), ih - h)
    return new_l, new_t, w, h

def clamp_crop_from_center(cx, cy, wz, hz, iw, ih, sx, sy):
    """Compute top-left from center + offsets and clamp to image bounds."""
    l2 = max(0, min(cx - wz // 2 + sx, iw - wz))
    t2 = max(0, min(cy - hz // 2 + sy, ih - hz))
    return l2, t2

def minimal_outside_rect_containing(w, h, ar):
    """
    Minimal rectangle of aspect 'ar' that contains a w×h rectangle.
    Returns (W,H).
    """
    # candidate widen by width
    H1 = max(h, math.ceil(w / ar))
    W1 = math.ceil(H1 * ar)
    # candidate widen by height
    W2 = max(w, math.ceil(h * ar))
    H2 = math.ceil(W2 / ar)
    # choose minimal area
    if W1 * H1 <= W2 * H2:
        return int(W1), int(H1)
    return int(W2), int(H2)

def maximal_inside_rect(w, h, ar):
    """
    Maximal rectangle of aspect 'ar' that fits inside a w×h rectangle.
    Returns (W,H).
    """
    if ar >= w / h:
        W = w
        H = int(round(W / ar))
    else:
        H = h
        W = int(round(H * ar))
    return int(W), int(H)

def centered_custom_base(rec: dict, img_w: int, img_h: int, cw: int, ch: int) -> Tuple[int,int,int,int]:
    """
    Start from guideline, then:
      - if custom area <= guideline output area -> INSIDE (trim), centered
      - else -> OUTSIDE (expand), centered (then clamped to image bounds)
    """
    l, t, w, h = compute_crop(rec, img_w, img_h)
    cx, cy = l + w // 2, t + h // 2
    ar = cw / ch

    # Compare target area vs guideline output area using effectivePpi
    ex, ey = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
    out_w = int(rec["frame"]["w"] * ex / 72)
    out_h = int(rec["frame"]["h"] * ey / 72)
    target_area = cw * ch
    guide_area  = out_w * out_h

    if target_area <= guide_area:
        W, H = maximal_inside_rect(w, h, ar)          # trim equally (centered)
    else:
        W, H = minimal_outside_rect_containing(w, h, ar)  # add around (centered)

    # Center on guideline center, clamp to image bounds
    L = max(0, min(int(round(cx - W / 2)), img_w - W))
    T = max(0, min(int(round(cy - H / 2)), img_h - H))
    return L, T, int(W), int(H)

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
# Tables
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
# Load image + faces
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
# UI: per-size FaceAware toggle, robust widgets, WYSIWYG
# ——————————————————————————————————————————————————————————
shifts = {}
if custom_sizes:
    st.subheader("Adjust Custom Crops")
    tabs = st.tabs([f"{w}×{h}" for w, h in custom_sizes])

    for ((cw, ch), tab) in zip(custom_sizes, tabs):
        key_root = f"{cw}x{ch}"
        base_key = f"base_{key_root}"
        fa_key   = f"face_{key_root}"
        z_key    = f"zoom_{key_root}"
        sx_key   = f"sx_{key_root}"
        sy_key   = f"sy_{key_root}"

        with tab:
            # Face-aware toggle per custom size
            face_on = st.checkbox("Face-aware for this size", value=True, key=fa_key)

            # Recompute base if not present OR if the toggle changed this run
            rec = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
            recompute_base = True
            if base_key in st.session_state and f"{base_key}_fa" in st.session_state:
                recompute_base = (st.session_state[f"{base_key}_fa"] != face_on)

            if recompute_base or base_key not in st.session_state:
                l0, t0, wb, hb = centered_custom_base(rec, iw, ih, cw, ch)
                if face_on:
                    chosen_face = choose_face_for_window(faces, l0, t0, wb, hb)
                    l0, t0, wb, hb = adjust_crop_to_include_face(l0, t0, wb, hb, chosen_face, iw, ih)
                st.session_state[base_key] = (l0, t0, wb, hb)
                st.session_state[f"{base_key}_fa"] = face_on
            else:
                l0, t0, wb, hb = st.session_state[base_key]

            # Defaults
            st.session_state.setdefault(z_key, 0)
            st.session_state.setdefault(sx_key, 0)
            st.session_state.setdefault(sy_key, 0)

            # ——— Widgets first ———
            # Zoom floor: keep window <= image
            zoom_floor = max(wb / iw, hb / ih, 1.0)
            zd_prev = int(st.session_state[z_key])
            zd_slider = st.slider("Zoom ±10%", -10, 10, zd_prev, 1, key=f"zslider_{key_root}")
            zd_num    = st.number_input("Zoom%", -10, 10, zd_slider, 1, key=f"znum_{key_root}", label_visibility="collapsed")
            zd = int(zd_num if zd_num != zd_slider else zd_slider)
            st.session_state[z_key] = zd

            zoom_user = 1 + zd / 100.0
            zoom = max(zoom_floor, zoom_user)

            # Derive window from base + zoom
            wz, hz = int(wb / zoom), int(hb / zoom)
            cx, cy = l0 + wb // 2, t0 + hb // 2

            # Legal movement ranges
            min_x = -cx + wz // 2
            max_x =  iw - (cx + wz // 2)
            min_y = -cy + hz // 2
            max_y =  ih - (cy + hz // 2)

            sx_prev = int(st.session_state[sx_key])
            sy_prev = int(st.session_state[sy_key])

            # Width controls
            if min_x < max_x:
                sx_slider = st.slider("Width Offset", min_x, max_x, max(min_x, min(sx_prev, max_x)), 1, key=f"sxslider_{key_root}")
                sx_num    = st.number_input("Width", min_x, max_x, sx_slider, 1, key=f"sxnum_{key_root}", label_visibility="collapsed")
                sx = int(sx_num if sx_num != sx_slider else sx_slider)
            else:
                sx = 0
            st.session_state[sx_key] = sx

            # Height controls
            if min_y < max_y:
                sy_slider = st.slider("Height Offset", min_y, max_y, max(min_y, min(sy_prev, max_y)), 1, key=f"syslider_{key_root}")
                sy_num    = st.number_input("Height", min_y, max_y, sy_slider, 1, key=f"synum_{key_root}", label_visibility="collapsed")
                sy = int(sy_num if sy_num != sy_slider else sy_slider)
            else:
                sy = 0
            st.session_state[sy_key] = sy

            # Preview (compute after widgets; clamp)
            l2, t2 = clamp_crop_from_center(cx, cy, wz, hz, iw, ih, sx, sy)
            prev = img.crop((l2, t2, l2 + wz, t2 + hz)).resize((cw, ch))
            st.image(prev, caption=f"Preview {cw}×{ch}", use_container_width=True)

            # Persist for export
            shifts[(cw, ch)] = (sx, sy, zoom, face_on)

# ——————————————————————————————————————————————————————————
# Generate ZIP (JPEG with ICC); uses SAME persisted base windows
# ——————————————————————————————————————————————————————————
zip_buf = BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    # Guideline crops
    for rec in guidelines:
        ow = int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72)
        oh = int(rec["frame"]["h"] * rec["effectivePpi"]["y"] / 72)
        l, t, wb, hb = compute_crop(rec, iw, ih)
        gcrop = img.crop((l, t, l + wb, t + hb)).resize((ow, oh))
        tmp = BytesIO()
        save_kwargs = {"quality": 95, "subsampling": 0}
        icc = img.info.get("icc_profile")
        if icc: save_kwargs["icc_profile"] = icc
        gcrop.save(tmp, format="JPEG", **save_kwargs)
        tmp.seek(0)
        zf.writestr(f"Guidelines/{rec['template']}_{ow}x{oh}.jpg", tmp.getvalue())

    # Custom crops (WYSIWYG + per-size Face-aware)
    for cw, ch in custom_sizes:
        key_root = f"{cw}x{ch}"
        base_key = f"base_{key_root}"
        fa_on = st.session_state.get(f"face_{key_root}", True)

        rec = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))

        if base_key in st.session_state:
            l0, t0, wb, hb = st.session_state[base_key]
        else:
            # Fallback: recompute with current toggle
            l0, t0, wb, hb = centered_custom_base(rec, iw, ih, cw, ch)
            if fa_on:
                chosen_face = choose_face_for_window(faces, l0, t0, wb, hb)
                l0, t0, wb, hb = adjust_crop_to_include_face(l0, t0, wb, hb, chosen_face, iw, ih)

        sx, sy, zoom, _fa = shifts.get((cw, ch), (0, 0, 1.0, fa_on))
        wz, hz = int(wb / zoom), int(hb / zoom)
        cx, cy = l0 + wb // 2, t0 + hb // 2
        l2, t2 = clamp_crop_from_center(cx, cy, wz, hz, iw, ih, sx, sy)
        ccrop = img.crop((l2, t2, l2 + wz, t2 + hz)).resize((cw, ch))

        tmp = BytesIO()
        save_kwargs = {"quality": 95, "subsampling": 0}
        icc = img.info.get("icc_profile")
        if icc: save_kwargs["icc_profile"] = icc
        ccrop.save(tmp, format="JPEG", **save_kwargs)
        tmp.seek(0)
        zf.writestr(f"Custom/{cw}x{ch}.jpg", tmp.getvalue())

zip_buf.seek(0)
st.download_button("Download Crops", zip_buf.getvalue(), file_name=f"crops_{image_file.name}.zip", mime="application/zip")
