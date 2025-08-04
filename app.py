import streamlit as st
from PIL import Image
import json, pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2

# ——————————————————————————————————————————————————————————
# Helper functions
# ——————————————————————————————————————————————————————————

def compute_crop(rec, img_w, img_h):
    """Return (left, top, width, height) crop rect (px) from guideline record,
    keeping the frame's intended aspect ratio."""
    off, fr = rec["imageOffset"], rec["frame"]
    fx, fy = abs(off["x"]), abs(off["y"])
    fw, fh = off["w"], off["h"]
    left = int((fx / fw) * img_w)
    top = int((fy / fh) * img_h)
    w = int((fr["w"] / fw) * img_w)
    h = int((fr["h"] / fh) * img_h)
    target_ar = fr["w"] / fr["h"]
    cur_ar = w / h if h else target_ar
    if abs(cur_ar - target_ar) > 1e-3:
        if cur_ar > target_ar:  # too wide – trim width
            new_w = int(h * target_ar)
            left += (w - new_w) // 2
            w = new_w
        else:  # too tall – trim height
            new_h = int(w / target_ar)
            top += (h - new_h) // 2
            h = new_h
    return left, top, w, h


def auto_custom_start(rec, img_w, img_h, cw, ch):
    """Get starting crop for custom WxH preserving width and centralising."""
    left, top, w, h = compute_crop(rec, img_w, img_h)
    tgt_ar = cw / ch
    new_h = int(w / tgt_ar)
    top += (h - new_h) // 2
    return left, top, w, new_h

# ——————————————————————————————————————————————————————————
# App setup
# ——————————————————————————————————————————————————————————

st.set_page_config(
    page_title="Smart Crop Automation Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {min-width: 450px !important; max-width: 450px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Crop Automation Prototype")

# ——————————————————————————————————————————————————————————
# Sidebar Inputs
# ——————————————————————————————————————————————————————————

st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload Cropping Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader("Upload Master Asset Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

with st.sidebar.expander("⚙️ Custom Crops", expanded=True):
    st.markdown("Add extra output dimensions here:")
    custom_df = st.data_editor(
        pd.DataFrame([{"Width_px": None, "Height_px": None}]),
        hide_index=True,
        num_rows="dynamic",
        key="custom_sizes_editor",
    )
    custom_sizes = [
        (int(r.Width_px), int(r.Height_px))
        for r in custom_df.itertuples()
        if pd.notna(r.Width_px) and pd.notna(r.Height_px)
    ]

# ——————————————————————————————————————————————————————————
# Load & match records
# ——————————————————————————————————————————————————————————

records: list[dict] = []
if json_file and image_file:
    data = json.load(json_file)
    fname = image_file.name
    for rec in data:
        spec = rec.get("filename") or rec.get("fileName") or rec.get("asset")
        if spec and (fname == spec or fname.startswith(spec)):
            records.append(rec)
    if not records:
        st.sidebar.error("No matching asset name found in JSON.")
else:
    st.sidebar.info("Upload both JSON & image to begin.")

# ——————————————————————————————————————————————————————————
# Main — only show once files loaded & matched
# ——————————————————————————————————————————————————————————

if records:
    guidelines = [r for r in records if not (abs(r["imageOffset"]["x"]) < 1e-6 and abs(r["imageOffset"]["y"]) < 1e-6)]

    # tables — guidelines
    st.subheader("Available Crops from Guidelines")
    g_rows = []
    for rec in guidelines:
        w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
        ex, ey = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px = int(w_pt * ex / 72)
        h_px = int(h_pt * ey / 72)
        g_rows.append({"Template": rec["template"], "Width_px": w_px, "Height_px": h_px, "Aspect Ratio": round(w_px/h_px,2)})
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True)

    # tables — custom
    st.subheader("Custom Crops")
    c_rows = []
    for cw, ch in custom_sizes:
        closest = min(records, key=lambda r: abs((cw/ch) - r["frame"]["w"] / r["frame"]["h"]))
        c_rows.append({"Template": f"{closest['template']} Custom {cw}×{ch}", "Width_px": cw, "Height_px": ch, "Aspect Ratio": round(cw/ch,2)})
    st.dataframe(pd.DataFrame(c_rows), use_container_width=True)

    # load master image
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size

    # ——————————————————————————————————————————
    # Custom crop adjust UI
    # ——————————————————————————————————————————

    custom_shifts = {}
    if custom_sizes:
        st.subheader("Adjust Custom Crops")
        tabs = st.tabs([f"{w}×{h}" for w,h in custom_sizes])
        for i, ((cw,ch), tab) in enumerate(zip(custom_sizes, tabs)):
            with tab:
                base_rec = min(records, key=lambda r: abs((cw/ch) - r["frame"]["w"] / r["frame"]["h"]))
                left, top, wb, hb = auto_custom_start(base_rec, img_w, img_h, cw, ch)

                # —— ZOOM ±10 % ——
                z_key = f"zoom_delta_{i}"
                st.session_state.setdefault(z_key, 0)
                colz1, colz2 = st.columns([3,1])
                with colz1:
                    z_delta = st.slider("Zoom (±10%)", -10, 10, st.session_state[z_key], 1, key=f"szoom_{i}")
                with colz2:
                    z_delta = st.number_input("", -10, 10, z_delta, 1, key=f"nzoom_{i}")
                st.session_state[z_key] = z_delta
                zoom = 1 + (z_delta / 100)

                # crop dims after zoom
                wz = int(wb / zoom)
                hz = int(hb / zoom)
                cx, cy = left + wb//2, top + hb//2
                left0, top0 = cx - wz//2, cy - hz//2

                # allowable offsets
                min_x, max_x = -left0, img_w - left0 - wz
                min_y, max_y = -top0,  img_h - top0 - hz

                # —— WIDTH OFFSET ——
                sx_key = f"sx_{i}"
                st.session_state.setdefault(sx_key, 0)
                colw1, colw2 = st.columns([3,1])
                with colw1:
                    sx = 0 if min_x==max_x else st.slider("Width Offset", min_x, max_x, st.session_state[sx_key], 1, key=f"sw_{i}")
                with colw2:
                    sx = st.number_input("", min_x, max_x, sx, 1, key=f"nw_{i}")
                st.session_state[sx_key] = sx

                # —— HEIGHT OFFSET ——
                sy_key = f"sy_{i}"
                st.session_state.setdefault(sy_key, 0)
                colh1, colh2 = st.columns([3,1])
                if min_y == max_y:
                    with colh1:
                        st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)
                    with colh2:
                        sy = st.number_input("Height Offset", value=0, disabled=True, key=f"nh_{i}")
                else:
                    with colh1:
                        sy = st.slider("Height Offset", min_y, max_y, st.session_state[sy_key], 1, key=f"sh_{i}")
                    with colh2:
                        sy = st.number_input("", min_y, max_y, sy, 1, key=f"nh_{i}")
                st.session_state[sy_key] = sy

                # preview
                x0, y0 = left0 + sx, top0 + sy
                preview = img_orig.crop((x0, y0, x0 + wz, y0 + hz)).resize((cw, ch), Image.LANCZOS)
                st.image(preview, caption=f"Preview {cw}×{ch}", width=600)

                custom_shifts[(cw,ch)] = (sx, sy, zoom)

    # ——————————————————————————————————————————
    # ZIP generation
    # ——————————————————————————————————————————

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # guideline crops
        for rec in guidelines:
            out_w = int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72)
            out_h = int(rec[
