import streamlit as st
from PIL import Image
import json, pandas as pd
from io import BytesIO
import zipfile
from typing import List, Tuple

# ——————————————————————————————————————————————————————————
# Helper functions
# ——————————————————————————————————————————————————————————

def compute_crop(rec: dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Compute a crop rectangle in the master image for a guideline record."""
    off, fr = rec["imageOffset"], rec["frame"]
    fx, fy = abs(off["x"]), abs(off["y"])
    fw, fh = off["w"], off["h"]
    l = int((fx / fw) * img_w)
    t = int((fy / fh) * img_h)
    w = int((fr["w"] / fw) * img_w)
    h = int((fr["h"] / fh) * img_h)
    tgt = fr["w"] / fr["h"]
    cur = w / h if h else tgt
    if abs(cur - tgt) > 1e-3:
        if cur > tgt:  # trim width
            new_w = int(h * tgt)
            l += (w - new_w) // 2
            w = new_w
        else:  # trim height
            new_h = int(w / tgt)
            t += (h - new_h) // 2
            h = new_h
    return l, t, w, h


def auto_custom_start(rec: dict, img_w: int, img_h: int, cw: int, ch: int) -> Tuple[int, int, int, int]:
    l, t, w, h = compute_crop(rec, img_w, img_h)
    new_h = int(w / (cw / ch))
    t += (h - new_h) // 2
    return l, t, w, new_h

# ——————————————————————————————————————————————————————————
# Streamlit setup
# ——————————————————————————————————————————————————————————

st.set_page_config(page_title="Smart Crop Automation Prototype", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      [data-testid="stSidebar"]{min-width:450px!important;max-width:450px!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Crop Automation Prototype")

# ——————————————————————————————————————————————————————————
# Sidebar inputs
# ——————————————————————————————————————————————————————————

json_file = st.sidebar.file_uploader("Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader("Master Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

with st.sidebar.expander("⚙️ Custom Crops", expanded=True):
    df_editor = st.data_editor(
        pd.DataFrame([{"Width_px": None, "Height_px": None}]),
        hide_index=True,
        num_rows="dynamic",
        key="custom_sizes_editor",
    )
    custom_sizes: List[Tuple[int, int]] = [
        (int(r.Width_px), int(r.Height_px))
        for r in df_editor.itertuples()
        if pd.notna(r.Width_px) and pd.notna(r.Height_px)
    ]

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

if records:
    guidelines = [r for r in records if not (abs(r["imageOffset"]["x"]) < 1e-6 and abs(r["imageOffset"]["y"]) < 1e-6)]

    # — Display guideline table
    st.subheader("Guideline Crops")
    g_rows = []
    for rec in guidelines:
        w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
        ex, ey = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px, h_px = int(w_pt * ex / 72), int(h_pt * ey / 72)
        g_rows.append({"Template": rec["template"], "Width_px": w_px, "Height_px": h_px, "AR": round(w_px / h_px, 2)})
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True)

    # — Display custom table
    st.subheader("Custom Crops")
    c_rows = []
    for cw, ch in custom_sizes:
        base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
        c_rows.append({"Template": f"{base['template']} {cw}×{ch}", "Width_px": cw, "Height_px": ch, "AR": round(cw / ch, 2)})
    st.dataframe(pd.DataFrame(c_rows), use_container_width=True)

    # Load master image
    img = Image.open(image_file)
    iw, ih = img.size

    # — Adjustment UI for custom crops
    shifts = {}
    if custom_sizes:
        st.subheader("Adjust Custom Crops")
        tabs = st.tabs([f"{w}×{h}" for w, h in custom_sizes])

        for idx, ((cw, ch), tab) in enumerate(zip(custom_sizes, tabs)):
            with tab:
                base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
                l0, t0, wb, hb = auto_custom_start(base, iw, ih, cw, ch)

                # Zoom (±10%)
                z_key = f"zoom_delta_{idx}"
                st.session_state.setdefault(z_key, 0)
                col_z1, col_z2 = st.columns([3, 1])
                with col_z1:
                    zd = st.slider("Zoom ±10%", -10, 10, st.session_state[z_key], 1, key=f"zoom_slider_{idx}")
                with col_z2:
                    zd = st.number_input("Zoom %", -10, 10, zd, 1, key=f"zoom_num_{idx}", label_visibility="collapsed")
                st.session_state[z_key] = zd
                zoom = 1 + zd / 100

                wz, hz = int(wb / zoom), int(hb / zoom)
                cx, cy = l0 + wb // 2, t0 + hb // 2
                ls, ts = cx - wz // 2, cy - hz // 2

                min_x, max_x = -ls, iw - ls - wz
                min_y, max_y = -ts, ih - ts - hz

                # Width Offset
                sx_key = f"width_off_{idx}"
                st.session_state.setdefault(sx_key, 0)
                col_w1, col_w2 = st.columns([3, 1])
                with col_w1:
                    sx = 0 if min_x == max_x else st.slider("Width Offset", min_x, max_x, st.session_state[sx_key], 1, key=f"width_slider_{idx}")
                with col_w2:
                    sx = st.number_input("Width", min_x, max_x, sx, 1, key=f"width_num_{idx}", label_visibility="collapsed")
                st.session_state[sx_key] = sx

                # Height Offset
                sy_key = f"height_off_{idx}"
                st.session_state.setdefault(sy_key, 0)
                col_h1, col_h2 = st.columns([3, 1])
                if min_y == max_y:
                    with col_h1:
                        st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)
                    with col_h2:
                        sy = st.number_input("Height", value=0, disabled=True, key=f"height_num_{idx}")
                else:
                    with col_h1:
                        sy = st.slider("Height Offset", min_y, max_y, st.session_state[sy_key], 1, key=f"height_slider_{idx}")
                    with col_h2:
                        sy = st.number_input("Height", min_y, max_y, sy, 1, key=f"height_num_{idx}", label_visibility="collapsed")
                st.session_state[sy_key] = sy

                # Preview
                x0, y0 = ls + sx, ts + sy
                prev = img.crop((x0, y0, x0 + wz, y0 + hz)).resize((cw, ch))
                st.image(prev, caption=f"Preview {cw}×{ch}", width=600)

                shifts[(cw, ch)] = (sx, sy, zoom)

    # — ZIP output
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for rec in guidelines:
            ow = int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72)
            oh = int(rec["frame"]["h"] * rec["effectivePpi"]["y"] / 72)
            l, t, wb, hb = compute_crop(rec, iw, ih)
            crop = img.crop((l, t, l + wb, t + hb)).resize((ow, oh))
            tmp = BytesIO(); crop.save(tmp, format="PNG"); tmp.seek(0)
            zf.writestr(f"Guidelines/{rec['template']}_{ow}x{oh}.png", tmp.getvalue())

        for cw, ch in custom_sizes:
            base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
            l, t, wb, hb = auto_custom_start(base, iw, ih, cw, ch)
            sx, sy, zoom = shifts.get((cw, ch), (0, 0, 1))
            wz, hz = int(wb / zoom), int(hb / zoom)
            cx, cy = l + wb // 2, t + hb // 2
            l2 = max(0, min(cx - wz // 2 + sx, iw - wz))
            t2 = max(0, min(cy - hz // 2 + sy, ih - hz))
            crop = img.crop((l2, t2
