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
    l, t, w, h = compute_crop(rec, img_w, img_h)
    new_h = int(w / (cw / ch))
    t += (h - new_h) // 2
    return l, t, w, new_h

# ——————————————————————————————————————————————————————————
# Streamlit page config
# ——————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Smart Crop Automation Prototype", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:450px!important;max-width:450px!important;}</style>", 
    unsafe_allow_html=True
)
st.title("Smart Crop Automation Prototype")

# ——————————————————————————————————————————————————————————
# Sidebar inputs
# ——————————————————————————————————————————————————————————
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader("Master Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

with st.sidebar.expander("⚙️ Custom Crops", expanded=True):
    df_editor = st.data_editor(
        pd.DataFrame([{"Width_px": None, "Height_px": None}]),
        hide_index=True,
        num_rows="dynamic"
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

if not records:
    st.stop()

# Remove master (offset=0) from guidelines list
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
# Load master image
# ——————————————————————————————————————————————————————————
img = Image.open(image_file)
iw, ih = img.size

# ——————————————————————————————————————————————————————————
# Custom adjustment UI
# ——————————————————————————————————————————————————————————
shifts = {}
if custom_sizes:
    st.subheader("Adjust Custom Crops")
    tabs = st.tabs([f"{w}×{h}" for w, h in custom_sizes])
    for idx, ((cw, ch), tab) in enumerate(zip(custom_sizes, tabs)):
        with tab:
            base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
            l0, t0, wb, hb = auto_custom_start(base, iw, ih, cw, ch)

            # Calculate default zoom and offset
            z_key = f"zoom_{idx}"; st.session_state.setdefault(z_key, 0)
            sx_key = f"sx_{idx}"; st.session_state.setdefault(sx_key, 0)
            sy_key = f"sy_{idx}"; st.session_state.setdefault(sy_key, 0)

            # Calculate crop window before adjustments
            zoom = 1 + st.session_state[z_key] / 100
            wz, hz = int(wb / zoom), int(hb / zoom)
            cx, cy = l0 + wb // 2, t0 + hb // 2
            left_start, top_start = cx - wz // 2, cy - hz // 2
            min_x, max_x = -left_start, iw - left_start - wz
            min_y, max_y = -top_start, ih - top_start - hz

            # Use session values for offset and zoom
            sx = st.session_state[sx_key]
            sy = st.session_state[sy_key]
            zd = st.session_state[z_key]

            # Preview before sliders
            x0, y0 = left_start + sx, top_start + sy
            prev = img.crop((x0, y0, x0 + wz, y0 + hz)).resize((cw, ch))
            st.image(prev, caption=f"Preview {cw}×{ch}", width=600)

            # Height offset sliders
            colh1, colh2 = st.columns([3, 1])
            if min_y == max_y:
                with colh1:
                    st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)
                with colh2:
                    sy = st.number_input("Height", value=0, disabled=True, key=f"synum_{idx}")
            else:
                with colh1:
                    sy = st.slider("Height Offset", min_y, max_y, st.session_state[sy_key], 1, key=f"syslider_{idx}")
                with colh2:
                    sy = st.number_input("Height", min_y, max_y, sy, 1, key=f"synum_{idx}", label_visibility="collapsed")
            st.session_state[sy_key] = sy

            # Width offset sliders
            colw1, colw2 = st.columns([3, 1])
            with colw1:
                sx = 0 if min_x == max_x else st.slider("Width Offset", min_x, max_x, st.session_state[sx_key], 1, key=f"sxslider_{idx}")
            with colw2:
                sx = st.number_input("Width", min_x, max_x, sx, 1, key=f"sxnum_{idx}", label_visibility="collapsed")
            st.session_state[sx_key] = sx

            # Zoom control sliders (always last)
            colz1, colz2 = st.columns([3, 1])
            with colz1:
                zd = st.slider("Zoom ±10%", -10, 10, st.session_state[z_key], 1, key=f"zslider_{idx}")
            with colz2:
                zd = st.number_input("Zoom%", -10, 10, zd, 1, key=f"znum_{idx}", label_visibility="collapsed")
            st.session_state[z_key] = zd

            # Save for ZIP generation
            shifts[(cw, ch)] = (sx, sy, 1 + zd / 100)

# ——————————————————————————————————————————————————————————
# Generate ZIP
# ——————————————————————————————————————————————————————————
zip_buf = BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    # Guideline crops
    for rec in guidelines:
        ow = int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72)
        oh = int(rec["frame"]["h"] * rec["effectivePpi"]["y"] / 72)
        l, t, wb, hb = compute_crop(rec, iw, ih)
        gcrop = img.crop((l, t, l + wb, t + hb)).resize((ow, oh))
        tmp = BytesIO(); gcrop.save(tmp, format="PNG"); tmp.seek(0)
        zf.writestr(f"Guidelines/{rec['template']}_{ow}x{oh}.png", tmp.getvalue())

    # Custom crops
    for cw, ch in custom_sizes:
        base = min(records, key=lambda r: abs((cw / ch) - r["frame"]["w"] / r["frame"]["h"]))
        l, t, wb, hb = auto_custom_start(base, iw, ih, cw, ch)
        sx, sy, zoom = shifts.get((cw, ch), (0, 0, 1))
        wz, hz = int(wb / zoom), int(hb / zoom)
        cx, cy = l + wb // 2, t + hb // 2
        l2 = max(0, min(cx - wz // 2 + sx, iw - wz))
        t2 = max(0, min(cy - hz // 2 + sy, ih - hz))
        ccrop = img.crop((l2, t2, l2 + wz, t2 + hz)).resize((cw, ch))
        tmp = BytesIO(); ccrop.save(tmp, format="PNG"); tmp.seek(0)
        zf.writestr(f"Custom/{cw}x{ch}.png", tmp.getvalue())

zip_buf.seek(0)
st.download_button("Download Crops", zip_buf.getvalue(), file_name=f"crops_{image_file.name}.zip", mime="application/zip")
