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
    off = rec['imageOffset']
    fr = rec['frame']
    fx, fy = abs(off['x']), abs(off['y'])
    fw, fh = off['w'], off['h']
    left = int((fx / fw) * img_w)
    top  = int((fy / fh) * img_h)
    w    = int((fr['w']  / fw) * img_w)
    h    = int((fr['h']  / fh) * img_h)
    target = fr['w'] / fr['h']
    current = w / h if h else target
    if abs(current - target) > 1e-3:
        if current > target:
            new_w = int(h * target)
            left += (w - new_w) // 2
            w = new_w
        else:
            new_h = int(w / target)
            top += (h - new_h) // 2
            h = new_h
    return left, top, w, h


def auto_custom_start(rec, img_w, img_h, cw, ch):
    left, top, w, h = compute_crop(rec, img_w, img_h)
    tgt = cw / ch
    new_h = int(w / tgt)
    top += (h - new_h) // 2
    return left, top, w, new_h

# ——————————————————————————————————————————————————————————
# App setup
# ——————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Smart Crop Automation Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar width fixed at 450px
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {
        min-width: 450px !important;
        max-width: 450px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Crop Automation Prototype")

# ——————————————————————————————————————————————————————————
# Sidebar: Inputs + Custom Sizes
# ——————————————————————————————————————————————————————————
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload Cropping Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader(
    "Upload Master Asset Image",
    type=["png","jpg","jpeg","tif","tiff"]
)

with st.sidebar.expander("⚙️ Custom Output Sizes", expanded=True):
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
# Load & match JSON → records
# ——————————————————————————————————————————————————————————
records = []
if json_file and image_file:
    data = json.load(json_file)
    fname = image_file.name
    for rec in data:
        spec = rec.get("filename") or rec.get("fileName") or rec.get("asset")
        if spec and (fname == spec or fname.startswith(spec)):
            records.append(rec)
    if not records:
        st.sidebar.error("Image name not found in JSON.")
else:
    st.sidebar.info("Please upload both JSON & image.")

if records:
    # ——————————————————————————————————————————————————————————
    # Available Crops from Guidelines (excluding master asset)
    # ——————————————————————————————————————————————————————————
    guidelines = [r for r in records if not (abs(r['imageOffset']['x'])<1e-6 and abs(r['imageOffset']['y'])<1e-6)]
    guideline_rows = []
    for rec in guidelines:
        w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
        eff_x, eff_y = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px = int(w_pt * eff_x / 72)
        h_px = int(h_pt * eff_y / 72)
        guideline_rows.append({
            "Include": True,
            "Template": rec["template"],
            "Width_px": w_px,
            "Height_px": h_px,
            "Aspect Ratio": round(w_px / h_px, 2)
        })
    st.subheader("Available Crops from Guidelines")
    df_guidelines = pd.DataFrame(guideline_rows)
    edited = st.data_editor(
        df_guidelines,
        hide_index=True,
        key="guidelines_editor",
        
    df_custom = pd.DataFrame(custom_rows)
    st.subheader("Custom Output Sizes")
    st.dataframe(df_custom, use_container_width=True)

    # ——————————————————————————————————————————————————————————
    # Prep image & face detection
    # ——————————————————————————————————————————————————————————
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size
    np_img = np.array(img_orig)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    dets = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    face_box = None
    if len(dets) > 0:
        xs, ys, ws, hs = zip(*dets)
        face_box = {
            "left": min(xs),
            "top": min(ys),
            "right": max(x+w for x,w in zip(xs,ws)),
            "bottom": max(y+h for y,h in zip(ys,hs)),
        }

    # ——————————————————————————————————————————————————————————
    # Manual Adjustments for Custom Crops
    # ——————————————————————————————————————————————————————————
    custom_shifts = {}
    if custom_sizes:
        st.subheader("Adjust Custom Crops")
        tabs = st.tabs([f"{cw}×{ch}" for cw,ch in custom_sizes])
        for i, ((cw, ch), tab) in enumerate(zip(custom_sizes, tabs)):
            with tab:
                rec = min(
                    records,
                    key=lambda r: abs((cw/ch) - r["frame"]["w"]/r["frame"]["h"])
                )
                left, top, w, h = auto_custom_start(rec, img_w, img_h, cw, ch)
                min_x, max_x = -left, img_w - left - w
                min_y, max_y = -top, img_h - top - h
                if min_x > max_x: min_x, max_x = max_x, min_x
                if min_y > max_y: min_y, max_y = max_y, min_y
                shift_x = 0 if min_x == max_x else st.slider("Shift left/right", min_x, max_x, 0, key=f"shiftx_{i}")
                shift_y = 0 if min_y == max_y else st.slider("Shift up/down", min_y, max_y, 0, key=f"shifty_{i}")
                x0, y0 = left + shift_x, top + shift_y
                preview = img_orig.crop((x0, y0, x0+w, y0+h)).resize((cw, ch), Image.LANCZOS)
                st.image(preview, caption=f"Preview {cw}×{ch}", width=900)
                custom_shifts[(cw, ch)] = (shift_x, shift_y)

    # ——————————————————————————————————————————————————————————
    # Generate & Download
    # ——————————————————————————————————————————————————————————
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # guidelines
        for rec in guidelines:
            if rec["template"] in selected_templates:
                out_w = int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72)
                out_h = int(rec["frame"]["h"] * rec["effectivePpi"]["y"] / 72)
                left, top, w, h = compute_crop(rec, img_w, img_h)
                crop = img_orig.crop((left, top, left+w, top+h)).resize((out_w, out_h), Image.LANCZOS)
                buf = BytesIO(); crop.save(buf, format="PNG"); buf.seek(0)
                zf.writestr(f"{rec['template']}_{out_w}x{out_h}.png", buf.getvalue())
        # customs
        for cw, ch in custom_sizes:
            tpl = f"custom_{cw}x{ch}"
            rec = min(
                records,
                key=lambda r: abs((cw/ch) - r["frame"]["w"]/r["frame"]["h"])
            )
            left, top, w, h = auto_custom_start(rec, img_w, img_h, cw, ch)
            sx, sy = custom_shifts.get((cw, ch), (0,0))
            left = max(0, min(left+sx, img_w - w))
            top  = max(0, min(top+sy, img_h - h))
            crop = img_orig.crop((left, top, left+w, top+h)).resize((cw, ch), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format="PNG"); buf.seek(0)
            zf.writestr(f"{tpl}.png", buf.getvalue())
    zip_buf.seek(0)
    st.download_button(
        "📥 Download All Crops",
        data=zip_buf.getvalue(),
        file_name=f"crops_{image_file.name}.zip",
        mime="application/zip",
    )
else:
    st.info("Upload JSON + image to see Output Sizes.")
