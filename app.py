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

# ——————————————————————————————————————————————————————————
# Available Crops from Guidelines (excluding master asset)
# filter out master asset by zero-offset
guidelines = [rec for rec in records if not (abs(rec['imageOffset']['x'])<1e-6 and abs(rec['imageOffset']['y'])<1e-6)]
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
# show editable table with Include checkbox
st.subheader("Available Crops from Guidelines")
df_guidelines = pd.DataFrame(guideline_rows)
edited = st.data_editor(
    df_guidelines,
    hide_index=True,
    key="guidelines_editor",
    column_config={"Include": st.column_config.ToggleColumn("Include crop?", default=True)}
)
st.dataframe(
    edited,
    use_container_width=True
)
# determine which to output
selected_templates = edited[edited["Include"]]["Template"].tolist()
# ——————————————————————————————————————————————————————————
# Custom Output Sizes
st.subheader("Custom Output Sizes")
custom_rows = []
for w, h in custom_sizes:
    ar = round(w / h, 2)
    # find base template
    base_tpl = min(records, key=lambda r: abs((w/h) - r["frame"]["w"]/r["frame"]["h"]))["template"]
    custom_rows.append({
        "Template": base_tpl + f" Custom {w}×{h}",
        "Width_px": w,
        "Height_px": h,
        "Aspect Ratio": ar
    })

df_custom = pd.DataFrame(custom_rows)
st.dataframe(df_custom, use_container_width=True)

# Generate combined outputs list
final_templates = [t for t in selected_templates]

# Now proceed to Face detection and Generation
    # Generate & Download
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # guidelines
        for rec in records:
            if rec['template'] in selected_templates:
                out_w = int(rec['frame']['w'] * rec['effectivePpi']['x'] / 72)
                out_h = int(rec['frame']['h'] * rec['effectivePpi']['y'] / 72)
                left, top, w, h = compute_crop(rec, img_w, img_h)
                crop = img_orig.crop((left, top, left+w, top+h)).resize((out_w, out_h), Image.LANCZOS)
                buf = BytesIO(); crop.save(buf, format="PNG"); buf.seek(0)
                zf.writestr(f"{rec['template']}_{out_w}x{out_h}.png", buf.getvalue())
        # customs
        for row in df_custom.itertuples():
            cw, ch = row.Width_px, row.Height_px
            tpl = row.Template.replace("×", "x")
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
