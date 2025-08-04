import streamlit as st
from PIL import Image
import json, pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2

# ——————————————————————————————————————————————————————————
# Helper functions (unchanged)
# ——————————————————————————————————————————————————————————
def compute_crop(rec, img_w, img_h):
    # … your compute_crop logic …
    pass

def auto_custom_start(rec, img_w, img_h, cw, ch):
    # … your start logic …
    pass

# ——————————————————————————————————————————————————————————
# App setup
# ——————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Smart Crop Automation Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)

# force sidebar to fixed 450px width via CSS
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

# Custom sizes editor in sidebar
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
# Build Final Output Sizes Table
# ——————————————————————————————————————————————————————————
if records:
    rows = []
    # built-in templates
    for rec in records:
        w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
        eff_x, eff_y = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px = int(w_pt * eff_x / 72)
        h_px = int(h_pt * eff_y / 72)
        rows.append({
            "Template": rec["template"],
            "Width_px": w_px,
            "Height_px": h_px,
            "Aspect Ratio": round(w_px / h_px, 2)
        })
    # custom entries
    for w, h in custom_sizes:
        rows.append({
            "Template": f"Custom {w}×{h}",
            "Width_px": w,
            "Height_px": h,
            "Aspect Ratio": round(w / h, 2)
        })

    st.subheader("Output Sizes")
    df_outputs = pd.DataFrame(rows)
    st.dataframe(df_outputs, use_container_width=True)

    # ————————————————————————————————————————————————
    # Face detection for custom crops
    # ————————————————————————————————————————————————
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size
    np_img = np.array(img_orig)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
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
    # Generate & Download
    # ——————————————————————————————————————————————————————————
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # templates
        for rec, row in zip(records, df_outputs.iloc[:len(records)].itertuples()):
            out_w, out_h = row.Width_px, row.Height_px
            left, top, w, h = compute_crop(rec, img_w, img_h)
            crop = img_orig.crop((left, top, left+w, top+h))
            crop = crop.resize((out_w, out_h), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format="PNG"); buf.seek(0)
            zf.writestr(f"{rec['template']}_{out_w}x{out_h}.png", buf.getvalue())

        # customs
        start_idx = len(records)
        for idx in range(start_idx, len(df_outputs)):
            row = df_outputs.iloc[idx]
            cw, ch = row.Width_px, row.Height_px
            rec = min(
                records,
                key=lambda r: abs((cw/ch) - r["frame"]["w"]/r["frame"]["h"])
            )
            left, top, w, h = auto_custom_start(rec, img_w, img_h, cw, ch)
            crop = img_orig.crop((left, top, left+w, top+h))
            crop = crop.resize((cw, ch), Image.LANCZOS)
            label = row.Template.replace("×", "x")
            buf = BytesIO(); crop.save(buf, format="PNG"); buf.seek(0)
            zf.writestr(f"{label}.png", buf.getvalue())

    zip_buf.seek(0)
    st.download_button(
        "📥 Download All Crops",
        data=zip_buf.getvalue(),
        file_name=f"crops_{image_file.name}.zip",
        mime="application/zip",
    )

else:
    st.info("Upload JSON + image to see Output Sizes.")
