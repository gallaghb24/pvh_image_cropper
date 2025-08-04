import streamlit as st
from PIL import Image
import json, pandas as pd
import numpy as np
import cv2
from io import BytesIO
import zipfile

# ——————————————————————————————————————————————————————————
# Helper functions
# ——————————————————————————————————————————————————————————
def compute_crop(rec, img_w, img_h):
    off, fr = rec['imageOffset'], rec['frame']
    fx, fy = abs(off['x']), abs(off['y'])
    fw, fh = off['w'], off['h']
    left  = int((fx/fw) * img_w)
    top   = int((fy/fh) * img_h)
    w     = int((fr['w']/fw) * img_w)
    h     = int((fr['h']/fh) * img_h)
    target = fr['w']/fr['h']
    cur    = w/h if h else target
    if abs(cur - target) > 1e-3:
        if cur > target:
            nw = int(h * target)
            left += (w-nw)//2; w = nw
        else:
            nh = int(w/target)
            top += (h-nh)//2; h = nh
    return left, top, w, h

def auto_custom_start(rec, img_w, img_h, cw, ch):
    left, top, w, h = compute_crop(rec, img_w, img_h)
    tgt = cw/ch
    nh  = int(w/tgt)
    top += (h-nh)//2
    return left, top, w, nh

# ——————————————————————————————————————————————————————————
# App setup
# ——————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Smart Crop Automation Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)

# force sidebar to fixed 450px width
st.markdown("""
  <style>
    [data-testid="stSidebar"] {
      min-width: 450px !important;
      max-width: 450px !important;
    }
  </style>
""", unsafe_allow_html=True)

st.title("Smart Crop Automation Prototype")

# ——————————————————————————————————————————————————————————
# Sidebar: Uploads + Custom Sizes
# ——————————————————————————————————————————————————————————
st.sidebar.header("Inputs")
json_file  = st.sidebar.file_uploader("Upload Cropping Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader(
    "Upload Master Asset Image",
    type=["png","jpg","jpeg","tif","tiff"]
)

with st.sidebar.expander("⚙️ Custom Crops", expanded=True):
    st.markdown("Add extra crop dimensions:")
    custom_df = st.data_editor(
        pd.DataFrame([{"Width_px": None, "Height_px": None}]),
        hide_index=True, num_rows="dynamic", key="custom_sizes"
    )
    custom_sizes = [
        (int(r.Width_px), int(r.Height_px))
        for r in custom_df.itertuples()
        if pd.notna(r.Width_px) and pd.notna(r.Height_px)
    ]

# ——————————————————————————————————————————————————————————
# Load JSON & match to uploaded image
# ——————————————————————————————————————————————————————————
records = []
if json_file and image_file:
    data  = json.load(json_file)
    fname = image_file.name
    for rec in data:
        spec = rec.get("filename") or rec.get("fileName") or rec.get("asset")
        if spec and (fname == spec or fname.startswith(spec)):
            records.append(rec)
    if not records:
        st.sidebar.error("No matching filename found in JSON.")
else:
    st.sidebar.info("Please upload both JSON & image.")

# ——————————————————————————————————————————————————————————
# Build & show Output Sizes table
# ——————————————————————————————————————————————————————————
if records:
    rows = []
    # built-in
    for rec in records:
        w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
        ex, ey     = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px = int(w_pt * ex/72)
        h_px = int(h_pt * ey/72)
        rows.append({
            "Template": rec["template"],
            "Width_px": w_px,
            "Height_px": h_px,
            "Aspect Ratio": round(w_px/h_px, 2)
        })
    # customs
    for w, h in custom_sizes:
        rows.append({
            "Template": f"Custom {w}×{h}",
            "Width_px": w,
            "Height_px": h,
            "Aspect Ratio": round(w/h, 2)
        })

    st.subheader("Output Sizes")
    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True)

    # ————————————————————————————————————————————————
    # Face detection (for potential future use)
    # ————————————————————————————————————————————————
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size
    gray  = cv2.cvtColor(np.array(img_orig), cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    dets = cascade.detectMultiScale(gray,1.1,5,minSize=(30,30))
    face_box = None
    if len(dets)>0:
        xs, ys, ws, hs = zip(*dets)
        face_box = dict(
            left   = min(xs),
            top    = min(ys),
            right  = max(x+w for x,w in zip(xs,ws)),
            bottom = max(y+h for y,h in zip(ys,hs))
        )

    # ————————————————————————————————————————————————
    # Custom‐crop UI
    # ————————————————————————————————————————————————
    if custom_sizes:
        st.subheader("Adjust Custom Crops")
        tabs = st.tabs([f"{w}×{h}" for w,h in custom_sizes])
        custom_params = {}
        for i, ((cw,ch), tab) in enumerate(zip(custom_sizes, tabs)):
            with tab:
                st.write(f"Fine-tune **{cw}×{ch}**")
                # pick base rec
                rec = min(
                    records,
                    key=lambda r: abs((cw/ch) - r["frame"]["w"]/r["frame"]["h"])
                )
                l, t, w0, h0 = auto_custom_start(rec, img_w, img_h, cw, ch)

                # compute pan bounds
                min_x, max_x = -l, img_w - l - w0
                min_y, max_y = -t, img_h - t - h0
                min_x, max_x = sorted((min_x, max_x))
                min_y, max_y = sorted((min_y, max_y))

                # Zoom ±10%
                zoom = st.slider(
                    "Zoom", 0.9, 1.1, 1.0, 0.01,
                    help="Scale crop ±10%", key=f"zoom_{i}"
                )

                # Pan slider + number input side-by-side
                col1, col2 = st.columns([4,1])
                label = "Width Offset" if w0>h0 else "Height Offset"
                with col1:
                    pan = st.slider(
                        label, min_x, max_x, 0, 1, key=f"pan_{i}"
                    )
                with col2:
                    pan = st.number_input(
                        "", min_value=min_x, max_value=max_x,
                        value=pan, step=1, key=f"pan_num_{i}"
                    )

                # apply zoom + pan
                cx, cy = l + w0//2, t + h0//2
                wz, hz = int(w0*zoom), int(h0*zoom)
                nx = cx - wz//2 + pan
                ny = cy - hz//2
                nx = max(0, min(nx, img_w-wz))
                ny = max(0, min(ny, img_h-hz))

                # preview
                preview = img_orig.crop((nx, ny, nx+wz, ny+hz)).resize((cw,ch), Image.LANCZOS)
                st.image(preview, width=600, caption=f"Preview {cw}×{ch}")

                custom_params[(cw,ch)] = (zoom, pan)

    # ——————————————————————————————————————————————————————————
    # Package & Download
    # ——————————————————————————————————————————————————————————
    zipbuf = BytesIO()
    with zipfile.ZipFile(zipbuf, "w") as zf:
        # guidelines folder
        for rec, row in zip(records, df_out.iloc[:len(records)].itertuples()):
            ow, oh = row.Width_px, row.Height_px
            l, t, w0, h0 = compute_crop(rec, img_w, img_h)
            crop = img_orig.crop((l, t, l+w0, t+h0)).resize((ow,oh), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf,"PNG"); buf.seek(0)
            zf.writestr(f"Guidelines/{rec['template']}_{ow}x{oh}.png", buf.getvalue())

        # custom folder
        for row in df_out.iloc[len(records):].itertuples():
            cw, ch = row.Width_px, row.Height_px
            rec = min(
                records,
                key=lambda r: abs((cw/ch) - r["frame"]["w"]/r["frame"]["h"])
            )
            l, t, w0, h0 = auto_custom_start(rec, img_w, img_h, cw, ch)
            zoom, pan = custom_params.get((cw,ch), (1.0, 0))
            # reapply
            cx, cy = l + w0//2, t + h0//2
            wz, hz = int(w0*zoom), int(h0*zoom)
            nx = max(0, min(cx-wz//2+pan, img_w-wz))
            ny = max(0, min(cy-hz//2, img_h-hz))
            crop = img_orig.crop((nx, ny, nx+wz, ny+hz)).resize((cw,ch), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf,"PNG"); buf.seek(0)
            label = row.Template.replace("×","x")
            zf.writestr(f"Custom/{label}.png", buf.getvalue())

    zipbuf.seek(0)
    st.download_button(
        "📥 Download All Crops",
        data=zipbuf.getvalue(),
        file_name=f"crops_{image_file.name}.zip",
        mime="application/zip",
    )

else:
    st.info("Upload JSON + image to see and generate crops.")
