import streamlit as st
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# --- App Setup ---
st.set_page_config(page_title="CropPack Tester", layout="wide")
st.title("CropPack Web App with Manual Canvas Cropping")

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload CropPack JSON", type=["json"])
image_file = st.sidebar.file_uploader("Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"])
st.sidebar.markdown("---")

# --- Select Page ---
pages, doc_data = [], []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec["page"] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
page = st.sidebar.selectbox("Select Page", pages) if pages else None

# --- Output Sizes Editor ---
st.subheader("Output Sizes Mapping")
size_mappings, custom_sizes = [], []
records = []
if page and doc_data and image_file:
    # Page records
    records = [r for r in doc_data if r["page"] == page]
    rows = []
    for rec in records:
        w_pt, h_pt = rec["frame"]["w"], rec["frame"]["h"]
        eff_x, eff_y = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px = int(w_pt * eff_x / 72)
        h_px = int(h_pt * eff_y / 72)
        rows.append({"Template": rec["template"], "Width_px": w_px, "Height_px": h_px})
    rows.append({"Template": "[CUSTOM]", "Width_px": None, "Height_px": None})
    df = pd.DataFrame(rows)
    edited = st.data_editor(df, hide_index=True, num_rows="dynamic", key="map_editor")
    # Build lists
    for i, row in edited.iloc[:len(records)].iterrows():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            size_mappings.append((records[i], [int(row.Width_px), int(row.Height_px)], False))
    for row in edited.iloc[len(records):].itertuples():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            custom_sizes.append((int(row.Width_px), int(row.Height_px)))
else:
    st.info("Upload JSON, select page, and map sizes to begin.")

# --- Face Detection ---
face_box = None
if image_file:
    img_orig = Image.open(image_file)
    np_img = np.array(img_orig)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(dets)>0:
        xs, ys, ws, hs = zip(*dets)
        face_box = {"left":min(xs), "top":min(ys), "right":max(x+w for x,w in zip(xs,ws)), "bottom":max(y+h for y,h in zip(ys,hs))}

# --- Generate Crops ---
if (size_mappings or custom_sizes) and image_file:
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    # Show master image
    st.image(img_orig, caption="Master Asset", use_column_width=True)
    img_w, img_h = img_orig.size

    # Prepare zip
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # Auto template crops
        for rec, out_size, is_custom in size_mappings:
            # compute and crop automatically
            left, top, w, h = compute_crop(rec, img_w, img_h)
            crop = img_orig.crop((left, top, left+w, top+h)).resize(tuple(out_size), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format="PNG"); buf.seek(0)
            fname = f"{rec['template']}_{out_size[0]}x{out_size[1]}.png"
            zf.writestr(fname, buf.getvalue())

        # Canvas-assisted custom crops
        for cw, ch in custom_sizes:
            st.subheader(f"Custom Crop: {cw}×{ch}")
            # initial guess: vertical-only face nudge
            init_left, init_top, init_w, init_h = auto_custom_box(face_box, img_w, img_h, cw, ch)
            # draw canvas
            canvas_result = st_canvas(
                fill_color="", stroke_width=2,
                background_image=img_orig,
                initial_drawing=[{
                    "type": "rect", "x": init_left, "y": init_top,
                    "width": init_w, "height": init_h,
                    "strokeColor": "#00FF00"
                }],
                height=img_h, width=img_w,
                drawing_mode="transform"
            )
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                obj = canvas_result.json_data["objects"][0]
                l = int(obj["left"]); t = int(obj["top"])
                cw_box = int(obj["width"]); ch_box = int(obj["height"])
                # final crop and resize
                final = img_orig.crop((l, t, l+cw_box, t+ch_box)).resize((cw, ch), Image.LANCZOS)
                buf = BytesIO(); final.save(buf, format="PNG"); buf.seek(0)
                zf.writestr(f"custom_{cw}x{ch}.png", buf.getvalue())
    zip_buf.seek(0)
    st.download_button("Download All Crops", zip_buf.getvalue(), file_name=f"crops_page_{page}.zip", mime="application/zip")
else:
    if page:
        st.warning("Define template and custom sizes to generate crops.")


# --- Helper Functions ---
def compute_crop(rec, img_w, img_h):
    off = rec["imageOffset"]; fr = rec["frame"]
    fx, fy = abs(off["x"]), abs(off["y"])
    fw, fh = off["w"], off["h"]
    left = int((fx/fw)*img_w); top = int((fy/fh)*img_h)
    w = int((fr["w"]/fw)*img_w); h = int((fr["h"]/fh)*img_h)
    tgt = fr["w"]/fr["h"]
    if abs((w/h)-tgt)>1e-3:
        if (w/h)>tgt:
            nw=int(h*tgt); left+=(w-nw)//2; w=nw
        else:
            nh=int(w/tgt); top+=(h-nh)//2; h=nh
    return left, top, w, h


def auto_custom_box(face_box, img_w, img_h, cw, ch):
    # start with full-frame
    left=0; top=0; w=img_w; h=img_h
    tgt = cw/ch
    # initial width-based crop
    crop_h = int(img_w/tgt); top=(img_h-crop_h)//2; h=crop_h
    # vertical nudge for faces
    if face_box:
        if face_box['top']<top: top=face_box['top']
        if face_box['bottom']>top+h: top=face_box['bottom']-h
        top = max(0, min(top, img_h - h))

    return left, top, w, h
