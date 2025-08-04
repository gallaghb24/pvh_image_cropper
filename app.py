import streamlit as st
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2

st.set_page_config(page_title="CropPack Tester", layout="wide")
st.title("CropPack Web App Prototype")

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload CropPack JSON", type=["json"])
image_file = st.sidebar.file_uploader(
    "Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"]
)

# --- Page Selection ---
st.sidebar.markdown("---")
pages, doc_data = [], []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec["page"] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
page = st.sidebar.selectbox("Select Page", pages) if pages else None

# --- Main: Mapping Table ---
st.subheader("Output Sizes Mapping")
size_mappings, custom_sizes, records = [], [], []

if page and doc_data and image_file:
    # filter page records
    records = [r for r in doc_data if r["page"] == page]
    # load image and detect faces
    img = Image.open(image_file)
    img_w, img_h = img.size
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    face_box = None
    if len(detections):
        xs, ys, ws, hs = zip(*detections)
        face_box = {
            "left":   int(min(xs)),
            "top":    int(min(ys)),
            "right":  int(max(x+w for x,w in zip(xs,ws))),
            "bottom": int(max(y+h for y,h in zip(ys,hs)))
        }
    # build DataFrame
    df_rows = []
    for rec in records:
        w_pt = rec["frame"]["w"]; h_pt = rec["frame"]["h"]
        eff_x = rec["effectivePpi"]["x"]; eff_y = rec["effectivePpi"]["y"]
        w_px = int(w_pt * eff_x / 72); h_px = int(h_pt * eff_y / 72)
        df_rows.append({
            "Template": rec["template"],
            "Width_pt": w_pt, "Height_pt": h_pt,
            "Width_px": w_px, "Height_px": h_px,
            "Aspect": rec.get("aspectRatio")
        })
    df_rows.append({"Template":"[CUSTOM]","Width_pt":None,"Height_pt":None,"Width_px":None,"Height_px":None,"Aspect":None})
    df_sizes = pd.DataFrame(df_rows)
    edited = st.data_editor(
        df_sizes,
        column_config={
            "Template": st.column_config.TextColumn("Template"),
            "Width_px": st.column_config.NumberColumn("Output Width (px)"),
            "Height_px": st.column_config.NumberColumn("Output Height (px)"),
            "Width_pt": st.column_config.NumberColumn("Frame Width (pt)", format="%.2f"),
            "Height_pt": st.column_config.NumberColumn("Frame Height (pt)", format="%.2f"),
            "Aspect": st.column_config.NumberColumn("Aspect", format="%.2f")
        },
        hide_index=True,
        key="size_editor",
        num_rows="dynamic"
    )
    # map rows
    for idx, row in edited.iloc[:len(records)].iterrows():
        w, h = row["Width_px"], row["Height_px"]
        if pd.notna(w) and pd.notna(h):
            size_mappings.append((records[idx], [int(w),int(h)], False))
    for _, row in edited.iloc[len(records):].iterrows():
        w, h = row["Width_px"], row["Height_px"]
        if pd.notna(w) and pd.notna(h):
            custom_sizes.append((int(w),int(h)))
else:
    st.info("Upload JSON, image and select a page to begin.")

# --- Main: Generate Crops ---
if size_mappings or custom_sizes:
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        tasks = size_mappings.copy()
        for cw,ch in custom_sizes:
            target_r = cw/ch
            best = min(records, key=lambda r: abs(r["aspectRatio"] - target_r))
            tasks.append((best, [cw,ch], True))
        for rec, out_size, is_custom in tasks:
            img_offset = rec["imageOffset"]; frame = rec["frame"]
            disp_w,disp_h = img_offset["w"], img_offset["h"]
            off_x,off_y = abs(img_offset["x"]), abs(img_offset["y"])
            frac_x,frac_y = off_x/disp_w, off_y/disp_h
            frac_w,frac_h = frame["w"]/disp_w, frame["h"]/disp_h
            img = Image.open(image_file)
            img_w, img_h = img.size
            left = int(frac_x * img_w); top = int(frac_y * img_h)
            w = int(frac_w * img_w); h = int(frac_h * img_h)
            target_ratio = out_size[0] / out_size[1]
            current_ratio = w / h if h else target_ratio
            # standard fit
            if not is_custom and abs(current_ratio - target_ratio) > 1e-3:
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left += (w - new_w)//2; w = new_w
                else:
                    new_h = int(w / target_ratio)
                    top += (h - new_h)//2; h = new_h
            # custom: nudge then center-crop
            if is_custom:
                orig_left, orig_top, orig_w, orig_h = left, top, w, h
                if face_box:
                    if face_box["left"] < orig_left: orig_left = max(0,face_box["left"])
                    if face_box["right"]>orig_left+orig_w: orig_left = min(img_w-orig_w,face_box["right"]-orig_w)
                    if face_box["top"] < orig_top: orig_top = max(0,face_box["top"])
                    if face_box["bottom"]>orig_top+orig_h: orig_top = min(img_h-orig_h,face_box["bottom"]-orig_h)
                if orig_w/orig_h > target_ratio:
                    crop_w = int(orig_h * target_ratio); crop_h = orig_h
                else:
                    crop_w = orig_w; crop_h = int(orig_w / target_ratio)
                left = orig_left + (orig_w - crop_w)//2
                top  = orig_top  + (orig_h - crop_h)//2
                w, h = crop_w, crop_h
            # clamp & crop
            left = max(0,min(left,img_w-w)); top = max(0,min(top,img_h-h))
            crop = img.crop((left,top,left+w,top+h))
            crop = crop.resize(tuple(out_size), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf,format="PNG"); buf.seek(0)
            fname = f"{rec['template']}_{out_size[0]}x{out_size[1]}.png"
            zf.writestr(fname, buf.getvalue())
    zip_buffer.seek(0)
    st.download_button("Download Crops", zip_buffer.getvalue(), file_name=f"page_{page}_crops.zip", mime="application/zip")
else:
    if page:
        st.warning("Define at least one size and upload an image to generate crops.")
