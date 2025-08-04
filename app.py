import streamlit as st
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2

# --- App Setup ---
st.set_page_config(page_title="CropPack Tester", layout="wide")
st.title("CropPack Web App Prototype")

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload CropPack JSON", type=["json"])
image_file = st.sidebar.file_uploader("Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"])
st.sidebar.markdown("---")

# --- Page Selection ---
pages, doc_data = [], []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec["page"] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
page = st.sidebar.selectbox("Select Page", pages) if pages else None

# --- Main: Size Mapping Editor ---
st.subheader("Output Sizes Mapping")
size_mappings, custom_sizes = [], []
records = []

if page is not None and doc_data and image_file:
    # Filter for page
    records = [r for r in doc_data if r["page"] == page]
    # Build mapping table
    rows = []
    for rec in records:
        w_pt = rec["frame"]["w"]; h_pt = rec["frame"]["h"]
        eff_x = rec["effectivePpi"]["x"]; eff_y = rec["effectivePpi"]["y"]
        rows.append({
            "Template": rec["template"],
            "Width_pt": w_pt, "Height_pt": h_pt,
            "Width_px": int(w_pt * eff_x / 72),
            "Height_px": int(h_pt * eff_y / 72),
            "Aspect": rec.get("aspectRatio")
        })
    rows.append({"Template":"[CUSTOM]","Width_pt":None,"Height_pt":None,"Width_px":None,"Height_px":None,"Aspect":None})
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        column_config={
            "Template": st.column_config.TextColumn("Template"),
            "Width_px": st.column_config.NumberColumn("Output Width (px)"),
            "Height_px": st.column_config.NumberColumn("Output Height (px)"),
            "Width_pt": st.column_config.NumberColumn("Frame Width (pt)", format="%.2f"),
            "Height_pt": st.column_config.NumberColumn("Frame Height (pt)", format="%.2f"),
            "Aspect": st.column_config.NumberColumn("Aspect Ratio", format="%.2f")
        },
        hide_index=True,
        num_rows="dynamic",
        key="size_editor"
    )
    # Collect mappings
    for i, row in edited.iloc[:len(records)].iterrows():
        if pd.notna(row["Width_px"]) and pd.notna(row["Height_px"]):
            size_mappings.append((records[i], [int(row["Width_px"]), int(row["Height_px"])], False))
    for row in edited.iloc[len(records):].itertuples():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            custom_sizes.append((int(row.Width_px), int(row.Height_px)))
else:
    st.info("Upload JSON, image, and select a page to begin.")

# --- Human-in-loop Slider for Custom ---
vertical_shift = 0
if custom_sizes:
    vertical_shift = st.slider(
        "Custom Crop: Vertical Shift (px)",
        min_value=-500, max_value=500, value=0, step=1
    )

# --- Crop Generation ---
if (size_mappings or custom_sizes) and image_file:
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        tasks = size_mappings.copy()
        for cw, ch in custom_sizes:
            ratio = cw/ch
            best = min(records, key=lambda r: abs(r.get("aspectRatio",1)-ratio))
            tasks.append((best, [cw, ch], True))

        # Face detection precompute
        np_img = np.array(img_orig)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        face_box = None
        if len(dets)>0:
            xs, ys, ws, hs = zip(*dets)
            face_box = {"left":min(xs),"top":min(ys),"right":max(x+w for x,w in zip(xs,ws)),"bottom":max(y+h for y,h in zip(ys,hs))}

        for rec, out_size, is_custom in tasks:
            off = rec["imageOffset"]; fr = rec["frame"]
            fx, fy = abs(off["x"]), abs(off["y"])
            fw, fh = off["w"], off["h"]
            left = int((fx/fw)*img_w); top = int((fy/fh)*img_h)
            w = int((fr["w"]/fw)*img_w); h = int((fr["h"]/fh)*img_h)

            # Exact template
            if not is_custom:
                tgt = out_size[0]/out_size[1]
                cur = w/h if h else tgt
                if abs(cur-tgt)>1e-3:
                    if cur>tgt:
                        nw = int(h*tgt)
                        left += (w-nw)//2; w=nw
                    else:
                        nh = int(w/tgt)
                        top += (h-nh)//2; h=nh

            # Custom: vertical-only nudge
            if is_custom:
                tgt = out_size[0]/out_size[1]
                # center-crop to ratio
                new_h = int(w/tgt)
                top += (h-new_h)//2
                h = new_h
                # vertical shift to include faces
                if face_box:
                    if face_box["top"] < top:
                        top = face_box["top"]
                    if face_box["bottom"] > top+h:
                        top = face_box["bottom"]-h
                # apply human slider
                top += vertical_shift

            # clamp & finalize
            left = max(0, min(left, img_w-w))
            top = max(0, min(top, img_h-h))
            crop = img_orig.crop((left, top, left+w, top+h))
            crop = crop.resize(tuple(out_size), Image.LANCZOS)

            buf = BytesIO()
            crop.save(buf, format="PNG")
            buf.seek(0)
            fname = f"{rec['template']}_{out_size[0]}x{out_size[1]}.png"
            zf.writestr(fname, buf.getvalue())

    zip_buf.seek(0)
    st.download_button(
        "Download Crops",
        zip_buf.getvalue(),
        file_name=f"page_{page}_crops.zip",
        mime="application/zip"
    )
else:
    if page is not None:
        st.warning("Define at least one size and upload an image to generate crops.")
