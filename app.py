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

# --- Sidebar: Upload Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload CropPack JSON", type=["json"])
image_file = st.sidebar.file_uploader("Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"])

# --- Select Page ---
st.sidebar.markdown("---")
pages, doc_data = [], []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec["page"] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
page = None
if pages:
    page = st.sidebar.selectbox("Select Page", pages)

# --- Main: Output Sizes Editor ---
st.subheader("Output Sizes Mapping")
size_mappings = []
custom_sizes = []
records = []

if page is not None and doc_data and image_file:
    # Filter records for selected page
    records = [r for r in doc_data if r["page"] == page]

    # Load master asset and detect faces
    img = Image.open(image_file)
    img_w, img_h = img.size
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(detections) > 0:
        xs, ys, ws, hs = zip(*detections)
        face_box = {
            "left": int(min(xs)),
            "top": int(min(ys)),
            "right": int(max(x+w for x,w in zip(xs,ws))),
            "bottom": int(max(y+h for y,h in zip(ys,hs)))
        }
    else:
        face_box = None

    # Build size mapping table
    df_rows = []
    for rec in records:
        w_pt = rec["frame"]["w"]
        h_pt = rec["frame"]["h"]
        eff_x = rec["effectivePpi"]["x"]
        eff_y = rec["effectivePpi"]["y"]
        w_px = int(w_pt * eff_x / 72)
        h_px = int(h_pt * eff_y / 72)
        df_rows.append({
            "Template": rec["template"],
            "Width_pt": w_pt,
            "Height_pt": h_pt,
            "Width_px": w_px,
            "Height_px": h_px,
            "Aspect": rec.get("aspectRatio")
        })
    # Add custom entry
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
            "Aspect": st.column_config.NumberColumn("Aspect Ratio", format="%.2f")
        },
        hide_index=True,
        key="size_editor",
        num_rows="dynamic"
    )

    # Map rows to crop tasks
    for idx, row in edited.iloc[:len(records)].iterrows():
        w = row["Width_px"]
        h = row["Height_px"]
        if pd.notna(w) and pd.notna(h):
            size_mappings.append((records[idx], [int(w), int(h)], False))
    for _, row in edited.iloc[len(records):].iterrows():
        w = row["Width_px"]
        h = row["Height_px"]
        if pd.notna(w) and pd.notna(h):
            custom_sizes.append((int(w), int(h)))
else:
    st.info("Upload JSON, image, and select a page to begin.")

# --- Main: Crop Download ---
if size_mappings or custom_sizes:
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        tasks = size_mappings.copy()
        # Add custom tasks
        for cw,ch in custom_sizes:
            target_r = cw / ch
            best = min(records, key=lambda r: abs(r["aspectRatio"] - target_r))
            tasks.append((best, [cw, ch], True))

        for rec, out_size, is_custom in tasks:
            img_offset = rec["imageOffset"]
            frame = rec["frame"]
            disp_w, disp_h = img_offset["w"], img_offset["h"]
            off_x, off_y = abs(img_offset["x"]), abs(img_offset["y"])
            frac_x, frac_y = off_x/disp_w, off_y/disp_h
            frac_w, frac_h = frame["w"]/disp_w, frame["h"]/disp_h

            img = Image.open(image_file)
            img_w, img_h = img.size
            left = int(frac_x * img_w)
            top  = int(frac_y * img_h)
            w    = int(frac_w * img_w)
            h    = int(frac_h * img_h)

            target_ratio  = out_size[0] / out_size[1]
            current_ratio = w / h if h else target_ratio

            # Standard template adjustment
            if not is_custom and abs(current_ratio - target_ratio) > 1e-3:
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left += (w - new_w) // 2
                    w = new_w
                else:
                    new_h = int(w / target_ratio)
                    top += (h - new_h) // 2
                    h = new_h

            # Custom: face-aware crop
            if is_custom and face_box:
                # 1) Compute face union
                f_l, f_t = face_box['left'], face_box['top']
                f_r, f_b = face_box['right'], face_box['bottom']
                f_w, f_h = f_r - f_l, f_b - f_t
                # 2) Expand to target ratio
                if f_w / f_h > target_ratio:
                    crop_w = f_w
                    crop_h = int(f_w / target_ratio)
                else:
                    crop_h = f_h
                    crop_w = int(f_h * target_ratio)
                # 3) Center on face midpoint
                cx, cy = f_l + f_w//2, f_t + f_h//2
                left = max(0, min(img_w - crop_w, cx - crop_w//2))
                top  = max(0, min(img_h - crop_h, cy - crop_h//2))
                w, h = crop_w, crop_h

            # Clamp & extract
            left = max(0, min(left, img_w - w))
            top  = max(0, min(top, img_h - h))
            crop = img.crop((left, top, left + w, top + h))

            # Resize everything to requested size
            crop = crop.resize(tuple(out_size), Image.LANCZOS)

            buf = BytesIO()
            crop.save(buf, format="PNG")
            buf.seek(0)
            fname = f"{rec['template']}_{out_size[0]}x{out_size[1]}.png"
            zf.writestr(fname, buf.getvalue())

    zip_buffer.seek(0)
    st.download_button(
        "Download Crops",
        zip_buffer.getvalue(),
        file_name=f"page_{page}_crops.zip",
        mime="application/zip"
    )
else:
    if page is not None:
        st.warning("Define at least one size and upload an image to generate crops.")
