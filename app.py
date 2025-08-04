import streamlit as st
from PIL import Image
import json
import pandas as pd
import io

st.set_page_config(page_title="CropPack Tester", layout="wide")
st.title("CropPack Web App Prototype")

# --- Sidebar: Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload CropPack JSON", type=["json"])
image_file = st.sidebar.file_uploader("Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"])

st.sidebar.markdown("---")

# Load CropPack JSON
doc_data = []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec['page'] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
        pages = []
else:
    pages = []
    doc_data = []

# Select page and template records
if pages:
    page = st.sidebar.selectbox("Select Page", pages)
    records = [r for r in doc_data if r['page'] == page]
    record_templates = [r['template'] for r in records]
else:
    page = None
    records = []
    record_templates = []

# --- Output Sizes Editor ---
st.sidebar.header("Output Sizes")
if records:
    df_sizes = pd.DataFrame({
        'Template': record_templates + ['[CUSTOM]'],
        'Width': [None] * (len(record_templates) + 1),
        'Height': [None] * (len(record_templates) + 1)
    })
    edited = st.sidebar.experimental_data_editor(
        df_sizes,
        num_rows="dynamic",
        key="size_editor"
    )
    # Build mapping list
    size_mappings = []
    custom_sizes = []
    for _, row in edited.iterrows():
        tpl = row['Template']
        w, h = row['Width'], row['Height']
        if pd.notna(w) and pd.notna(h):
            if tpl == '[CUSTOM]':
                custom_sizes.append((int(w), int(h)))
            else:
                size_mappings.append({'template': tpl, 'size': [int(w), int(h)]})
else:
    size_mappings = []
    custom_sizes = []

# Main display
if records and image_file and (size_mappings or custom_sizes):
    st.header(f"Crops for Page {page}")
    img = Image.open(image_file)

    crops_to_show = []
    # Exact template crops
    for rec in records:
        for m in size_mappings:
            if m['template'] == rec['template']:
                crops_to_show.append((rec, m['size'], False))
    # Custom sizes crops
    for cw, ch in custom_sizes:
        target_r = cw / ch
        # find record with closest aspectRatio
        best = min(
            records,
            key=lambda r: abs(r.get('aspectRatio', (r['frame']['w']/r['frame']['h'])) - target_r)
        )
        crops_to_show.append((best, [cw, ch], True))

    cols = st.columns(len(crops_to_show) or 1)
    for idx, (rec, out_size, is_custom) in enumerate(crops_to_show):
        offs = rec['imageOffset']
        # Master crop region from JSON (points == pixels)
        base_left = offs['x']
        base_top = offs['y']
        base_w = offs['w']
        base_h = offs['h']

        if not is_custom:
            # Direct crop
            left, top, w, h = base_left, base_top, base_w, base_h
        else:
            # Custom: center new region within base region
            cw, ch = out_size
            target_ratio = cw / ch
            # Determine new region size in base pixels
            # Fit inside base_w x base_h
            new_w = base_w
            new_h = new_w / target_ratio
            if new_h > base_h:
                new_h = base_h
                new_w = new_h * target_ratio
            # Center region
            xc = base_left + base_w / 2
            yc = base_top + base_h / 2
            left = max(0, xc - new_w/2)
            top = max(0, yc - new_h/2)
            w, h = new_w, new_h

        # Perform crop and resize
        crop_img = img.crop((left, top, left + w, top + h))
        crop_img = crop_img.resize(tuple(out_size), Image.LANCZOS)

        tpl_label = rec['template'] + (" (custom)" if is_custom else "")
        col = cols[idx % len(cols)]
        col.subheader(f"{tpl_label} → {out_size[0]}×{out_size[1]}")
        col.image(crop_img, use_column_width=False)
else:
    st.info("Upload JSON, image, and define at least one output size.")
