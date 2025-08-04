import streamlit as st
from PIL import Image
import json
import pandas as pd

st.set_page_config(page_title="CropPack Tester", layout="wide")
st.title("CropPack Web App Prototype")

# --- Sidebar: Upload Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader(
    "Upload CropPack JSON", type=["json"]
)
image_file = st.sidebar.file_uploader(
    "Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"]
)

# Select Page
st.sidebar.markdown("---")
pages = []
doc_data = []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec['page'] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

page = None
if pages:
    page = st.sidebar.selectbox("Select Page", pages)

# Main: Output Sizes Editor
st.subheader("Output Sizes Mapping")
size_mappings = []
custom_sizes = []
records = []
if page is not None and doc_data:
    records = [r for r in doc_data if r['page'] == page]
    # Build default DataFrame
    df_sizes = pd.DataFrame([
        {
            'Template': rec['template'],
            'Width': rec['frame']['w'],
            'Height': rec['frame']['h']
        }
        for rec in records
    ] + [{'Template':'[CUSTOM]', 'Width':None, 'Height':None}])
    edited = st.data_editor(
        df_sizes,
        key="size_editor",
        num_rows="dynamic"
    )
    # Extract mappings
    for _, row in edited.iterrows():
        tpl, w, h = row['Template'], row['Width'], row['Height']
        if pd.notna(w) and pd.notna(h):
            if tpl == '[CUSTOM]':
                custom_sizes.append((int(w), int(h)))
            else:
                size_mappings.append({'template': tpl, 'size': [int(w), int(h)]})
else:
    st.info("Upload JSON and select a page to define output sizes.")

# Main: Crop Display
if page is not None and image_file and (size_mappings or custom_sizes):
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    img = Image.open(image_file)

    # Prepare crops
    crops_to_show = []
    for rec in records:
        for m in size_mappings:
            if m['template'] == rec['template']:
                crops_to_show.append((rec, m['size'], False))
    for cw, ch in custom_sizes:
        target_r = cw / ch
        best = min(
            records,
            key=lambda r: abs(r.get('aspectRatio', r['frame']['w']/r['frame']['h']) - target_r)
        )
        crops_to_show.append((best, [cw, ch], True))

    cols = st.columns(len(crops_to_show) or 1)
    for idx, (rec, out_size, is_custom) in enumerate(crops_to_show):
        offs = rec['imageOffset']
        base_left = offs['x']
        base_top = offs['y']
        base_w = offs['w']
        base_h = offs['h']

        if not is_custom:
            left, top, w, h = base_left, base_top, base_w, base_h
        else:
            cw, ch = out_size
            # center custom region in base
            target_ratio = cw/ch
            new_w = base_w
            new_h = new_w / target_ratio
            if new_h > base_h:
                new_h = base_h
                new_w = new_h * target_ratio
            xc = base_left + base_w/2
            yc = base_top + base_h/2
            left = max(0, xc - new_w/2)
            top  = max(0, yc - new_h/2)
            w, h = new_w, new_h

        crop_img = img.crop((left, top, left + w, top + h))
        crop_img = crop_img.resize(tuple(out_size), Image.LANCZOS)
        tpl_label = rec['template'] + (" (custom)" if is_custom else "")
        col = cols[idx % len(cols)]
        col.subheader(f"{tpl_label} → {out_size[0]}×{out_size[1]}")
        col.image(crop_img, use_column_width=False)
else:
    if page is not None:
        st.warning("Please upload an image and define at least one size.")
