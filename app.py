import streamlit as st
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2

# --- Helper Functions ---
def compute_crop(rec, img_w, img_h):
    off = rec['imageOffset']
    fr = rec['frame']
    fx, fy = abs(off['x']), abs(off['y'])
    fw, fh = off['w'], off['h']
    left = int((fx/fw) * img_w)
    top  = int((fy/fh) * img_h)
    w    = int((fr['w']/fw) * img_w)
    h    = int((fr['h']/fh) * img_h)
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

# Custom: initial ratio-based crop
# then allow manual nudges via sliders
def auto_custom_start(rec, img_w, img_h, cw, ch):
    # start from template rec window
    left, top, w, h = compute_crop(rec, img_w, img_h)
    tgt = cw / ch
    new_h = int(w / tgt)
    top = top + (h - new_h) // 2
    return left, top, w, new_h

# --- App Setup ---
st.set_page_config(page_title='CropPack Tester', layout='wide')
st.title('Smart Crop Automation Prototype')

# --- Sidebar Inputs ---
st.sidebar.header('Inputs')
json_file = st.sidebar.file_uploader('Upload Cropping Guidelines JSON', type=['json'])
image_file = st.sidebar.file_uploader('Upload Master Asset Image', type=['png','jpg','jpeg','tif','tiff'])
st.sidebar.markdown('---')

# --- Load data ---
doc_data = []
if json_file:
    try:
        doc_data = json.load(json_file)
    except Exception as e:
        st.sidebar.error(f'Invalid JSON: {e}')

# --- Determine records by filename ---
st.subheader('Detected Asset & Crops')
records = []
if doc_data and image_file:
    fname = image_file.name
    # match rec['filename'] or rec.get('filename') to uploaded name
    for rec in doc_data:
        f = rec.get('filename') or rec.get('fileName') or rec.get('asset') or rec.get('template')
        if f and (fname == f or fname.startswith(f)):
            records.append(rec)
    if not records:
        st.error('Uploaded image filename not found in JSON. Please ensure the JSON has a matching filename field.')
else:
    st.info('Upload both JSON and an image to detect crops.')

# --- Output Sizes Mapping ---
st.subheader('Output Sizes Mapping')
size_mappings, custom_sizes = [], []
if records:
    # Build DataFrame of templates and a custom placeholder
    rows = []
    for rec in records:
        w_pt, h_pt = rec['frame']['w'], rec['frame']['h']
        eff_x, eff_y = rec['effectivePpi']['x'], rec['effectivePpi']['y']
        w_px = int(w_pt * eff_x / 72)
        h_px = int(h_pt * eff_y / 72)
        ar = round(w_px / h_px, 2) if h_px else None
        rows.append({
            'Template': rec['template'],
            'Width_px': w_px,
            'Height_px': h_px,
            'Aspect Ratio': ar
        })
    rows.append({'Template':'[CUSTOM]', 'Width_px':None, 'Height_px':None, 'Aspect Ratio':None})
    df_sizes = pd.DataFrame(rows)
    edited_df = st.data_editor(df_sizes, hide_index=True, num_rows='dynamic', key='map_editor')
    # Extract mappings
    size_mappings = []
    custom_sizes = []
    # Templates
    for i, rec in enumerate(records):
        w = edited_df.at[i, 'Width_px']
        h = edited_df.at[i, 'Height_px']
        if pd.notna(w) and pd.notna(h):
            size_mappings.append((rec, [int(w), int(h)], False))
    # Customs
    for j in range(len(records), len(edited_df)):
        w = edited_df.at[j, 'Width_px']
        h = edited_df.at[j, 'Height_px']
        if pd.notna(w) and pd.notna(h):
            custom_sizes.append((int(w), int(h)))
else:
    st.info('Upload image & JSON to configure sizes.')

# --- Face Detection ---
face_box = None
if image_file:
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size
    np_img = np.array(img_orig)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(dets)>0:
        xs, ys, ws, hs = zip(*dets)
        face_box = {
            'left': min(xs), 'top': min(ys),
            'right': max(x+w for x,w in zip(xs,ws)),
            'bottom': max(y+h for y,h in zip(ys,hs))
        }

# --- Manual Nudges Tabs for Custom Sizes ---
custom_shifts = {}
if custom_sizes and image_file:
    tabs = st.tabs([f"{cw}×{ch}" for cw,ch in custom_sizes])
    for i, ((cw, ch), tab) in enumerate(zip(custom_sizes, tabs)):
        with tab:
            st.write(f"Adjust crop for **{cw}×{ch}**")
            # pick best template ratio
            rec = min(
                records,
                key=lambda r: abs((cw/ch) - r.get("aspectRatio", r["frame"]["w"]/r["frame"]["h"]))
            )
            # initial custom window
            init_l, init_t, init_w, init_h = auto_custom_start(rec, img_w, img_h, cw, ch)
            # compute shift bounds
            min_x, max_x = -init_l, img_w - init_l - init_w
            min_y, max_y = -init_t, img_h - init_t - init_h
            # clamp any inverted bounds
            if min_x > max_x: min_x, max_x = max_x, min_x
            if min_y > max_y: min_y, max_y = max_y, min_y
            # user inputs
            shift_x = st.number_input(
                "Shift left/right (px)", min_value=min_x, max_value=max_x,
                step=1, key=f"shiftx_{cw}_{ch}_{i}"
            )
            shift_y = st.number_input(
                "Shift up/down (px)", min_value=min_y, max_value=max_y,
                step=1, key=f"shifty_{cw}_{ch}_{i}"
            )
            # generate preview once
            x0, y0 = init_l + shift_x, init_t + shift_y
            x1, y1 = x0 + init_w, y0 + init_h
            preview = img_orig.crop((x0, y0, x1, y1)).resize((cw, ch), Image.LANCZOS)
            st.image(preview, caption=f"Preview {cw}×{ch}", width=900)
            custom_shifts[(cw, ch)] = (shift_x, shift_y)

# --- Generate & Download ---
if (size_mappings or custom_sizes) and image_file:
    st.markdown('---')
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        # template crops
        for rec, out_size, _ in size_mappings:
            left, top, w, h = compute_crop(rec, img_w, img_h)
            crop = img_orig.crop((left, top, left+w, top+h)).resize(tuple(out_size), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
            fname = f"{rec['template']}_{out_size[0]}x{out_size[1]}.png"
            zf.writestr(fname, buf.getvalue())
        # custom crops with manual shifts
        for cw, ch in custom_sizes:
            rec = min(
                records,
                key=lambda r: abs((cw/ch) - r.get('aspectRatio', r['frame']['w']/r['frame']['h']))
            )
            # compute initial custom window and apply shifts
            left, top, w, h = auto_custom_start(rec, img_w, img_h, cw, ch)
            sx, sy = custom_shifts.get((cw, ch), (0,0))
            left = max(0, min(left + sx, img_w - w))
            top  = max(0, min(top + sy, img_h - h))
            # crop and resize
            crop = img_orig.crop((left, top, left + w, top + h)).resize((cw, ch), Image.LANCZOS)
            buf = BytesIO()
            crop.save(buf, format='PNG')
            buf.seek(0)
            # build label and filename
            label = f"{rec['template']} Custom {cw}x{ch}"
            fname = f"{label}.png"
            zf.writestr(fname, buf.getvalue())
