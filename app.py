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
    rows = []
    for rec in records:
        w_pt, h_pt = rec['frame']['w'], rec['frame']['h']
        eff_x, eff_y = rec['effectivePpi']['x'], rec['effectivePpi']['y']
        rows.append({
            'Template': rec['template'],
            'Width_px': int(w_pt * eff_x / 72),
            'Height_px': int(h_pt * eff_y / 72)
        })
    rows.append({'Template':'[CUSTOM]', 'Width_px':None, 'Height_px':None})
    df = pd.DataFrame(rows)
    edited = st.data_editor(df, hide_index=True, num_rows='dynamic', key='map_editor')
    for i, row in edited.iloc[:len(records)].iterrows():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            size_mappings.append((records[i], [int(row.Width_px), int(row.Height_px)], False))
    for row in edited.iloc[len(records):].itertuples():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            custom_sizes.append((int(row.Width_px), int(row.Height_px)))
else:
    size_mappings, custom_sizes = [], []

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
    for (cw, ch), tab in zip(custom_sizes, tabs):
        with tab:
            st.write(f"Adjust crop for **{cw}×{ch}**")
            rec = min(records, key=lambda r: abs((cw/ch) - r.get("aspectRatio", r["frame"]["w"]/r["frame"]["h"])))
            init_l, init_t, init_w, init_h = auto_custom_start(rec, img_w, img_h, cw, ch)

            # Compute shift bounds
            min_x = -init_l
            max_x = img_w - init_l - init_w
            min_y = -init_t
            max_y = img_h - init_t - init_h

                                    # Inputs with automatic default (ensure min<=max)
            if min_x > max_x:
                max_x = min_x
            if min_y > max_y:
                min_y = max_y
            shift_x = st.number_input(
                "Shift left/right (px)",
                min_value=min_x,
                max_value=max_x,
                step=1,
                key=f"shiftx_{cw}_{ch}"
            )
            shift_y = st.number_input(
                "Shift up/down (px)",
                min_value=min_y,
                max_value=max_y,
                step=1,
                key=f"shifty_{cw}_{ch}"
            )

            # Compute preview crop and show
            x0 = init_l + shift_x
            y0 = init_t + shift_y
            x1 = x0 + init_w
            y1 = y0 + init_h
            crop_preview = img_orig.crop((x0, y0, x1, y1))
            crop_preview = crop_preview.resize((cw, ch), Image.LANCZOS)
            st.image(crop_preview, caption=f"Preview {cw}×{ch}", use_container_width=True)

            custom_shifts[(cw, ch)] = (shift_x, shift_y) (ensure min<=max)
            if min_x > max_x:
                max_x = min_x
            if min_y > max_y:
                min_y = max_y
            shift_x = st.number_input(
                "Shift left/right (px)",
                min_value=min_x,
                max_value=max_x,
                step=1,
                key=f"shiftx_{cw}_{ch}"
            )
            shift_y = st.number_input(
                "Shift up/down (px)",
                min_value=min_y,
                max_value=max_y,
                step=1,
                key=f"shifty_{cw}_{ch}"
            )

            # Show preview
            crop = img_orig.crop((
                init_l + shift_x,
                init_t + shift_y,
                init_l + shift_x + init_w,
                init_t + shift_y + init_h
            ))[(cw, ch)] = (shift_x, shift_y)

# --- Generate & Download ---[(cw, ch)] = (shift_x, shift_y)

        # --- Generate & Download ---[(cw, ch)] = (shift_x, shift_y)

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
            rec = min(records, key=lambda r: abs((cw/ch) - r.get('aspectRatio', r['frame']['w']/r['frame']['h'])))
            left, top, w, h = auto_custom_start(rec, img_w, img_h, cw, ch)
            sx, sy = custom_shifts.get((cw, ch), (0,0))
            left = max(0, min(left+sx, img_w-w))
            top  = max(0, min(top+sy, img_h-h))
            crop = img_orig.crop((left, top, left+w, top+h)).resize((cw, ch), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
            fname = f"custom_{cw}x{ch}.png"
            zf.writestr(fname, buf.getvalue())
    zip_buf.seek(0)
    st.download_button('Download Crops', zip_buf.getvalue(), file_name=f'crops_{image_file.name}.zip', mime='application/zip')
else:
    if image_file:
        st.warning('Define at least one output size to generate crops.')
