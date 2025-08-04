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

# --- Streamlit App ---
st.set_page_config(page_title='CropPack Tester', layout='wide')
st.title('CropPack Web App Prototype')

# --- Sidebar Inputs ---
st.sidebar.header('Inputs')
json_file = st.sidebar.file_uploader('Upload CropPack JSON', type=['json'])
image_file = st.sidebar.file_uploader('Upload Master Asset Image', type=['png','jpg','jpeg','tif','tiff'])

# --- Page Selection ---
pages, doc_data = [], []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec['page'] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f'Invalid JSON: {e}')
page = st.sidebar.selectbox('Select Page', pages) if pages else None

# --- Size Mapping Editor ---
st.subheader('Output Sizes Mapping')
size_mappings, custom_sizes = [], []
if page and doc_data:
    records = [r for r in doc_data if r['page'] == page]
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
    st.info('Upload JSON and select a page to define sizes.')

# --- Load Master Image & Detect Faces ---
if image_file:
    img_orig = Image.open(image_file)
    img_w, img_h = img_orig.size
    np_img = np.array(img_orig)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(dets) > 0:
        xs, ys, ws, hs = zip(*dets)
        face_box = {
            'left': min(xs), 'top': min(ys),
            'right': max(x+w for x,w in zip(xs,ws)),
            'bottom': max(y+h for y,h in zip(ys,hs))
        }
    else:
        face_box = None
else:
    face_box = None

# --- Crop Generation & Manual Adjust ---
if (size_mappings or custom_sizes) and image_file:
    st.markdown('---')
    st.header(f'Crops for Page {page}')
    st.image(img_orig, caption='Master Asset', use_container_width=True)
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        # Automatic template crops
        for rec, out_size, _ in size_mappings:
            left, top, w, h = compute_crop(rec, img_w, img_h)
            crop = img_orig.crop((left, top, left+w, top+h)).resize(tuple(out_size), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
            zf.writestr(f"{rec['template']}_{out_size[0]}x{out_size[1]}.png", buf.getvalue())

        # Manual-adjust custom crops
        st.subheader('Custom Crop Adjustments')
        manual_settings = {}
        for cw, ch in custom_sizes:
            st.markdown(f'**Custom Size: {cw}×{ch}**')
            # Initial window
            init_left, init_top, init_w, init_h = 0, 0, img_w, img_h
            target_ratio = cw/ch
            # compute initial width-based crop
            init_crop_h = int(init_w / target_ratio)
            init_top = (img_h - init_crop_h)//2
            init_h = init_crop_h
            # sliders
            max_shift_vert = img_h - init_h
            shift_vert = st.slider(f'Vertical shift for {cw}×{ch}', -init_top, max_shift_vert - init_top, 0)
            max_shift_horiz = img_w - init_w
            shift_horiz = st.slider(f'Horizontal shift for {cw}×{ch}', -init_left, max_shift_horiz - init_left, 0)
            # apply shifts
            left = max(0, min(init_left + shift_horiz, img_w - init_w))
            top  = max(0, min(init_top + shift_vert, img_h - init_h))
            # preview
            preview = img_orig.crop((left, top, left+init_w, top+init_h)).resize((200, int(200*init_h/init_w)), Image.LANCZOS)
            st.image(preview, caption='Preview', use_container_width=False)
            manual_settings[f'{cw}x{ch}'] = (left, top, init_w, init_h)

        # Save manual crops
        for key, (left, top, w, h) in manual_settings.items():
            cw, ch = map(int, key.split('x'))
            final = img_orig.crop((left, top, left+w, top+h)).resize((cw, ch), Image.LANCZOS)
            buf = BytesIO(); final.save(buf, format='PNG'); buf.seek(0)
            zf.writestr(f'custom_{key}.png', buf.getvalue())

    zip_buf.seek(0)
    st.download_button('Download All Crops', zip_buf.getvalue(), file_name=f'crops_page_{page}.zip', mime='application/zip')
else:
    if page:
        st.warning('Define sizes and upload an image to generate crops.')
