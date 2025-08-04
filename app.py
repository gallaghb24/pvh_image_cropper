import streamlit as st
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2
import base64
from streamlit_drawable_canvas import st_canvas

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


def auto_custom_box(face_box, img_w, img_h, cw, ch):
    # initial full-width crop to aspect
    target = cw / ch
    crop_h = int(img_w / target)
    left = 0
    w = img_w
    top = max(0, (img_h - crop_h) // 2)
    h = crop_h
    # vertical nudge to include faces
    if face_box:
        if face_box['top'] < top:
            top = face_box['top']
        if face_box['bottom'] > top + h:
            top = face_box['bottom'] - h
        top = max(0, min(top, img_h - h))
    return left, top, w, h

# --- App Setup ---
st.set_page_config(page_title='CropPack Tester', layout='wide')
st.title('CropPack Web App with Manual Cropping')

# --- Sidebar Inputs ---
st.sidebar.header('Inputs')
json_file = st.sidebar.file_uploader('Upload CropPack JSON', type=['json'])
image_file = st.sidebar.file_uploader('Upload Master Asset Image', type=['png','jpg','jpeg','tif','tiff'])
st.sidebar.markdown('---')

# --- Page Selection ---
pages, doc_data = [], []
if json_file:
    try:
        doc_data = json.load(json_file)
        pages = sorted({rec['page'] for rec in doc_data})
    except Exception as e:
        st.sidebar.error(f'Invalid JSON: {e}')
page = st.sidebar.selectbox('Select Page', pages) if pages else None

# --- Output Sizes Mapping ---
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
    # build tasks
    for i, row in edited.iloc[:len(records)].iterrows():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            size_mappings.append((records[i], [int(row.Width_px), int(row.Height_px)], False))
    for row in edited.iloc[len(records):].itertuples():
        if pd.notna(row.Width_px) and pd.notna(row.Height_px):
            custom_sizes.append((int(row.Width_px), int(row.Height_px)))
else:
    st.info('Upload JSON and select a page to define sizes.')

# --- Precompute Face Box ---
face_box = None
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

# --- Crop Generation ---
if (size_mappings or custom_sizes) and image_file:
    st.markdown('---')
    st.header(f'Crops for Page {page}')
    st.image(img_orig, caption='Master Asset', use_container_width=True)
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        # Template crops
        for rec, out_size, _ in size_mappings:
            left, top, w, h = compute_crop(rec, img_w, img_h)
            crop = img_orig.crop((left, top, left+w, top+h)).resize(tuple(out_size), Image.LANCZOS)
            buf = BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
            zf.writestr(f"{rec['template']}_{out_size[0]}x{out_size[1]}.png", buf.getvalue())
                        # Canvas-assisted custom crops
        for cw, ch in custom_sizes:
            st.subheader(f'Custom Crop: {cw}×{ch}')
            init_l, init_t, init_w, init_h = auto_custom_box(face_box, img_w, img_h, cw, ch)
            # encode image as base64 data URI
            buf = BytesIO()
            img_orig.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            bg_url = f"data:image/png;base64,{b64}"
            # launch drawable canvas with URL
            canvas_data = st_canvas(
                fill_color='', stroke_width=2,
                background_image_url=bg_url,
                width=img_w, height=img_h,
                initial_drawing=[{'type':'rect','x':init_l,'y':init_t,'width':init_w,'height':init_h,'strokeColor':'#00FF00'}],
                drawing_mode='transform'
            )
                fill_color='', stroke_width=2,
                background_image=bg_bytes,
                width=img_w, height=img_h,
                initial_drawing=[{'type':'rect','x':init_l,'y':init_t,'width':init_w,'height':init_h,'strokeColor':'#00FF00'}],
                drawing_mode='transform'
            )
            if canvas_data.json_data and canvas_data.json_data.get('objects'):
                obj = canvas_data.json_data['objects'][0]
                l = int(obj['left']); t = int(obj['top'])
                cw_box = int(obj['width']); ch_box = int(obj['height'])
                final = img_orig.crop((l, t, l+cw_box, t+ch_box)).resize((cw, ch), Image.LANCZOS)
                buf = BytesIO(); final.save(buf, format='PNG'); buf.seek(0)
                zf.writestr(f'custom_{cw}x{ch}.png', buf.getvalue())
    zip_buf.seek(0)
    st.download_button('Download All Crops', zip_buf.getvalue(), file_name=f'crops_page_{page}.zip', mime='application/zip')
else:
    if page:
        st.warning('Define sizes to generate crops.')
