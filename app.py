import streamlit as st
from PIL import Image
import json, pandas as pd
from io import BytesIO
import zipfile
import numpy as np
import cv2

# FULL APP (syntax‑checked end‑to‑end)
# ────────────────────────────────────────────────────────────────────────────────

def compute_crop(rec, img_w, img_h):
    off, fr = rec["imageOffset"], rec["frame"]
    fx, fy = abs(off["x"]), abs(off["y"])
    fw, fh = off["w"], off["h"]
    left = int((fx / fw) * img_w)
    top = int((fy / fh) * img_h)
    w = int((fr["w"] / fw) * img_w)
    h = int((fr["h"] / fh) * img_h)
    target_ar = fr["w"] / fr["h"]
    cur_ar = w / h if h else target_ar
    if abs(cur_ar - target_ar) > 1e-3:
        if cur_ar > target_ar:
            new_w = int(h * target_ar)
            left += (w - new_w) // 2
            w = new_w
        else:
            new_h = int(w / target_ar)
            top += (h - new_h) // 2
            h = new_h
    return left, top, w, h


def auto_custom_start(rec, img_w, img_h, cw, ch):
    l, t, w, h = compute_crop(rec, img_w, img_h)
    new_h = int(w / (cw / ch))
    t += (h - new_h) // 2
    return l, t, w, new_h

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit config
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Smart Crop", layout="wide")
st.markdown("<style>[data-testid=\"stSidebar\"]{min-width:450px!important;max-width:450px!important;}</style>", unsafe_allow_html=True)

# Sidebar
json_file = st.sidebar.file_uploader("Guidelines JSON", type="json")
image_file = st.sidebar.file_uploader("Master Image", type=["png","jpg","jpeg","tif","tiff"])

with st.sidebar.expander("Custom Crops", True):
    df = st.data_editor(pd.DataFrame([{"Width_px":None,"Height_px":None}]), hide_index=True, num_rows="dynamic")
    custom_sizes = [(int(r.Width_px), int(r.Height_px)) for r in df.itertuples() if pd.notna(r.Width_px) and pd.notna(r.Height_px)]

# Load records
records = []
if json_file and image_file:
    data = json.load(json_file)
    fname = image_file.name
    for rec in data:
        spec = rec.get("filename") or rec.get("fileName") or rec.get("asset")
        if spec and (fname == spec or fname.startswith(spec)):
            records.append(rec)
else:
    st.info("Upload JSON and image to get started.")

if records:
    guidelines = [r for r in records if not (abs(r["imageOffset"]["x"])<1e-6 and abs(r["imageOffset"]["y"])<1e-6)]

    # Tables
    gtable = []
    for rec in guidelines:
        w_pt,h_pt = rec["frame"]["w"], rec["frame"]["h"]
        ex,ey = rec["effectivePpi"]["x"], rec["effectivePpi"]["y"]
        w_px,h_px = int(w_pt*ex/72), int(h_pt*ey/72)
        gtable.append({"Template": rec["template"],"Width_px":w_px,"Height_px":h_px,"AR":round(w_px/h_px,2)})
    st.dataframe(pd.DataFrame(gtable), use_container_width=True)

    ctable = []
    for cw,ch in custom_sizes:
        base = min(records, key=lambda r: abs((cw/ch)-r["frame"]["w"] / r["frame"]["h"]))
        ctable.append({"Template":f"{base['template']} {cw}×{ch}","Width_px":cw,"Height_px":ch,"AR":round(cw/ch,2)})
    st.dataframe(pd.DataFrame(ctable), use_container_width=True)

    img = Image.open(image_file)
    iw, ih = img.size

    # Adjustment UI
    shifts = {}
    if custom_sizes:
        tabs = st.tabs([f"{w}×{h}" for w,h in custom_sizes])
        for i,((cw,ch),tab) in enumerate(zip(custom_sizes,tabs)):
            with tab:
                base = min(records, key=lambda r: abs((cw/ch)-r["frame"]["w"] / r["frame"]["h"]))
                l0,t0,wb,hb = auto_custom_start(base, iw, ih, cw, ch)
                z_key = f"z{i}"; st.session_state.setdefault(z_key,0)
                colz1,colz2=st.columns([3,1])
                with colz1: zd = st.slider("Zoom ±10%", -10,10,st.session_state[z_key],1)
                with colz2: zd = st.number_input("", -10,10,zd,1)
                st.session_state[z_key]=zd
                zoom=1+zd/100
                wz,hz = int(wb/zoom),int(hb/zoom)
                cx,cy = l0+wb//2,t0+hb//2
                ls,ts = cx-wz//2, cy-hz//2
                min_x,max_x = -ls, iw-ls-wz
                min_y,max_y = -ts, ih-ts-hz
                sx_key=f"sx{i}"; st.session_state.setdefault(sx_key,0)
                sy_key=f"sy{i}"; st.session_state.setdefault(sy_key,0)
                colw1,colw2=st.columns([3,1])
                with colw1: sx=0 if min_x==max_x else st.slider("Width Offset",min_x,max_x,st.session_state[sx_key],1)
                with colw2: sx=st.number_input("",min_x,max_x,sx,1)
                st.session_state[sx_key]=sx
                colh1,colh2=st.columns([3,1])
                if min_y==max_y:
                    with colh1: st.markdown("<div style='height:35px'></div>",unsafe_allow_html=True)
                    with colh2: sy=st.number_input("Height Offset",value=0,disabled=True)
                else:
                    with colh1: sy=st.slider("Height Offset",min_y,max_y,st.session_state[sy_key],1)
                    with colh2: sy=st.number_input("",min_y,max_y,sy,1)
                st.session_state[sy_key]=sy
                x0,y0 = ls+sx, ts+sy
                prev = img.crop((x0,y0,x0+wz,y0+hz)).resize((cw,ch),Image.LANCZOS)
                st.image(prev,width=600)
                shifts[(cw,ch)] = (sx,sy,zoom)

    # ZIP
    buf=BytesIO()
    with zipfile.ZipFile(buf,"w") as zf:
        for rec in guidelines:
            ow=int(rec["frame"]["w"]*rec["effectivePpi"]["x"]/72)
            oh=int(rec["frame"]["h"]*rec["effectivePpi"]["y"]/72)
            l,t,wb,hb = compute_crop(rec,iw,ih)
            c=img.crop((l,t,l+wb,t+hb)).resize((ow,oh),Image.LANCZOS)
            b=BytesIO();c.save(b,format="PNG");b.seek(0)
            zf.writestr(f"Guidelines/{rec['template']}_{ow}x{oh}.png",b.getvalue())
        for cw,ch in custom_sizes:
            base = min(records, key=lambda r: abs((cw/ch)-r["frame"]["w"] / r["frame"]["h"]))
            l,t,wb,hb = auto_custom_start(base,iw,ih,cw,ch)
            sx,sy,zoom = shifts.get((cw,ch),(0,0,1))
            wz,hz = int(wb/zoom),int(hb/zoom)
            cx,cy=l+wb//2,t+hb//2
            l2=max(0,min(cx-wz//2+sx,iw-wz))
            t2=max(0,min(cy-hz//2+sy,ih-hz))
            c=img.crop((l2,t2,l2+wz,t2+hz)).resize((cw,ch),Image.LANCZOS)
            b=BytesIO();c.save(b,format="PNG");b.seek(0)
            zf.writestr(f"Custom/{cw}x{ch}.png",b.getvalue())
    buf.seek(0)
    st.download_button("Download Crops",buf.getvalue(),file_name=f"crops_{image_file.name}.zip",mime="application/zip")
