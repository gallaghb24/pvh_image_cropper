import streamlit as st
from PIL import Image
import json
import pandas as pd
from io import BytesIO
import zipfile

st.set_page_config(page_title="CropPack Tester", layout="wide")
st.title("CropPack Web App Prototype")

# --- Sidebar: Upload Inputs ---
st.sidebar.header("Inputs")
json_file = st.sidebar.file_uploader("Upload CropPack JSON", type=["json"])
image_file = st.sidebar.file_uploader(
    "Upload Master Asset Image", type=["png","jpg","jpeg","tif","tiff"]
)

# Select Page
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
size_mappings, custom_sizes, records = [], [], []
if page is not None and doc_data:
    records = [r for r in doc_data if r["page"] == page]
    # Build DataFrame prefilled with pixel dims (pts → px via effective PPI)
    df_sizes = pd.DataFrame([
        {
            "Template": rec["template"],
            "Width": int(rec["frame"]["w"] * rec["effectivePpi"]["x"] / 72),
            "Height": int(rec["frame"]["h"] * rec["effectivePpi"]["y"] / 72)
        }
        for rec in records
    ] + [{"Template": "[CUSTOM]", "Width": None, "Height": None}])
    edited = st.data_editor(df_sizes, key="size_editor", num_rows="dynamic")
    for idx, row in edited.iterrows():
        tpl, w, h = row["Template"], row["Width"], row["Height"]
        if pd.notna(w) and pd.notna(h):
            # For duplicates, each row maps separately
            if tpl == "[CUSTOM]":
                custom_sizes.append((int(w), int(h)))
            else:
                size_mappings.append({"template": tpl, "size": [int(w), int(h)]})
else:
    st.info("Upload JSON and select a page to define output sizes.")
                size_mappings.append({"template": tpl, "size": [int(w), int(h)]})
else:
    st.info("Upload JSON and select a page to define output sizes.")

# --- Main: Crop Download ---
if page is not None and image_file and (size_mappings or custom_sizes):
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    img = Image.open(image_file)
    img_w, img_h = img.size

    # Prepare crop tasks
    crops_to_generate = []
    for rec in records:
        for m in size_mappings:
            if m["template"] == rec["template"]:
                crops_to_generate.append((rec, m["size"], False))
    for cw, ch in custom_sizes:
        target_r = cw / ch
        best = min(records, key=lambda r: abs(r["aspectRatio"] - target_r))
        crops_to_generate.append((best, [cw, ch], True))

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for rec, out_size, is_custom in crops_to_generate:
            # Compute fraction of original image shown
            img_offset = rec["imageOffset"]
            frame = rec["frame"]
            # displayed image size in pts
            disp_w = img_offset["w"]
            disp_h = img_offset["h"]
            # offset of visible window within displayed image
            off_x = abs(img_offset["x"])
            off_y = abs(img_offset["y"])
            # fraction
            frac_x = off_x / disp_w
            frac_y = off_y / disp_h
            frac_w = frame["w"] / disp_w
            frac_h = frame["h"] / disp_h
            # compute pixel coords on original high-res
            ox_px = int(frac_x * img_w)
            oy_px = int(frac_y * img_h)
            ow_px = int(frac_w * img_w)
            oh_px = int(frac_h * img_h)

            left, top, w, h = ox_px, oy_px, ow_px, oh_px
            target_ratio = out_size[0] / out_size[1]
            current_ratio = w / h if h else target_ratio
            # exact
            if not is_custom and abs(current_ratio - target_ratio) > 1e-3:
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left += (w - new_w) // 2
                    w = new_w
                else:
                    new_h = int(w / target_ratio)
                    top += (h - new_h) // 2
                    h = new_h
            # custom
            if is_custom:
                new_w = w
                new_h = int(new_w / target_ratio)
                if new_h > h:
                    new_h = h
                    new_w = int(new_h * target_ratio)
                xc = left + w // 2
                yc = top + h // 2
                left = max(0, xc - new_w // 2)
                top = max(0, yc - new_h // 2)
                w, h = new_w, new_h

            # clamp and crop
            left = max(0, min(left, img_w-1))
            top  = max(0, min(top, img_h-1))
            w    = max(1, min(w, img_w-left))
            h    = max(1, min(h, img_h-top))
            crop_img = img.crop((left, top, left+w, top+h))
            crop_img = crop_img.resize(tuple(out_size), Image.LANCZOS)

            tpl_label = rec["template"] + ("_custom" if is_custom else "")
            img_bytes = BytesIO()
            crop_img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            fname = f"{tpl_label}_{out_size[0]}x{out_size[1]}.png"
            zf.writestr(fname, img_bytes.getvalue())

    zip_buffer.seek(0)
    st.download_button(
        "Download Crops",
        data=zip_buffer.getvalue(),
        file_name=f"page_{page}_crops.zip",
        mime="application/zip",
    )
else:
    if page is not None:
        st.warning("Please upload an image and define at least one size.")
