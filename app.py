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
# We’ll map each record individually rather than grouping by template
size_mappings = []
custom_sizes = []
records = []
if page is not None and doc_data:
    # Filter to this page
    records = [r for r in doc_data if r["page"] == page]
    # Build DataFrame rows matching each record
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
    # Add a CUSTOM row at end
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
    # Map each of the first N rows to its record
    for idx, row in edited.iloc[:len(records)].iterrows():
        tpl = row["Template"]
        w = row["Width_px"]
        h = row["Height_px"]
        if pd.notna(w) and pd.notna(h):
            size_mappings.append((records[idx], [int(w), int(h)], False))
    # CUSTOM rows
    for _, row in edited.iloc[len(records):].iterrows():
        w = row["Width_px"]
        h = row["Height_px"]
        if pd.notna(w) and pd.notna(h):
            custom_sizes.append((int(w), int(h)))
else:
    st.info("Upload JSON and select a page to define output sizes.")

# --- Main: Crop Download ---
if page is not None and image_file and (size_mappings or custom_sizes):
    st.markdown("---")
    st.header(f"Crops for Page {page}")
    img = Image.open(image_file)
    img_w, img_h = img.size

    # Combine exact and custom tasks
    crops_to_generate = []
    # exact crops: size_mappings now holds (record, size, False)
    crops_to_generate.extend(size_mappings)
    # custom crops
    for cw, ch in custom_sizes:
        target_r = cw / ch
        # pick the best matching record
        best = min(records, key=lambda r: abs(r["aspectRatio"] - target_r))
        crops_to_generate.append((best, [cw, ch], True))

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for rec, out_size, is_custom in crops_to_generate:
            img_offset = rec["imageOffset"]
            frame = rec["frame"]
            disp_w = img_offset["w"]
            disp_h = img_offset["h"]
            off_x = abs(img_offset["x"])
            off_y = abs(img_offset["y"])
            frac_x = off_x / disp_w
            frac_y = off_y / disp_h
            frac_w = frame["w"] / disp_w
            frac_h = frame["h"] / disp_h
            ox_px = int(frac_x * img_w)
            oy_px = int(frac_y * img_h)
            ow_px = int(frac_w * img_w)
            oh_px = int(frac_h * img_h)

            left, top, w, h = ox_px, oy_px, ow_px, oh_px
            target_ratio = out_size[0] / out_size[1]
            current_ratio = w / h if h else target_ratio

            # exact template centering
            if not is_custom and abs(current_ratio - target_ratio) > 1e-3:
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left += (w - new_w) // 2
                    w = new_w
                else:
                    new_h = int(w / target_ratio)
                    top += (h - new_h) // 2
                    h = new_h

            # custom size centering
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

            # clamp
            left = max(0, min(left, img_w - 1))
            top = max(0, min(top, img_h - 1))
            w = max(1, min(w, img_w - left))
            h = max(1, min(h, img_h - top))

            crop_img = img.crop((left, top, left + w, top + h))
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
                    new_h = int(w / target_ratio)
                    top += (h - new_h) // 2
                    h = new_h

            # custom size centering
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

            # clamp
            left = max(0, min(left, img_w - 1))
            top = max(0, min(top, img_h - 1))
            w = max(1, min(w, img_w - left))
            h = max(1, min(h, img_h - top))

            crop_img = img.crop((left, top, left + w, top + h))
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
