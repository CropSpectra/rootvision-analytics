import io
import zipfile
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize

st.set_page_config(page_title="RootVision", page_icon="🌱", layout="wide")
st.title("🌱 RootVision — Root Image Measurements")
st.warning("Experimental tool, not validated for research accuracy. Inspect every mask.")
st.caption("Single and batch images use the same real pixel-measurement pipeline.")


def measure(mask):
    s = skeletonize(mask)
    h = np.count_nonzero(s[:, :-1] & s[:, 1:])
    v = np.count_nonzero(s[:-1, :] & s[1:, :])
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    length = float(h + v + np.sqrt(2) * (d1.sum() + d2.sum()))
    ys, xs = np.nonzero(mask)
    return s, length, int(mask.sum()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)

mode = st.radio("Workflow", ["Single image", "Batch images"], horizontal=True)
files = st.file_uploader("Upload root images", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)
if not files:
    st.info("Upload images to begin. TIFF files: only the first frame is analyzed.")
    st.stop()
if mode == "Single image" and len(files) > 1:
    st.error("Choose Batch images to process multiple files, or upload one file.")
    st.stop()
st.info("Configure each image below. Crop out rulers, labels, shoots and seeds. Scale is entered separately for each image; no calibration is assumed.")
rows = []
artifacts = {}
for index, uploaded in enumerate(files):
    raw = uploaded.getvalue()
    key = f"{index}_{hashlib.sha256(raw).hexdigest()[:12]}"
    with st.expander(f"{index+1}. {uploaded.name}", expanded=len(files)==1):
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            rgb = np.asarray(image)
            height, width = rgb.shape[:2]
            st.caption(f"Original oriented image: {width} × {height} pixels. Analysis uses full resolution; preview may be resized.")
            st.image(image, caption="Original", use_column_width=True)
            c1, c2 = st.columns(2)
            xr = c1.slider("Horizontal crop (%)", 0, 100, (0, 100), key=key+"x")
            yr = c2.slider("Vertical crop (%)", 0, 100, (0, 100), key=key+"y")
            x0, x1 = [int(width*x/100) for x in xr]
            y0, y1 = [int(height*y/100) for y in yr]
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Crop must have positive width and height.")
            crop = rgb[y0:y1, x0:x1]
            calibrated = st.checkbox("Apply physical scale", key=key+"cal")
            scale = None
            if calibrated:
                st.caption("Measure a known ruler interval in ORIGINAL oriented-image pixels using an image editor. Enter the Euclidean pixel distance and its physical distance. Ruler and roots must be in the same plane; perspective distortion is not corrected.")
                a, b = st.columns(2)
                px = a.number_input("Ruler interval (pixels)", min_value=0.01, value=100.0, key=key+"px")
                cm = b.number_input("Same interval (cm)", min_value=0.001, value=1.0, format="%.3f", key=key+"cm")
                scale = px/cm
                st.write(f"Scale: {scale:.3f} pixels/cm")
            polarity = st.selectbox("Root appearance", ["Light roots on dark background", "Dark roots on light background"], key=key+"pol")
            gray = np.dot(crop[..., :3], [0.2126, 0.7152, 0.0722])
            automatic = st.checkbox("Automatic brightness threshold", value=True, key=key+"auto")
            if automatic:
                threshold = float(threshold_otsu(gray))
                st.caption(f"Automatic threshold: {threshold:.1f}/255")
            else:
                threshold = st.slider("Brightness threshold", 0, 255, 150, key=key+"thr")
            minimum = st.number_input("Remove isolated objects smaller than (pixels)", min_value=1, value=30, key=key+"min")
            mask = gray > threshold if polarity.startswith("Light") else gray < threshold
            mask = remove_small_objects(mask, min_size=int(minimum))
            if not mask.any():
                raise ValueError("No selected root pixels. Change crop, polarity or threshold.")
            skeleton, length, area, span_x, span_y = measure(mask)
            overlay = crop.copy()
            overlay[mask] = (overlay[mask]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
            overlay[skeleton] = [255,0,0]
            a, b = st.columns(2)
            a.image(mask.astype(np.uint8)*255, caption="White = measured pixels", use_column_width=True)
            b.image(overlay, caption="Green = mask; red = skeleton", use_column_width=True)
            border = bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
            if border:
                st.warning("Selected pixels touch the crop border: roots may be cut off or background may be selected.")
            reviewed = st.checkbox("I reviewed this mask: roots are adequately included and non-root objects excluded", key=key+"review")
            st.caption("Review is your visual judgment, not an automated accuracy certification. Recheck after any setting change.")
            divisor = scale if scale else 1.0
            unit = "cm" if scale else "px"
            c1,c2 = st.columns(2)
            c1.metric("Estimated visible root length", f"{length/divisor:.2f} {unit}")
            c2.metric("Projected selected area", f"{area/divisor**2:.2f} {unit}²")
            c1,c2 = st.columns(2)
            c1.metric("Horizontal span", f"{span_x/divisor:.2f} {unit}")
            c2.metric("Vertical span", f"{span_y/divisor:.2f} {unit}")
            rows.append(dict(image_index=index+1, image=uploaded.name, status="reviewed" if reviewed else "needs_visual_review", crop_border_contact=border, estimated_visible_length=length/divisor, projected_area=area/divisor**2, horizontal_span=span_x/divisor, vertical_span=span_y/divisor, length_unit=unit, area_unit=unit+"^2", pixels_per_cm=scale, threshold=threshold, polarity=polarity, minimum_object_pixels=minimum, crop_x0=x0, crop_x1=x1, crop_y0=y0, crop_y1=y1, original_width=width, original_height=height, image_sha256=hashlib.sha256(raw).hexdigest()))
            for name, arr in [("mask", mask.astype(np.uint8)*255), ("overlay", overlay)]:
                buffer = io.BytesIO()
                Image.fromarray(arr).save(buffer, format="PNG")
                artifacts[f"image_{index+1:03d}_{name}.png"] = buffer.getvalue()
        except Exception as exc:
            st.error(f"Image could not be measured: {exc}")
            rows.append(dict(image_index=index+1, image=uploaded.name, status="failed", error=str(exc)))

st.subheader("Results and exports")
st.caption("Pixel and centimeter rows must not be pooled. Failed images remain in the CSV. Review flags are exported; unreviewed rows are provisional.")
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download measurements CSV", csv, "root_measurements.csv", "text/csv")
notes = """RootVision experimental image analysis
No trained AI model. No simulated measurements.
Length: sum of orthogonal (1 px) and unbridged diagonal (sqrt(2) px) links on a binary skeleton.
Area: selected foreground pixel count, converted using scale squared if calibrated.
Spans: selected-mask bounding-box dimensions, not soil rooting depth.
Projected area is NOT cylindrical surface area. No volume, root density, biological root count, root order or health score is inferred.
Overlaps hide roots; segmentation errors and skeleton discretization affect length. Small-object filtering may discard fine or detached roots.
Root hairs, shadows, seeds, debris and background texture can be selected as roots.
Uniform scale assumes roots and ruler in the same plane, negligible perspective distortion and no image resizing after calibration.
User review is not validation. Compare with manually traced references and known dimensions before scientific use.
Per-image SHA256 identifies input files. Crop coordinates refer to the EXIF-oriented original image.
Masks and overlays correspond to the crop, not the full image. TIFF: only first frame analyzed.
Results are computed in memory, not stored as analysis history. Download before leaving.
"""
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("measurements.csv", csv)
    z.writestr("METHOD_AND_LIMITATIONS.txt", notes)
    for name, data in artifacts.items():
        z.writestr(name, data)
st.download_button("Download CSV + masks + overlays (ZIP)", archive.getvalue(), "root_analysis.zip", "application/zip")
st.subheader("Interpretation limits")
st.write("Length estimates represent visible skeleton paths, not hidden or overlapping root lengths. Projected area is not root surface area. Spans are image-plane dimensions, not soil depth. This app does not infer root counts, root orders, volume, health or volumetric density.")
