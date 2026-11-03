import hashlib
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="RootVision", page_icon="🌱", layout="wide")
st.title("🌱 RootVision — Root Image Measurements")
st.warning("Not validated for research accuracy.")
st.caption("Measure root images individually or in batches using image processing.")
st.info("Keep the ruler in the original photograph. Select two ruler marks and choose centimeters or inches for calibration. Then crop the root-analysis region to exclude the ruler and other non-root objects.")


def measure(mask):
    s = skeletonize(mask)
    horizontal = np.count_nonzero(s[:, :-1] & s[:, 1:])
    vertical = np.count_nonzero(s[:-1, :] & s[1:, :])
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    length = float(horizontal + vertical + np.sqrt(2) * (d1.sum() + d2.sum()))
    ys, xs = np.nonzero(mask)
    return s, length, int(mask.sum()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


workflow = st.radio("Analysis mode", ["Single image", "Batch images"], horizontal=True)
files = st.file_uploader("Upload root images", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)
if not files:
    st.info("Upload an image to begin.")
    st.stop()
if workflow == "Single image" and len(files) > 1:
    st.error("Select Batch images to process multiple files, or upload one image.")
    st.stop()
st.caption("Each image has independent calibration and analysis settings. TIFF: only the first frame is analyzed.")
rows, artifacts = [], {}
for index, uploaded in enumerate(files, start=1):
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    key = f"image_{index}_{digest[:16]}"
    with st.expander(f"{index}. {uploaded.name}", expanded=len(files)==1):
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            width, height = image.size
            st.caption(f"Original oriented image: {width} × {height} pixels.")
            st.subheader("1. Ruler calibration")
            calibrated = st.checkbox("Calibrate using the ruler", value=True, key=key+"cal")
            scale = None
            points = None
            entered = None
            input_unit = None
            known_cm = None
            if calibrated:
                units = st.selectbox("Ruler units", ["Centimeters (cm)", "Inches (in)"], key=key+"unit")
                input_unit = "in" if units == "Inches (in)" else "cm"
                entered = st.number_input(f"Distance between selected ruler marks ({input_unit})", min_value=0.001, value=1.0, format="%.3f", key=key+"distance_"+input_unit)
                known_cm = entered*2.54 if input_unit == "in" else entered
                st.caption("Enter the separation, not the number at the second mark. Marks at 2 inches and 5 inches are 3 inches apart.")
                st.write(f"Selected interval: {entered:g} {input_unit} = {known_cm:g} cm")
                st.caption("Click the first mark in the left image and the second mark in the right image. Use well-separated marks.")
                pw = min(width, 450)
                ph = max(1, round(height*pw/width))
                preview = image.resize((pw, ph), Image.Resampling.LANCZOS)
                epoch_key = key+"epoch"
                if epoch_key not in st.session_state:
                    st.session_state[epoch_key] = 0
                if st.button("Reset ruler points", key=key+"reset"):
                    st.session_state[epoch_key] += 1
                    st.rerun()
                epoch = st.session_state[epoch_key]
                left, right = st.columns(2)
                with left:
                    st.write("First ruler mark")
                    a = streamlit_image_coordinates(preview, key=f"{key}_a_{epoch}")
                with right:
                    st.write("Second ruler mark")
                    b = streamlit_image_coordinates(preview, key=f"{key}_b_{epoch}")
                if not a or not b:
                    st.info("Select both marks to continue, or turn calibration off for pixel measurements.")
                    rows.append(dict(image_index=index, image=uploaded.name, status="awaiting_calibration"))
                    continue
                points = ((a["x"]*width/pw, a["y"]*height/ph), (b["x"]*width/pw, b["y"]*height/ph))
                distance_px = float(np.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1]))
                if distance_px <= 0:
                    raise ValueError("Select two different ruler marks.")
                scale = distance_px/known_cm
                marked = preview.copy()
                draw = ImageDraw.Draw(marked)
                draw.line([(a["x"],a["y"]),(b["x"],b["y"])], fill="red", width=3)
                for p in (a,b):
                    x,y = p["x"],p["y"]
                    draw.ellipse((x-5,y-5,x+5,y+5), outline="red", width=3)
                st.image(marked, caption=f"Check red endpoints. Scale: {scale:.3f} pixels/cm.")
                st.caption("Preview coordinates are mapped to original dimensions. Ruler and roots should be in the same plane; perspective is not corrected.")
                artifacts[f"image_{index:03d}_calibration.png"] = png_bytes(np.asarray(marked))
            else:
                st.image(image, caption="Original image — pixel measurements", use_column_width=True)
            st.subheader("2. Select the root-analysis region")
            st.caption("Exclude ruler, labels, shoots, seed and other non-root objects. Cropping does not resize the image or change its scale.")
            left,right = st.columns(2)
            xr = left.slider("Horizontal crop (%)",0,100,(0,100),key=key+"x")
            yr = right.slider("Vertical crop (%)",0,100,(0,100),key=key+"y")
            x0,x1 = [int(width*x/100) for x in xr]
            y0,y1 = [int(height*y/100) for y in yr]
            if x1<=x0 or y1<=y0:
                raise ValueError("Select a crop with nonzero width and height.")
            crop = np.asarray(image)[y0:y1,x0:x1]
            st.image(crop, caption="Selected analysis region", use_column_width=True)
            st.subheader("3. Separate roots from background")
            gray = np.dot(crop,[0.2126,0.7152,0.0722])
            light = st.checkbox("Light roots on a dark background",value=True,key=key+"light")
            automatic = st.checkbox("Use automatic brightness threshold",value=True,key=key+"auto")
            if automatic:
                threshold = float(threshold_otsu(gray))
                st.caption(f"Automatic threshold: {threshold:.1f}/255")
            else:
                threshold = st.slider("Brightness threshold",0,255,150,key=key+"thr")
            minimum = st.number_input("Remove isolated objects smaller than this many pixels",min_value=1,value=30,step=5,key=key+"min")
            mask = gray>threshold if light else gray<threshold
            mask = remove_small_objects(mask,min_size=int(minimum))
            if not mask.any():
                raise ValueError("No root pixels selected. Adjust crop, threshold or root appearance.")
            skeleton,length_px,area_px,span_x,span_y = measure(mask)
            overlay = crop.copy()
            overlay[mask] = (overlay[mask]*0.5+np.array([0,255,0])*0.5).astype(np.uint8)
            overlay[skeleton] = [255,0,0]
            left,right = st.columns(2)
            left.image(mask.astype(np.uint8)*255,caption="White pixels are measured",use_column_width=True)
            right.image(overlay,caption="Green mask; red skeleton",use_column_width=True)
            border = bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
            if border:
                st.warning("Selected pixels touch the crop boundary. Check for cut-off roots or included background.")
            signature = hashlib.sha256(mask.tobytes()+str((x0,x1,y0,y1,scale,points,threshold,minimum,light)).encode()).hexdigest()[:20]
            reviewed = st.checkbox("I have reviewed the root mask and calibration",key=key+signature+"review")
            divisor = scale if scale is not None else 1.0
            unit = "cm" if scale is not None else "px"
            values = dict(estimated_visible_length=length_px/divisor,projected_area=area_px/divisor**2,horizontal_span=span_x/divisor,vertical_span=span_y/divisor)
            st.subheader("4. Measurements")
            left,right = st.columns(2)
            left.metric("Estimated visible root length",f"{values['estimated_visible_length']:.2f} {unit}")
            right.metric("Projected root area",f"{values['projected_area']:.2f} {unit}²")
            left,right = st.columns(2)
            left.metric("Horizontal span",f"{values['horizontal_span']:.2f} {unit}")
            right.metric("Vertical span",f"{values['vertical_span']:.2f} {unit}")
            if calibrated:
                st.caption("Results use cm and cm² whether calibration was entered in inches or centimeters.")
            record = dict(image_index=index,image=uploaded.name,status="reviewed" if reviewed else "needs_review",length_unit=unit,area_unit=unit+"^2",pixels_per_cm=scale,calibration_input_distance=entered,calibration_input_unit=input_unit,calibration_distance_cm=known_cm,crop_border_contact=border,brightness_threshold=threshold,minimum_object_pixels=int(minimum),light_roots=light,crop_x0=x0,crop_x1=x1,crop_y0=y0,crop_y1=y1,original_width=width,original_height=height,image_sha256=digest,**values)
            if points is not None:
                record.update(calibration_ax=points[0][0],calibration_ay=points[0][1],calibration_bx=points[1][0],calibration_by=points[1][1])
            rows.append(record)
            artifacts[f"image_{index:03d}_mask.png"] = png_bytes(mask.astype(np.uint8)*255)
            artifacts[f"image_{index:03d}_overlay.png"] = png_bytes(overlay)
        except Exception as error:
            st.error(f"This image could not be measured: {error}")
            rows.append(dict(image_index=index,image=uploaded.name,status="failed",error=str(error)))

st.subheader("Results and downloads")
st.caption("Calibrated rows use cm and cm²; uncalibrated rows use pixels. Do not combine different units. Unreviewed results are provisional.")
df = pd.DataFrame(rows)
st.dataframe(df,use_container_width=True,hide_index=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download measurements CSV",csv,"root_measurements.csv","text/csv")
notes = """RootVision image analysis
Calibration accepts cm or inches; inch inputs are converted using 1 inch = 2.54 cm.
Calibrated results use cm and cm^2. Uncalibrated results use pixels.
Preview coordinates map to EXIF-oriented original image dimensions.
Scale assumes coplanar ruler and roots and negligible perspective distortion.
Length is estimated from weighted skeleton links; area is projected selected-pixel area.
Spans are image-plane bounding-box dimensions, not soil depth.
Projected area is not cylindrical surface area. No biological root counts, root orders, volume, health scores or volumetric density are calculated.
Inspect masks and calibration. Overlap, missed roots and selected background affect results.
User review is not accuracy validation. Compare against reference measurements before research use.
Masks/overlays correspond to crops; calibration previews show endpoints.
TIFF: first frame only. No permanent analysis history; download results before leaving.
"""
archive = io.BytesIO()
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    z.writestr("measurements.csv",csv)
    z.writestr("METHOD_AND_LIMITATIONS.txt",notes)
    for name,data in artifacts.items():
        z.writestr(name,data)
st.download_button("Download CSV, calibration previews, masks and overlays",archive.getvalue(),"root_analysis.zip","application/zip")
st.caption("Measurements describe selected visible pixels. Overlapping or obscured roots cannot be fully measured. Download results before leaving.")
