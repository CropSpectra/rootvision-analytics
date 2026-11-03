import io
import hashlib
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="RootVision", layout="wide")
st.title("🌱 RootVision — Root Image Measurements")
st.warning("Not validated for research accuracy.")
st.caption("Measure root images individually or in batches using image processing.")
st.info("Keep the ruler in the photograph. Calibrate by selecting two ruler marks, then crop the root-analysis region to exclude the ruler and other non-root objects.")
files = st.file_uploader("Upload one or more images", type=["jpg","jpeg","png","tif","tiff"], accept_multiple_files=True)
rows, assets = [], {}
for i, f in enumerate(files or []):
    raw = f.getvalue()
    key = str(i)+hashlib.sha256(raw).hexdigest()[:12]
    with st.expander(f"{i+1}. {f.name}", expanded=len(files)==1):
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            w,h = image.size
            st.subheader("Ruler calibration")
            calibrated = st.checkbox("Calibrate from ruler", value=True, key=key+"cal")
            scale = None
            points = [None,None]
            if calibrated:
                st.caption("Select one mark in the first image and a second mark in the second image. Use well-separated marks. This is manual point selection, not automatic ruler recognition.")
                preview_width = min(w, 700)
                preview_height = max(1, round(h*preview_width/w))
                preview = image.resize((preview_width,preview_height))
                epoch = st.session_state.get(key+"epoch",0)
                if st.button("Reset calibration points",key=key+"reset"):
                    st.session_state[key+"epoch"] = epoch+1
                    st.rerun()
                a,b = st.columns(2)
                with a:
                    st.write("1. Click first ruler mark")
                    p1 = streamlit_image_coordinates(preview, key=key+"p1"+str(epoch))
                with b:
                    st.write("2. Click second ruler mark")
                    p2 = streamlit_image_coordinates(preview, key=key+"p2"+str(epoch))
                known = st.number_input("Distance between selected marks (cm)", min_value=0.001, value=5.0, format="%.3f", key=key+"known")
                st.caption("If your ruler is marked in inches, convert the selected interval to centimeters first. Ruler and roots must be in the same plane; perspective is not corrected.")
                if not p1 or not p2:
                    st.info("Select both ruler marks to calculate measurements, or disable calibration for pixel units.")
                    rows.append(dict(image=f.name,status="awaiting_calibration"))
                    continue
                points = [(p["x"]*w/preview_width,p["y"]*h/preview_height) for p in (p1,p2)]
                distance = float(np.hypot(points[1][0]-points[0][0],points[1][1]-points[0][1]))
                if distance <= 0:
                    raise ValueError("Choose two different ruler marks.")
                scale = distance/known
                marked = preview.copy()
                draw = ImageDraw.Draw(marked)
                draw.line([(p1["x"],p1["y"]),(p2["x"],p2["y"])],fill="red",width=3)
                for p in (p1,p2):
                    x,y = p["x"],p["y"]
                    draw.ellipse((x-5,y-5,x+5,y+5),outline="red",width=3)
                st.image(marked,caption=f"Check endpoints: {distance:.2f} original pixels = {known:g} cm; {scale:.3f} px/cm")
            else:
                st.image(image, use_column_width=True)
            st.subheader("Root-analysis region")
            xr = st.slider("Horizontal crop (%)",0,100,(0,100),key=key+"x")
            yr = st.slider("Vertical crop (%)",0,100,(0,100),key=key+"y")
            x0,x1 = [int(w*x/100) for x in xr]
            y0,y1 = [int(h*y/100) for y in yr]
            if x1<=x0 or y1<=y0:
                raise ValueError("Select a nonempty crop.")
            crop = np.asarray(image)[y0:y1,x0:x1]
            gray = np.dot(crop,[0.2126,0.7152,0.0722])
            light = st.checkbox("Light roots on dark background",value=True,key=key+"light")
            auto = st.checkbox("Automatic threshold",value=True,key=key+"auto")
            threshold = float(threshold_otsu(gray)) if auto else st.slider("Brightness threshold",0,255,150,key=key+"thr")
            size = st.number_input("Remove isolated objects smaller than (pixels)",min_value=1,value=30,key=key+"size")
            mask = gray>threshold if light else gray<threshold
            mask = remove_small_objects(mask,min_size=int(size))
            if not mask.any():
                raise ValueError("No root pixels selected; adjust threshold, polarity or crop.")
            s = skeletonize(mask)
            links = np.count_nonzero(s[:,:-1]&s[:,1:])+np.count_nonzero(s[:-1,:]&s[1:,:])
            d1 = s[:-1,:-1]&s[1:,1:]&~s[:-1,1:]&~s[1:,:-1]
            d2 = s[:-1,1:]&s[1:,:-1]&~s[:-1,:-1]&~s[1:,1:]
            length = float(links+np.sqrt(2)*(d1.sum()+d2.sum()))
            overlay = crop.copy()
            overlay[mask] = (overlay[mask]*0.5+np.array([0,255,0])*0.5).astype(np.uint8)
            overlay[s] = [255,0,0]
            a,b = st.columns(2)
            a.image(mask.astype(np.uint8)*255,caption="White pixels are measured",use_column_width=True)
            b.image(overlay,caption="Green mask; red skeleton",use_column_width=True)
            border = bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
            if border:
                st.warning("Selected pixels touch crop border. Check for cut-off roots or selected background.")
            signature = hashlib.sha256(mask.tobytes()+str((scale,points)).encode()).hexdigest()[:16]
            reviewed = st.checkbox("I reviewed this mask and calibration",key=key+signature+"review")
            divisor = scale if scale else 1.0
            unit = "cm" if scale else "px"
            ys,xs = np.nonzero(mask)
            values = dict(estimated_visible_length=length/divisor,projected_area=float(mask.sum())/divisor**2,horizontal_span=float(xs.max()-xs.min()+1)/divisor,vertical_span=float(ys.max()-ys.min()+1)/divisor)
            a,b = st.columns(2)
            a.metric("Estimated visible length",f"{values['estimated_visible_length']:.2f} {unit}")
            b.metric("Projected area",f"{values['projected_area']:.2f} {unit}²")
            record = dict(image=f.name,status="reviewed" if reviewed else "needs_review",length_unit=unit,area_unit=unit+"^2",pixels_per_cm=scale,crop_border_contact=border,threshold=threshold,minimum_object_pixels=size,light_roots=light,crop_x0=x0,crop_x1=x1,crop_y0=y0,crop_y1=y1,image_sha256=hashlib.sha256(raw).hexdigest(),**values)
            if calibrated:
                record.update(calibration_ax=points[0][0],calibration_ay=points[0][1],calibration_bx=points[1][0],calibration_by=points[1][1],calibration_cm=known)
            rows.append(record)
            for label,arr in [("mask",mask.astype(np.uint8)*255),("overlay",overlay)]:
                buf=io.BytesIO()
                Image.fromarray(arr).save(buf,format="PNG")
                assets[f"image_{i+1:03d}_{label}.png"]=buf.getvalue()
        except Exception as exc:
            st.error(str(exc))
            rows.append(dict(image=f.name,status="failed",error=str(exc)))
if rows:
    st.subheader("Batch results and exports")
    df=pd.DataFrame(rows)
    st.dataframe(df,use_container_width=True,hide_index=True)
    csv=df.to_csv(index=False).encode()
    st.download_button("Download CSV",csv,"root_measurements.csv","text/csv")
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("measurements.csv",csv)
        z.writestr("METHOD.txt","Experimental brightness segmentation and weighted skeleton-path length. No trained AI. Inspect masks. Area is projected pixel area, not root surface area. Spans are image-plane dimensions, not soil depth. No biological counts or volume. Overlaps, shadows, debris and background texture affect measurements. Scale assumes coplanar ruler and roots and negligible perspective. Calibration points are mapped from displayed preview to EXIF-oriented original dimensions. First TIFF frame only. No persistent storage. Validate against reference measurements before research use.")
        for name,data in assets.items(): z.writestr(name,data)
    st.download_button("Download CSV, masks and overlays ZIP",out.getvalue(),"root_analysis.zip","application/zip")
st.caption("Length estimates omit hidden overlapping roots. Projected area is not root surface area; spans are not soil rooting depth. No biological root counts, volume or health scores are inferred. Review masks and download results before leaving.")
