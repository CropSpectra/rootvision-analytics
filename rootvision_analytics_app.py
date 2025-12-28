import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image
import io

st.set_page_config(
    page_title="🌱 RootVision Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #208090;
        font-weight: bold;
    }
    .header-section {
        background: linear-gradient(135deg, #208090 0%, #32b8c6 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .cv-highlight {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌱 RootVision Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🔬 Root Analysis",
    "📊 Dashboard",
    "📋 Trial Management",
    "⚙️ Settings"
])

@st.cache_data
def generate_trial_data():
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    trials = ['Trial A', 'Trial B', 'Trial C', 'Trial D']
    data = []
    
    for trial in trials:
        base_depth = np.random.uniform(2, 4)
        for i, date in enumerate(dates):
            depth = base_depth + (i * np.random.uniform(0.25, 0.35))
            data.append({
                'Date': date,
                'Trial': trial,
                'Root Depth (cm)': depth + np.random.normal(0, 0.3),
                'Root Surface Area (cm²)': depth * 18 + np.random.normal(0, 5),
                'Root Density (per cm³)': depth * 2.5 + np.random.normal(0, 0.2),
                'Health Score': min(10, 5 + (depth / 2) + np.random.normal(0, 0.5))
            })
    
    return pd.DataFrame(data)

trial_data = generate_trial_data()

if page == "🔬 Root Analysis":
    st.markdown("""
    <div class="header-section">
        <h1>🔬 Root Image Analysis with Computer Vision</h1>
        <p>AI-Powered Root Phenotyping & Automated Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="cv-highlight">
        <h3>🤖 Computer Vision Pipeline</h3>
        <p>Upload root images for automated analysis using YOLOv8 object detection and semantic segmentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Single Image Analysis", "Batch Processing", "Model Settings", "Analysis History"])
    
    with tabs[0]:
        st.subheader("📸 Single Image Analysis")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.write("**Upload root image for analysis**")
            uploaded_file = st.file_uploader(
                "Choose a root image (JPG, PNG, TIFF)",
                type=["jpg", "png", "tiff", "jpeg"],
                key="single_image"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Root Image", use_container_width=True)
                st.success("✅ Image loaded successfully!")
        
        with col2:
            st.write("**Analysis Parameters**")
            confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.65, 0.05)
            model_type = st.selectbox("Model Type", ["YOLOv8", "YOLOv5", "Mask R-CNN"])
            iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
            
            st.write("**Processing Options**")
            show_bbox = st.checkbox("Show Bounding Boxes", value=True)
            show_segmentation = st.checkbox("Show Segmentation Mask", value=False)
            extract_metrics = st.checkbox("Extract Root Metrics", value=True)
        
        st.markdown("---")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze Root Image", use_container_width=True, type="primary"):
                st.info("⏳ Processing image with computer vision model...")
                
                progress = st.progress(0)
                for i in range(100):
                    progress.progress(i + 1)
                
                st.success("✅ Analysis Complete!")
                
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Roots Detected", "47", "+3 from previous")
                
                with col2:
                    st.metric("Total Length", "1,847.3 cm", "+82.2 cm")
                
                with col3:
                    st.metric("Surface Area", "4,287.3 cm²", "+251.1 cm²")
                
                with col4:
                    st.metric("Root Density", "28.4/cm³", "+1.8/cm³")
                
                st.markdown("---")
                
                st.subheader("🌿 Root System Summary")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("Primary Roots", "8", "Main axes")
                
                with col2:
                    st.metric("Secondary Roots", "24", "Lateral branches")
                
                with col3:
                    st.metric("Tertiary Roots", "156", "Fine roots")
                
                with col4:
                    st.metric("Root Depth", "42.3 cm", "Penetration")
                
                with col5:
                    st.metric("Avg Diameter", "2.1 mm", "All roots")
                
                with col6:
                    st.metric("Branching Index", "6.2", "Complexity")
                
                st.markdown("---")
                
                st.subheader("🔍 Primary Root Details")
                
                results_data = {
                    'Root ID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
                    'Type': ['Primary', 'Primary', 'Primary', 'Primary', 'Primary', 'Primary', 'Primary', 'Primary'],
                    'Length (cm)': [42.3, 38.9, 41.2, 35.7, 39.8, 37.5, 40.1, 36.2],
                    'Diameter (mm)': [3.2, 2.9, 3.1, 2.8, 3.0, 2.7, 3.2, 2.9],
                    'Surface Area (cm²)': [428.5, 356.2, 398.7, 314.5, 376.8, 320.1, 412.3, 332.7],
                    'Volume (cm³)': [85.7, 62.3, 76.2, 52.1, 68.9, 48.3, 82.4, 56.8],
                    'Angle (degrees)': [45.2, 32.1, 58.4, 41.7, 39.8, 48.2, 35.9, 43.1],
                    'Secondary Branches': [3, 3, 3, 2, 3, 2, 3, 3],
                    'Confidence': [0.98, 0.96, 0.99, 0.94, 0.97, 0.95, 0.99, 0.96]
                }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                st.subheader("🌾 Secondary Root Details")
                
                secondary_data = {
                    'Root ID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006'],
                    'Parent Root': ['P001', 'P001', 'P002', 'P002', 'P003', 'P003'],
                    'Length (cm)': [18.5, 16.2, 17.8, 15.3, 19.2, 17.1],
                    'Diameter (mm)': [1.8, 1.6, 1.9, 1.5, 2.0, 1.7],
                    'Insertion Angle (°)': [65.3, 72.1, 58.9, 68.4, 71.2, 64.8],
                    'Distance from Base (cm)': [8.2, 12.5, 7.8, 15.3, 9.1, 13.7],
                    'Surface Area (cm²)': [104.8, 81.3, 106.5, 72.1, 121.3, 91.8]
                }
                
                secondary_df = pd.DataFrame(secondary_data)
                st.dataframe(secondary_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                st.subheader("📈 Root Distribution Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_length = px.histogram(
                        results_df,
                        x='Length (cm)',
                        nbins=8,
                        title='Primary Root Length Distribution',
                        labels={'Length (cm)': 'Length (cm)', 'count': 'Frequency'},
                        color_discrete_sequence=['#208090']
                    )
                    st.plotly_chart(fig_length, use_container_width=True)
                
                with col2:
                    fig_diameter = px.histogram(
                        results_df,
                        x='Diameter (mm)',
                        nbins=8,
                        title='Primary Root Diameter Distribution',
                        labels={'Diameter (mm)': 'Diameter (mm)', 'count': 'Frequency'},
                        color_discrete_sequence=['#059669']
                    )
                    st.plotly_chart(fig_diameter, use_container_width=True)
                
                with col3:
                    fig_angle = px.histogram(
                        results_df,
                        x='Angle (degrees)',
                        nbins=8,
                        title='Root Emergence Angle Distribution',
                        labels={'Angle (degrees)': 'Angle (°)', 'count': 'Frequency'},
                        color_discrete_sequence=['#f59e0b']
                    )
                    st.plotly_chart(fig_angle, use_container_width=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_ld = px.scatter(
                        results_df,
                        x='Length (cm)',
                        y='Diameter (mm)',
                        size='Surface Area (cm²)',
                        color='Secondary Branches',
                        title='Root Length vs Diameter (colored by branching)',
                        color_continuous_scale=['#ef4444', '#f59e0b', '#10b981']
                    )
                    st.plotly_chart(fig_ld, use_container_width=True)
                
                with col2:
                    fig_ab = px.scatter(
                        results_df,
                        x='Angle (degrees)',
                        y='Secondary Branches',
                        size='Length (cm)',
                        color='Confidence',
                        title='Emergence Angle vs Secondary Branching',
                        color_continuous_scale=['#ef4444', '#f59e0b', '#10b981']
                    )
                    st.plotly_chart(fig_ab, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("🏗️ Root Architecture Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Total Root System Metrics:**
                    - **Total Root Length:** 1,847.3 cm
                    - **Total Surface Area:** 4,287.3 cm²
                    - **Total Root Volume:** 612.8 cm³
                    - **Maximum Depth:** 42.3 cm
                    - **Average Diameter:** 2.1 mm
                    - **Root Density Index:** 28.4 roots/cm³
                    """)
                
                with col2:
                    st.markdown("""
                    **Branching Architecture:**
                    - **Primary Roots:** 8 main axes
                    - **Secondary Roots:** 24 lateral branches
                    - **Tertiary Roots:** 156 fine roots
                    - **Branching Complexity Index:** 6.2
                    - **Avg Branching Angle:** 52.8°
                    - **Root System Spread:** 38.7 cm (horizontal)
                    """)
                
                st.markdown("---")
                
                st.subheader("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Primary Roots CSV",
                        data=csv,
                        file_name=f"primary_roots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_sec = secondary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Secondary Roots CSV",
                        data=csv_sec,
                        file_name=f"secondary_roots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    st.button("📧 Email Complete Report")
    
    with tabs[1]:
        st.subheader("📂 Batch Image Processing")
        
        st.write("Upload multiple root images for batch analysis")
        
        batch_files = st.file_uploader(
            "Choose multiple root images",
            type=["jpg", "png", "tiff", "jpeg"],
            accept_multiple_files=True,
            key="batch_images"
        )
        
        if batch_files:
            st.write(f"📊 {len(batch_files)} images selected")
            
            col1, col2 = st.columns(2)
            
            with col1:
                batch_confidence = st.slider("Batch Confidence Threshold", 0.0, 1.0, 0.65)
            
            with col2:
                batch_model = st.selectbox("Batch Model", ["YOLOv8", "YOLOv5", "Mask R-CNN"])
            
            if st.button("🚀 Process Batch", use_container_width=True, type="primary"):
                st.info("⏳ Processing batch images...")
                
                progress_bar = st.progress(0)
                for i in range(len(batch_files)):
                    progress_bar.progress((i + 1) / len(batch_files))
                
                st.success(f"✅ Processed {len(batch_files)} images successfully!")
                
                batch_summary = {
                    'Image': [f.name for f in batch_files[:5]],
                    'Total Roots': [np.random.randint(30, 60) for _ in range(min(5, len(batch_files)))],
                    'Total Length (cm)': [np.random.uniform(1200, 2000) for _ in range(min(5, len(batch_files)))],
                    'Avg Diameter (mm)': [np.random.uniform(1.8, 2.5) for _ in range(min(5, len(batch_files)))],
                    'Max Depth (cm)': [np.random.uniform(35, 45) for _ in range(min(5, len(batch_files)))],
                    'Avg Confidence': [np.random.uniform(0.92, 0.99) for _ in range(min(5, len(batch_files)))]
                }
                
                batch_df = pd.DataFrame(batch_summary)
                st.dataframe(batch_df, use_container_width=True, hide_index=True)
    
    with tabs[2]:
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Selection**")
            model_version = st.selectbox("Model Version", ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
            pretrained = st.checkbox("Use Pretrained Weights", value=True)
            
            st.write("**Detection Parameters**")
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
            iou = st.slider("IOU Threshold", 0.0, 1.0, 0.45)
            max_detections = st.slider("Max Detections", 10, 500, 300)
        
        with col2:
            st.write("**Segmentation Settings**")
            enable_segmentation = st.checkbox("Enable Semantic Segmentation", value=True)
            seg_model = st.selectbox("Segmentation Model", ["U-Net", "Mask R-CNN", "DeepLab"])
            
            st.write("**Processing**")
            device = st.radio("Processing Device", ["GPU", "CPU"])
            batch_size = st.slider("Batch Size", 1, 32, 8)
        
        st.markdown("---")
        
        if st.button("💾 Save Model Configuration", use_container_width=True):
            st.success("✅ Configuration saved!")
    
    with tabs[3]:
        st.subheader("📋 Analysis History")
        
        history_data = {
            'Date': pd.date_range('2025-01-20', periods=8),
            'Image': [f"root_sample_{i}.jpg" for i in range(8)],
            'Total Roots': np.random.randint(30, 60, 8),
            'Total Length (cm)': np.random.uniform(1200, 2000, 8),
            'Max Depth (cm)': np.random.uniform(35, 45, 8),
            'Processing Time (s)': np.random.uniform(2, 8, 8),
            'Model Used': ['YOLOv8'] * 8
        }
        
        history_df = pd.DataFrame(history_data)
        history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.subheader("📊 Analysis Trends")
        
        fig = px.line(
            history_df,
            x='Date',
            y='Total Roots',
            title='Root Detection Trend Over Time',
            markers=True
        )
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Dashboard":
    st.markdown("""
    <div class="header-section">
        <h1>📊 RootVision Analytics Dashboard</h1>
        <p>Trial Overview & Growth Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Active Trials", value="4", delta="On track")
    
    with col2:
        avg_depth = trial_data['Root Depth (cm)'].mean()
        st.metric(label="Avg Root Depth", value=f"{avg_depth:.1f} cm", delta="+0.8 cm this week")
    
    with col3:
        avg_health = trial_data['Health Score'].mean()
        st.metric(label="Overall Health", value=f"{avg_health:.1f}/10", delta="+0.5")
    
    with col4:
        total_plants = len(trial_data['Trial'].unique()) * 12
        st.metric(label="Plants Analyzed", value=total_plants, delta="+48 this week")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_depth = go.Figure()
        for trial in trial_data['Trial'].unique():
            trial_subset = trial_data[trial_data['Trial'] == trial]
            fig_depth.add_trace(go.Scatter(
                x=trial_subset['Date'],
                y=trial_subset['Root Depth (cm)'],
                mode='lines+markers',
                name=trial,
                line=dict(width=3),
                marker=dict(size=5)
            ))
        
        fig_depth.update_layout(
            title="Root Depth Growth Over Time",
            xaxis_title="Date",
            yaxis_title="Depth (cm)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_depth, use_container_width=True)
    
    with col2:
        health_by_trial = trial_data.groupby('Trial')['Health Score'].mean().sort_values(ascending=False)
        fig_health = px.bar(
            x=health_by_trial.index,
            y=health_by_trial.values,
            labels={'x': 'Trial', 'y': 'Health Score (0-10)'},
            title='Plant Health Score by Trial',
            color=health_by_trial.values,
            color_continuous_scale=['#ef4444', '#f59e0b', '#10b981']
        )
        fig_health.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_health, use_container_width=True)

elif page == "📋 Trial Management":
    st.title("📋 Trial Management")
    st.markdown("---")
    
    tabs = st.tabs(["Active Trials", "Create New Trial", "Trial History"])
    
    with tabs[0]:
        st.subheader("Active Trials")
        trial_list = [
            {"name": "Trial A", "status": "Active", "start": "2025-01-01", "plants": 12},
            {"name": "Trial B", "status": "Active", "start": "2025-01-05", "plants": 12},
            {"name": "Trial C", "status": "Pending", "start": "2025-01-15", "plants": 12},
            {"name": "Trial D", "status": "Active", "start": "2025-01-20", "plants": 12},
        ]
        
        for trial in trial_list:
            st.write(f"**{trial['name']}** | {trial['status']} | {trial['start']} | {trial['plants']} plants")
            st.divider()
    
    with tabs[1]:
        st.subheader("Create New Trial")
        trial_name = st.text_input("Trial Name")
        num_plants = st.number_input("Number of Plants", min_value=1, max_value=100, value=12)
        trial_type = st.selectbox("Trial Type", ["Growth Comparison", "Disease Resistance", "Breeding Selection"])
        start_date = st.date_input("Start Date")
        
        if st.button("Create Trial", use_container_width=True):
            st.success(f"✅ Trial '{trial_name}' created successfully!")
    
    with tabs[2]:
        st.subheader("Trial History")
        st.info("No completed trials yet")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Display Settings")
        theme = st.radio("Theme", ["Light", "Dark", "Auto"])
        language = st.selectbox("Language", ["English", "Spanish", "French"])
    
    with col2:
        st.subheader("Analysis Settings")
        units = st.radio("Measurement Units", ["Metric (cm)", "Imperial (in)"])
        auto_save = st.checkbox("Auto-save Results", value=True)
    
    st.markdown("---")
    st.subheader("About RootVision Analytics")
    st.markdown("**v0.4.0** | AI-Powered Root Phenotyping with Advanced Morphology Analysis | Built with Streamlit & Python")
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved!")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.85rem;'>🌱 RootVision Analytics | AgriVision Analytics | v0.4.0</div>", unsafe_allow_html=True)
