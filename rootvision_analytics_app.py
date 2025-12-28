import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="🌱 RootVision Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🌱 RootVision Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "📈 Growth Tracking",
    "🔬 Root Analysis",
    "📋 Trial Management",
    "⚙️ Settings"
])

# Generate sample data
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

# DASHBOARD PAGE
if page == "📊 Dashboard":
    st.markdown("""
    <div class="header-section">
        <h1>🌱 RootVision Analytics Dashboard</h1>
        <p>AI-Powered Root Phenotyping & Growth Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Metrics
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
    
    # Charts
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

# GROWTH TRACKING PAGE
elif page == "📈 Growth Tracking":
    st.title("📈 Plant Growth Tracking")
    st.markdown("---")
    
    selected_trial = st.selectbox("Select Trial:", trial_data['Trial'].unique())
    trial_subset = trial_data[trial_data['Trial'] == selected_trial]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        final_depth = trial_subset['Root Depth (cm)'].iloc[-1]
        initial_depth = trial_subset['Root Depth (cm)'].iloc[0]
        growth = final_depth - initial_depth
        st.metric("Root Depth", f"{final_depth:.1f} cm", f"+{growth:.1f} cm growth")
    
    with col2:
        avg_health = trial_subset['Health Score'].mean()
        st.metric("Average Health", f"{avg_health:.1f}/10", "Excellent")
    
    with col3:
        total_days = (trial_subset['Date'].max() - trial_subset['Date'].min()).days
        st.metric("Days Tracked", total_days, "Complete data")
    
    st.markdown("---")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trial_subset['Date'],
        y=trial_subset['Root Depth (cm)'],
        name='Root Depth (cm)',
        mode='lines+markers',
        line=dict(color='#208090', width=3)
    ))
    
    fig.update_layout(
        title=f"Growth Metrics - {selected_trial}",
        xaxis_title="Date",
        yaxis_title="Root Depth (cm)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detailed Measurements")
    display_data = trial_subset[['Date', 'Root Depth (cm)', 'Health Score']].copy()
    display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_data, use_container_width=True, hide_index=True)

# ROOT ANALYSIS PAGE
elif page == "🔬 Root Analysis":
    st.title("🔬 Root Image Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Root Image")
        uploaded_file = st.file_uploader("Choose a root image (JPG, PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            st.success("✅ Image uploaded successfully!")
    
    with col2:
        st.subheader("Analysis Settings")
        threshold = st.slider("Threshold", 0, 255, 127)
        min_width = st.slider("Min Root Width (px)", 1, 50, 5)
        sensitivity = st.slider("Sensitivity", 0.0, 1.0, 0.7)
        
        if st.button("🔍 Analyze Root Image", use_container_width=True):
            st.success("✅ Analysis Complete!")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.metric("Total Length", "47.3 cm", "+2.1 cm")
            with col_2:
                st.metric("Surface Area", "156.8 cm²", "+8.2 cm²")
            with col_3:
                st.metric("Density", "8.4/cm³", "+0.3/cm³")

# TRIAL MANAGEMENT PAGE
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

# SETTINGS PAGE
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
    st.markdown("**v0.2.0** | AI-Powered Root Phenotyping | Built with Streamlit & Python")
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved!")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.85rem;'>🌱 RootVision Analytics | AgriVision Analytics | v0.2.0</div>", unsafe_allow_html=True)
