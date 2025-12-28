import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
        font-size: 2rem;
        color: #208090;
    }
    .metric-card {
        background: linear-gradient(135deg, #208090 0%, #32b8c6 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🌱 RootVision Analytics")
page = st.sidebar.radio("Navigation", ["Dashboard", "Root Analysis", "Growth Tracking", "Settings"])

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "Dashboard":
    st.title("🌱 RootVision Analytics Dashboard")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Plants Analyzed",
            value="247",
            delta="+12 this week",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Avg Root Depth",
            value="8.3 cm",
            delta="+0.5 cm",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Health Score",
            value="8.7/10",
            delta="+0.3",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Active Trials",
            value="5",
            delta="On track",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Root depth over time
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        depths = np.cumsum(np.random.normal(0.3, 0.1, 30)) + 2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=depths,
            mode='lines+markers',
            name='Root Depth',
            line=dict(color='#208090', width=3),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Root Depth Growth (30 Days)",
            xaxis_title="Date",
            yaxis_title="Depth (cm)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Health distribution
        health_data = {
            'Excellent': 85,
            'Good': 95,
            'Fair': 45,
            'Poor': 22
        }
        
        fig = px.bar(
            x=list(health_data.keys()),
            y=list(health_data.values()),
            labels={'x': 'Health Status', 'y': 'Count'},
            title='Plant Health Distribution',
            color_discrete_sequence=['#208090', '#32b8c6', '#a8d5d5', '#e8f4f4']
        )
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ROOT ANALYSIS PAGE
# ============================================================================
elif page == "Root Analysis":
    st.title("📊 Root Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Root Image")
        uploaded_file = st.file_uploader("Choose a root image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            st.success("✅ Image uploaded successfully!")
            st.info("📝 Note: Computer vision analysis coming soon!")
    
    with col2:
        st.subheader("Analysis Parameters")
        
        st.slider("Image Threshold", 0, 255, 127)
        st.slider("Minimum Root Width (px)", 1, 50, 5)
        st.slider("Sensitivity", 0.0, 1.0, 0.7)
        
        if st.button("🔍 Analyze Root Image"):
            st.success("Analysis complete!")
            
            # Mock results
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Root Length", "47.3 cm", "+2.1 cm")
            with col_b:
                st.metric("Root Surface Area", "156.8 cm²", "+8.2 cm²")
            with col_c:
                st.metric("Root Density", "8.4/cm³", "+0.3/cm³")

# ============================================================================
# GROWTH TRACKING PAGE
# ============================================================================
elif page == "Growth Tracking":
    st.title("📈 Growth Tracking")
    st.markdown("---")
    
    # Sample data
    growth_data = pd.DataFrame({
        'Day': range(1, 31),
        'Root Length (cm)': np.cumsum(np.random.normal(0.4, 0.15, 30)) + 2,
        'Shoot Height (cm)': np.cumsum(np.random.normal(0.5, 0.2, 30)) + 5,
        'Biomass (g)': np.cumsum(np.random.normal(0.08, 0.03, 30)) + 0.5
    })
    
    # Display data table
    st.dataframe(growth_data, use_container_width=True)
    
    # Growth chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=growth_data['Day'],
        y=growth_data['Root Length (cm)'],
        name='Root Length',
        mode='lines+markers',
        line=dict(color='#208090', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=growth_data['Day'],
        y=growth_data['Shoot Height (cm)'],
        name='Shoot Height',
        mode='lines+markers',
        line=dict(color='#32b8c6', width=2)
    ))
    
    fig.update_layout(
        title='Plant Growth Over Time',
        xaxis_title='Days',
        yaxis_title='Measurement (cm)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SETTINGS PAGE
# ============================================================================
elif page == "Settings":
    st.title("⚙️ Settings")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Display Settings")
        theme = st.radio("Theme", ["Light", "Dark"])
        language = st.selectbox("Language", ["English", "Spanish", "French"])
    
    with col2:
        st.subheader("Data Settings")
        units = st.radio("Measurement Units", ["Metric (cm, g)", "Imperial (in, oz)"])
        precision = st.slider("Decimal Precision", 1, 5, 2)
    
    st.markdown("---")
    
    st.subheader("About RootVision Analytics")
    st.markdown("""
    🌱 **RootVision Analytics v0.1.0**
    
    Empowering plant breeders and researchers with AI-powered root phenotyping.
    
    **Features:**
    - 📊 Real-time root analysis and measurement
    - 📈 Growth tracking and visualization
    - 🤖 AI-powered image analysis (coming soon)
    - 📱 Mobile-friendly interface
    
    **Built with:** Streamlit, Python, Computer Vision
    """)
    
    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
🌱 RootVision Analytics | Powered by AgriVision Analytics | v0.1.0
</div>
""", unsafe_allow_html=True)
