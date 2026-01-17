#!/usr/bin/env python3
"""
Smart Jal - Streamlit Dashboard
Interactive visualization of groundwater predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path
import json
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Page config
st.set_page_config(
    page_title="Smart Jal - Groundwater Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1e88e5;
    }
    .risk-critical { background-color: #ffebee; border-color: #c62828; }
    .risk-high { background-color: #fff3e0; border-color: #ef6c00; }
    .risk-moderate { background-color: #fff8e1; border-color: #f9a825; }
    .risk-low { background-color: #e8f5e9; border-color: #2e7d32; }
    .stMetric > div { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = Path(__file__).parent.parent / "downloaded_data"


@st.cache_data
def load_latest_predictions():
    """Load most recent prediction results."""
    pred_files = sorted(OUTPUT_DIR.glob("predictions_*.csv"), reverse=True)
    if pred_files:
        return pd.read_csv(pred_files[0])
    return None


@st.cache_data
def load_latest_risk():
    """Load most recent risk classification."""
    risk_files = sorted(OUTPUT_DIR.glob("risk_classification_*.csv"), reverse=True)
    if risk_files:
        return pd.read_csv(risk_files[0])
    return None


@st.cache_data
def load_latest_geojson():
    """Load most recent GeoJSON with full village data including coordinates."""
    import geopandas as gpd
    geojson_files = sorted(OUTPUT_DIR.glob("villages_*.geojson"), reverse=True)
    if geojson_files:
        gdf = gpd.read_file(geojson_files[0])
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(gdf.drop(columns=['geometry']))
        # Add risk tier label if not present
        if 'risk_tier' in df.columns and 'risk_tier_label' not in df.columns:
            tier_map = {4: 'Critical', 3: 'High', 2: 'Moderate', 1: 'Low'}
            df['risk_tier_label'] = df['risk_tier'].map(tier_map).fillna('Unknown')
        return df
    return None


@st.cache_data
def load_grace_data():
    """Load GRACE time series."""
    grace_file = DATA_DIR / "grace" / "grace_krishna_proxy.csv"
    if grace_file.exists():
        return pd.read_csv(grace_file, parse_dates=['date'])
    return None


@st.cache_data
def load_alerts():
    """Load latest alerts."""
    alert_files = sorted(OUTPUT_DIR.glob("alerts_*.json"), reverse=True)
    if alert_files:
        with open(alert_files[0]) as f:
            return json.load(f)
    return []


def create_risk_map(df: pd.DataFrame):
    """Create interactive Folium map of risk zones."""
    # Center on Krishna district
    center_lat = df['centroid_lat'].mean() if 'centroid_lat' in df.columns else 16.25
    center_lon = df['centroid_lon'].mean() if 'centroid_lon' in df.columns else 80.75

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='cartodbpositron'
    )

    # Color mapping
    color_map = {
        4: '#c62828',  # Critical - Red
        3: '#ef6c00',  # High - Orange
        2: '#f9a825',  # Moderate - Yellow
        1: '#2e7d32'   # Low - Green
    }

    # Add markers for each village
    for _, row in df.iterrows():
        if 'centroid_lat' in row and 'centroid_lon' in row:
            lat = row['centroid_lat']
            lon = row['centroid_lon']

            if pd.notna(lat) and pd.notna(lon):
                risk_tier = row.get('risk_tier', 2)
                color = color_map.get(risk_tier, '#9e9e9e')

                popup_html = f"""
                <div style="font-family: Arial; min-width: 200px;">
                    <h4 style="margin: 5px 0;">{row.get('village', 'Unknown')}</h4>
                    <p style="margin: 3px 0;"><b>Risk Tier:</b> {row.get('risk_tier_label', 'N/A')}</p>
                    <p style="margin: 3px 0;"><b>Water Level:</b> {row.get('prediction', 0):.1f}m</p>
                    <p style="margin: 3px 0;"><b>Risk Score:</b> {row.get('risk_score', 0):.2f}</p>
                    <p style="margin: 3px 0;"><b>Aquifer:</b> {row.get('geo_class', 'N/A')}</p>
                </div>
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 2px solid gray;">
        <p style="margin: 0 0 5px 0; font-weight: bold;">Risk Levels</p>
        <p style="margin: 2px;"><span style="background:#c62828; padding: 2px 10px; border-radius: 3px;">&nbsp;</span> Critical</p>
        <p style="margin: 2px;"><span style="background:#ef6c00; padding: 2px 10px; border-radius: 3px;">&nbsp;</span> High</p>
        <p style="margin: 2px;"><span style="background:#f9a825; padding: 2px 10px; border-radius: 3px;">&nbsp;</span> Moderate</p>
        <p style="margin: 2px;"><span style="background:#2e7d32; padding: 2px 10px; border-radius: 3px;">&nbsp;</span> Low</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_prediction_histogram(df: pd.DataFrame):
    """Create histogram of water level predictions."""
    fig = px.histogram(
        df,
        x='prediction',
        nbins=30,
        color='risk_tier_label' if 'risk_tier_label' in df.columns else None,
        color_discrete_map={
            'Critical': '#c62828',
            'High': '#ef6c00',
            'Moderate': '#f9a825',
            'Low': '#2e7d32'
        },
        title='Distribution of Water Level Predictions'
    )
    fig.update_layout(
        xaxis_title='Predicted Water Level (m below ground)',
        yaxis_title='Number of Villages',
        showlegend=True
    )
    return fig


def create_risk_distribution_pie(df: pd.DataFrame):
    """Create pie chart of risk distribution."""
    if 'risk_tier_label' not in df.columns:
        return None

    risk_counts = df['risk_tier_label'].value_counts()

    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            'Critical': '#c62828',
            'High': '#ef6c00',
            'Moderate': '#f9a825',
            'Low': '#2e7d32'
        },
        title='Risk Tier Distribution'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_aquifer_comparison(df: pd.DataFrame):
    """Create bar chart comparing aquifer types."""
    if 'geo_class' not in df.columns:
        return None

    aquifer_stats = df.groupby('geo_class').agg({
        'prediction': 'mean',
        'risk_score': 'mean',
        'village': 'count'
    }).reset_index()
    aquifer_stats.columns = ['Aquifer', 'Avg Water Level', 'Avg Risk Score', 'Villages']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(name='Avg Water Level', x=aquifer_stats['Aquifer'], y=aquifer_stats['Avg Water Level']),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(name='Avg Risk Score', x=aquifer_stats['Aquifer'], y=aquifer_stats['Avg Risk Score'],
                   mode='lines+markers', line=dict(color='red', width=2)),
        secondary_y=True
    )

    fig.update_layout(title='Water Level and Risk by Aquifer Type')
    fig.update_yaxes(title_text='Water Level (m)', secondary_y=False)
    fig.update_yaxes(title_text='Risk Score', secondary_y=True)

    return fig


def create_grace_timeseries(grace_df: pd.DataFrame):
    """Create GRACE time series plot."""
    fig = px.line(
        grace_df,
        x='date',
        y='tws_anomaly_cm',
        title='GRACE Satellite: Regional Water Storage Anomaly'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='TWS Anomaly (cm)'
    )
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">💧 Smart Jal - Groundwater Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">AI-Powered Village-Level Groundwater Prediction for Krishna District</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Risk Map", "Predictions", "GRACE Satellite", "Alerts", "Scenario Analysis"]
    )

    # Load data
    predictions = load_latest_predictions()
    risk_data = load_latest_risk()
    grace_data = load_grace_data()
    alerts = load_alerts()

    if page == "Dashboard":
        st.header("Executive Dashboard")

        if risk_data is not None:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total = len(risk_data)
                st.metric("Total Villages", f"{total:,}")

            with col2:
                critical = (risk_data['risk_tier'] == 4).sum()
                st.metric("Critical Risk", f"{critical}", delta=f"{100*critical/total:.1f}%", delta_color="inverse")

            with col3:
                high = (risk_data['risk_tier'] == 3).sum()
                st.metric("High Risk", f"{high}", delta=f"{100*high/total:.1f}%", delta_color="inverse")

            with col4:
                mean_wl = risk_data['prediction'].mean()
                st.metric("Avg Water Level", f"{mean_wl:.1f}m")

            st.divider()

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                pie_fig = create_risk_distribution_pie(risk_data)
                if pie_fig:
                    st.plotly_chart(pie_fig, use_container_width=True)

            with col2:
                hist_fig = create_prediction_histogram(risk_data)
                if hist_fig:
                    st.plotly_chart(hist_fig, use_container_width=True)

            # Aquifer comparison
            aquifer_fig = create_aquifer_comparison(risk_data)
            if aquifer_fig:
                st.plotly_chart(aquifer_fig, use_container_width=True)

        else:
            st.warning("No prediction data available. Please run the pipeline first.")
            st.code("python backend/pipeline.py", language="bash")

    elif page == "Risk Map":
        st.header("Risk Map")

        # Load GeoJSON which has coordinates
        map_data = load_latest_geojson()
        if map_data is not None and 'centroid_lat' in map_data.columns:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Tier",
                    options=['Critical', 'High', 'Moderate', 'Low'],
                    default=['Critical', 'High', 'Moderate', 'Low']
                )
            with col2:
                if 'geo_class' in map_data.columns:
                    aquifer_filter = st.multiselect(
                        "Filter by Aquifer",
                        options=map_data['geo_class'].unique().tolist(),
                        default=map_data['geo_class'].unique().tolist()
                    )
                else:
                    aquifer_filter = None

            # Apply filters
            filtered = map_data[map_data['risk_tier_label'].isin(risk_filter)]
            if aquifer_filter:
                filtered = filtered[filtered['geo_class'].isin(aquifer_filter)]

            st.info(f"Showing {len(filtered)} of {len(map_data)} villages")

            # Display map
            risk_map = create_risk_map(filtered)
            st_folium(risk_map, width=None, height=600)

        else:
            st.warning("Location data not available for mapping.")

    elif page == "Predictions":
        st.header("Village Predictions")

        if predictions is not None:
            # Search and filter
            col1, col2, col3 = st.columns(3)

            with col1:
                search = st.text_input("Search Village", "")

            with col2:
                if 'district' in predictions.columns:
                    district = st.selectbox(
                        "District",
                        options=['All'] + predictions['district'].dropna().unique().tolist()
                    )
                else:
                    district = 'All'

            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['Risk Score', 'Water Level', 'Uncertainty']
                )

            # Apply filters
            filtered = predictions.copy()

            if search:
                filtered = filtered[filtered['village'].str.contains(search, case=False, na=False)]

            if district != 'All':
                filtered = filtered[filtered['district'] == district]

            # Sort
            sort_map = {
                'Risk Score': 'risk_score' if 'risk_score' in filtered.columns else 'prediction',
                'Water Level': 'prediction',
                'Uncertainty': 'uncertainty'
            }
            if sort_map[sort_by] in filtered.columns:
                filtered = filtered.sort_values(sort_map[sort_by], ascending=False)

            # Display
            st.dataframe(
                filtered[[c for c in ['village', 'district', 'mandal', 'prediction', 'uncertainty',
                                     'risk_score', 'risk_tier_label', 'geo_class'] if c in filtered.columns]],
                use_container_width=True,
                height=500
            )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        else:
            st.warning("No predictions available.")

    elif page == "GRACE Satellite":
        st.header("GRACE Satellite Data")

        if grace_data is not None:
            st.info("""
            **GRACE (Gravity Recovery and Climate Experiment)** satellites measure changes in Earth's
            gravitational field to detect water storage changes. This data provides a regional constraint
            for our village-level predictions, ensuring physical consistency.
            """)

            # Time series
            fig = create_grace_timeseries(grace_data)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Data Period", f"{grace_data['date'].min().strftime('%Y-%m')} to {grace_data['date'].max().strftime('%Y-%m')}")

            with col2:
                st.metric("Mean Anomaly", f"{grace_data['tws_anomaly_cm'].mean():.2f} cm")

            with col3:
                st.metric("Seasonal Amplitude", f"{grace_data['tws_anomaly_cm'].max() - grace_data['tws_anomaly_cm'].min():.1f} cm")

            # Show data table
            with st.expander("View Raw Data"):
                st.dataframe(grace_data, use_container_width=True)

        else:
            st.warning("GRACE data not available.")

    elif page == "Alerts":
        st.header("Active Alerts")

        if alerts:
            # Summary
            critical_alerts = [a for a in alerts if a['risk_tier'] == 4]
            high_alerts = [a for a in alerts if a['risk_tier'] == 3]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Critical Alerts", len(critical_alerts))
            with col2:
                st.metric("High Priority Alerts", len(high_alerts))

            st.divider()

            # Filter
            alert_filter = st.selectbox(
                "Show alerts",
                ["All", "Critical Only", "Critical + High"]
            )

            if alert_filter == "Critical Only":
                display_alerts = critical_alerts
            elif alert_filter == "Critical + High":
                display_alerts = critical_alerts + high_alerts
            else:
                display_alerts = alerts

            # Display alerts
            for alert in display_alerts[:50]:  # Limit to 50
                color_map = {
                    4: '🔴',
                    3: '🟠',
                    2: '🟡',
                    1: '🟢'
                }
                icon = color_map.get(alert['risk_tier'], '⚪')

                with st.expander(f"{icon} {alert['village']} - {alert['alert_level']}"):
                    st.write(f"**District:** {alert.get('district', 'N/A')}")
                    st.write(f"**Mandal:** {alert.get('mandal', 'N/A')}")
                    st.write(f"**Risk Score:** {alert['risk_score']:.2f}")
                    st.write(f"**Water Level:** {alert['water_level_prediction']:.1f}m")
                    st.write(f"**Message:** {alert['message']}")
                    st.write("**Recommended Actions:**")
                    for action in alert['recommended_actions']:
                        st.write(f"- {action}")

        else:
            st.info("No alerts generated yet. Run the pipeline to generate alerts.")

    elif page == "Scenario Analysis":
        st.header("What-If Scenario Analysis")

        st.info("""
        Explore how water levels might change under different scenarios.
        This helps with contingency planning and policy decisions.
        """)

        # Scenario inputs
        col1, col2 = st.columns(2)

        with col1:
            rainfall_factor = st.slider(
                "Rainfall Factor",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.05,
                help="1.0 = normal rainfall, 0.5 = drought, 1.5 = excess rainfall"
            )

        with col2:
            extraction_factor = st.slider(
                "Extraction Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="1.0 = current extraction, 2.0 = doubled extraction"
            )

        if st.button("Run Scenario", type="primary"):
            st.info("Scenario analysis would run here with the configured parameters.")

            # Simulated results
            if predictions is not None:
                # Simulate scenario impact
                scenario_preds = predictions.copy()
                scenario_preds['prediction'] = predictions['prediction'] - (1 - rainfall_factor) * 3 + (extraction_factor - 1) * 2

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Baseline")
                    st.metric("Mean Water Level", f"{predictions['prediction'].mean():.1f}m")

                with col2:
                    st.subheader("Scenario")
                    st.metric("Mean Water Level", f"{scenario_preds['prediction'].mean():.1f}m",
                             delta=f"{scenario_preds['prediction'].mean() - predictions['prediction'].mean():.1f}m")

                # Impact summary
                st.write(f"""
                **Scenario Impact:**
                - Rainfall at {rainfall_factor*100:.0f}% of normal
                - Extraction at {extraction_factor*100:.0f}% of current
                - Average water level change: {scenario_preds['prediction'].mean() - predictions['prediction'].mean():.2f}m
                """)

    # Footer
    st.divider()
    st.markdown("""
    <p style="text-align: center; color: gray; font-size: 0.8rem;">
    Smart Jal - Hackathon Prototype | Hierarchical Physics-Informed Groundwater Prediction
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
