"""
Smart Jal - Map View Component
Folium map for risk visualization.
"""

import folium
import pandas as pd


def create_risk_map(df: pd.DataFrame) -> folium.Map:
    """
    Create interactive Folium map of risk zones.

    Args:
        df: DataFrame with village data including centroid_lat, centroid_lon, risk_tier

    Returns:
        Folium Map object
    """
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

    return m
