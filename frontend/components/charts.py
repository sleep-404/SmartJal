"""
Smart Jal - Chart Components
Plotly charts for data visualization.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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


def create_timeseries_chart(village_name: str, historical_df: pd.DataFrame):
    """Create time series chart for a specific village."""
    fig = px.line(
        historical_df,
        x='date',
        y='water_level',
        title=f'Water Level History - {village_name}'
    )

    # Add trend line if available
    if 'trend' in historical_df.columns:
        fig.add_trace(
            go.Scatter(x=historical_df['date'], y=historical_df['trend'],
                       name='Trend', line=dict(dash='dash', color='red'))
        )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Water Level (m below ground)',
        yaxis_autorange='reversed'  # Lower values (shallow) at top
    )

    return fig
