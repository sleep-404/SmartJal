"""
Smart Jal - Risk Summary Cards Component
"""

import streamlit as st
import pandas as pd


def create_risk_summary_cards(df: pd.DataFrame):
    """
    Create summary cards showing risk tier counts.

    Args:
        df: DataFrame with risk_tier column
    """
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)

    with col1:
        critical = (df['risk_tier'] == 4).sum() if 'risk_tier' in df.columns else 0
        st.metric(
            "Critical",
            f"{critical}",
            delta=f"{100*critical/total:.1f}%" if total > 0 else "0%",
            delta_color="inverse"
        )

    with col2:
        high = (df['risk_tier'] == 3).sum() if 'risk_tier' in df.columns else 0
        st.metric(
            "High Risk",
            f"{high}",
            delta=f"{100*high/total:.1f}%" if total > 0 else "0%",
            delta_color="inverse"
        )

    with col3:
        moderate = (df['risk_tier'] == 2).sum() if 'risk_tier' in df.columns else 0
        st.metric(
            "Moderate",
            f"{moderate}",
            delta=f"{100*moderate/total:.1f}%" if total > 0 else "0%"
        )

    with col4:
        low = (df['risk_tier'] == 1).sum() if 'risk_tier' in df.columns else 0
        st.metric(
            "Low Risk",
            f"{low}",
            delta=f"{100*low/total:.1f}%" if total > 0 else "0%",
            delta_color="normal"
        )
