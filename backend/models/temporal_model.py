#!/usr/bin/env python3
"""
Smart Jal - Temporal Model Module
Time series decomposition for water level prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Try to import statsmodels
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class WaterLevelDecomposer:
    """
    Decompose water level time series into components:

    Water_Level = Baseline + Seasonal + Trend + Anomaly

    Where:
    - Baseline: Long-term average level for the aquifer
    - Seasonal: Annual monsoon-driven cycle
    - Trend: Multi-year decline/recovery
    - Anomaly: Residual (drought/flood effects)
    """

    def __init__(self, period: int = 12):
        """
        Initialize decomposer.

        Args:
            period: Seasonal period (12 for monthly data)
        """
        self.period = period
        self.baseline = None
        self.seasonal_pattern = None
        self.trend_slope = None
        self.decomposition = None

    def decompose(self, series: pd.Series) -> Dict[str, np.ndarray]:
        """
        Decompose time series into components.

        Args:
            series: Time series with DatetimeIndex

        Returns:
            Dict with 'seasonal', 'trend', 'residual' components
        """
        # Handle missing values
        series_clean = series.interpolate(method='linear', limit=3)

        # Calculate baseline
        self.baseline = series_clean.mean()

        if STATSMODELS_AVAILABLE and len(series_clean) >= 2 * self.period:
            # Use STL decomposition
            stl = STL(series_clean, period=self.period, robust=True)
            result = stl.fit()

            self.decomposition = {
                'seasonal': result.seasonal.values,
                'trend': result.trend.values,
                'residual': result.resid.values,
                'observed': series_clean.values
            }

            # Extract seasonal pattern (average seasonal component by month)
            months = series_clean.index.month
            self.seasonal_pattern = pd.Series(
                result.seasonal.values, index=months
            ).groupby(level=0).mean()

            # Extract trend slope
            x = np.arange(len(result.trend))
            valid = ~np.isnan(result.trend.values)
            if valid.sum() > 2:
                self.trend_slope, _ = np.polyfit(x[valid], result.trend.values[valid], 1)
            else:
                self.trend_slope = 0
        else:
            # Simple decomposition fallback
            self.decomposition = self._simple_decomposition(series_clean)

        return self.decomposition

    def _simple_decomposition(self, series: pd.Series) -> Dict[str, np.ndarray]:
        """
        Simple decomposition when statsmodels not available.
        """
        n = len(series)

        # Seasonal: monthly averages
        monthly_avg = series.groupby(series.index.month).transform('mean')
        seasonal = (monthly_avg - monthly_avg.mean()).values

        # Trend: linear fit
        x = np.arange(n)
        y = series.values
        valid = ~np.isnan(y)
        if valid.sum() > 2:
            slope, intercept = np.polyfit(x[valid], y[valid], 1)
            trend = slope * x + intercept
            self.trend_slope = slope
        else:
            trend = np.full(n, series.mean())
            self.trend_slope = 0

        # Residual
        residual = series.values - seasonal - trend

        # Store seasonal pattern
        self.seasonal_pattern = series.groupby(series.index.month).mean() - self.baseline

        return {
            'seasonal': seasonal,
            'trend': trend,
            'residual': residual,
            'observed': series.values
        }

    def forecast(self,
                 n_periods: int,
                 last_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future values.

        Args:
            n_periods: Number of periods to forecast
            last_date: Last date in training data

        Returns:
            Tuple of (forecasts, uncertainties)
        """
        if self.decomposition is None:
            raise ValueError("Must call decompose() first")

        forecasts = []
        uncertainties = []

        # Get last trend value
        last_trend = self.decomposition['trend'][-1]

        # Residual std for uncertainty
        residual_std = np.nanstd(self.decomposition['residual'])

        for i in range(1, n_periods + 1):
            # Future month
            future_date = last_date + pd.DateOffset(months=i)
            month = future_date.month

            # Seasonal component
            if self.seasonal_pattern is not None and month in self.seasonal_pattern.index:
                seasonal = self.seasonal_pattern[month]
            else:
                seasonal = 0

            # Trend component (extrapolate)
            trend = last_trend + self.trend_slope * i

            # Forecast
            forecast = self.baseline + seasonal + (trend - self.baseline)
            forecasts.append(forecast)

            # Uncertainty grows with horizon
            uncertainty = residual_std * np.sqrt(i)
            uncertainties.append(uncertainty)

        return np.array(forecasts), np.array(uncertainties)


class PiezometerForecaster:
    """
    Forecast water levels for each piezometer using time series analysis.
    """

    def __init__(self):
        self.decomposers = {}
        self.forecasts = {}

    def fit(self,
            water_levels: pd.DataFrame,
            min_observations: int = 24) -> 'PiezometerForecaster':
        """
        Fit time series models for each piezometer.

        Args:
            water_levels: DataFrame with 'piezo_id', 'date', 'water_level'
            min_observations: Minimum observations required

        Returns:
            self
        """
        print("Fitting temporal models for piezometers...")

        id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'
        piezometers = water_levels[id_col].unique()

        fitted = 0
        skipped = 0

        for piezo in piezometers:
            piezo_data = water_levels[water_levels[id_col] == piezo].copy()
            piezo_data = piezo_data.sort_values('date')

            # Check if enough data
            valid_data = piezo_data.dropna(subset=['water_level'])
            if len(valid_data) < min_observations:
                skipped += 1
                continue

            # Create series with DatetimeIndex
            series = pd.Series(
                valid_data['water_level'].values,
                index=valid_data['date']
            )

            # Decompose
            decomposer = WaterLevelDecomposer(period=12)
            try:
                decomposer.decompose(series)
                self.decomposers[piezo] = decomposer
                fitted += 1
            except Exception as e:
                skipped += 1
                continue

        print(f"  Fitted: {fitted} piezometers")
        print(f"  Skipped: {skipped} (insufficient data)")

        return self

    def forecast_all(self,
                     target_date: pd.Timestamp,
                     water_levels: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast water levels for all piezometers.

        Args:
            target_date: Date to forecast
            water_levels: Historical water levels

        Returns:
            DataFrame with forecasts
        """
        print(f"Forecasting for {target_date}...")

        id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'
        results = []

        for piezo, decomposer in self.decomposers.items():
            # Get last date for this piezometer
            piezo_data = water_levels[water_levels[id_col] == piezo]
            last_date = piezo_data['date'].max()

            # Calculate periods to forecast
            n_periods = ((target_date.year - last_date.year) * 12 +
                        (target_date.month - last_date.month))

            if n_periods <= 0:
                # Use decomposition directly
                forecast = decomposer.baseline
                uncertainty = np.nanstd(decomposer.decomposition['residual'])
            else:
                # Forecast
                forecasts, uncertainties = decomposer.forecast(n_periods, last_date)
                forecast = forecasts[-1]
                uncertainty = uncertainties[-1]

            results.append({
                id_col: piezo,
                'forecast': forecast,
                'uncertainty': uncertainty,
                'baseline': decomposer.baseline,
                'trend_slope': decomposer.trend_slope,
                'periods_ahead': max(0, n_periods)
            })

        return pd.DataFrame(results)

    def get_seasonal_patterns(self) -> pd.DataFrame:
        """
        Get seasonal patterns for all piezometers.

        Returns:
            DataFrame with monthly patterns
        """
        patterns = []

        for piezo, decomposer in self.decomposers.items():
            if decomposer.seasonal_pattern is not None:
                for month, value in decomposer.seasonal_pattern.items():
                    patterns.append({
                        'piezo_id': piezo,
                        'month': month,
                        'seasonal_effect': value
                    })

        return pd.DataFrame(patterns)

    def get_trends(self) -> pd.DataFrame:
        """
        Get trend information for all piezometers.

        Returns:
            DataFrame with trend slopes and baselines
        """
        trends = []

        for piezo, decomposer in self.decomposers.items():
            trends.append({
                'piezo_id': piezo,
                'baseline': decomposer.baseline,
                'trend_slope': decomposer.trend_slope,
                'annual_change': decomposer.trend_slope * 12
            })

        return pd.DataFrame(trends)


class SeasonalAdjuster:
    """
    Adjust predictions for seasonal effects.
    """

    def __init__(self):
        self.seasonal_factors = None

    def fit(self, water_levels: pd.DataFrame) -> 'SeasonalAdjuster':
        """
        Learn seasonal adjustment factors from data.

        Args:
            water_levels: DataFrame with water level time series

        Returns:
            self
        """
        print("Learning seasonal adjustment factors...")

        # Calculate monthly averages across all piezometers
        water_levels = water_levels.copy()
        water_levels['month'] = water_levels['date'].dt.month

        # Overall mean
        overall_mean = water_levels['water_level'].mean()

        # Monthly deviations
        monthly_means = water_levels.groupby('month')['water_level'].mean()
        self.seasonal_factors = monthly_means - overall_mean

        print(f"  Seasonal amplitude: {self.seasonal_factors.max() - self.seasonal_factors.min():.2f}m")

        return self

    def adjust(self,
               predictions: np.ndarray,
               target_month: int) -> np.ndarray:
        """
        Apply seasonal adjustment to predictions.

        Args:
            predictions: Base predictions
            target_month: Month to adjust for

        Returns:
            Adjusted predictions
        """
        if self.seasonal_factors is None:
            return predictions

        adjustment = self.seasonal_factors.get(target_month, 0)
        return predictions + adjustment

    def get_factors(self) -> pd.Series:
        """Get seasonal adjustment factors."""
        return self.seasonal_factors


def detect_anomalies(series: pd.Series,
                     window: int = 12,
                     n_std: float = 2.0) -> pd.Series:
    """
    Detect anomalies in water level time series.

    Args:
        series: Water level time series
        window: Rolling window size
        n_std: Number of standard deviations for threshold

    Returns:
        Boolean series marking anomalies
    """
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()

    # Mark anomalies
    lower_bound = rolling_mean - n_std * rolling_std
    upper_bound = rolling_mean + n_std * rolling_std

    anomalies = (series < lower_bound) | (series > upper_bound)

    return anomalies


def calculate_trend_significance(series: pd.Series) -> Dict:
    """
    Calculate trend and its statistical significance.

    Args:
        series: Water level time series

    Returns:
        Dict with trend statistics
    """
    x = np.arange(len(series))
    y = series.values

    # Remove NaN
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return {'slope': 0, 'significant': False, 'p_value': 1.0}

    x_valid = x[valid]
    y_valid = y[valid]

    # Linear regression
    n = len(y_valid)
    x_mean = x_valid.mean()
    y_mean = y_valid.mean()

    numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
    denominator = np.sum((x_valid - x_mean) ** 2)

    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean

    # Calculate residuals and standard error
    y_pred = slope * x_valid + intercept
    residuals = y_valid - y_pred
    sse = np.sum(residuals ** 2)
    mse = sse / (n - 2) if n > 2 else 0
    se_slope = np.sqrt(mse / denominator) if denominator > 0 else float('inf')

    # t-statistic
    t_stat = slope / se_slope if se_slope > 0 else 0

    # Approximate p-value (two-tailed)
    from scipy import stats
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1.0

    return {
        'slope': slope,
        'annual_change': slope * 12,  # Convert monthly to annual
        'intercept': intercept,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing temporal model...")

    np.random.seed(42)

    # Generate synthetic water level time series
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='MS')
    n = len(dates)

    # Components
    baseline = 15.0
    seasonal = 5 * np.cos(2 * np.pi * (dates.month - 10) / 12)  # Peak in October
    trend = -0.05 * np.arange(n)  # Slight decline
    noise = np.random.normal(0, 1, n)

    water_level = baseline + seasonal + trend + noise

    # Create DataFrame
    wl_df = pd.DataFrame({
        'piezo_id': 'P001',
        'date': dates,
        'water_level': water_level
    })

    # Test decomposer
    series = pd.Series(water_level, index=dates)
    decomposer = WaterLevelDecomposer()
    components = decomposer.decompose(series)

    print(f"\nDecomposition results:")
    print(f"  Baseline: {decomposer.baseline:.2f}m")
    print(f"  Trend slope: {decomposer.trend_slope:.4f}m/month ({decomposer.trend_slope*12:.2f}m/year)")

    # Forecast
    forecasts, uncertainties = decomposer.forecast(6, dates[-1])
    print(f"\n6-month forecast:")
    print(f"  Values: {forecasts}")
    print(f"  Uncertainties: {uncertainties}")

    # Test PiezometerForecaster
    forecaster = PiezometerForecaster()
    forecaster.fit(wl_df)

    forecast_df = forecaster.forecast_all(pd.Timestamp('2025-06-01'), wl_df)
    print(f"\nForecast DataFrame:")
    print(forecast_df)
