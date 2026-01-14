#!/usr/bin/env python3
"""
Smart Jal - Risk Classification Module
Assigns risk tiers to villages based on predictions and vulnerability factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RiskClassifier:
    """
    Multi-factor risk classification for villages.

    Risk tiers:
    - Critical (4): Immediate intervention needed
    - High (3): Close monitoring required
    - Moderate (2): Standard management
    - Low (1): Sustainable status

    Factors considered:
    1. Current water level (depth below ground)
    2. Trend (declining, stable, rising)
    3. Extraction intensity
    4. Aquifer vulnerability
    5. Prediction uncertainty
    """

    # Default thresholds (can be calibrated)
    DEFAULT_THRESHOLDS = {
        'water_level': {
            'critical': 20,  # meters below ground
            'high': 15,
            'moderate': 10
        },
        'trend': {
            'critical': -1.0,  # meters per year decline
            'high': -0.5,
            'moderate': -0.2
        },
        'extraction': {
            'critical': 30,  # wells per km²
            'high': 20,
            'moderate': 10
        }
    }

    # Aquifer vulnerability scores
    AQUIFER_VULNERABILITY = {
        'Granite': 0.8,  # Hard rock, limited storage
        'Granite gneiss': 0.75,
        'Basalt': 0.7,
        'Quartzite': 0.6,
        'Limestone': 0.5,
        'Sandstone': 0.4,
        'Alluvium': 0.3,  # High storage, resilient
        'default': 0.5
    }

    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize classifier.

        Args:
            thresholds: Custom threshold values
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.risk_scores = None

    def classify(self,
                 predictions: pd.DataFrame,
                 villages: pd.DataFrame = None) -> pd.DataFrame:
        """
        Classify villages into risk tiers.

        Args:
            predictions: DataFrame with predictions and uncertainties
            villages: Optional DataFrame with additional village features

        Returns:
            DataFrame with risk classifications
        """
        print("Classifying risk tiers...")

        results = predictions.copy()

        # 1. Water level risk score (0-1)
        results['wl_risk_score'] = self._calculate_water_level_risk(
            results['prediction'].values
        )

        # 2. Uncertainty risk factor
        results['uncertainty_factor'] = self._calculate_uncertainty_factor(
            results['uncertainty'].values
        )

        # 3. Aquifer vulnerability
        if 'geo_class' in results.columns:
            results['aquifer_vulnerability'] = results['geo_class'].map(
                self.AQUIFER_VULNERABILITY
            ).fillna(0.5)
        else:
            results['aquifer_vulnerability'] = 0.5

        # 4. Extraction risk (if available)
        if villages is not None and 'well_density' in villages.columns:
            results['extraction_risk'] = self._calculate_extraction_risk(
                villages['well_density'].values
            )
        elif 'well_density' in results.columns:
            results['extraction_risk'] = self._calculate_extraction_risk(
                results['well_density'].values
            )
        else:
            results['extraction_risk'] = 0.5

        # 5. Calculate composite risk score
        results['risk_score'] = self._calculate_composite_score(results)

        # 6. Assign risk tier
        results['risk_tier'] = self._assign_risk_tier(results['risk_score'].values)
        results['risk_tier_label'] = results['risk_tier'].map({
            4: 'Critical',
            3: 'High',
            2: 'Moderate',
            1: 'Low'
        })

        # Summary
        tier_counts = results['risk_tier_label'].value_counts()
        print("\nRisk Tier Distribution:")
        for tier, count in tier_counts.items():
            pct = 100 * count / len(results)
            print(f"  {tier}: {count} villages ({pct:.1f}%)")

        self.risk_scores = results

        return results

    def _calculate_water_level_risk(self, water_levels: np.ndarray) -> np.ndarray:
        """
        Calculate risk score based on water level depth.

        Deeper water levels = higher risk.
        """
        thresholds = self.thresholds['water_level']

        risk = np.zeros(len(water_levels))

        # Score based on thresholds
        risk[water_levels >= thresholds['critical']] = 1.0
        risk[(water_levels >= thresholds['high']) & (water_levels < thresholds['critical'])] = 0.75
        risk[(water_levels >= thresholds['moderate']) & (water_levels < thresholds['high'])] = 0.5
        risk[water_levels < thresholds['moderate']] = 0.25

        return risk

    def _calculate_uncertainty_factor(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Calculate risk factor based on prediction uncertainty.

        Higher uncertainty = higher risk (precautionary principle).
        """
        # Normalize to 0-1 range
        max_unc = np.percentile(uncertainties, 95)
        normalized = np.clip(uncertainties / max_unc, 0, 1)

        # Higher uncertainty slightly increases risk
        return 1 + 0.2 * normalized

    def _calculate_extraction_risk(self, well_density: np.ndarray) -> np.ndarray:
        """
        Calculate risk based on extraction intensity.
        """
        thresholds = self.thresholds['extraction']

        risk = np.zeros(len(well_density))

        risk[well_density >= thresholds['critical']] = 1.0
        risk[(well_density >= thresholds['high']) & (well_density < thresholds['critical'])] = 0.75
        risk[(well_density >= thresholds['moderate']) & (well_density < thresholds['high'])] = 0.5
        risk[well_density < thresholds['moderate']] = 0.25

        return risk

    def _calculate_composite_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate weighted composite risk score.
        """
        weights = {
            'wl_risk_score': 0.40,
            'aquifer_vulnerability': 0.25,
            'extraction_risk': 0.20,
            'uncertainty_factor': 0.15
        }

        score = np.zeros(len(df))

        for col, weight in weights.items():
            if col in df.columns:
                if col == 'uncertainty_factor':
                    # Uncertainty is a multiplier, not additive
                    continue
                values = df[col].fillna(0.5).values
                score += weight * values

        # Apply uncertainty factor
        if 'uncertainty_factor' in df.columns:
            score = score * df['uncertainty_factor'].values

        # Normalize to 0-1
        score = np.clip(score, 0, 1)

        return score

    def _assign_risk_tier(self, scores: np.ndarray) -> np.ndarray:
        """
        Assign risk tier based on composite score.
        """
        tiers = np.ones(len(scores), dtype=int)  # Default: Low

        tiers[scores >= 0.75] = 4  # Critical
        tiers[(scores >= 0.5) & (scores < 0.75)] = 3  # High
        tiers[(scores >= 0.25) & (scores < 0.5)] = 2  # Moderate
        # scores < 0.25 remain as 1 (Low)

        return tiers

    def get_critical_villages(self,
                              predictions: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get list of critical risk villages.
        """
        if predictions is not None:
            df = self.classify(predictions)
        elif self.risk_scores is not None:
            df = self.risk_scores
        else:
            raise ValueError("No risk scores available")

        critical = df[df['risk_tier'] == 4].copy()
        return critical.sort_values('risk_score', ascending=False)

    def get_risk_summary(self) -> Dict:
        """
        Get summary statistics of risk classification.
        """
        if self.risk_scores is None:
            return {}

        df = self.risk_scores

        summary = {
            'total_villages': len(df),
            'tier_distribution': df['risk_tier_label'].value_counts().to_dict(),
            'mean_risk_score': df['risk_score'].mean(),
            'critical_count': (df['risk_tier'] == 4).sum(),
            'high_count': (df['risk_tier'] == 3).sum(),
            'risk_by_aquifer': df.groupby('geo_class')['risk_score'].mean().to_dict() if 'geo_class' in df.columns else {}
        }

        return summary


class VulnerabilityAssessor:
    """
    Assess village vulnerability to water stress.
    """

    @staticmethod
    def calculate_vulnerability_index(df: pd.DataFrame) -> pd.Series:
        """
        Calculate comprehensive vulnerability index.

        Factors:
        - Physical: aquifer type, depth, recharge potential
        - Socio-economic: population density, agriculture dependency
        - Adaptive capacity: infrastructure, alternatives
        """
        # Physical vulnerability
        physical = df.get('aquifer_vulnerability', 0.5)

        # Extraction pressure
        extraction = df.get('extraction_risk', 0.5)

        # Combine into index
        vulnerability = 0.6 * physical + 0.4 * extraction

        return vulnerability


class AlertGenerator:
    """
    Generate alerts based on risk classification.
    """

    ALERT_TEMPLATES = {
        4: {
            'level': 'CRITICAL',
            'color': 'red',
            'message': 'Immediate intervention required. Water level critically low.',
            'actions': [
                'Implement emergency water supply',
                'Restrict new bore well permits',
                'Activate demand management measures'
            ]
        },
        3: {
            'level': 'HIGH',
            'color': 'orange',
            'message': 'Close monitoring required. Significant stress detected.',
            'actions': [
                'Increase monitoring frequency',
                'Review extraction permits',
                'Plan contingency water supply'
            ]
        },
        2: {
            'level': 'MODERATE',
            'color': 'yellow',
            'message': 'Standard management with attention to trends.',
            'actions': [
                'Continue regular monitoring',
                'Promote water conservation',
                'Review demand patterns'
            ]
        },
        1: {
            'level': 'LOW',
            'color': 'green',
            'message': 'Sustainable water levels. Continue current practices.',
            'actions': [
                'Maintain current management',
                'Document successful practices',
                'Share learnings with neighbors'
            ]
        }
    }

    @classmethod
    def generate_alerts(cls, risk_df: pd.DataFrame) -> List[Dict]:
        """
        Generate alerts for villages based on risk tier.

        Args:
            risk_df: DataFrame with risk classifications

        Returns:
            List of alert dictionaries
        """
        alerts = []

        for _, row in risk_df.iterrows():
            tier = row['risk_tier']
            template = cls.ALERT_TEMPLATES[tier]

            alert = {
                'village': row.get('village', 'Unknown'),
                'district': row.get('district', 'Unknown'),
                'mandal': row.get('mandal', 'Unknown'),
                'risk_tier': tier,
                'risk_score': row['risk_score'],
                'alert_level': template['level'],
                'color': template['color'],
                'message': template['message'],
                'recommended_actions': template['actions'],
                'water_level_prediction': row['prediction'],
                'uncertainty': row['uncertainty'],
                'aquifer_type': row.get('geo_class', 'Unknown')
            }

            alerts.append(alert)

        # Sort by risk score (highest first)
        alerts.sort(key=lambda x: x['risk_score'], reverse=True)

        return alerts

    @classmethod
    def generate_summary_report(cls, risk_df: pd.DataFrame) -> str:
        """
        Generate summary report of risk status.
        """
        total = len(risk_df)
        critical = (risk_df['risk_tier'] == 4).sum()
        high = (risk_df['risk_tier'] == 3).sum()
        moderate = (risk_df['risk_tier'] == 2).sum()
        low = (risk_df['risk_tier'] == 1).sum()

        report = f"""
SMART JAL - RISK STATUS REPORT
==============================

Total Villages Assessed: {total}

RISK TIER DISTRIBUTION:
  🔴 Critical: {critical} ({100*critical/total:.1f}%)
  🟠 High:     {high} ({100*high/total:.1f}%)
  🟡 Moderate: {moderate} ({100*moderate/total:.1f}%)
  🟢 Low:      {low} ({100*low/total:.1f}%)

IMMEDIATE ATTENTION REQUIRED:
  Villages in Critical/High tiers: {critical + high}

KEY STATISTICS:
  Mean Risk Score: {risk_df['risk_score'].mean():.2f}
  Mean Water Level: {risk_df['prediction'].mean():.1f}m
  Mean Uncertainty: {risk_df['uncertainty'].mean():.2f}m
"""

        if 'geo_class' in risk_df.columns:
            report += "\nRISK BY AQUIFER TYPE:\n"
            for aquifer, group in risk_df.groupby('geo_class'):
                mean_risk = group['risk_score'].mean()
                report += f"  {aquifer}: {mean_risk:.2f}\n"

        return report


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing risk classifier...")

    np.random.seed(42)
    n_villages = 100

    # Create synthetic predictions
    predictions = pd.DataFrame({
        'village': [f'Village_{i}' for i in range(n_villages)],
        'prediction': np.random.uniform(5, 25, n_villages),
        'uncertainty': np.random.uniform(0.5, 5, n_villages),
        'geo_class': np.random.choice(['Granite', 'Basalt', 'Alluvium'], n_villages),
        'well_density': np.random.uniform(0, 40, n_villages)
    })

    # Classify
    classifier = RiskClassifier()
    results = classifier.classify(predictions)

    # Get critical villages
    critical = classifier.get_critical_villages()
    print(f"\nCritical villages: {len(critical)}")

    # Generate alerts
    alerts = AlertGenerator.generate_alerts(results)
    print(f"\nGenerated {len(alerts)} alerts")

    # Summary report
    report = AlertGenerator.generate_summary_report(results)
    print(report)
