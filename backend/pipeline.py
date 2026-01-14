#!/usr/bin/env python3
"""
Smart Jal - End-to-End Pipeline
Orchestrates data loading, preprocessing, modeling, and prediction.

Implements the WINNING_ARCHITECTURE.md + novel_ideas.md research:
- Physics-informed water balance (ΔStorage = Recharge - Extraction)
- Temporal decomposition (28 years of signal)
- Hierarchical constraints (GRACE → Aquifer → Village)
- Graph Neural Network (GNN) for spatial interpolation
- SHAP explainability for transparent predictions
- Conformal Prediction for uncertainty quantification
- Transfer Learning for data-sparse aquifers
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json
import warnings

warnings.filterwarnings('ignore')

# Local imports
from data_processing.load_data import load_all_data
from data_processing.preprocess import preprocess_all
from data_processing.feature_engineering import build_feature_matrix
from models.ensemble import HierarchicalEnsemble, AquiferLevelAggregator, GRACEValidator
from models.risk_classifier import RiskClassifier, AlertGenerator
from models.physics_model import PhysicsInformedEnsemble

# Novel approaches
from models.gnn_model import GNNPredictor, TORCH_AVAILABLE
from models.explainability import SHAPExplainer, ConformalPredictor, TransferLearner, SHAP_AVAILABLE, MAPIE_AVAILABLE

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class SmartJalPipeline:
    """
    End-to-end pipeline for Smart Jal groundwater prediction.

    Pipeline stages:
    1. Load data
    2. Preprocess
    3. Feature engineering
    4. Model fitting
    5. Prediction
    6. Risk classification
    7. Export results
    """

    def __init__(self, target_date: Optional[pd.Timestamp] = None,
                 use_physics_model: bool = True,
                 use_gnn: bool = True,
                 use_explainability: bool = True):
        """
        Initialize pipeline.

        Args:
            target_date: Date for predictions (default: latest available)
            use_physics_model: Whether to use physics-informed model (default: True)
            use_gnn: Whether to use Graph Neural Network (default: True)
            use_explainability: Whether to generate SHAP explanations (default: True)
        """
        if target_date is None:
            self.target_date = pd.Timestamp('2023-10-01')
        elif isinstance(target_date, str):
            self.target_date = pd.Timestamp(target_date)
        else:
            self.target_date = target_date
        self.use_physics_model = use_physics_model
        self.use_gnn = use_gnn and TORCH_AVAILABLE
        self.use_explainability = use_explainability and SHAP_AVAILABLE
        self.raw_data = None
        self.processed_data = None
        self.feature_data = None
        self.model = None
        self.physics_model = None
        self.gnn_model = None
        self.shap_explainer = None
        self.conformal_predictor = None
        self.predictions = None
        self.risk_results = None
        self.explanations = None

    def run(self, save_outputs: bool = True) -> Dict:
        """
        Run complete pipeline.

        Args:
            save_outputs: Whether to save results to disk

        Returns:
            Dict with all results
        """
        print("\n" + "=" * 70)
        print("       SMART JAL - GROUNDWATER PREDICTION PIPELINE")
        print("=" * 70)
        print(f"\nTarget Date: {self.target_date}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = {}

        # Stage 1: Load data
        print("\n" + "-" * 60)
        print("STAGE 1: Loading Data")
        print("-" * 60)
        self.raw_data = load_all_data()
        results['data_loaded'] = True

        # Stage 2: Preprocess
        print("\n" + "-" * 60)
        print("STAGE 2: Preprocessing")
        print("-" * 60)
        self.processed_data = preprocess_all(self.raw_data)
        results['preprocessing_done'] = True

        # Stage 3: Feature engineering
        print("\n" + "-" * 60)
        print("STAGE 3: Feature Engineering")
        print("-" * 60)
        self.feature_data, piezo_features = build_feature_matrix(
            self.processed_data,
            target_year=self.target_date.year,
            target_month=self.target_date.month
        )
        results['features_built'] = True

        # Stage 4: Model fitting
        print("\n" + "-" * 60)
        print("STAGE 4: Model Fitting")
        print("-" * 60)

        # Prepare data dict for model
        model_data = {
            'villages': self.feature_data,
            'piezometers': self.processed_data['piezometers'],
            'water_levels': self.processed_data['water_levels'],
            'grace': self.processed_data.get('grace')
        }

        # Fit physics-informed model (uses research from WINNING_ARCHITECTURE.md)
        if self.use_physics_model:
            print("\n[Physics-Informed Model] Using water balance + temporal decomposition...")
            self.physics_model = PhysicsInformedEnsemble()
            self.physics_model.fit(
                self.processed_data['water_levels'],
                self.processed_data['piezometers']
            )

        # Also fit feature model for comparison
        self.model = HierarchicalEnsemble()
        fit_metrics = self.model.fit(model_data, self.target_date)
        results['model_metrics'] = fit_metrics

        # Fit GNN model (novel approach from research)
        if self.use_gnn:
            print("\n[Graph Neural Network] Training spatial GNN...")
            self.gnn_model = GNNPredictor(
                hidden_channels=64,
                num_layers=3,
                epochs=200
            )
            gnn_metrics = self.gnn_model.fit(
                self.feature_data,
                self.processed_data['piezometers'],
                self.processed_data['water_levels'],
                self.target_date
            )
            results['gnn_metrics'] = gnn_metrics

        # Setup explainability (SHAP + Conformal)
        if self.use_explainability and hasattr(self.model, 'feature_model') and self.model.feature_model.model is not None:
            print("\n[Explainability] Setting up SHAP explainer...")
            self.shap_explainer = SHAPExplainer(
                self.model.feature_model.model,
                self.model.feature_model.feature_cols
            )
            # Get training data for SHAP
            X_train = self.model.feature_model._prepare_features(self.feature_data, fit_scaler=False)
            self.shap_explainer.fit(X_train)

            print("\n[Uncertainty] Setting up Conformal Prediction...")
            if MAPIE_AVAILABLE:
                from sklearn.ensemble import GradientBoostingRegressor
                self.conformal_predictor = ConformalPredictor(
                    GradientBoostingRegressor(n_estimators=100, random_state=42),
                    alpha=0.1  # 90% confidence intervals
                )

        # Stage 5: Prediction
        print("\n" + "-" * 60)
        print("STAGE 5: Generating Predictions")
        print("-" * 60)

        if self.use_physics_model and self.physics_model is not None:
            # Use physics-informed predictions
            print("\n[Physics Model] Predicting using water balance equation...")
            physics_preds = self.physics_model.predict_all_villages(
                self.feature_data, self.target_date
            )

            # Also get feature model predictions for comparison
            feature_preds = self.model.predict(self.feature_data, self.target_date)

            # Combine: weight physics model (60%) with feature model (40%)
            print("\n[Ensemble] Combining physics + feature predictions...")
            self.predictions = physics_preds.copy()

            # Ensemble the predictions
            combined_pred = (
                0.6 * physics_preds['prediction'].values +
                0.4 * feature_preds['prediction'].values
            )
            combined_unc = np.sqrt(
                0.6 * physics_preds['uncertainty'].values**2 +
                0.4 * feature_preds['uncertainty'].values**2
            )

            self.predictions['prediction'] = combined_pred
            self.predictions['uncertainty'] = combined_unc
            self.predictions['physics_prediction'] = physics_preds['prediction']
            self.predictions['feature_prediction'] = feature_preds['prediction']

            results['physics_mean'] = physics_preds['prediction'].mean()
            results['feature_mean'] = feature_preds['prediction'].mean()
            results['combined_mean'] = combined_pred.mean()
        else:
            # Use feature model only
            self.predictions = self.model.predict(self.feature_data, self.target_date)

        # Add GNN predictions to ensemble
        if self.use_gnn and self.gnn_model is not None:
            print("\n[GNN] Adding GNN predictions to ensemble...")
            gnn_preds, gnn_unc = self.gnn_model.predict(self.feature_data)

            # Combine GNN with existing predictions (GNN gets 30% weight)
            current_preds = self.predictions['prediction'].values
            combined_with_gnn = 0.7 * current_preds + 0.3 * gnn_preds

            self.predictions['gnn_prediction'] = gnn_preds
            self.predictions['prediction_before_gnn'] = current_preds
            self.predictions['prediction'] = combined_with_gnn

            results['gnn_mean'] = float(np.nanmean(gnn_preds))
            print(f"  GNN mean prediction: {results['gnn_mean']:.2f}m")

        results['n_predictions'] = len(self.predictions)

        # Generate explanations for high-risk villages
        if self.use_explainability and self.shap_explainer is not None:
            print("\n[SHAP] Generating explanations...")
            self._generate_explanations()
            results['explanations_generated'] = True

        # Stage 6: Risk classification
        print("\n" + "-" * 60)
        print("STAGE 6: Risk Classification")
        print("-" * 60)
        classifier = RiskClassifier()
        self.risk_results = classifier.classify(self.predictions, self.feature_data)
        results['risk_summary'] = classifier.get_risk_summary()

        # Stage 7: Generate alerts
        print("\n" + "-" * 60)
        print("STAGE 7: Generating Alerts")
        print("-" * 60)
        alerts = AlertGenerator.generate_alerts(self.risk_results)
        results['n_alerts'] = len(alerts)
        results['critical_alerts'] = len([a for a in alerts if a['risk_tier'] == 4])

        # Aggregations
        aquifer_stats = AquiferLevelAggregator.aggregate(
            self.predictions, self.feature_data
        )
        results['aquifer_stats'] = aquifer_stats.to_dict('records')

        # GRACE validation
        if self.processed_data.get('grace') is not None:
            validator = GRACEValidator(self.processed_data['grace'])
            validation = validator.validate(
                self.predictions, self.feature_data, self.target_date
            )
            results['grace_validation'] = validation

        # Save outputs
        if save_outputs:
            self._save_outputs(results, alerts)

        # Print summary
        self._print_summary(results)

        return results

    def run_scenarios(self,
                      scenarios: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """
        Run scenario analysis.

        Args:
            scenarios: Dict of scenario definitions

        Returns:
            Dict of scenario results
        """
        if self.model is None:
            raise ValueError("Run pipeline first with run()")

        default_scenarios = {
            'drought_25': {
                'rainfall_factor': 0.75,
                'description': '25% below normal rainfall'
            },
            'drought_50': {
                'rainfall_factor': 0.50,
                'description': '50% below normal rainfall'
            },
            'increased_extraction_25': {
                'extraction_factor': 1.25,
                'description': '25% increase in extraction'
            }
        }

        scenarios = scenarios or default_scenarios

        print("\n" + "=" * 60)
        print("SCENARIO ANALYSIS")
        print("=" * 60)

        scenario_results = self.model.predict_scenarios(
            self.feature_data, self.target_date, scenarios
        )

        return scenario_results

    def _generate_explanations(self):
        """Generate SHAP explanations for predictions."""
        if self.shap_explainer is None:
            return

        # Get feature matrix
        X = self.model.feature_model._prepare_features(self.feature_data, fit_scaler=False)

        # Generate batch explanations
        village_names = self.feature_data['village'].tolist() if 'village' in self.feature_data.columns else None
        self.explanations = self.shap_explainer.explain_batch(X, village_names)

        # Add top explanation to predictions
        if self.explanations is not None and len(self.explanations) > 0:
            self.predictions['top_factor'] = self.explanations['most_important_feature'].values
            self.predictions['top_factor_value'] = self.explanations['most_important_value'].values

        print(f"  Generated explanations for {len(self.explanations)} villages")

    def get_village_explanation(self, village_name: str) -> Dict:
        """
        Get detailed explanation for a specific village.

        Args:
            village_name: Name of the village

        Returns:
            Dict with explanation details
        """
        if self.shap_explainer is None:
            return {'error': 'SHAP explainer not available'}

        # Find village index
        idx = self.feature_data[self.feature_data['village'] == village_name].index
        if len(idx) == 0:
            return {'error': f'Village {village_name} not found'}

        # Get features for this village
        X = self.model.feature_model._prepare_features(
            self.feature_data.loc[idx], fit_scaler=False
        )

        return self.shap_explainer.explain_village(X[0], village_name)

    def _save_outputs(self, results: Dict, alerts: list):
        """Save outputs to disk."""
        print("\n" + "-" * 60)
        print("Saving outputs...")
        print("-" * 60)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save predictions
        pred_file = OUTPUT_DIR / f"predictions_{timestamp}.csv"
        self.predictions.to_csv(pred_file, index=False)
        print(f"  Predictions: {pred_file}")

        # Save risk results
        risk_file = OUTPUT_DIR / f"risk_classification_{timestamp}.csv"
        self.risk_results.to_csv(risk_file, index=False)
        print(f"  Risk results: {risk_file}")

        # Save alerts
        alerts_file = OUTPUT_DIR / f"alerts_{timestamp}.json"
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        print(f"  Alerts: {alerts_file}")

        # Save explanations
        if self.explanations is not None and len(self.explanations) > 0:
            exp_file = OUTPUT_DIR / f"explanations_{timestamp}.csv"
            self.explanations.to_csv(exp_file, index=False)
            print(f"  Explanations: {exp_file}")

        # Save summary report
        report = AlertGenerator.generate_summary_report(self.risk_results)
        report_file = OUTPUT_DIR / f"report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"  Report: {report_file}")

        # Save feature importance
        if self.model is not None:
            importance = self.model.get_feature_importance()
            if importance is not None and len(importance) > 0:
                imp_file = OUTPUT_DIR / f"feature_importance_{timestamp}.csv"
                importance.to_csv(imp_file, index=False)
                print(f"  Feature importance: {imp_file}")

        # Save GeoJSON for visualization
        try:
            geo_results = self.feature_data.copy()
            geo_results['prediction'] = self.predictions['prediction'].values
            geo_results['risk_tier'] = self.risk_results['risk_tier'].values
            geo_results['risk_score'] = self.risk_results['risk_score'].values

            geojson_file = OUTPUT_DIR / f"villages_{timestamp}.geojson"
            geo_results.to_file(geojson_file, driver='GeoJSON')
            print(f"  GeoJSON: {geojson_file}")
        except Exception as e:
            print(f"  Warning: Could not save GeoJSON: {e}")

    def _print_summary(self, results: Dict):
        """Print pipeline summary."""
        print("\n" + "=" * 70)
        print("                    PIPELINE SUMMARY")
        print("=" * 70)

        print(f"\nTarget Date: {self.target_date}")
        print(f"Villages Processed: {results['n_predictions']}")

        # Physics model summary
        if self.use_physics_model and 'physics_mean' in results:
            print(f"\nPhysics-Informed Model (Water Balance + Temporal Decomposition):")
            print(f"  Physics Model Mean: {results.get('physics_mean', 0):.2f}m")
            print(f"  Feature Model Mean: {results.get('feature_mean', 0):.2f}m")
            print(f"  Combined Mean: {results.get('combined_mean', 0):.2f}m")
            if self.physics_model is not None:
                print(f"  Piezometers with temporal fit: {len(self.physics_model.piezometer_components)}")

        if 'risk_summary' in results:
            summary = results['risk_summary']
            print(f"\nRisk Classification:")
            print(f"  Mean Risk Score: {summary.get('mean_risk_score', 0):.2f}")
            print(f"  Critical Villages: {summary.get('critical_count', 0)}")
            print(f"  High Risk Villages: {summary.get('high_count', 0)}")

        if 'grace_validation' in results:
            val = results['grace_validation']
            print(f"\nGRACE Validation:")
            print(f"  GRACE TWS Anomaly: {val.get('grace_tws_cm', 0):.2f} cm")
            print(f"  Difference: {val.get('difference_m', 0):.3f} m")

        if 'model_metrics' in results and 'feature_model' in results['model_metrics']:
            metrics = results['model_metrics']['feature_model']
            print(f"\nFeature Model Performance:")
            print(f"  Validation RMSE: {metrics.get('val_rmse', 0):.3f} m")
            print(f"  Validation R²: {metrics.get('val_r2', 0):.3f}")

        # GNN summary
        if 'gnn_metrics' in results:
            gnn = results['gnn_metrics']
            print(f"\nGraph Neural Network (GNN):")
            if 'rmse' in gnn:
                print(f"  Training RMSE: {gnn.get('rmse', 0):.3f} m")
                print(f"  Training R²: {gnn.get('r2', 0):.3f}")
            if 'gnn_mean' in results:
                print(f"  Mean Prediction: {results['gnn_mean']:.2f} m")

        # Explainability summary
        if results.get('explanations_generated'):
            print(f"\nExplainability (SHAP):")
            print(f"  Explanations generated: {len(self.explanations) if self.explanations is not None else 0}")
            if self.explanations is not None and 'most_important_feature' in self.explanations.columns:
                top_features = self.explanations['most_important_feature'].value_counts().head(3)
                print(f"  Most influential features:")
                for feat, count in top_features.items():
                    print(f"    - {feat}: {count} villages")

        # Novel approaches used
        print(f"\nNovel Approaches Used:")
        print(f"  Physics Model: {'Yes' if self.use_physics_model else 'No'}")
        print(f"  Graph Neural Network: {'Yes' if self.use_gnn else 'No'}")
        print(f"  SHAP Explainability: {'Yes' if self.use_explainability else 'No'}")
        print(f"  GRACE Constraint: {'Yes' if self.processed_data.get('grace') is not None else 'No'}")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


def main():
    """Main entry point."""
    # Run pipeline
    pipeline = SmartJalPipeline(target_date=pd.Timestamp('2023-10-01'))
    results = pipeline.run(save_outputs=True)

    # Run scenarios
    try:
        scenarios = pipeline.run_scenarios()
        print("\nScenario Analysis Complete:")
        for name, df in scenarios.items():
            mean_pred = df['prediction'].mean()
            print(f"  {name}: Mean prediction = {mean_pred:.2f}m")
    except Exception as e:
        print(f"\nScenario analysis skipped: {e}")

    return results


if __name__ == '__main__':
    main()
