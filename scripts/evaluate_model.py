#!/usr/bin/env python3
"""
Smart Jal - Model Evaluation

Evaluate model performance using leave-one-out cross-validation on piezometers.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from utils.constants import VALIDATION_TARGETS


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate evaluation metrics.

    Args:
        y_true: Actual water levels
        y_pred: Predicted water levels

    Returns:
        Dictionary with MAE, RMSE, R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def check_targets(metrics: dict) -> dict:
    """
    Check if metrics meet validation targets.

    Args:
        metrics: Dictionary with MAE, RMSE, R2

    Returns:
        Dictionary with pass/fail status for each metric
    """
    results = {}

    results['MAE'] = {
        'value': metrics['MAE'],
        'target': VALIDATION_TARGETS['mae'],
        'passed': metrics['MAE'] <= VALIDATION_TARGETS['mae']
    }

    results['RMSE'] = {
        'value': metrics['RMSE'],
        'target': VALIDATION_TARGETS['rmse'],
        'passed': metrics['RMSE'] <= VALIDATION_TARGETS['rmse']
    }

    results['R2'] = {
        'value': metrics['R2'],
        'target': VALIDATION_TARGETS['r2'],
        'passed': metrics['R2'] >= VALIDATION_TARGETS['r2']
    }

    return results


def print_evaluation_report(results: dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    for metric, data in results.items():
        status = "PASS" if data['passed'] else "FAIL"
        print(f"\n{metric}:")
        print(f"  Value:  {data['value']:.3f}")
        print(f"  Target: {data['target']:.3f}")
        print(f"  Status: {status}")

    print("\n" + "=" * 60)

    all_passed = all(d['passed'] for d in results.values())
    if all_passed:
        print("OVERALL: ALL TARGETS MET")
    else:
        failed = [k for k, v in results.items() if not v['passed']]
        print(f"OVERALL: FAILED - {', '.join(failed)} below target")

    print("=" * 60)


def main():
    """Run model evaluation."""
    print("Smart Jal - Model Evaluation")
    print("-" * 40)

    # Load predictions (from run_pipeline output)
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    predictions_file = output_dir / 'predictions.csv'

    if not predictions_file.exists():
        print(f"ERROR: Predictions file not found at {predictions_file}")
        print("Run 'python scripts/run_pipeline.py' first.")
        return

    df = pd.read_csv(predictions_file)

    # For validation, we need piezometer locations with actual values
    # This assumes predictions.csv has both actual and predicted for validation points
    if 'actual_level' in df.columns and 'predicted_level' in df.columns:
        valid_df = df.dropna(subset=['actual_level', 'predicted_level'])

        metrics = evaluate_predictions(
            valid_df['actual_level'],
            valid_df['predicted_level']
        )

        results = check_targets(metrics)
        print_evaluation_report(results)
    else:
        print("WARNING: No validation data found in predictions.")
        print("Columns available:", df.columns.tolist())


if __name__ == '__main__':
    main()
