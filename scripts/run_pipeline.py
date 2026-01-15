#!/usr/bin/env python3
"""
Smart Jal - Run Full Pipeline

Execute the complete pipeline:
1. Load all data
2. Preprocess and join
3. Engineer features
4. Train models
5. Generate predictions for all villages
6. Classify risks
7. Save outputs
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from pipeline import SmartJalPipeline
import pandas as pd


def main():
    """Run the full pipeline."""
    print("=" * 70)
    print("SMART JAL - FULL PIPELINE")
    print("=" * 70)

    # Create and run pipeline
    pipeline = SmartJalPipeline(
        target_date=pd.Timestamp('2024-05-01'),  # Pre-monsoon 2024
        use_physics_model=True,
        use_gnn=False,  # Disable GNN for faster runs
        use_explainability=True
    )

    results = pipeline.run(save_outputs=True)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    # Summary
    if pipeline.predictions is not None:
        print(f"\nPredictions generated for {len(pipeline.predictions)} villages")

    if pipeline.risk_results is not None:
        print(f"\nRisk classification:")
        print(pipeline.risk_results['risk_tier_label'].value_counts())

    return results


if __name__ == '__main__':
    main()
