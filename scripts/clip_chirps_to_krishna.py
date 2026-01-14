#!/usr/bin/env python3
"""
Clip CHIRPS global rainfall data to Krishna District bounds.
This reduces file sizes and speeds up processing.
"""

import gzip
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path
import numpy as np

# Krishna District bounds
KRISHNA_BOUNDS = {
    'west': 80.0,
    'south': 15.5,
    'east': 81.5,
    'north': 17.0
}

def clip_chirps_to_krishna():
    chirps_dir = Path('/Users/jeevan/RealTimeGovernance/prototypes/SmartJal/downloaded_data/rainfall/chirps')
    output_dir = chirps_dir / 'krishna_clipped'
    output_dir.mkdir(exist_ok=True)

    gz_files = sorted(chirps_dir.glob('chirps-v2.0.*.tif.gz'))
    print(f"Found {len(gz_files)} CHIRPS files to process")

    for i, gz_file in enumerate(gz_files):
        # Extract year and month from filename (chirps-v2.0.YYYY.MM.tif.gz)
        parts = gz_file.stem.replace('.tif', '').split('.')
        year = parts[2]  # YYYY
        month = parts[3]  # MM
        year_month = f"{year}.{month}"
        output_file = output_dir / f"chirps_krishna_{year_month}.tif"

        if output_file.exists():
            print(f"  [{i+1}/{len(gz_files)}] {year_month} - already clipped, skipping")
            continue

        print(f"  [{i+1}/{len(gz_files)}] Processing {year_month}...", end=' ')

        try:
            # Decompress and open
            with gzip.open(gz_file, 'rb') as f_in:
                with rasterio.open(f_in) as src:
                    # Get window for Krishna bounds
                    window = from_bounds(
                        KRISHNA_BOUNDS['west'],
                        KRISHNA_BOUNDS['south'],
                        KRISHNA_BOUNDS['east'],
                        KRISHNA_BOUNDS['north'],
                        src.transform
                    )

                    # Read clipped data
                    data = src.read(1, window=window)

                    # Calculate new transform
                    transform = src.window_transform(window)

                    # Update profile
                    profile = src.profile.copy()
                    profile.update({
                        'height': data.shape[0],
                        'width': data.shape[1],
                        'transform': transform,
                        'compress': 'deflate'
                    })

                    # Write clipped data
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(data, 1)

            size_kb = output_file.stat().st_size / 1024
            print(f"✓ ({size_kb:.0f} KB)")

        except Exception as e:
            print(f"✗ Error: {e}")

    # Summary
    clipped_files = list(output_dir.glob('*.tif'))
    total_size = sum(f.stat().st_size for f in clipped_files) / 1024 / 1024
    print(f"\nDone! {len(clipped_files)} files clipped to Krishna district")
    print(f"Total size: {total_size:.1f} MB (vs ~786 MB global files)")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    clip_chirps_to_krishna()
