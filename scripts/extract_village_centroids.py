#!/usr/bin/env python3
"""
Extract village centroids from GTWells bore well data.
Since each village has multiple bore wells with coordinates,
we calculate the centroid (mean lat/long) for each village.
"""

import pandas as pd
from pathlib import Path
import json

def extract_village_centroids():
    # Read GTWells data
    data_file = Path('/Users/jeevan/RealTimeGovernance/prototypes/SmartJal/GTWells_Krishna/GTWells/kris.csv')
    df = pd.read_csv(data_file)

    print(f"Total bore well records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Clean up - remove rows without coordinates
    df_valid = df.dropna(subset=['Lat', 'Long'])
    print(f"Records with valid coordinates: {len(df_valid):,}")

    # Group by Village Name and Mandal Name, calculate centroid
    villages = df_valid.groupby(['Village Name', 'Mandal Name']).agg({
        'Lat': 'mean',
        'Long': 'mean',
        'District Name': 'first',
        'Bore Well Working': 'count'  # Count of bore wells
    }).reset_index()

    villages.columns = ['village_name', 'mandal_name', 'lat', 'lon', 'district_name', 'bore_well_count']

    print(f"\nUnique villages: {len(villages)}")
    print(f"Unique mandals: {villages['mandal_name'].nunique()}")

    # Summary stats
    print(f"\nBore wells per village:")
    print(f"  Min: {villages['bore_well_count'].min()}")
    print(f"  Max: {villages['bore_well_count'].max()}")
    print(f"  Mean: {villages['bore_well_count'].mean():.1f}")

    # Check bounds
    print(f"\nCoordinate bounds:")
    print(f"  Lat: {villages['lat'].min():.4f} to {villages['lat'].max():.4f}")
    print(f"  Lon: {villages['lon'].min():.4f} to {villages['lon'].max():.4f}")

    # Output directory
    output_dir = Path('/Users/jeevan/RealTimeGovernance/prototypes/SmartJal/downloaded_data/villages')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_file = output_dir / 'krishna_village_centroids.csv'
    villages.to_csv(csv_file, index=False)
    print(f"\nSaved CSV: {csv_file}")

    # Save as GeoJSON for mapping
    features = []
    for _, row in villages.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['lon'], row['lat']]
            },
            "properties": {
                "village_name": row['village_name'],
                "mandal_name": row['mandal_name'],
                "district_name": row['district_name'],
                "bore_well_count": int(row['bore_well_count'])
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    geojson_file = output_dir / 'krishna_village_centroids.geojson'
    with open(geojson_file, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved GeoJSON: {geojson_file}")

    # Show sample villages
    print(f"\nSample villages:")
    print(villages[['village_name', 'mandal_name', 'lat', 'lon', 'bore_well_count']].head(10).to_string(index=False))

    return villages

if __name__ == '__main__':
    extract_village_centroids()
