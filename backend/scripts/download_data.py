"""
SmartJal Data Downloader

Downloads all required datasets for Krishna District, Andhra Pradesh.
Krishna District Bounds: 80.0°E - 81.5°E, 15.5°N - 17.0°N

Usage:
    python download_data.py --all
    python download_data.py --dem
    python download_data.py --rainfall
    python download_data.py --soil
    python download_data.py --landcover
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io

# Krishna District bounds
KRISHNA_BOUNDS = {
    "min_lon": 80.0,
    "max_lon": 81.5,
    "min_lat": 15.5,
    "max_lat": 17.0
}

# Output directory
DATA_DIR = Path(__file__).parent.parent.parent / "downloaded_data"


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, output_path, desc="Downloading"):
    """Download a file with progress."""
    print(f"{desc}: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    print(f"\n  Saved to: {output_path}")
    return output_path


# =============================================================================
# 1. DEM (SRTM 30m) - via OpenTopography
# =============================================================================
def download_dem():
    """
    Download SRTM DEM from OpenTopography.
    Note: Requires free API key from https://opentopography.org/
    """
    print("\n" + "="*60)
    print("DOWNLOADING DEM (SRTM 30m)")
    print("="*60)

    output_dir = ensure_dir(DATA_DIR / "dem")

    # OpenTopography requires API key
    api_key = os.environ.get("OPENTOPO_API_KEY", "")

    if not api_key:
        print("""
ERROR: OpenTopography API key not found.

To get a free API key:
1. Go to https://opentopography.org/
2. Create a free account
3. Get your API key from your profile
4. Set it as environment variable:
   export OPENTOPO_API_KEY=your_key_here

Alternative: Download manually from:
- https://earthexplorer.usgs.gov (USGS EarthExplorer)
- Search for SRTM 1 Arc-Second Global
- Select tiles covering 80-82E, 15-17N
""")
        return None

    # OpenTopography Global DEM API
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",
        "south": KRISHNA_BOUNDS["min_lat"],
        "north": KRISHNA_BOUNDS["max_lat"],
        "west": KRISHNA_BOUNDS["min_lon"],
        "east": KRISHNA_BOUNDS["max_lon"],
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    output_path = output_dir / "krishna_srtm_30m.tif"

    response = requests.get(url, params=params, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"DEM saved to: {output_path}")
        return output_path
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# =============================================================================
# 2. RAINFALL (CHIRPS) - via IRI Data Library
# =============================================================================
def download_rainfall():
    """
    Download CHIRPS rainfall data for recent period.
    CHIRPS provides daily rainfall at ~5km resolution.
    """
    print("\n" + "="*60)
    print("DOWNLOADING RAINFALL (CHIRPS)")
    print("="*60)

    output_dir = ensure_dir(DATA_DIR / "rainfall")

    # Calculate date range (last 2 years for training, recent months for prediction)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years

    # CHIRPS via IRI Data Library (Columbia University)
    # This provides easy subsetting
    base_url = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp"

    # Build the data URL with subsetting
    url = f"{base_url}/X/({KRISHNA_BOUNDS['min_lon']})/({KRISHNA_BOUNDS['max_lon']})/RANGE"
    url += f"/Y/({KRISHNA_BOUNDS['min_lat']})/({KRISHNA_BOUNDS['max_lat']})/RANGE"
    url += f"/T/({start_date.strftime('%d %b %Y')})/({end_date.strftime('%d %b %Y')})/RANGE"
    url += "/data.nc"

    output_path = output_dir / "krishna_chirps_rainfall.nc"

    print(f"Downloading CHIRPS data from {start_date.date()} to {end_date.date()}")
    print(f"URL: {url[:100]}...")

    try:
        response = requests.get(url, stream=True, timeout=300)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Rainfall data saved to: {output_path}")
            return output_path
        else:
            print(f"Error: {response.status_code}")
            print("\nAlternative: Download from https://data.chc.ucsb.edu/products/CHIRPS-2.0/")
            return None
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/")
        print("2. Download files for recent years")
        return None


# =============================================================================
# 3. SOIL (ISRIC SoilGrids) - via REST API
# =============================================================================
def download_soil():
    """
    Download soil properties from ISRIC SoilGrids.
    250m resolution, various soil properties.
    """
    print("\n" + "="*60)
    print("DOWNLOADING SOIL DATA (ISRIC SoilGrids)")
    print("="*60)

    output_dir = ensure_dir(DATA_DIR / "soil")

    # SoilGrids WCS (Web Coverage Service)
    # Properties we need: sand, clay, silt, bulk density, hydraulic conductivity
    properties = {
        "sand": "sand",      # Sand content (%)
        "clay": "clay",      # Clay content (%)
        "soc": "soc",        # Soil organic carbon
        "bdod": "bdod",      # Bulk density
    }

    depths = ["0-5cm", "5-15cm", "15-30cm"]

    # SoilGrids REST API
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    print("Downloading soil properties for Krishna District...")
    print("Note: This may take a few minutes per property")

    # Create sample points grid
    import numpy as np
    lons = np.arange(KRISHNA_BOUNDS["min_lon"], KRISHNA_BOUNDS["max_lon"], 0.1)
    lats = np.arange(KRISHNA_BOUNDS["min_lat"], KRISHNA_BOUNDS["max_lat"], 0.1)

    results = []
    total_points = len(lons) * len(lats)

    print(f"Querying {total_points} points...")

    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            params = {
                "lon": lon,
                "lat": lat,
                "property": list(properties.keys()),
                "depth": depths,
                "value": "mean"
            }

            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "lon": lon,
                        "lat": lat,
                        "data": data
                    })
            except:
                pass

            # Progress
            done = i * len(lats) + j + 1
            if done % 10 == 0:
                print(f"\r  Progress: {done}/{total_points} ({100*done/total_points:.1f}%)", end="", flush=True)

    print()

    # Save results
    import json
    output_path = output_dir / "krishna_soil_properties.json"
    with open(output_path, 'w') as f:
        json.dump(results, f)

    print(f"Soil data saved to: {output_path}")
    print(f"Downloaded {len(results)} points")

    return output_path


# =============================================================================
# 4. LAND COVER (ESA WorldCover) - Direct tile download
# =============================================================================
def download_landcover():
    """
    Download ESA WorldCover 2021 tiles for Krishna District.
    10m resolution, 11 land cover classes.
    """
    print("\n" + "="*60)
    print("DOWNLOADING LAND COVER (ESA WorldCover 2021)")
    print("="*60)

    output_dir = ensure_dir(DATA_DIR / "landcover")

    # ESA WorldCover tiles are 3x3 degrees
    # Krishna District (80-81.5E, 15.5-17N) needs tiles:
    # - N15E078 (covers 78-81E, 15-18N)
    # - N15E081 (covers 81-84E, 15-18N)

    tiles = ["N15E078", "N15E081"]

    # ESA WorldCover AWS bucket (direct download)
    base_url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

    downloaded = []
    for tile in tiles:
        # Full tile path
        url = f"{base_url}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        output_path = output_dir / f"worldcover_{tile}.tif"

        print(f"\nDownloading tile {tile}...")
        print(f"URL: {url}")

        try:
            response = requests.get(url, stream=True, timeout=600)
            if response.status_code == 200:
                total = int(response.headers.get('content-length', 0))
                downloaded_bytes = 0

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total:
                            pct = (downloaded_bytes / total) * 100
                            print(f"\r  Progress: {pct:.1f}% ({downloaded_bytes//1024//1024}MB)", end="", flush=True)

                print(f"\n  Saved: {output_path}")
                downloaded.append(output_path)
            else:
                print(f"  Error: {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")

    if downloaded:
        print(f"\nDownloaded {len(downloaded)} tiles")
    else:
        print("\nManual download instructions:")
        print("1. Go to https://worldcover2021.esa.int/download")
        print("2. Select tiles covering India (78-84E, 15-18N)")
        print("3. Download and place in downloaded_data/landcover/")

    return downloaded


# =============================================================================
# 5. Village Boundaries (DataMeet)
# =============================================================================
def download_villages():
    """
    Download village boundaries from DataMeet GitHub.
    """
    print("\n" + "="*60)
    print("DOWNLOADING VILLAGE BOUNDARIES")
    print("="*60)

    output_dir = ensure_dir(DATA_DIR / "villages")

    # DataMeet provides Indian administrative boundaries
    # Andhra Pradesh villages
    url = "https://github.com/datameet/maps/raw/master/Villages/andhra-pradesh.geojson"

    output_path = output_dir / "andhra_pradesh_villages.geojson"

    print("Downloading AP village boundaries from DataMeet...")

    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved to: {output_path}")
            return output_path
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nAlternative sources:")
    print("1. https://github.com/datameet/maps")
    print("2. https://bhuvan.nrsc.gov.in (requires login)")
    print("3. Census of India maps")

    return None


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Download data for SmartJal")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dem", action="store_true", help="Download DEM (SRTM)")
    parser.add_argument("--rainfall", action="store_true", help="Download rainfall (CHIRPS)")
    parser.add_argument("--soil", action="store_true", help="Download soil (ISRIC)")
    parser.add_argument("--landcover", action="store_true", help="Download land cover (ESA)")
    parser.add_argument("--villages", action="store_true", help="Download village boundaries")

    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.dem, args.rainfall, args.soil, args.landcover, args.villages]):
        args.all = True

    print("="*60)
    print("SmartJal Data Downloader")
    print("="*60)
    print(f"Output directory: {DATA_DIR}")
    print(f"Krishna District bounds: {KRISHNA_BOUNDS}")

    ensure_dir(DATA_DIR)

    results = {}

    if args.all or args.dem:
        results["dem"] = download_dem()

    if args.all or args.rainfall:
        results["rainfall"] = download_rainfall()

    if args.all or args.soil:
        results["soil"] = download_soil()

    if args.all or args.landcover:
        results["landcover"] = download_landcover()

    if args.all or args.villages:
        results["villages"] = download_villages()

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    for name, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nData saved to: {DATA_DIR}")
    print("\nNext steps:")
    print("1. Check downloaded files in the downloaded_data/ folder")
    print("2. For failed downloads, follow manual instructions above")
    print("3. Run: python -m backend.scripts.train_models")


if __name__ == "__main__":
    main()
