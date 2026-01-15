"""
Smart Jal - Geospatial Utility Functions

Functions for spatial operations, coordinate transformations, and distance calculations.
"""

import numpy as np
from typing import Tuple, List
import geopandas as gpd
from shapely.geometry import Point


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth (in km).

    Args:
        lat1, lon1: Coordinates of first point (decimal degrees)
        lat2, lon2: Coordinates of second point (decimal degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def find_nearest_point(target_lat: float, target_lon: float,
                       points_df, lat_col: str = 'lat', lon_col: str = 'lon') -> Tuple[int, float]:
    """
    Find the nearest point from a DataFrame of points.

    Args:
        target_lat, target_lon: Target coordinates
        points_df: DataFrame with point coordinates
        lat_col, lon_col: Column names for coordinates

    Returns:
        Tuple of (index of nearest point, distance in km)
    """
    distances = points_df.apply(
        lambda row: haversine_distance(target_lat, target_lon, row[lat_col], row[lon_col]),
        axis=1
    )
    min_idx = distances.idxmin()
    return min_idx, distances[min_idx]


def find_points_within_radius(target_lat: float, target_lon: float,
                              points_df, radius_km: float,
                              lat_col: str = 'lat', lon_col: str = 'lon') -> List[int]:
    """
    Find all points within a given radius.

    Args:
        target_lat, target_lon: Target coordinates
        points_df: DataFrame with point coordinates
        radius_km: Search radius in kilometers
        lat_col, lon_col: Column names for coordinates

    Returns:
        List of indices of points within radius
    """
    distances = points_df.apply(
        lambda row: haversine_distance(target_lat, target_lon, row[lat_col], row[lon_col]),
        axis=1
    )
    return distances[distances <= radius_km].index.tolist()


def point_in_polygon(lat: float, lon: float, gdf: gpd.GeoDataFrame) -> int:
    """
    Find which polygon a point falls within.

    Args:
        lat, lon: Point coordinates
        gdf: GeoDataFrame with polygon geometries

    Returns:
        Index of the polygon containing the point, or -1 if not found
    """
    point = Point(lon, lat)
    for idx, row in gdf.iterrows():
        if row.geometry.contains(point):
            return idx
    return -1


def get_centroid(geometry) -> Tuple[float, float]:
    """
    Get the centroid of a geometry.

    Args:
        geometry: Shapely geometry object

    Returns:
        Tuple of (latitude, longitude)
    """
    centroid = geometry.centroid
    return centroid.y, centroid.x


def spatial_join_nearest(points_gdf: gpd.GeoDataFrame,
                         polygons_gdf: gpd.GeoDataFrame,
                         polygon_cols: List[str]) -> gpd.GeoDataFrame:
    """
    Join points to nearest polygon, handling points that don't fall within any polygon.

    Args:
        points_gdf: GeoDataFrame of points
        polygons_gdf: GeoDataFrame of polygons
        polygon_cols: Columns from polygons to join

    Returns:
        points_gdf with joined polygon columns
    """
    # First try regular spatial join
    joined = gpd.sjoin(points_gdf, polygons_gdf[polygon_cols + ['geometry']],
                       how='left', predicate='within')

    # For points that didn't match, find nearest polygon
    unmatched = joined[joined[polygon_cols[0]].isna()]

    for idx in unmatched.index:
        point = points_gdf.loc[idx, 'geometry']
        distances = polygons_gdf.geometry.distance(point)
        nearest_idx = distances.idxmin()
        for col in polygon_cols:
            joined.loc[idx, col] = polygons_gdf.loc[nearest_idx, col]

    return joined.drop(columns=['index_right'], errors='ignore')


def calculate_slope_from_dem(dem_array: np.ndarray, resolution: float = 30) -> np.ndarray:
    """
    Calculate slope from DEM array.

    Args:
        dem_array: 2D numpy array of elevation values
        resolution: Cell size in meters

    Returns:
        2D numpy array of slope in degrees
    """
    # Calculate gradients
    dy, dx = np.gradient(dem_array, resolution)

    # Calculate slope
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi

    return slope


def extract_value_at_point(raster, lat: float, lon: float):
    """
    Extract raster value at a specific point.

    Args:
        raster: Open rasterio dataset
        lat, lon: Point coordinates

    Returns:
        Raster value at point, or None if out of bounds
    """
    try:
        py, px = raster.index(lon, lat)
        if 0 <= py < raster.height and 0 <= px < raster.width:
            return float(raster.read(1)[py, px])
    except:
        pass
    return None
