import requests
from obstore.store import S3Store, S3Config
import argparse
import logging
from pystac_client import Client
import shapely
from sentinel_tiles import sentinel_tiles, UTC_to_solar
from functools import reduce
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import tempfile
import stac_geoparquet
import os
from typing import Any
from pyproj import Transformer
import lazycogs
import xarray as xr
import geopandas as gpd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Check if we are on JupyterHub
import os
is_jupyter_hub = "jupyter" in os.environ.get("HOSTNAME", default="")

# LPCLOUD constants
LPCLOUD_AWS_ENDPOINT="https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
LPCLOUD_BUCKET="lp-prod-protected"
LPCLOUD_STAC_URL="https://cmr.earthdata.nasa.gov/stac/LPCLOUD"

# ECOSTRESS collections to search
ECO_COLLECTIONS=[
    "ECO_L3T_MET_002",
    "ECO_L2T_LSTE_002",
    "ECO_L2T_STARS_002"
]
ECO_ASSETS = [
    "Ta", "RH", "NDVI", "LST", "cloud", "QC"
]

# Search constants
SEARCH_TEMPORAL_RANGE="2024-01/2025-01"
# SEARCH_TEMPORAL_RANGE=None

# Filter constants
AFTERNOON_HOURS=list(range(12, 19))

# Output constants
# These match the grid of LCMS detections
OUTPUT_CRS="EPSG:5071"
OUTPUT_RES=300 # m

# Datasets with objects
DETECTIONS = gpd.read_parquet("data_working/detections.parquet")
TILES = gpd.read_file("data_working/sentinel2_tiles_world_with_land.geojson")

def _get_lpcloud_s3_obstore() -> S3Store:
    creds = requests.get(LPCLOUD_AWS_ENDPOINT).json()
    s3_config = S3Config(
        access_key_id=creds["accessKeyId"],
        secret_access_key=creds["secretAccessKey"],
        session_token=creds["sessionToken"]
    )
    store = S3Store(
        config=s3_config,
        bucket=LPCLOUD_BUCKET,
        region="us-west-2",
        request_payer=True
    )
    return store

def _get_granule_local_hour(feature: dict[str, Any]) -> int:
    '''
    Calculate the hour a granule was observed in local
    solar time, based on the center longitude.
    
    :param feature: STAC item serialized as a dictionary
    :type feature: dict[str, Any]
    :return: Hour of the starting datetime in local solar time.
    :rtype: int
    '''
    middle_lon = (feature["bbox"][2] + feature["bbox"][0])/2
    datetime_utc = pd.to_datetime(feature["properties"]["datetime"])
    datetime_solar = UTC_to_solar(datetime_utc, middle_lon)
    return datetime_solar.hour

def _sanitize_item_asset_keys(feature: dict[str, Any], use_s3=is_jupyter_hub) -> dict[str, Any]:
    '''
    The CMR STAC uses filename-like asset keys, which is problematic for
    lazycogs to understand. This function parses a meaningful variable name
    for relevant assets.
    
    :param feature: STAC item serialized as a dictionary
    :type feature: dict[str, Any]
    :return: The same STAC item with asset keys sanitized
    :rtype: dict[str, Any]
    '''
    new_asset_dict = {}
    for key in feature["assets"]:
        # Keep s3 keys if we are using s3, otherwise keep the keys
        # for http assets.
        if use_s3 == key.startswith("s3"):
            new_key = key.split("_")[-1]
            new_asset_dict[new_key] = deepcopy(feature["assets"][key])

    # Copy all other properties over
    new_feature = deepcopy(feature)
    new_feature["assets"] = new_asset_dict

    return new_feature

def search_lpcloud_stac(collections: list[str], **kwargs) -> list[dict[str, Any]]:
    client = Client.open(LPCLOUD_STAC_URL)
    items = client.search(
        collections=collections,
        **kwargs
    ).item_collection_as_dict()
    return items["features"]

def get_tileset_data_array(tileset: str, tile_geoms: gpd.GeoDataFrame) -> xr.DataArray:
    # Parse tiles
    tiles = tileset.split("-")
            
    # Convert to geometries
    tile_bounds = tile_geoms[tile_geoms["Name"].isin(tiles)]
    
    # Search for granules
    logger.info("Searching for granules...")
    search_tasks = [
        search_lpcloud_stac(
            ECO_COLLECTIONS, 
            intersects=shapely.to_geojson(centroid),
            datetime=SEARCH_TEMPORAL_RANGE
        )
        for centroid in tile_bounds.geometry.centroid
    ]
    search_results = reduce(list.__add__, search_tasks)
    logger.info(f"Found {len(search_results)} granules in search")
    if len(search_results) == 0:
        logger.error("Found no granules in search, exiting.")
        sys.exit(1)
    
    granules_by_collection = {
        collection: list(filter(
            lambda x: x["collection"] == collection,
            search_results
        ))
        for collection in ECO_COLLECTIONS
    }
    
    # MET/LST granules need to be in the afternoon and come from
    # the same sensing time.    
    granules_by_collection["ECO_L3T_MET_002"] = list(filter(
        lambda x: _get_granule_local_hour(x) in AFTERNOON_HOURS,
        granules_by_collection["ECO_L3T_MET_002"]
    ))
    
    # Only keep LST granules that share a datetime with an afternoon MET granule
    met_datetimes = {
        f["id"].split("_")[-3]
        for f in granules_by_collection["ECO_L3T_MET_002"]
    }
    granules_by_collection["ECO_L2T_LSTE_002"] = list(filter(
        lambda x: x["id"].split("_")[-3] in met_datetimes,
        granules_by_collection["ECO_L2T_LSTE_002"]
    ))
    
    # Collect all granules back together
    search_results_filter = list(reduce(list.__add__, granules_by_collection.values()))
    logger.info("After filtering, granule distribution is")
    logger.info("; ".join([f"{c}: {len(granules_by_collection[c])}" for c in ECO_COLLECTIONS]))
    
    if len(granules_by_collection["ECO_L2T_LSTE_002"]) != len(granules_by_collection["ECO_L3T_MET_002"]):
        logger.warning("LST and MET have differing numbers of granules!")
    
    # Sanitize asset keys for loading
    search_results_sanitized = list(map(
        _sanitize_item_asset_keys,
        search_results_filter
    ))
    
    # Save to geoparquet
    tempdir = tempfile.gettempdir()
    parquet_path = os.path.join(tempdir, f"items-{tileset}.parquet")
    stac_geoparquet.to_geodataframe(search_results_sanitized).to_parquet(parquet_path)
    
    # Determine total boundary of output
    total_bbox = tile_bounds.geometry.to_crs(OUTPUT_CRS).total_bounds
    
    # Load data lazily
    if is_jupyter_hub:
        s3_store = _get_lpcloud_s3_obstore()
    else:
        logging.error("Cannot access granules over HTTP, exiting.")
        sys.exit(0)
    
    tile_da = lazycogs.open(
        parquet_path,
        bands=ECO_ASSETS,
        bbox=total_bbox,
        crs=OUTPUT_CRS,
        resolution=OUTPUT_RES,
        store=s3_store,
        nodata=np.nan
    )
    
    return tile_da
    
def get_objects_in_tiles(tile_ids: str, tile_geoms: gpd.GeoDataFrame, objects: gpd.GeoDataFrame):
    '''
    Returns a subset of `objects` that are within the MGRS grids in `tileset`.
    '''
    tileset = tile_ids.split("-")
    tile_geom_subset = tile_geoms[tile_geoms["Name"].isin(tileset)]
    object_subset = objects[objects["geometry"].intersects(tile_geom_subset.to_crs(objects.crs).union_all(), align=False)]
    return object_subset
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileset", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Received tileset: {args.tileset}")

    objects = get_objects_in_tiles(
        args.tileset,
        TILES,
        DETECTIONS
    )
    
    n_detections = (objects["tmin"] != -1).sum()
    n_nondetections = (objects["tmin"] == -1).sum()
    logger.info(f"Found {n_detections} detections and {n_nondetections} nondetections.")
    
    da = get_tileset_data_array(args.tileset, TILES)