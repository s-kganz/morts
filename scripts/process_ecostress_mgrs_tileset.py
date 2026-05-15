import requests
from obstore.store import S3Store
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Check if we are on JupyterHub
import os
#is_jupyter_hub = "jupyter" in os.environ.get("HOSTNAME", default="")
is_jupyter_hub = True

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
SEARCH_TEMPORAL_RANGE="2024-06/2024-07"

# Filter constants
AFTERNOON_HOURS=[12, 13, 14, 15, 16]

# Output constants
OUTPUT_CRS="EPSG:5071"
OUTPUT_RES=70 # m

def _get_lpcloud_s3_obstore() -> S3Store:
    creds = requests.get(LPCLOUD_AWS_ENDPOINT).json()
    s3_config = dict(
        aws_access_key_id=creds["accessKeyId"],
        aws_secret_access_key=creds["secretAccessKey"],
        aws_session_token=creds["sessionToken"]
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
    
def sentinel_tile_to_shapely_geom(tile: str, buffer: float=-10_000) -> shapely.Geometry:
    return shapely.geometry.box(
        *sentinel_tiles.bbox(tile).buffer(buffer).transform("EPSG:4326")
    )

def get_tileset_data_array(tileset: str) -> xr.DataArray:
    # Parse tiles
    tiles = tileset.split("-")
    logger.info(f"Received tileset: {tiles}")
    
    # Get detections/nondetections in this tileset
    
    # Convert to geometries
    tile_bounds = list(map(sentinel_tile_to_shapely_geom, tiles))
    
    # Search for granules
    logger.info("Searching for granules...")
    search_tasks = [
        search_lpcloud_stac(
            ECO_COLLECTIONS, 
            bbox=tuple(shapely.bounds(tile)),
            datetime=SEARCH_TEMPORAL_RANGE
        )
        for tile in tile_bounds
    ]
    search_results = reduce(list.__add__, search_tasks)
    logger.info(f"Found {len(search_results)} granules in search")
    if len(search_results) == 0:
        logger.error("Found no granules in search, exiting.")
        sys.exit(1)

    # Filter air temperature granules for afternoon observations
    search_results_filter = list(filter(
        lambda x: _get_granule_local_hour(x) in AFTERNOON_HOURS,
        search_results
    ))
    logger.info(f"After afternoon filter, {len(search_results_filter)} granules left")
    
    # Sanitize assets
    search_results_sanitized = list(map(
        _sanitize_item_asset_keys,
        search_results_filter
    ))
    
    # Save to geoparquet
    tempdir = tempfile.gettempdir()
    parquet_path = os.path.join(tempdir, f"items-{tileset}.parquet")
    stac_geoparquet.to_geodataframe(search_results_sanitized).to_parquet(parquet_path)
    
    # Load data lazily
    if is_jupyter_hub:
        s3_store = _get_lpcloud_s3_obstore()
    else:
        s3_store = None
    transformer = Transformer.from_crs("EPSG:4326", OUTPUT_CRS, always_xy=True)
    
    total_bbox = shapely.transform(
        shapely.union_all(tile_bounds),
        transformation=transformer.transform,
        interleaved=False
    )
    total_bbox = shapely.buffer(total_bbox, 10_000)
    
    tile_da = lazycogs.open(
        parquet_path,
        bands=ECO_ASSETS,
        bbox=shapely.bounds(total_bbox),
        crs=OUTPUT_CRS,
        resolution=OUTPUT_RES,
        store=s3_store,
        nodata=np.nan
    )
    
    return tile_da
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileset", type=str, required=True)
    args = parser.parse_args()
    da = get_tileset_data_array(args.tileset)