from obstore.store import S3Store
import argparse
import logging
from pystac_client import Client
import shapely
import sys
import numpy as np
import pandas as pd
import tempfile
import stac_geoparquet
import os
from typing import Any
import lazycogs
import xarray as xr
import geopandas as gpd
import warnings
import configparser
import time
import xvec
import rioxarray

from store_rate_limit import ObjectStoreRateLimiter, RateLimitedStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

lazycogs_logger = logging.getLogger("lazycogs")

# Ignore warnings
# Centroid from square tiles in a geographic CRS
warnings.filterwarnings("ignore", message=".*Geographic CRS.*")
# Casting warnings when ingesting data
warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")

# Check if we are on JupyterHub
import os
is_jupyter_hub = "jupyter" in os.environ.get("HOSTNAME", default="")

# CDSE constants
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1/"
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
CDSE_BUCKET = "eodata"
# Max. number of requests per minute. Use this to strategically
# call time.sleep so that we do not get a bunch of nan data back.
# See https://github.com/developmentseed/lazycogs/issues/80.
CDSE_S3_RATE_LIMIT = 2000

# Search constants
# SEARCH_TEMPORAL_RANGE="2024"
SEARCH_TEMPORAL_RANGE=None

COLLECTIONS=[
    "sentinel-1-global-mosaics"
]
ASSETS = [
    "VV", "VH"
]
SAR_QUERY = {
    "sar:instrument_mode": {"eq": "IW"},
    "sar:polarizations": {"eq": ["VH", "VV"]}
}

# Data files
TILES_PATH = "data_working/sentinel2_tiles_world_with_land.geojson"
STAC_CACHE_DIR = "data_working/stac_cache/"

def _get_cdse_s3_obstore() -> S3Store:
    config = configparser.ConfigParser()
    if is_jupyter_hub:
        config.read("/home/jovyan/.s3cfg")
    else:
        config.read(".s3cfg")

    store = S3Store(
        aws_access_key_id=config.get("cdse", "access_key"),
        aws_secret_access_key=config.get("cdse", "secret_key"),
        bucket=CDSE_BUCKET,
        endpoint=CDSE_S3_ENDPOINT
    )

    limiter = ObjectStoreRateLimiter(rate=CDSE_S3_RATE_LIMIT*0.9, per=60, burst=1)
    limited_store = RateLimitedStore(store, limiter)
    
    return limited_store

def _sanitize_item_datetime(features: list[dict[str, Any]]) -> None:
    '''
    Pop start_datetime and end_datetime from each item. This is due to a bug in range searches
    in rustac.
    https://github.com/stac-utils/rustac/issues/1070
    
    :param feature: STAC item serialized as a dictionary
    :type feature: dict[str, Any]
    :return: The same STAC item with asset keys sanitized
    :rtype: dict[str, Any]
    '''
    for item in features:
        item["properties"].pop("end_datetime", None)
        item["properties"].pop("start_datetime", None)


def search_cdse_stac(collections: list[str], **kwargs) -> list[dict[str, Any]]:
    client = Client.open(CDSE_STAC)
    items = client.search(
        collections=collections,
        **kwargs
    ).item_collection_as_dict()
    return items["features"]
    

def get_tile_parquet(tile: str, tile_geoms: gpd.GeoDataFrame) -> None:
    fname = os.path.join(STAC_CACHE_DIR, f"{tile}-s1-mosaic.parquet")
    if os.path.exists(fname):
        # Cache hit! No need to search
        logger.info(f"{fname} found in cache directory")
        return
    logger.info(f"{fname} not found in cache. Searching...")
    
    # Convert to geometries
    tile_center = tile_geoms[tile_geoms["Name"].isin([tile])].geometry.centroid.iloc[0]
    
    # Search for granules
    search_results = search_cdse_stac(
        COLLECTIONS, 
        intersects=shapely.to_geojson(tile_center),
        datetime=SEARCH_TEMPORAL_RANGE,
        query=SAR_QUERY
    )
    
    logger.info(f"Found {len(search_results)} granules in search")
    if len(search_results) == 0:
        logger.error("Found no granules in search, exiting.")
        sys.exit(1)
    
    # Sanitize asset keys for loading
    # _sanitize_item_datetime(search_results)
    
    # Save to geoparquet
    stac_geoparquet.to_geodataframe(search_results, dtype_backend="numpy_nullable").to_parquet(fname)
    
def get_tileset_data_array(full_parquet_path: str, **lazycogs_args) -> xr.Dataset | None:
    # Load data lazily
    s3_store = _get_cdse_s3_obstore()
        
    tile_da = lazycogs.open(
        full_parquet_path,
        bands=ASSETS,
        store=s3_store,
        mosaic_method=lazycogs.MeanMethod,
        **lazycogs_args
    )

    plan = tile_da.lazycogs.explain()
    n_reads = plan.total_cog_reads

    if n_reads > CDSE_S3_RATE_LIMIT:
        logger.warn("Loading this array may result in too many requests!")
    
    #sleep_time = 60 * n_reads / CDSE_S3_RATE_LIMIT * 0.9
    #logger.info(f"Need to make {n_reads} requests. Sleeping for {sleep_time:.2f} seconds to reduce request rate")
    #time.sleep(sleep_time)

    tile_da = tile_da.load().to_dataset("band")
    
    return tile_da
    
def get_timeseries_at_objects(objects: gpd.GeoDataFrame, da: xr.Dataset) -> xr.Dataset:
    '''
    Acquire a time series of all bands in `da` at each geometry in `objects`.
    '''
    return da.xvec.zonal_stats(objects.geometry, x_coords="x", y_coords="y")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--where", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--raster_crs", type=str, default="EPSG:5071")
    parser.add_argument("--raster_res", type=int, default=70)
    args = parser.parse_args()
    logger.info(f"Processing input {args.input} with subset argument {args.where}")

    # Make sure output directory is available
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data and subset to the subtree
    objects = gpd.read_file(args.input).query(args.where).set_index(args.index)
    
    # Log stats about the number of objects in this subtree
    logger.info(f"Found {objects.shape[0]} objects.")
    if objects.shape[0] == 0:
        logger.error("No objects found, exiting.")
        sys.exit(1)
        
    # Figure out what tiles are in this subtree
    tiles = gpd.read_file(TILES_PATH)
    tiles = tiles[tiles.geometry.intersects(objects.geometry.to_crs(tiles.crs).union_all())]
    tileset : list[str] = list(tiles["Name"])
    if len(tileset) == 0:
        logger.error("Objects do not intersect any MGRS tiles, exiting.")
        sys.exit(1)
    logger.info(f"Intersecting MGRS tiles: {tileset}")
        
    # Make sure the search results for each tile are available
    stac_parquet_objects: list[gpd.GeoDataFrame] = []
    for tile in tileset:
        get_tile_parquet(tile, tiles)
        stac_parquet_objects.append(gpd.read_parquet(os.path.join(STAC_CACHE_DIR, f"{tile}-s1-mosaic.parquet")))
        
    # Combine all the tiles into one parquet, and save that to a tempfile
    # for loading with lazycogs. All of these are STAC items in 4326 so we can concat
    # without worrying about projection differences.
    tempdir = tempfile.gettempdir()
    full_parquet_path = os.path.join(tempdir, f"stac-subset-{os.getpid()}.parquet")
    pd.concat(stac_parquet_objects).to_parquet(full_parquet_path)
    
    # Load the tileset. Small geometries can be problematic for xvec so buffer a little bit
    buffered_footprint = shapely.bounds(objects.geometry.union_all().buffer(args.raster_res))
    ds = get_tileset_data_array(
        full_parquet_path, 
        bbox=buffered_footprint,
        crs=args.raster_crs,
        resolution=args.raster_res
    )
    
    if ds is None:
        sys.exit(0)
    else:
        try:
            ds = ds.load()
        except TypeError as e:
            if "429 Too Many Requests" in str(e):
                print("Failed to read from too many requests")
                sys.exit(1)
            else:
                raise e

    # Has illegal data type for saving
    del ds.attrs["zarr_conventions"]
        
    logger.info("Computing time series for each object")
    ts_dataset = get_timeseries_at_objects(objects, ds).set_index(geometry=args.index)
    
    # Print proportion NA pixels for each band
    logger.info("Proportion NA pixels per band across all objects:")
    prop_nan_by_band = ts_dataset[ASSETS].isnull().mean(dim=["geometry", "time"])
    for band in ASSETS:
        logger.info(f"\t{band}: {prop_nan_by_band[band].data:.3f}")
    
    # Save output
    out_path = os.path.join(args.out_dir, f"{args.out_name}.nc")
    logger.info(f"Saving output to {out_path}")
    ts_dataset.to_netcdf(out_path)
    