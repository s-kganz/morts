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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# When too many requests hit the CDSE API, we want to immediately stop
# and force a retry through snakemake. Since those warnings are dispatched
# through the logger we cannot use the warnings module.
# See 
# https://stackoverflow.com/questions/76194992/
# capturing-a-warning-sent-using-logging-warning-from-a-library-function-python
lazycogs_logger = logging.getLogger("lazycogs")

def too_many_requests_filter(record):
    print(record.msg)
    print(record.levelname)
    print(module)
    if (
        "Too Many Requests" in record.msg and 
        record.levelname == "WARNING" and 
        record.module == "lazycogs"
    ):
        raise RuntimeError("Hit CDSE request limit")
    return True  # Pass through

lazycogs_logger.addFilter(too_many_requests_filter)

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

# S1 collections
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
DETECTIONS_PATH = "data_working/detections_labeled.parquet"
TILES_PATH = "data_working/sentinel2_tiles_world_with_land.geojson"
STAC_CACHE_DIR = "data_working/stac_cache/"

# Search constants
# SEARCH_TEMPORAL_RANGE="2024"
SEARCH_TEMPORAL_RANGE=None

# Output constants
# These match the grid of LCMS detections but use a coarser
# overview resolution in the S1 COGs
OUTPUT_CRS="EPSG:5071"
OUTPUT_RES=80 # m
OUTPUT_DIRECTORY="data_working/s1_timeseries/"

# CDSE has quota restrictions for the provider-pays bucket. We have to 
# slow lazycogs down to stay under.
MAX_CONCURRENT_READS = 1

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
    
    return store

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
    fname = os.path.join(STAC_CACHE_DIR, f"{tile}.parquet")
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
    
def get_tileset_data_array(full_parquet_path: str, bbox: tuple[float, float, float, float]) -> xr.Dataset | None:
    # Load data lazily
    s3_store = _get_cdse_s3_obstore()
        
    tile_da = lazycogs.open(
        full_parquet_path,
        bands=ASSETS,
        bbox=bbox,
        crs=OUTPUT_CRS,
        resolution=OUTPUT_RES,
        store=s3_store,
        nodata=np.nan,
        max_concurrent_reads=MAX_CONCURRENT_READS,
        mosaic_method=lazycogs.MeanMethod
    )

    plan = tile_da.lazycogs.explain()
    n_reads = plan.total_cog_reads

    if n_reads > CDSE_S3_RATE_LIMIT:
        logger.warn("Loading this array may result in too many requests!")
    
    sleep_time = 60 * n_reads / CDSE_S3_RATE_LIMIT * 0.9
    logger.info(f"Need to make {n_reads} requests. Sleeping for {sleep_time:.2f} seconds to reduce request rate")
    time.sleep(sleep_time)

    tile_da = tile_da.load().to_dataset("band")
    
    return tile_da
    
def get_timeseries_at_objects(objects: gpd.GeoDataFrame, da: xr.Dataset) -> xr.Dataset:
    '''
    Acquire a time series of all bands in `da` at each geometry in `objects`.
    '''
    ts_objects: list[xr.Dataset] = []
    for idx, data in objects.geometry.bounds.iterrows():
        slicer = dict(
            # Slice objects are inclusive so subtract off one cell
            # on both axes.
            x=slice(data["minx"], data["maxx"]-OUTPUT_RES),
            # Maxy goes first because of decreasing y coordinate
            y=slice(data["maxy"], data["miny"]+OUTPUT_RES)
        )
        
        this_ts = da.sel(**slicer).mean(dim=["x", "y"])\
            .assign(tmin=objects["tmin"][idx])\
            .assign(tmax=objects["tmax"][idx])\
            .expand_dims(sample=[idx])

        ts_objects.append(this_ts)
        
    return xr.combine_by_coords(ts_objects)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtree", type=int, required=True)
    args = parser.parse_args()
    logger.info(f"Processing subtree {args.subtree}")

    # Make sure output directory is available
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Load data and subset to the subtree
    objects = gpd.read_parquet(DETECTIONS_PATH)
    objects = objects[objects.subtree == args.subtree]
    
    # Log stats about the number of objects in this subtree
    n_detections = (objects["tmin"] != -1).sum()
    n_nondetections = (objects["tmin"] == -1).sum()
    logger.info(f"Found {n_detections} detections and {n_nondetections} nondetections.")
    if n_detections + n_nondetections == 0:
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
        stac_parquet_objects.append(gpd.read_parquet(os.path.join(STAC_CACHE_DIR, f"{tile}.parquet")))
        
    # Combine all the tiles into one parquet, and save that to a tempfile
    # for loading with lazycogs. All of these are STAC items in 4326 so we can concat
    # without worrying about projection differences.
    tempdir = tempfile.gettempdir()
    full_parquet_path = os.path.join(tempdir, f"subtree-{args.subtree}.parquet")
    pd.concat(stac_parquet_objects).to_parquet(full_parquet_path)
    
    # Load the tileset and remove problematic attrs
    ds = get_tileset_data_array(full_parquet_path, objects.geometry.total_bounds)
    
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
    ts_dataset = get_timeseries_at_objects(objects, ds)
    
    # Print proportion NA pixels for each band
    logger.info("Proportion NA pixels per band across all objects:")
    prop_nan_by_band = ts_dataset[ASSETS].isnull().mean(dim=["sample", "time"])
    for band in ASSETS:
        logger.info(f"\t{band}: {prop_nan_by_band[band].data:.3f}")

    # Check if any objects were fully nan, which would indicate we computed the
    # boundary wrong.
    # prop_nan_by_sample = ts_dataset["].isnull().mean(dim=["time"])
    # if (prop_nan_by_sample == 1).any():
    #    logger.error("Some objects had no valid pixels!")
    
    # Save output
    out_path = os.path.join(OUTPUT_DIRECTORY, f"timeseries-subtree-{args.subtree}.nc")
    logger.info(f"Saving output to {out_path}")
    ts_dataset.to_netcdf(out_path)
    