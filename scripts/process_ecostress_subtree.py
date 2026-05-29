import requests
from obstore.store import S3Store
import argparse
import logging
from pystac_client import Client
import shapely
from sentinel_tiles import UTC_to_solar
from functools import reduce
from copy import deepcopy
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
print(__name__)

# Ignore warnings
# Centroid from square tiles in a geographic CRS
warnings.filterwarnings("ignore", message=".*Geographic CRS.*")
# Casting warnings when ingesting data
warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")

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

# Data files
DETECTIONS_PATH = "data_working/detections_labeled.parquet"
TILES_PATH = "data_working/sentinel2_tiles_world_with_land.geojson"
STAC_CACHE_DIR = "data_working/stac_cache/"

# Search constants
# SEARCH_TEMPORAL_RANGE="2024-01/2025-01"
SEARCH_TEMPORAL_RANGE=None

# Filter constants
AFTERNOON_HOURS=list(range(12, 19))

# Output constants
# These match the grid of LCMS detections
OUTPUT_CRS="EPSG:5071"
OUTPUT_RES=70 # m, native ecostress resolution
OUTPUT_DIRECTORY="data_working/eco_timeseries/"

# Approximate memory footprint per sq km of ecostress data
# at native resolution
ECO_MB_PER_SQ_KM = 5

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
    

def get_ecostress_tile_parquet(tile: str, tile_geoms: gpd.GeoDataFrame) -> None:
    fname = os.path.join(STAC_CACHE_DIR, f"{tile}.parquet")
    if os.path.exists(fname):
        # Cache hit! No need to search
        logger.info(f"{fname} found in cache directory")
        return
    logger.info(f"{fname} not found in cache. Searching...")
    
    # Convert to geometries
    tile_center = tile_geoms[tile_geoms["Name"].isin([tile])].geometry.centroid.iloc[0]
    
    # Search for granules
    search_results = search_lpcloud_stac(
        ECO_COLLECTIONS, 
        intersects=shapely.to_geojson(tile_center),
        datetime=SEARCH_TEMPORAL_RANGE
    )
    
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
    stac_geoparquet.to_geodataframe(search_results_sanitized, dtype_backend="numpy_nullable").to_parquet(fname)
    
def get_tileset_data_array(full_parquet_path: str, bbox: tuple[float, float, float, float]) -> xr.DataArray | None:
    # Load data lazily
    if is_jupyter_hub:
        s3_store = _get_lpcloud_s3_obstore()
    else:
        logging.error("Cannot access granules over HTTP, exiting.")
        return None
        
    tile_da = lazycogs.open(
        full_parquet_path,
        bands=ECO_ASSETS,
        bbox=bbox,
        crs=OUTPUT_CRS,
        resolution=OUTPUT_RES,
        store=s3_store,
        nodata=np.nan
    )
    
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

def mask_lst(da: xr.DataArray) -> xr.Dataset:
    '''
    Mask LST pixels with degraded quality. Also converts
    remaining bands to variables in a dataset.
    
    :param da: Data array containing a QC and LST band
    :type da: xr.DataArray
    :return: Description
    :rtype: DataArray
    '''
    ds = da.to_dataset(dim="band")
    ds["QC"] = ds["QC"].astype(np.int32)
    ds["LST"] = ds["LST"].where(ds["QC"] & 0b11 == 0)
    return ds

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
        get_ecostress_tile_parquet(tile, tiles)
        stac_parquet_objects.append(gpd.read_parquet(os.path.join(STAC_CACHE_DIR, f"{tile}.parquet")))
        
    # Combine all the tiles into one parquet, and save that to a tempfile
    # for loading with lazycogs. All of these are STAC items in 4326 so we can concat
    # without worrying about projection differences.
    tempdir = tempfile.gettempdir()
    full_parquet_path = os.path.join(tempdir, f"subtree-{args.subtree}.parquet")
    pd.concat(stac_parquet_objects).to_parquet(full_parquet_path)
    
    # Load the tileset and remove problematic attrs
    est_footprint = (shapely.geometry.box(*objects.geometry.total_bounds).area / 1e6) * ECO_MB_PER_SQ_KM
    logger.info(f"Estimated array size (GB): {est_footprint / 1e3:.2f}")
    da = get_tileset_data_array(full_parquet_path, objects.geometry.total_bounds)
    
    if da is None:
        sys.exit(0)
    else:
        da = da.load()
    
    del da.attrs["_stac_backend"]
    del da.attrs["_stac_time_coords"]
    del da.attrs["zarr_conventions"]
        
    logger.info("Masking LST product")
    ds = mask_lst(da)
    del da # save memory

    logger.info("Computing time series for each object")
    ts_dataset = get_timeseries_at_objects(objects, ds)
    
    # Print proportion NA pixels for each band
    logger.info("Proportion NA pixels per band across all objects:")
    prop_nan_by_band = ts_dataset[ECO_ASSETS].isnull().mean(dim=["sample", "time"])
    for band in prop_nan_by_band.variables.keys():
        logger.info(f"\t{band}: {prop_nan_by_band[band].data:.3f}")

    # Check if any objects were fully nan, which would indicate we computed the
    # boundary wrong.
    prop_nan_by_sample = ts_dataset[ECO_ASSETS].isnull().mean(dim=["time"])
    if (prop_nan_by_sample == 1).any():
        logger.warning("Some objects had no valid pixels!")
    
    # Save output
    out_path = os.path.join(OUTPUT_DIRECTORY, f"timeseries-subtree-{args.subtree}.nc")
    logger.info(f"Saving output to {out_path}")
    ts_dataset.to_netcdf(out_path)
    