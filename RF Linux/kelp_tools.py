import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import re
#from scipy.ndimage import binary_dilation, uniform_filter
from rasterio.errors import RasterioIOError
from skimage import io

import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
#from cuml.model_selection import train_test_split
from scipy.stats import randint
from cupyx.scipy.ndimage import binary_dilation, convolve
import time

import sys
import pickle
import csv
import cupy as cp
import random

# def load_granule_data(path):
#     granules = os.listdir(path)
#     for item in granules:
#         if os.path.isdir(os.path.join(path, item)):
#             img_path = os.path.join(path, item)
#             granule = item
#             break
#     files = os.listdir(img_path)
#     return img_path, granule, files

def get_sensor(granule):
    file_data = granule.split('.')
    return file_data[1]

def get_sensor_bands(granule):
    file_data = granule.split('.')
    sensor = file_data[1]
    if sensor == 'L30':
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
    else:
        return ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    
def load_rf(rf_path):
    with open(rf_path, 'rb') as f:
        cu_rf = pickle.load(f)
    return cu_rf

def filter_and_sort_files(img_path, granule):
    # Split the granule to get the sensor information
    sensor_bands = get_sensor_bands(granule)
    
    # Get all files in the directory
    img_files = os.listdir(img_path)
    
    # Filter files to keep only those that are in the sensor_bands list
    filtered_files = [f for f in img_files if any(band in f for band in sensor_bands)]
    
    # Remove files that don't match the bands in sensor_bands
    final_files = []
    for band in sensor_bands:
        for file in filtered_files:
            if band in file:
                final_files.append(file)
    
    # Sort files based on the order in sensor_bands
    def sort_key(filename):
        for band in sensor_bands:
            if band in filename:
                return sensor_bands.index(band)
        return len(sensor_bands)  # Place files with no band information at the end
    
    sorted_files = sorted(final_files, key=sort_key)
    
    return sorted_files

def valid_granule(img_files, sensor_bands, files, item):
    f_mask = [f for f in files if re.search(r'Fmask\.tif$', f)]
    if not f_mask or len(img_files) != len(sensor_bands):
        print(f"Incomplete or invalid granule: {item}")
        return False
    return True
   
def create_qa_mask(land_mask, img_path):
    files = os.listdir(img_path)
    f_mask = [f for f in files if re.search(r'Fmask\.tif$', f)]
    if not f_mask:
        return False

##==========Fmask Cloud mask==========##
    #bitwise operations are weird. Far outside my comfort zone. Need to take CS33 first.........
    try:
        with rasterio.open(os.path.join(img_path,f_mask[0])) as fmask:
            qa_band = fmask.read(1)
        # qa_bit = (1 << 1) - 1
        # qa_cloud_mask = ((qa_band >> 1) & qa_bit) == 1  # Bit 1 for cloud
        # qa_adjacent_to_cloud_mask = ((qa_band >> 2) & qa_bit) == 1  # Bit 2 for cloud adjacent
        # qa_cloud_shadow = ((qa_band >> 3) & qa_bit) == 1 
        # qa_ice = ((qa_band >> 4) & qa_bit) == 1 
        # qa_aerosol = (((qa_band >> 6) & 1) == 1) & (((qa_band >> 7) & 1) == 1)
        # cloud_mask = qa_cloud_mask | qa_adjacent_to_cloud_mask | qa_cloud_shadow | qa_ice | qa_aerosol#Mask out Clouds and cloud-adjacent pixels 
        cloud_mask = ((qa_band >> 1) & 1) #| ((qa_band >> 2) & 1) | ((qa_band >> 3) & 1) | ((qa_band >> 4) & 1) | (((qa_band >> 6) & 1) & ((qa_band >> 7) & 1))
        #cloud_mask_2D = cloud_mask.reshape(-1).T
    except RasterioIOError as e:
        print(f"Error reading file {f_mask}: {e}")
        return False  # Skip to the next granule if a file cannot be read
    #may not be necessary to mask out the cloud-adjacent pixels 

##========== Determine percentage of ocean covered by clouds ==========##
    cloud_land_mask = cp.asarray(cloud_mask | land_mask)
    cloud_but_not_land_mask = cp.asarray(cloud_mask & ~land_mask)
    num_pixels_cloud_not_land = cp.count_nonzero(cloud_but_not_land_mask)
    num_pixels_not_land = np.count_nonzero(~land_mask)
    percent_cloud_covered = num_pixels_cloud_not_land/num_pixels_not_land
    # print(f'{granule} Percent cloud covered: {percent_cloud_covered}')
    return cloud_land_mask, cloud_but_not_land_mask, percent_cloud_covered

def calculate_local_variance(image_gpu, window_size):

    mean_kernel = cp.ones((window_size, window_size), dtype=cp.float32) / (window_size * window_size)
    local_mean_gpu = convolve(image_gpu.astype(cp.float32), mean_kernel, mode='constant', cval=0.0)

    squared_image_gpu = cp.square(image_gpu.astype(cp.float32))
    mean_squared_gpu = convolve(squared_image_gpu, mean_kernel, mode='constant', cval=0.0)
    local_variance_gpu = mean_squared_gpu - cp.square(local_mean_gpu)
    
    return local_variance_gpu

def reproject_dem_to_hls(hls_path, dem_path='/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff'):
    with rasterio.open(hls_path) as dst:
        hls = dst.read()
        dem = rasterio.open(dem_path)
        if dem.crs != dst.crs:
            reprojected_dem = np.zeros((hls.shape[1], hls.shape[2]), dtype=hls.dtype)
            reproject(
                source=dem.read(),
                destination=reprojected_dem,
                src_transform=dem.transform,
                src_crs=dem.crs,
                dst_transform=dst.transform,
                dst_crs=dst.crs,
                resampling=Resampling.bilinear)
            if reprojected_dem.any():
                return reprojected_dem
            else:
                return False

def generate_land_mask(reprojected_dem, land_dilation=7, show_image=False):
    if reprojected_dem.any():
        struct = cp.ones((land_dilation, land_dilation))
        reprojected_dem_gpu = cp.asarray(reprojected_dem)
        land_mask = binary_dilation(reprojected_dem_gpu > 0, structure=struct)
        land_mask = cp.asnumpy(land_mask)
        if show_image:
            plt.figure(figsize=(6, 6))
            plt.imshow(land_mask, cmap='gray')
            plt.show()
        return land_mask
    else:
        print("Something failed, you better go check...")
        sys.exit()

def create_land_mask(hls_path, dem_path='/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff'):
    reprojected_dem = reproject_dem_to_hls(hls_path, dem_path)
    return generate_land_mask(reprojected_dem)

def create_mesma_mask(
    classified_img,
    img,
    land_mask,
    cloud_but_not_land_mask,
    ocean_dilation_size=100,  # Struct size for dilation
    kelp_neighborhood=5,
    min_kelp_count=4,
    kelp_dilation_size=15,
    variance_window_size=15,
    variance_threshold=0.95
):
    ocean_dilation = cp.ones((ocean_dilation_size, ocean_dilation_size))
    structuring_element = cp.ones((kelp_dilation_size, kelp_dilation_size))
    
    #time_st = time.time()
    classified_img_gpu = cp.array(classified_img)
    land_dilated_gpu = cp.asarray(land_mask)
    #time_val = time.time()
    cloud_not_land_gpu = cp.asarray(cloud_but_not_land_mask)
    #land_dilated_gpu = cp.where(land_mask, True, False)
    clouds_dilated_gpu = cp.where(classified_img_gpu == 2, True, cloud_not_land_gpu)
    land_dilated_gpu = binary_dilation(land_dilated_gpu, structure=ocean_dilation)
    #print(f'land finished: {time.time()-time_val}')

    ocean_dilated_gpu = land_dilated_gpu | clouds_dilated_gpu 

    kelp_dilated_gpu = cp.where(classified_img_gpu == 0, True, False)  # This is expanding the kelp_mask so the TF is reversed
    kernel = cp.ones((kelp_neighborhood, kelp_neighborhood), dtype=cp.int32)

    time_val = time.time()
    kelp_count_gpu = convolve(kelp_dilated_gpu.astype(cp.int32), kernel, mode='constant', cval=0.0)
    #print(f'kelp moving average finished: {time.time()-time_val}')

    kelp_dilated_gpu = cp.where(((~kelp_dilated_gpu) | (kelp_count_gpu <= min_kelp_count)), 0, 1)  # If there's no kelp, or the kelp count is <=4, set pixel == false
    #time_val = time.time()
    kelp_dilated_gpu = binary_dilation(kelp_dilated_gpu, structure=structuring_element)  # I may not want to do this. we'll see
    #print(f'kelp dilation finished: {time.time()-time_val}')
    #time_val = time.time()
    kelp_mask = [None] * 4
    ocean_mask = [None] * 4

    for i in range(4):
        band_data = img[i]
        band_data_gpu = cp.asarray(band_data)  # Ensure it's a CuPy array

        kmask_gpu = cp.where(kelp_dilated_gpu == 1, band_data_gpu, cp.nan)

        local_variance_gpu = calculate_local_variance(band_data_gpu, variance_window_size)
        max_local_variance = cp.percentile(local_variance_gpu, 100 * variance_threshold)

        variance_mask_gpu = cp.where(local_variance_gpu > max_local_variance, cp.nan, band_data_gpu)


        final_omask_gpu = cp.where((ocean_dilated_gpu == True) | cp.isnan(variance_mask_gpu), cp.nan, band_data_gpu)

        kelp_mask[i] = kmask_gpu
        ocean_mask[i] = final_omask_gpu
    #print(f'kBand masking and variance masking complete: {time.time()-time_val}')
    kelp_mask = cp.stack(kelp_mask, axis=0)
    ocean_mask = cp.stack(ocean_mask,axis=0)
    return kelp_mask, ocean_mask


def normalize_img(img, flatten=True):
    img_2D = img.reshape(img.shape[0], -1).T
    img_sum = img_2D.sum(axis=1)
    #print(img_sum.shape)
    epsilon = 1e-10  
    #mask = img_sum[:, None] != 0
    #mask = cp.broadcast_to(mask, img_2D.shape)
    #print(mask.shape)
    #print(type(mask))
    img_2D_nor = cp.divide(img_2D, img_sum[:, None] + epsilon)#, where=mask)
    img_2D_nor = (img_2D_nor * 255).astype(cp.uint8)
    img_2D_nor = img_2D_nor.T
    if flatten:
        img_data= img_2D_nor.reshape(img_2D_nor.shape[0], -1).T
        return img_data
    return img_2D_nor
   

def select_ocean_endmembers(ocean_mask, print_average=False, n=30, min_pixels=1000):
    ocean_EM_n = 0
    ocean_data = ocean_mask.reshape(ocean_mask.shape[0], -1)
    nan_columns = cp.isnan(ocean_data).any(axis=0)  # Remove columns with nan 
    ocean_EM_stack =[]
    filtered_ocean = ocean_data[:, ~nan_columns]
    if len(filtered_ocean[0,:]) < min_pixels:
        print("Too few valid ocean end-members")
        return None
    
    for i in range(n):
        index = random.randint(0,len(filtered_ocean[0])-1)
        ocean_EM_stack.append(filtered_ocean[:,index])
    ocean_EM = cp.stack(ocean_EM_stack, axis=1)
    #print(ocean_EM_array)
    if(print_average):
        average_val = cp.nanmean(filtered_ocean, axis=1)
        average_endmember = cp.nanmean(ocean_EM, axis=1)
        print(f"average EM Val: {average_endmember}")
        print(f"average    Val: {average_val}")
    return ocean_EM

def run_mesma(kelp_mask, ocean_EM, n=30, kelp_EM=[459, 556, 437, 1227]):
    height = kelp_mask.shape[1]
    width = kelp_mask.shape[2]
    kelp_data = kelp_mask.reshape(kelp_mask.shape[0], -1)


    frac1 = cp.full((height, width, n), cp.nan)
    frac2 = cp.full((height, width, n), cp.nan)
    rmse = cp.full((height, width, n), cp.nan)

    kelp_mask = cp.asarray(kelp_mask)
    ocean_EM = cp.asarray(ocean_EM)
    kelp_EM = cp.asarray(kelp_EM)
    kelp_data = cp.asarray(kelp_data)
    print(rmse.shape)

    print("Running MESMA")
    for k in range(n):
        B = cp.column_stack((ocean_EM[:, k], kelp_EM))
        U, S, Vt = cp.linalg.svd(B, full_matrices=False)
        IS = Vt.T / S
        em_inv = IS @ U.T
        F = em_inv @ kelp_data
        model = (F.T @ B.T).T
        resids = (kelp_data - model) / 10000
        rmse[:, :, k] = cp.sqrt(cp.mean(resids**2, axis=0)).reshape(height, width)
        frac1[:, :, k] = F[0, :].reshape(height, width)
        frac2[:, :, k] = F[1, :].reshape(height, width)

    minVals = cp.nanmin(rmse, axis=2)
    PageIdx = cp.nanargmin(rmse, axis=2)
    rows, cols = cp.meshgrid(cp.arange(rmse.shape[0]), cp.arange(rmse.shape[1]), indexing='ij')
    Zindex = cp.ravel_multi_index((rows, cols, PageIdx), dims=rmse.shape)
    Mes2 = frac2.ravel()[Zindex]
    Mes2 = Mes2.T
    Mes2 = -0.229 * Mes2**2 + 1.449 * Mes2 - 0.018 #Landsat mesma corrections 
    Mes2 = cp.clip(Mes2, 0, None)  # Ensure no negative values
    Mes2 = cp.round(Mes2 * 100).astype(cp.int16)
    return Mes2, minVals


def get_metadata(path, files=None):
    if files==None:
        files = os.listdir(path)
    metadata_file = [f for f in files if re.search(r'metadata\.csv$', f)]
    if metadata_file:
        with open(os.path.join(path, metadata_file[0]), mode='r') as file:
            csv_reader = csv.reader(file)
            keys = next(csv_reader)  
            values = next(csv_reader) 
        return dict(zip(keys, values))