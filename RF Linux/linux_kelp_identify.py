import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from rasterio.errors import RasterioIOError
import csv
from skimage import io
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

import cudf
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.model_selection import train_test_split
from scipy.stats import randint

import pickle
import cupy as cp
import random

retrain = False
reclassify = True #Reclassify previously classified images
show_image = True
num_classify = 20
classified_path = r'/mnt/c/users/attic/hls_kelp/imagery/rf_classified_cuML'
tile = '11SKU'
location = 'Isla_Vista_Kelp'
cloud_cover_threshold = .5
#save_mask = True
#save_classification = True
#remask = False
path = os.path.join(r'/mnt/c/Users/attic/HLS_Kelp/imagery',location,tile)
num_iterations = 1000
#unclassified_path = r'/mnt/c/users/attic/hls_kelp/imagery/rf_prepped_v2'
#unclassified_files = os.listdir(unclassified_path)



### ==================== Create DEM mask ====================  ####
granules = os.listdir(path)
for item in granules:
    if os.path.isdir(os.path.join(path,item)):
        img_path = os.path.join(path,item)
        granule = item
        break
    else:
        continue
files = os.listdir(img_path)
file_data = granule.split('.')
sensor = file_data[1]
if(sensor == 'L30'):
    sensor_bands = ['B02','B03','B04','B05','B06','B07'] #2,3,4,5,6,7]
else:
    sensor_bands = ['B02','B03','B04','B08A','B11','B12']
    
pattern = re.compile(r'\.(' + '|'.join(sensor_bands) + r')\.tif$')
img_files = [f for f in files if re.search(pattern, f)]
geotiff_path = os.path.join(img_path, img_files[0])

with rasterio.open(geotiff_path) as dst:   
    hls = dst.read()
    dem_path = r'/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff'
    dem = rasterio.open(dem_path)
    if (dem.crs != dst.crs):
        reprojected_dem = np.zeros((hls.shape[1], hls.shape[2]), dtype=hls.dtype)
        reproject(
            source=dem.read(),
            destination=reprojected_dem,
            src_transform=dem.transform,
            src_crs=dem.crs,
            dst_transform=dst.transform,
            dst_crs=dst.crs,
            resampling=Resampling.bilinear)
    hls_flat = np.squeeze(hls, axis=0)   

if reprojected_dem.any():
    struct = np.ones((5,5))
    land_mask = binary_dilation(reprojected_dem > 0, structure = struct)
    ocean_mask = binary_dilation(reprojected_dem < -60 , structure = struct)
    full_mask = land_mask + ocean_mask
    plt.figure(figsize=(6, 6))
    plt.imshow(land_mask, cmap='gray')
    plt.show()    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(full_mask, cmap='gray')
    # plt.show()
    if save_mask:
        mask_path = os.path.join(path,f'{tile}_fullmask.tif')
        transform = dst.transform  
        height, width = full_mask.shape
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 1,  # one band
            'dtype': rasterio.uint8,  # assuming binary mask, adjust dtype if needed
            'crs': dst.crs,
            'transform': transform,
            'nodata': 0  # assuming no data is 0
        }

        # Write the land mask array to GeoTIFF
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(full_mask.astype(rasterio.uint8), 1)
else:
    print("Something failed, you better go check...")

### ==================== Load RF Classifier ================== ###
rf_path = r'/mnt/c/users/attic/hls_kelp/random_forest/cu_rf2'
with open(rf_path, 'rb') as f:
    cu_rf = pickle.load(f)



### ====================  Begin processing each file  ====================  ####
iterations = 0
print(f"Granules to process: {len(granules)}")
print("Img process beginning...")
for item in granules:
    if iterations > num_iterations:
        break
 ##==========Select Granule and Get File Names==========##

# Check sensor type and define bands for L8 and S2
    if os.path.isdir(os.path.join(path,item)):
        img_path = os.path.join(path,item)
    else:
        continue
    files = os.listdir(img_path)
    file_data = item.split('.')
    sensor = file_data[1]
    if(sensor == 'L30'):
        sensor_bands = ['B02','B03','B04','B05','B06','B07'] #2,3,4,5,6,7]
    else:
        sensor_bands = ['B02','B03','B04','B8A','B11','B12']
    pattern = re.compile(r'\.(' + '|'.join(sensor_bands) + r')\.tif$')

# Get File names
    img_files = [f for f in files if re.search(pattern, f)]
    f_mask = [f for f in files if re.search(r'Fmask\.tif$', f)]
    if not f_mask:
        print(f"Invalid granule: {item}")
        continue
    if not len(img_files)  == 6:
        print(f"incomplete file download: {item}")
        continue

    img_bands = []
    metadata =[]
    metadata_file = [f for f in files if re.search(r'metadata\.csv$', f)]
    if metadata_file :
        with open(os.path.join(path,item, metadata_file[0]), mode='r') as file:
            csv_reader = csv.reader(file)
            keys = next(csv_reader)  
            values = next(csv_reader) 
        metadata = dict(zip(keys, values)) #Load metadata into dictionary 


##==========Fmask Cloud mask==========##
    #bitwise operations are weird. Far outside my comfort zone. Need to take CS33 first.........
    try:
        with rasterio.open(os.path.join(img_path,f_mask[0])) as fmask:
            qa_band = fmask.read(1)
        qa_bit = (1 << 1) - 1
        qa_cloud_mask = ((qa_band >> 1) & qa_bit) == 1  # Bit 1 for cloud
        qa_adjacent_to_cloud_mask = ((qa_band >> 2) & qa_bit) == 1  # Bit 2 for cloud adjacent
        qa_cloud_shadow = ((qa_band >> 3) & qa_bit) == 1 
        qa_ice = ((qa_band >> 4) & qa_bit) == 1 
        #qa_water = ((qa_band >> 5) & qa_bit) == 1
        qa_aerosol = (((qa_band >> 6) & 1) == 1) & (((qa_band >> 7) & 1) == 1)
        cloud_mask = qa_cloud_mask | qa_adjacent_to_cloud_mask | qa_cloud_shadow | qa_ice | qa_aerosol#Mask out Clouds and cloud-adjacent pixels 
        cloud_mask_2D = cloud_mask.reshape(-1).T
    except RasterioIOError as e:
        print(f"Error reading file {file} in granule {item}: {e}")
        continue  # Skip to the next granule if a file cannot be read
    #may not be necessary to mask out the cloud-adjacent pixels 

##========== Determine percentage of ocean covered by clouds ==========##
    cloud_land_mask = cloud_mask | land_mask
    cloud_but_not_land_mask = cloud_mask & ~land_mask
    num_pixels_cloud_not_land = np.count_nonzero(cloud_but_not_land_mask)
    num_pixels_not_land = np.count_nonzero(~land_mask)
    percent_cloud_covered = num_pixels_cloud_not_land/num_pixels_not_land
    if(percent_cloud_covered > cloud_cover_threshold):
        continue
    print(f'{granule} Percent cloud covered: {percent_cloud_covered}')
    
 ##==========Create stacked np array, Apply landmask==========##
    try:
        for file in img_files:
            with rasterio.open(os.path.join(img_path, file)) as src:
                img_bands.append(np.where(cloud_land_mask, 0, src.read(1)))  # Create image with the various bands
    except RasterioIOError as e:
        print(f"Error reading file {file} in granule {item}: {e}")
        continue  # Skip to the next granule if a file cannot be read
    img = np.stack(img_bands, axis=0)
    n_bands, height, width = img.shape
    img_2D = img.reshape(img.shape[0], -1).T #classifier takes 2D array of band values for each pixel 

    #normalized_img_bands = np.column_stack((img_2D, cloud_mask_2D))
 ##========== Normalize multi-spectral data ==========##

 ##========== Add masked file-folder to directory, if needed ==========##
   # if not os.path.isdir (r'/mnt/c/Users/attic/HLS_Kelp/imagery/rf_prepped_v2'):
   #     os.mkdir(r'/mnt/c/Users/attic/HLS_Kelp/imagery/rf_prepped_v2')
   # classification_path = os.path.join(r'/mnt/c/Users/attic/HLS_Kelp/imagery/rf_prepped_v2',f'{item}_rf_ready.tif')


#  ##========== Save masked file ==========##

#     num_bands = 6
#     data_type = np.int16
#     profile = {
#         'driver': 'GTiff',
#         'width': width,
#         'height': height,
#         'count': 6,  # one band
#         'dtype': data_type,  # assuming binary mask, adjust dtype if needed
#         'crs': dst.crs,
#         'transform': transform,
#         'nodata': 0,  # assuming no data is 0
#         'cloud_cover': percent_cloud_covered
#     }
#     # Write the land mask array to GeoTIFF
#     with rasterio.open(classification_path, 'w', **profile) as dst:
#         for i in range(num_bands):
#             dst.write(normalized_img_bands[:,:,i].astype(data_type), i + 1)



    # file_name = file.split('_')
    # if not reclassify and os.path.isfile(os.path.join(classified_path, f'{file_name[0]}_kelp_classified.tif')):
    #     print(f"{file} already classified")
    #     continue
    #file_img =[]
    #with rasterio.open(os.path.join(unclassified_path,file)) as src:
        #file_img = src.read(indexes=[1, 2, 3, 4, 5, 6])
        #img = np.stack(file_img, axis=0)
        #n_bands, height, width = img.shape
        #img_2D = img.reshape(img.shape[0], -1).T #classifier takes 2D array of band values for each pixel 
    #normalized_img_bands = np.column_stack((img_2D, cloud_mask_2D))
 ##========== Normalize multi-spectral data ==========##

    img_sum = img_2D.sum(axis=1)
    epsilon = 1e-10  
    img_2D_nor = np.divide(img_2D, img_sum[:, None] + epsilon, where=(img_sum[:, None] != 0))
    img_2D_nor = (img_2D_nor * 255).astype(np.uint8)
    #img_normalized = img_2D_normalized.reshape((height, width))
        # img_sum_nonzero = np.where(img_sum == 0, 1, img_sum)
        # img_2D_normalized = img_2D / img_sum_nonzero[:, None] #divide value by sum of pixel band values
        # print(img_2D_normalized.shape)
        # img_2D_normalized = (img_2D_normalized * 255)
        # img_2D_normalized = img_2D_normalized.astype(np.uint8)

    #img_data= file_img.reshape(file_img.shape[0], -1).T
    img_data = cudf.DataFrame(img_2D_nor)
    img_data = img_data.astype(np.float32)

    kelp_pred = cu_rf.predict(img_data)
    kelp_img = kelp_pred.values_host.reshape(width,height)

    if show_image:
        print(file)
        plt.figure(figsize=(25, 25)) 
        plt.subplot(2, 1, 1)  
        plt.imshow(kelp_img[2700:3400, 600:2000])
        plt.title(file)
        r_nor = img_2D_nor[:,2].reshape((height, width))
        g_nor = img_2D_nor[:,1].reshape((height, width))
        b_nor = img_2D_nor[:,0].reshape((height, width))
        rgb_nor = np.stack([r_nor,g_nor,b_nor], axis=-1)  
        rgb_cropped = rgb_nor[2700:3400, 600:2000]
        plt.subplot(2, 1, 2) 
        plt.imshow(rgb_cropped)
        plt.title("RGB Cropped Image")
        #plt.colorbar()
        plt.show()

    # Write the land mask array to GeoTIFF
    # with rasterio.open(os.path.join(classified_path, f'{file_name[0]}_kelp_classified.tif'), 'w', **profile) as dst:
    #         dst.write(file_img[0].astype(data_type), 1)
    #         dst.write(file_img[1].astype(data_type), 2)
    #         dst.write(file_img[2].astype(data_type), 3)
    #         dst.write(file_img[3].astype(data_type), 4)
    #         dst.write(kelp_img.astype(rasterio.uint8), 5)
    # iterations = iterations + 1
    # print(f"{iterations}/{len(num_iterations)}")

    path = r'/mnt/c/Users/attic/HLS_Kelp/imagery/rf_classified_cuML/HLS.L30.T11SKU.2019077T183342.v2.0_rf_ready.tif_kelp_classified.tif'
    ocean_dilation = np.ones((100,100)) #Struct for dilation (increase to enlarge non-ocean mask) larger --> takes longer
    kelp_dilation = np.ones((4,4))
    with rasterio.open(path) as imagery:
        classified_img = imagery.read(5)
        kelp_mask  = []
        ocean_mask = []
        ocean_dilated = np.where(classified_img == 1, False, True)
        ocean_dilated = binary_dilation(ocean_dilated, structure=ocean_dilation) #This takes ~25 seconds. Should look to optimize 
        kelp_dilated = np.where(classified_img == 0, True, False) #This is expanding hte kelp_mask so the TF is reversed
        kelp_dilated = binary_dilation(kelp_dilated,structure=kelp_dilation) #I may not want to do this. we'll see
        for i in range(4):
            band_data = imagery.read(i + 1)
            kmask = np.where(kelp_dilated == True, band_data, np.nan)
            omask = np.where(ocean_dilated == False, band_data, np.nan)
            kelp_mask.append(kmask)
            ocean_mask.append(omask)

        kelp_mask = np.array(kelp_mask)
        ocean_mask = np.array(ocean_mask)
        #print(ocean_mask)

    rgb_nor = np.stack([ocean_mask[2]/600,ocean_mask[0]/600,ocean_mask[1]/600], axis=-1)
    rgb_nor_cropped = rgb_nor
    #print(kelp_mask)
    rgb_nor_cropped = np.ma.masked_where(np.isnan(rgb_nor_cropped), rgb_nor_cropped)
    image = kelp_mask[1]#,2500:3500,800:1800]
    plt.figure(figsize=(30, 30), dpi=200)
    plt.imshow(image, alpha=1)
    plt.imshow(rgb_nor_cropped, alpha=1)
    plt.colorbar()
    plt.show()

    ocean_EM_stack = []
    kelp_EM = [459, 556, 437, 1227]

    n_bands, height, width = kelp_mask.shape
    ocean_EM_n = 0
    ocean_data = ocean_mask.reshape(ocean_mask.shape[0], -1)
    kelp_data = kelp_mask.reshape(kelp_mask.shape[0],-1)

    nan_columns = np.isnan(ocean_data).any(axis=0)  # Remove columns with nan 
   
    filtered_ocean = ocean_data[:, ~nan_columns]
    for i in range(30):
        index = random.randint(0,len(filtered_ocean[0])-1)
        ocean_EM_stack.append(filtered_ocean[:,index])
    ocean_EM = np.stack(ocean_EM_stack, axis=1)
    #print(ocean_EM_array)

    average_val = np.nanmean(filtered_ocean, axis=1)
    average_endmember = np.nanmean(ocean_EM, axis=1)
    print(f"average EM Val: {average_endmember}")
    print(f"average    Val: {average_val}")

    kelp_mask = cp.asarray(kelp_mask)
    ocean_EM = cp.asarray(ocean_EM)
    kelp_EM = cp.asarray(kelp_EM)
    kelp_data = cp.asarray(kelp_data)

    frac1 = cp.full((kelp_mask.shape[1], kelp_mask.shape[2], 30), cp.nan)
    frac2 = cp.full((kelp_mask.shape[1], kelp_mask.shape[2], 30), cp.nan)
    rmse = cp.full((kelp_mask.shape[1], kelp_mask.shape[2], 30), cp.nan)
    print(rmse.shape)

    print("Running MESMA")
    for k in range(30):
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


        data_type = rasterio.int16
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 5,  # one band  B02, B03, B04, and B05, classified (Blue, Green, Red, and NIR).
            'dtype': data_type,  # assuming binary mask, adjust dtype if needed
            'crs': src.crs,
            'transform': src.transform,
            'nodata': 0  # assuming no data is 0
        }



        minVals = cp.nanmin(rmse, axis=2)
    PageIdx = cp.nanargmin(rmse, axis=2)
    rows, cols = cp.meshgrid(cp.arange(rmse.shape[0]), cp.arange(rmse.shape[1]), indexing='ij')
    Zindex = cp.ravel_multi_index((rows, cols, PageIdx), dims=rmse.shape)
    Mes2 = frac2.ravel()[Zindex]
    Mes2 = Mes2.T
    Mes2 = -0.229 * Mes2**2 + 1.449 * Mes2 - 0.018 #Landsat mesma corrections 
    Mes2 = cp.clip(Mes2, 0, None)  # Ensure no negative values
    Mes2 = cp.round(Mes2 * 100).astype(cp.int16)


    #Mes2 = Mes2.astype(cp.float32)
    #Mes2 = Mes2.where(Mes2 == 0, cp.nan)
    Mes_array = cp.asnumpy(Mes2).T
    if img_show:
        plt.figure(figsize=(20, 20), dpi=200)
        plt.imshow(Mes_array[2700:3400,800:1600], alpha=1)
        plt.colorbar()
        plt.show()
    break

    #  ##========== Save masked file ==========##

#     num_bands = 6
#     data_type = np.int16
#     profile = {
#         'driver': 'GTiff',
#         'width': width,
#         'height': height,
#         'count': 6,  # one band
#         'dtype': data_type,  # assuming binary mask, adjust dtype if needed
#         'crs': dst.crs,
#         'transform': transform,
#         'nodata': 0,  # assuming no data is 0
#         'cloud_cover': percent_cloud_covered
#     }
#     # Write the land mask array to GeoTIFF
#     with rasterio.open(classification_path, 'w', **profile) as dst:
#         for i in range(num_bands):
#             dst.write(normalized_img_bands[:,:,i].astype(data_type), i + 1)
    