import os
import geopandas as gp
import matplotlib.pyplot as plt
import json
from contextlib import redirect_stdout
import csv
import earthaccess

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation, uniform_filter
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from rasterio.errors import RasterioIOError
import csv
from skimage import io
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import sys
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.model_selection import train_test_split
from scipy.stats import randint
import shutil
import pickle
import cupy as cp
import random
import time



geojson_path = '/mnt/c/Users/attic/HLS_Kelp/maps/Isla_Vista_Kelp.geojson'

load_num = -1 #sets number of granules to load, this should generally be >> than num_download;  -1 loads all 
specific_tile = True #set true if you only want specific tile 
retrain = False
reclassify = True #Reclassify previously classified images
show_image = True
save_final_data = True
save_to_path = '/mnt/c/Users/attic/HLS_Kelp/imagery/' # + tile_processed
num_classify =300

tile = '10SGD'
location = 'Isla_Vista_Kelp'
cloud_cover_threshold = .5
version = 1
temporal = ("2019-1-01T00:00:00", "2020-1-01T00:00:00") #
dem_path = '/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff'
rf_path = r'/mnt/c/users/attic/hls_kelp/random_forest/cu_rf5'
sample_geotiff_path = '/mnt/c/Users/attic/HLS_Kelp/imagery/Isla_Vista_Kelp/10SGD/HLS.L30.T10SGD.2013108T183619.v2.0/HLS.L30.T10SGD.2013108T183619.v2.0.B02.tif'

earthaccess.login(persist=True)

field = gp.read_file(geojson_path)
bbox = tuple(list(field.total_bounds))
bbox #Display coordinate bounds
with open(geojson_path, 'r') as f:
    data = json.load(f)
# Extract the name

#Search for satellite data from  Landsat 30m and Sentinel 30m
results = earthaccess.search_data(
    short_name=['HLSL30','HLSS30'],
    bounding_box=bbox,
    temporal=temporal,
     cloud_cover=0, #Determine cloud cover
    count=load_num
)

#print(results[0])

folder_path = os.path.join(f'/mnt/c/Users/attic/HLS_Kelp/imagery',location)
temp_folder = os.path.join(os.path.join(folder_path),'temp')

## ======= create location folder path ======= ##
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
if os.path.isdir(temp_folder):
    shutil.rmtree(temp_folder)
os.mkdir(temp_folder)
iterations = 0

### ==================== Load RF Classifier ================== ###
with open(rf_path, 'rb') as f:
    cu_rf = pickle.load(f)

### ==================== Create DEM mask ====================  ####
time_st = time.time()
print(f"Building DEM for {tile}")
with rasterio.open(sample_geotiff_path) as dst:
    hls = dst.read()
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

if not reprojected_dem.any():
    print("Something failed, you better go check...")
    sys.exit()
else:
    struct = np.ones((5,5))
    land_mask = binary_dilation(reprojected_dem > 0, structure = struct)
time_1 = time.time()
print(f"Bathymetry and landmask built. Beginning individual granule processing | {time_1 - time_st} ")

for i, granule in enumerate(results):
    time_it = time.time()
    if(iterations >num_classify):
        break
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)
    ## ======= Parse metadata ======= ##
    json_str = json.dumps(granule.__dict__)
    metadata = json.loads(json_str) 
    meta = metadata['render_dict']['meta']
    name = meta['native-id']
    #print(f"Building {name} | {time_it-time_st}")
    #For some reason, attributes are parsed into a list in the HLS metadata. This reformats it into a dictionary.
    attributes_list = metadata['render_dict']['umm']['AdditionalAttributes']

    attributes = {attr['Name']: attr['Values'][0] for attr in attributes_list}
    #print(attributes['MGRS_TILE_ID'])
    tile_type = attributes['MGRS_TILE_ID']
    if(int(attributes['CLOUD_COVERAGE']) > 50): #Reject granules with large cloud cover, for now
        print("Overall Cloud coverage >50%")
        shutil.rmtree(temp_folder)
        continue
    time = attributes['SENSING_TIME']
    tile_folder = f"{tile_type}_Classified_v{version}"
    if specific_tile and not tile_type == tile:
        shutil.rmtree(temp_folder)
        continue
    ## ======= Create file directory, if needed  ======= ##
    tile_path = os.path.join(folder_path,tile_folder)
    if not os.path.isdir(tile_path):
         os.mkdir(tile_path)

    ## ======= download granule ======= ##
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        downloadPath = earthaccess.download(granule, temp_folder)
    
    print(f'{name} Downloaded | LT: {time.time() - time_it}')


### ====================  Begin processing each file  ====================  ####

    print(f"Beginning {name} processing")

##==========Select Granule and Get File Names==========##

    # Check sensor type and define bands for L8 and S2
    
    files = os.listdir(temp_folder)
    file_data = name.split('.')
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
        print(f"Invalid granule: {name}")
        shutil.rmtree(temp_folder)
        continue
    if not len(img_files)  == 6:
        print(f"incomplete file download: {name}")
        shutil.rmtree(temp_folder)
        continue

    img_bands = []

##==========Fmask Cloud mask==========##
    #bitwise operations are weird. Far outside my comfort zone. Need to take CS33 first.........
    try:
        with rasterio.open(os.path.join(temp_folder,f_mask[0])) as fmask:
            qa_band = fmask.read(1)
        # qa_bit = (1 << 1) - 1
        # qa_cloud_mask = ((qa_band >> 1) & qa_bit) == 1  # Bit 1 for cloud
        # qa_adjacent_to_cloud_mask = ((qa_band >> 2) & qa_bit) == 1  # Bit 2 for cloud adjacent
        # qa_cloud_shadow = ((qa_band >> 3) & qa_bit) == 1 
        # qa_ice = ((qa_band >> 4) & qa_bit) == 1 
        # #qa_water = ((qa_band >> 5) & qa_bit) == 1
        # qa_aerosol = (((qa_band >> 6) & 1) == 1) & (((qa_band >> 7) & 1) == 1)
        # cloud_mask = qa_cloud_mask | qa_cloud_shadow | qa_ice | qa_aerosol #Mask out Clouds and cloud-adjacent pixels 
        cloud_mask = (qa_band >> 1) & 1 | (qa_band >> 2) & 1 | (qa_band >> 3) & 1 | (qa_band >> 4) & 1
        cloud_mask = cloud_mask == 1
        cloud_mask_2D = cloud_mask.reshape(-1).T
    except RasterioIOError as e:
        print(f"Error reading file {file} in granule {name}: {e}")
        shutil.rmtree(temp_folder)
        continue  # Skip to the next granule if a file cannot be read
    #may not be necessary to mask out the cloud-adjacent pixels 

##========== Determine percentage of ocean covered by clouds ==========##
    cloud_land_mask = cloud_mask | land_mask
    cloud_but_not_land_mask = cloud_mask & ~land_mask
    num_pixels_cloud_not_land = np.count_nonzero(cloud_but_not_land_mask)
    num_pixels_not_land = np.count_nonzero(~land_mask)
    percent_cloud_covered = num_pixels_cloud_not_land/num_pixels_not_land
    if(percent_cloud_covered > cloud_cover_threshold):
        print(f"Percent clouds greater than threshold: {percent_cloud_covered}. Moving to next granule...")
        shutil.rmtree(temp_folder)
        continue
    print(f'{name} Percent cloud covered: {percent_cloud_covered}')
    
##==========Create stacked np array, Apply landmask==========##
    try:
        for file in img_files:
            with rasterio.open(os.path.join(temp_folder, file)) as src:
                img_bands.append(np.where(cloud_land_mask, 0, src.read(1)))  # Create image with the various bands
    except RasterioIOError as e:
        print(f"Error reading file {file} in granule {name}: {e}")
        shutil.rmtree(temp_folder)
        continue  # Skip to the next granule if a file cannot be read
    img = np.stack(img_bands, axis=0)
    n_bands, height, width = img.shape
    img_2D = img.reshape(img.shape[0], -1).T #classifier takes 2D array of band values for each pixel 

 ##========== Normalize multi-spectral data ==========##

    img_sum = img_2D.sum(axis=1)
    epsilon = 1e-10  
    img_2D_nor = np.divide(img_2D, img_sum[:, None] + epsilon, where=(img_sum[:, None] != 0))
    img_2D_nor = (img_2D_nor * 255).astype(np.uint8)

 ##========== Classify Image with random forest ==========##
    print(f"Beginning kelp classification | LT: {time.time() - time_it}")
    img_data = cudf.DataFrame(img_2D_nor)
    img_data = img_data.astype(np.float32)

    classification_pred = cu_rf.predict(img_data)
    classified_img = classification_pred.values_host.reshape(width,height)

    print(f"Classification finished | LT: {time.time() - time_it}")
 ##========== Display image, if asked to ==========##
    if show_image:
        print(file)
        plt.figure(figsize=(25, 25)) 
        plt.subplot(2, 1, 1)  
        plt.imshow(classified_img)#[2700:3400, 600:2000])
        plt.colorbar()
        plt.title(file)
        r_nor = img_2D_nor[:,2].reshape((height, width))
        g_nor = img_2D_nor[:,1].reshape((height, width))
        b_nor = img_2D_nor[:,0].reshape((height, width))
        rgb_nor = np.stack([r_nor,g_nor,b_nor], axis=-1)  
        rgb_cropped = rgb_nor#[2700:3400, 600:2000]
        plt.subplot(2, 1, 2) 
        plt.imshow(rgb_cropped)
        plt.title("RGB Cropped Image")
        #plt.colorbar()
        plt.show()
 ##========== Prep for Mesma ==========##
    ocean_dilation = np.ones((100,100)) #Struct for dilation (increase to enlarge non-ocean mask) larger --> takes longer
    kelp_dilation = np.ones((15,15))
    kelp_neighborhood_min = 5
    min_kelp_count = 4
    kelp_mask  = []
    ocean_mask = []
    print(f"Masking image for MESMA | LT: {time.time() - time_it}")
##========== Create mask for kelp and ocean ==========##

    ocean_dilated = np.where(classified_img == 1, False, True)
    ocean_dilated = binary_dilation(ocean_dilated, structure=ocean_dilation) #This takes ~25 seconds. Should look to optimize 
    kelp_dilated = np.where(classified_img == 0, True, False) #This is expanding hte kelp_mask so the TF is reversed
    kelp_count = uniform_filter(kelp_dilated.astype(int),size=kelp_neighborhood_min)
    kelp_dilated = np.where(((~kelp_dilated) | (kelp_count >= min_kelp_count)), 0, 1) #If theres no kelp, or the kelp count is <=4, set pixel == false
    kelp_dilated = binary_dilation(kelp_dilated,structure=kelp_dilation) 
    for i in range(4):
        kmask = np.where(kelp_dilated == True, img[i],np.nan)
        omask = np.where(ocean_dilated == False, img[i], np.nan)
        kelp_mask.append(kmask)
        ocean_mask.append(omask)

    kelp_mask = np.array(kelp_mask)
    ocean_mask = np.array(ocean_mask)

    rgb_nor = np.stack([ocean_mask[2]/600,ocean_mask[0]/600,ocean_mask[1]/600], axis=-1)
    rgb_nor_cropped = rgb_nor
    #print(kelp_mask)
    rgb_nor_cropped = np.ma.masked_where(np.isnan(rgb_nor_cropped), rgb_nor_cropped)
    print(f"Generated final kelp & ocean mask | LT: {time.time() - time_it}")
##========== Display masked image, if asked to ==========##
    if show_image:
        display_image = kelp_mask[1]#,2500:3500,800:1800]
        plt.figure(figsize=(30, 30), dpi=200)
        plt.imshow(display_image, alpha=1)
        plt.imshow(rgb_nor_cropped, alpha=1)
        plt.colorbar()
        plt.show()
 ##========== Prepare kelp and ocean endmembers ==========##
    print(f"Gathering Endmembers | LT: {time.time() - time_it}")
    ocean_EM_stack = []
    kelp_EM = [459, 556, 437, 1227]

    n_bands, height, width = kelp_mask.shape
    ocean_EM_n = 0
    ocean_data = ocean_mask.reshape(ocean_mask.shape[0], -1)
    kelp_data = kelp_mask.reshape(kelp_mask.shape[0],-1)

    nan_columns = np.isnan(ocean_data).all(axis=0)  # Remove columns with nan 

    filtered_ocean = ocean_data[:, ~nan_columns]
    if(len(filtered_ocean[0,:]) < 100):
    
        print(f"Insufficient number of ocean pixels: {len(filtered_ocean[0,:])}")
        shutil.rmtree(temp_folder)
        continue

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
 ###========== Start MESMA ==========##
    print(f"Running MESMA | LT: {time.time() - time_it}")
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

    print(f"MESMA Complete, building final image | LT: {time.time() - time_it}")

    minVals = cp.nanmin(rmse, axis=2)
    PageIdx = cp.nanargmin(rmse, axis=2)
    rows, cols = cp.meshgrid(cp.arange(rmse.shape[0]), cp.arange(rmse.shape[1]), indexing='ij')
    Zindex = cp.ravel_multi_index((rows, cols, PageIdx), dims=rmse.shape)
    Mes2 = frac2.ravel()[Zindex]
    Mes2 = Mes2.T
    Mes2 = -0.229 * Mes2**2 + 1.449 * Mes2 - 0.018 #Landsat mesma corrections 
    Mes2 = cp.clip(Mes2, 0, None)  # Ensure no negative values
    Mes2 = cp.round(Mes2 * 100).astype(cp.int16)
    if show_image:
        kelp_img = cp.asnumpy(kelp_mask)
        Mes_array = cp.asnumpy(Mes2).T
        Mes_array_vis = np.where(Mes_array == 0, np.nan, Mes_array)
        kelp_vis = np.where(kelp_img == 0, np.nan, kelp_img)
        plt.figure(figsize=(20, 20), dpi=200)
        plt.imshow(rgb_nor[1,2800:3100,800:1400])
        plt.imshow(kelp_img[1,2800:3100,800:1400] , cmap='Greys', alpha=1)
        plt.imshow(Mes_array_vis[2800:3100,800:1400], alpha=1)
        plt.colorbar()
        plt.show()
    
    if save_final_data:
        num_bands = 6
        data_type = rasterio.int16
        profile = {
            'driver': 'GTiff',
            'width': width,
            'height': height,
            'count': 6,  # one band  B02, B03, B04, and B05, classified (Blue, Green, Red, and NIR).
            'dtype': data_type,  # assuming binary mask, adjust dtype if needed
            'crs': src.crs,
            'transform': src.transform,
            'nodata': 0  # assuming no data is 0
        }
        if not os.path.isdir(os.path.join(save_to_path,tile_type)):
            os.mkdir(os.path.join(save_to_path,tile_type))
        img_path = os.path.join(save_to_path,tile_type,f'{name}_processed.tif')

        # Write the land mask array to GeoTIFF
        with rasterio.open(img_path, 'w', **profile) as dst:
                dst.write(img[0].astype(data_type), 1)
                dst.write(img[1].astype(data_type), 2)
                dst.write(img[2].astype(data_type), 3)
                dst.write(img[3].astype(data_type), 4)
                dst.write(classified_img.astype(data_type), 5)
                dst.write(Mes_array.astype(data_type), 6)
                
        iterations = iterations + 1
        print(f"{iterations}/{len(results)}")

    print(f"{name} Processing complete. Processing Duration: {time.time() - time_it}")
    shutil.rmtree(temp_folder)
