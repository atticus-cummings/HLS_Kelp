#Import libraries
import os
import pandas as pd
import geopandas as gp
from skimage import io
import matplotlib.pyplot as plt
from osgeo import gdal
import json
import csv
import earthaccess
from contextlib import redirect_stdout

earthaccess.login(persist=True)

field = gp.read_file('./maps/Isla_Vista_Kelp.geojson')
bbox = tuple(list(field.total_bounds))
bbox #Display coordinate bounds
with open('./maps/Isla_Vista_Kelp.geojson', 'r') as f:
    data = json.load(f)
# Extract the name
location = (data['name']).replace('.kmz', '').replace(' ', '_')
temporal = ("2017-03-01T00:00:00", "2024-7-01T00:00:00") #
location
dem_name = 'dem.tif'

results = earthaccess.search_data(
    short_name=['HLSL30','HLSS30'],
    bounding_box=bbox,
    temporal=temporal,
     cloud_cover=.15,
    count=-1
)

dem_results = earthaccess.search_data(
    short_name="ASTGTM",
    bounding_box=bbox)

print(results[0])
print(dem_results[0])


folder_path = os.path.join(os.getcwd(),(f'imagery/{location}'))
if not os.path.isfile(os.path.join(folder_path, dem_name)):
        dem_path = earthaccess.download(dem_results[0], local_path=folder_path)
        os.rename(dem_path[0], os.path.join(folder_path,'dem.tif'))
        os.rename(dem_path[1], os.path.join(folder_path, 'num.tif'))
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
for i, granule in enumerate(results):
    #if i > 20:
    #    break
    json_str = json.dumps(granule.__dict__)
    metadata = json.loads(json_str) 
    meta = metadata['render_dict']['meta']
    name = meta['native-id']

    #For some reason, attributes are parsed into a list in the HLS metadata. This reformats it into a dictionary.
    attributes_list = metadata['render_dict']['umm']['AdditionalAttributes']

    attributes = {attr['Name']: attr['Values'][0] for attr in attributes_list}
    print(attributes['MGRS_TILE_ID'])
    if not attributes['MGRS_TILE_ID'] == '11SKU':
        pass

    if(int(attributes['CLOUD_COVERAGE']) > 50): #Reject granules with large cloud cover, for now
        pass
    time = attributes['SENSING_TIME']
    tile_folder = attributes['MGRS_TILE_ID']
    tile_path = os.path.join(folder_path,tile_folder)
    if not os.path.isdir(tile_path):
         os.mkdir(tile_path)
    folder_name = (f'{name}')
    file_path = os.path.join(tile_path,folder_name)
    if not os.path.isdir(file_path):
        os.mkdir(file_path) #Make folder for granule 
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        downloadPath = earthaccess.download(granule, local_path=file_path, threads=10)

    csv_file = os.path.join(file_path, (f'{folder_name}_metadata.csv'))
    metadata_full_dict = {**attributes, **meta}
    metadata_full_dict['data_vis_url'] = granule.dataviz_links()

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metadata_full_dict.keys())
        writer.writerow(metadata_full_dict.values())

