import os
import geopandas as gp
import json
import csv
import earthaccess

earthaccess.login(persist=True)

field = gp.read_file('./Isla_Vista_Kelp.geojson')
bbox = tuple(list(field.total_bounds))
bbox #Display coordinate bounds
with open('./Isla_Vista_Kelp.geojson', 'r') as f:
    data = json.load(f)
# Extract the name
location = (data['name']).replace('.kmz', '').replace(' ', '_')
temporal = ("2017-06-01T00:00:00", "2018-01-01T00:00:00")

results = earthaccess.search_data(
    short_name=['HLSL30','HLSS30'],
    bounding_box=bbox,
    temporal=temporal,
    cloud_cover=.05,
    count=100
)




folder_path = os.path.join(os.getcwd(),(f'imagery/{location}'))
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
for i, granule in enumerate(results):
    if i > 2:
        break
    json_str = json.dumps(granule.__dict__)
    metadata = json.loads(json_str) 
    meta = metadata['render_dict']['meta']
    name = meta['native-id']

    #For some reason, attributes are parsed into a list in the HLS metadata. This reformats it into a dictionary.
    attributes_list = metadata['render_dict']['umm']['AdditionalAttributes']

    attributes = {attr['Name']: attr['Values'][0] for attr in attributes_list}
    if(int(attributes['CLOUD_COVERAGE']) < 10): #Reject granules with large cloud cover, for now
        continue
    time = attributes['SENSING_TIME']

    folder_name = (f'{name}')
    file_path = os.path.join(folder_path,folder_name)
    if not os.path.isdir(file_path):
        os.mkdir(file_path) #Make folder for granule 

    downloadPath = earthaccess.download(granule, local_path=file_path)
    csv_file = os.path.join(file_path, (f'{folder_name}_metadata.csv'))
    metadata_full_dict = {**attributes, **meta}
    metadata_full_dict['data_vis_url'] = granule.dataviz_links()

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metadata_full_dict.keys())
        writer.writerow(metadata_full_dict.values())