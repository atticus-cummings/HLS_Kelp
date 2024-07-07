frame_urls = frame.dataviz_links()
for num, url in enumerate(frame_urls):
    image = io.imread(url)  # Load jpg browse image into memory
    # Basic plot of the image
    plt.figure(figsize=(10,10))              
    plt.imshow(image)
    plt.show()
    print(num)


    
# Set GDAL configuration options for HTTP handling and file extensions
#gdal.SetConfigOption("GDAL_HTTP_COOKIEFILE", "../cookies.txt") # I have tried this being commented in and out
#gdal.SetConfigOption("GDAL_HTTP_COOKIEJAR", "../cookies.txt")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "TIF")
gdal.SetConfigOption("GDAL_HTTP_UNSAFESSL", "YES")
gdal.SetConfigOption('CPL_DEBUG', 'ON')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'ON')

import os
from datetime import datetime
import requests as r
import numpy as np
import pandas as pd
import geopandas as gp
from skimage import io
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import hvplot.xarray
import hvplot.pandas
import json
import panel as pn
import csv
import geoviews
import earthaccess
import string as str