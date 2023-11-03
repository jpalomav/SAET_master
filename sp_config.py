############################ CONFIGURATION FILE #######################################

import os
from pathlib import Path

# variables declared with os.getenv(key,default=value) are exposed to change its value
# from other scripts without modify the configuration file

# CREDENTIALS FOR IMAGERY SERVERS (You must set your own credentials.) ################
os.environ['USER_ESA'] = os.getenv('USER_ESA', 'jmpalomav@gmail.com')
os.environ['PASS_ESA'] = os.getenv('PASS_ESA', 'kutuCoperDS@2018')
os.environ['USER_USGS'] = os.getenv('USER_USGS', 'jpalomav')
os.environ['PASS_USGS'] = os.getenv('PASS_USGS', 'kutuUsgs#2018')

# HOME FOLDER #########################################################################
os.environ["SAET_HOME_PATH"] = str(
    Path(os.path.dirname(os.path.realpath(__file__))))

# AUXILIARY DATA ######################################################################
# Folder for auxiliary shapefiles needed for SAET (relative paths)
os.environ['AUX_DATA_FOLDER_PATH'] = str(
    Path(os.path.join(os.getenv("SAET_HOME_PATH"), 'aux_data')))
# Names of every auxiliary shapefile
os.environ['SHP_BEACHES_PATH'] = str(
    Path(os.path.join(os.getenv("AUX_DATA_FOLDER_PATH"), 'beaches.shp')))
os.environ['SHP_LANDSAT_GRID_PATH'] = str(
    Path(os.path.join(os.getenv("AUX_DATA_FOLDER_PATH"), 'landsat_grid.shp')))
os.environ['SHP_SENTINEL2_GRID_PATH'] = str(
    Path(os.path.join(os.getenv("AUX_DATA_FOLDER_PATH"), 'sentinel2_grid.shp')))

# LOGGING #############################################################################
# Level for logging
# {NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50}
os.environ['LOG_LEVEL'] = os.getenv('LOG_LEVEL', '20')

# RESULTS #############################################################################
# Way to see the results in search mode
# {CONSOLE:0, TXT FILE:1, JSON FILE:2}
os.environ['OUT_RES'] = os.getenv('OUT_RES', '0')
