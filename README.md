# SAET-Pro
SHORELINE ANALYSIS AND EXTRACTION TOOL.
**Version adapted to the new Copernicus Data Space Ecosystem (CDSE). https://dataspace.copernicus.eu**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7488654.svg)](https://doi.org/10.5281/zenodo.7488654)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## INDEX<a name="id0"></a>
1. [INTRODUCTION](#id1)
2. [WORKFLOW](#id2)
3. [REQUIREMENTS](#id3)
4. [FOLDER STRUCTURE](#id4)
5. [RUNNING SAET](#id5)
6. [OUTPUTS](#id6)
7. [CONSIDERATIONS](#id7)
8. [INSTALLATION](#id8)

## 1. INTRODUCTION<a name="id1"> </a><small>[(index)](#id0)</small>

SAET is a software for the extraction and analysis of shorelines using satellite images from the Sentinel-2 series (levels 1C and 2A) and Landsat 8 and 9 (collection 2, levels 1 and 2). It is primarily focused on studying the impact of coastal storms, allowing the determination of the shoreline position before and after the storm to evaluate its effects. Although this is its main function, SAET can also be used for temporal analysis and to study the evolution of any event (natural or anthropogenic). The main features are as follows:

- Direct access to official satellite image servers: ESA and USGS.
- Download and processing of entire scenes, allowing coverage of large areas affected by storms.
- Numerous configuration parameters (different types of water indices, segmentation thresholds, etc.) that make SAET a highly flexible software capable of being adapted to different scene conditions.
- Sub-pixel algorithm for shoreline extraction. 
- Visualization of QUICKLOOK images in HTML format.
- Updated to the new Copernicus Data Space Ecosystem (CDSE) features, allowing access to the Sentinel-2 images in a more optimised way.
- Output of the shoreline positions in shapefile format (point and line) in the WGS84 spatial reference system (EPSG: 4326).

This software has been developed as part of the H2020 EU project ECFAS (a proof-of-concept for the implementation of a European Copernicus Coastal Flood Awareness System, GA n° 101004211) by the Geo-Environmental Cartography and Remote Sensing Group (CGAT) at the Universitat Politècnica de València, Spain. It contains the core algorithm for shoreline extraction at a sub-pixel level. For detailed information on the tool and the algorithm, please refer to the following papers:

- Palomar-Vázquez, J.; Pardo-Pascual, J.E.; Almonacid-Caballer, J.; Cabezas-Rabadán, C. Shoreline Analysis and Extraction Tool (SAET): A New Tool for the Automatic Extraction of Satellite-Derived Shorelines with Subpixel Accuracy. *Remote Sens.* 2023, 15, 3198. https://doi.org/10.3390/rs15123198
- Pardo-Pascual, J.E., Almonacid-Caballer, J., Ruiz, L.A., Palomar-Vázquez, J. Automatic extraction of shorelines from Landsat TM and ETM multi-temporal images with subpixel precision. *Remote Sensing of Environment*, 123, 2012. https://doi.org/10.1016/j.rse.2012.02.024
- Pardo-Pascual, J.E., Sánchez-García, E., Almonacid-Caballer, J., Palomar-Vázquez, J.M., Priego de los Santos, E., Fernández-Sarría, A., Balaguer-Beser, Á. Assessing the Accuracy of Automatically Extracted Shorelines on Microtidal Beaches from Landsat 7, Landsat 8 and Sentinel-2 Imagery. *Remote Sensing*, 10(2), 326, 2018. https://doi.org/10.3390/rs10020326
- Sánchez-García, E., Palomar-Vázquez, J. M., Pardo-Pascual, J. E., Almonacid-Caballer, J., Cabezas-Rabadán, C., & Gómez-Pujol, L. An efficient protocol for accurate and massive shoreline definition from mid-resolution satellite imagery. *Coastal Engineering*, 103732, 2020. https://doi.org/10.1016/j.coastaleng.2020.103732
- Cabezas-Rabadán, C., Pardo-Pascual, J. E., & Palomar-Vázquez, J. Characterizing the relationship between the sediment grain size and the shoreline variability defined from sentinel-2 derived shorelines. *Remote Sensing*, 13(14), 2829. 2021. https://doi.org/10.3390/rs13142829

- CGAT: https://cgat.webs.upv.es
- ECFAS: https://www.ecfas.eu
- SAET [ECFAS E-learning module](https://youtu.be/ygLoW4z1K4w)

## Copyright and License
This open-source software is distributed under the GNU license and is the copyright of the UPV. It has been developed within the framework of the European ECFAS project by the following authors: Jesús Palomar Vázquez, Jaime Almonacid Caballer, Josep E. Pardo Pascual, and Carlos Cabezas Rabadán.

This project has received funding from the Horizon 2020 research and innovation programme under grant agreement No. 101004211

Please note that this software is designed specifically for the automatic extraction of shorelines in pre-storm and post-storm events and is not intended for massive extraction purposes.

## How to Cite SAET
To cite SAET in your research, please use the following reference:

J. Palomar-Vázquez, J. Almonacid-Caballer, J.E. Pardo-Pascual, and C. Cabezas-Rabadán (2021).
SAET (V 1.0). Open-source code. Universitat Politècnica de València. http://www.upv.es


## 2. WORKFLOW<a name="id2"> </a><small>[(index)](#id0)</small>

In this image, we can see the general workflow of the algorithm. The parameters can change depending on the expected results (see section 5).

![Alt text](https://github.com/jpalomav/SAET_master/blob/main/doc/images/workflow.jpg)


## 3. REQUIREMENTS<a name="id3"> </a><small>[(index)](#id0)</small>

The tool uses Sentinel 2, Landsat 8 and Landsat 9 images as input. In this way, the first thing needed is to have username and password from Copernicus Data Space Ecosystem (CDSE) and USGS Landsat Explorer servers:

- **Access ESA-CDSE:** before downloading S2 images, first you must obtain your credentials by registering on the website https://dataspace.copernicus.eu/. 

- **Access USGS Landsat Explorer service:** In this case, you need to do two things: to register on the Landsat Explorer website and to make a request to access the service “machine to machine” (m2m). For the first requirement, you must register on the website https://ers.cr.usgs.gov/register. Once you have your credentials, access the website https://earthexplorer.usgs.gov, and go to your profile settings. Click on the button “Go to Profile” and finally, on the option “Access Request”. There you can make a new request to the m2m service by filling out a form.
Once you have your credentials for both data source providers you can edit the file “saet_config.py” (see structure section) by changing the asterisks with your own credentials:

```
os.environ['USER_ESA'] = os.getenv('USER_ESA', '********')
os.environ['PASS_ESA'] = os.getenv('PASS_ESA', '********')
os.environ['USER_USGS'] = os.getenv('USER_USGS', '********')
os.environ['PASS_USGS'] = os.getenv('PASS_USGS', '********')
```

## 4. FOLDER STRUCTURE<a name="id4"> </a><small>[(index)](#id0)</small>

The folder SAET contains the following files and subfolders:
-	sp_config. py file that contains several configuration variables (credentials for the satellite imagery servers, folder structure for results, etc.).
-	sp_searching_run.py. Main script for searching images. It must be executed in command line mode.
-	sp_searching_functions.py. Module that contains all functions needed to run the searching algorithm.
-	sp_downloading_run.py. Main script for downloading images. It must be executed in command line mode.
-	sp_downloading_functions.py. Module that contains all functions needed to run the downloading algorithm.
-	sp_processing_run.py. Main script for processing images. It must be executed in command line mode.
-	sp_processing_functions.py. Module that contains all functions needed to run the processing algorithm.
-	sp_parameters_validation.py. Module that contains all functions needed to validate the input parameters from the command line.
-	sp_basic_functions.py. Module that contains all functions common in all modules.
-	polynomial.py. Module with extra functions for surface interpolation.
-	landsatxplore2. This folder contains the needed files for a modified version of the landsatxplore API to access to the Landsat imagery from the USGS server. This have been necessary to update this library to the new Collection 2 of Landsat, which includes Landsat-9 product.

-	examples_of_use.txt. Text file that contains several examples of use of the tool.

-	aux_data. Folder containing:
      * beaches.shp. Shapefile with all the European areas classified as beaches. Based on “Coastal uses 2018” dataset (https://land.copernicus.eu/local/coastal-zones/coastal-zones-2018). This shapefile contains the field named “BEACH_CODE” that will be copied to the final shoreline shapefile. This file is used to focus the shoreline extraction into those areas.
      * landsat_grid.shp. Shapefile including Landsat-8 footprints (https://www.usgs.gov/media/files/landsat-wrs-2-descending-path-row-shapefile). This shapefile has two fields named “PATH” and “ROW” that will be used to select a scene by its identifier.
      * sentinel2_grid.shp. Shapefile including Sentinel-2 footprints (based on the .kml file https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2/data-products).
      * map_director.qgz. Project file for QGIS with the three previous .shp files. It is useful for planning the process of downloading scenes.
      * SAET.pdf. Document explaining the tool.

-	search_data. Folder containing the output files after running the "searching" algorithm. There are two type of files: .html files, for the visualization of the image quicklooks, and .txt files containing the metadata of the found images. These text files will be used as input in the download process.


## Configuration file

The configuration file (sp_config.py) contains some parameters that allow controlling the access to the imagery servers and modify the algorithm workflow. Normally it will not be needed to modify this file apart from the credential values, but if you want to do it, you must take in account this explanation about each section:
-	Section “credentials”. **Change the asterisks characters by your own credentials to properly run SAET (see section 3).**
-	Section “home folder”. It represents the path where SAET will be installed. All other subfolders will depend on it by employing relative paths.
-	Section “auxiliary data”. Relative path to the auxiliar data needed to SAET. The name of each shapefile can be changed if it is required.
-	Section “logging”. This section should be changed only by expert users. It controls the level of messages (severity) that SAET can return. For testing and debugging purposes, set this level to 10.

## 5. RUNNING SAET<a name="id5"> </a><small>[(index)](#id0)</small>

SAET works following the next steps in order:
1. Search for images: run the script sp_searching_run.py.
2. Download images: run the script sp_downloading_run.py.
3. Process images: run the script sp_processing_run.py

One way to do this is by opening a PowerShell window or a command window (cmd). In Windows, go to the search bar and type "powershell" or "cmd". Run the script saet_run.py with parameters:
```
python sp_searching_run.py --parameter=value
```
**Note:** whether using one or the other method (powershel or cmd), it would be a good idea to open these tools as administrator.

## Parameters
* Parameters for searching images (sp_searching_run.py)

|     Parameter    	|     Description    	|     Required    	|     Usage / examples    	|     Default value    	|
|---	|---	|---	|---	|---	|
|     --help    	|     Shows   the tool help message.    	|          	|     --h   / --help    	|          	|
|     --fp    	|     Footprint   for scene searching. This parameter has two ways to be used: by   coordinates or using NONE value.           - Coordinates   as point, with longitude and latitude, separated by “comma”.      - Coordinates   as AOI (area of interest), with latitude and longitude coordinates separated   by “comma” with the format min_long, min_lat, max_long, max_lat.            -   using NONE value to avoid searching by coordinates. In this case, we can   activate the searching by scene or tile identifiers (parameters --ll and   --sl).          	|     Yes     	|     fp=0.28,39.23     fp= 0.28,39.23,0.39,39.33     fp =   NONE    	|     NONE    	|
|     --sd    	|     Start   date for searching scenes in format (YYYYMMDD).    	|     Yes     	|     --sd=20211001    	|          	|
|     --cd    	|     Central   date for searching scenes in format (YYYYMMDD). It is assumed to be the   central date of the storm.    	|     Yes     	|     --cd=20211001    	|          	|
|     --ed    	|     End   date for searching scenes in format (YYYYMMDD).    	|     Yes    	|     --ed=20211001.    	|          	|
|     --mc    	|     Maximum   percentage of cloud coverage in the scene. It must be a number between 0 and   100.    	|     No    	|     --mc=10    	|     100    	|
|     --lp    	|     Product   type for Landsat scenes. By default, the tool uses the Collection 2 (level 1)   to search Landsat-8 images (landsat_ot_c2_l1), but it also can search Landsat   8 and Landsat 9 images from Collection 2 at level 2 (landsat_ot_c2_l2). This   parameter can be set up to “NONE”, to avoid Landsat scenes processing.    	|     Yes    	|     --lp=   landsat_ot_c2_l1     --lp=   landsat_ot_c2_l2     --lp=NONE    	|          	|
|     --ll    	|     Scene   list identifiers for Landsat images. It must be a list of numbers of 6   digits. If there is more than one identifier, they must be separated by the   “comma” character. The value NONE means that the search by AOI will have   priority.    	|     Yes    	|     --ll=198032     --ll=198032,199031     --ll=NONE    	|          	|
|     --sp    	|     product   type for Sentinel-2 scenes (S2). The tool uses 1C (S2MSI1C) and 2A (S2MSI2A)   products. This parameter can be set up to “NONE”, to avoid S2 scenes   processing.    	|     Yes    	|     --sp=   S2MSI1C     --sp=   S2MSI2A     --sp=NONE    	|          	|
|     --sl    	|     Scene   list identifiers for S2. It must be a list of alphanumeric characters (named   “tile” identifier) composed of two numbers and three capital letters. If   there is more than one identifier, they must be separated by the “comma”   character. The value NONE means that the search by AOI will have priority.    	|     Yes    	|     --sl=30TYJ     --sl=30TYJ,30TYK     --sl=NONE    	|          	|
|     --so    	|     This   parameter only affects S2 images. It allows filtering of images containing areas   with no data values. The value 0 indicates that all images will be returned. The   value 1 indicates that images containing areas with no data values will be   filtered.    	|     No    	|     --so=0     --so=1    	|     1    	|

* Parameters for downloading images (sp_downloading_run.py)

|     Parameter    	|     Description    	|     Required    	|     Usage / examples    	|     Default value    	|
|---	|---	|---	|---	|---	|
|     --help    	|     Shows   the tool help message.    	|          	|     --h   / --help    	|          	|
|     --sp    	|     Satellite   platform. Allows choosing between S2 (‘s’ value) and L8-9 images (‘l’ value).       	|     Yes    	|     --sp=s     --so=l    	|          	|

* Parameters for processing images (sp_processing_run.py)

|     Parameter    	|     Description    	|     Required    	|     Usage / examples    	|     Default value    	|
|---	|---	|---	|---	|---	|
|     --wi    	|     Water   index type. SAET supports these indices: aweinsh, aweish, mndwi, kmeans   (K-means it is not a water index, but also leads to a classification mask. In   this case it is not needed a threshold value).    	|     No    	|     --wi=aweinsh     --wi=aweish     --wi=mndwi     --wi=kmeans    	|     aweinsh    	|
|     --th    	|     Threshold   method to obtain the water-land mask from the water index. SAET supports   three methods: standard 0 value, Otsu bimodal and Otsu multimodal with three   classes. These methods are applied for all type of index except kmeans.    	|     No    	|     --th=0   (standard 0 value)     --th=1   (Otsu bimodal)     --th=2   (Otsu multimodal)    	|     0    	|
|     --mm    	|     Morphological   method. To generate the shoreline at pixel level (SPL) from the water-land   mask. SAET can apply two methods: erosion and dilation.    	|     No    	|     --mm=erosion     --mm=dilation    	|     dilation    	|
|     --cl    	|     Cloud   masking severity. This parameter controls what kind of clouds will be used to   mask the SPL. SAET supports three levels of severity: low (SAET don’t use   cloud mask), medium (only opaque clouds) and high (opaque clouds, cirrus, and   cloud shadows).     Note:   Landsat   8-9 Collection 2 and Sentinel-2 use algorithms to classify clouds. SAET uses   these classification layers. You must assume that sometimes this   classification can fail. This will directly affect the result.    	|     No    	|     --cl=0   (low)     --cl=1   (medium)     --cl=2   (high)    	|     0    	|
|     --ks    	|     Kernel   size. The main algorithm for shoreline extraction uses a kernel analysis over   each pixel in the SPL. Users can control this size, choosing between 3 or 5   pixels.    	|     No    	|     --ks=3     --ks=5    	|     3    	|
|     --bc    	|     Beach   code list to filter the extraction process for a group of beaches. This code   is related to the “BEACH_CODE” field in the shapefile “Beaches.shp”. The   default value is NONE, which means that all beaches in the image will be   processed. In case you want process a group of beaches, you must indicate a   list of codes, separated by “comma”. If some code in the list is not correct,   it will not be considered. If all codes are incorrect, all beaches will be   processed.    	|     No    	|     --bc=1683     --bc=1683,2485,758     --bc=NONE    	|     NONE    	|

This is the text of help that appears when you run SAET for searching with the --h parameter (python sp_searching_run.py --h):
```
usage: sp_searching_run.py [-h] --fp FP --sd SD --cd CD --ed ED [--mc [0-100]] --lp
                           {landsat_ot_c2_l1,landsat_ot_c2_l2,NONE} --ll LL --sp {S2MSI1C,S2MSI2A,NONE} --sl SL
                           [--so [0-1]]

optional arguments:
  -h, --help            show this help message and exit
  --fp FP               Coordinates long/lat in these formats: (POINT) fp=long,lat; (AOI)
                        fp=min_long,min_lat,max_long,max_lat. Default: NONE
  --sd SD               Start date for searching scenes (YYYYMMDD). --sd=20210101. Default:20200101
  --cd CD               Central date for storm (YYYYMMDD). --sd=20210101. Default:20200102
  --ed ED               End date for searching scenes (YYYYMMDD). --sd=20210101. Default:20200103
  --mc [0-100]          maximum cloud coverture for the whole scene [0-100]. --mc=10. Default 100
  --lp {landsat_ot_c2_l1,landsat_ot_c2_l2,NONE}
                        Landsat 8 product type. landsat_ot_c2_l1 or landsat_ot_c2_l2 or NONE. Default:
                        landsat_ot_c2_l1
  --ll LL               List of scenes for Landsat 8 (number of 6 digits). --ll=198032,199031. Default: NONE
  --sp {S2MSI1C,S2MSI2A,NONE}
                        Sentinel 2 product type (S2MSI1C / S2MSI2A). --s2=S2MSI1C / --s2=S2MSI2A / NONE. Default:
                        S2MSI1C
  --sl SL               List of scenes for Sentinel 2 (string of 5 characters). --sl=31TCF,30TYK. Default: NONE
  --so [0-1]            Exclude images with NO DATA values [0-1]. --so=1. Default: 1
```

## Examples of use

* Searching for all Sentinel-2 (level 1C) scenes inside an area of interest (tile 30SYJ), with less than 15% of cloud coverage and within the date range from 01-04-2023 to 30-04-2023 (central date or storm peak 15-04-2023):
```
python sp_searching_run.py --fp=NONE --sd=20230401 --cd=20230415 --ed=20230430 --mc=15 --lp=NONE --ll=NONE --sp=S2MSI1C --sl=30SYJ

2023-10-19 15:39:22,620 INFO Starting searching SAET_pro algorithm...

[0] Scene: S2B_MSIL1C_20230420T104619_N0509_R051_T30SYJ_20230420T125145 Cloud coverage: 9.2% 5 days
[*******] Central date:20230415
[1] Scene: S2A_MSIL1C_20230405T105031_N0509_R051_T30SYJ_20230405T160934 Cloud coverage: 0.01% -10 days


2023-10-19 15:39:23,218 INFO SAET_pro searching algorithm have finished successfully.
```
Results show a list with the found images with the identifier (number of order in the list), the name, the cloud coverage percentage, and the difference in days between the central date and de image date. Besides, an .html file with the quicklook images is opened automatically in the default browser. 

**Note:** sometimes firefox may experiment problems showing quicklooks. If this is the case, try chrome as default browser.

Finally, as result of the searching process a .txt file called "search_result_s2.txt" has been created. This file contains the metadata of every S2 image found. In this case, this is the content for this file:

```
{"9c7a0978-7606-4025-9243-d9b2c8c53134": {"id": "9c7a0978-7606-4025-9243-d9b2c8c53134", "name": "S2B_MSIL1C_20230420T104619_N0509_R051_T30SYJ_20230420T125145.SAFE", "online": true, "quicklook": "https://catalogue.dataspace.copernicus.eu/odata/v1/Assets(92324096-429d-4c46-997e-e3770bddc86c)/$value", "corners": 5, "cloud_cover": 9.19609423989967, "days_off": 5}, "9a9002e1-2f86-4dd6-8340-8697c1837e25": {"id": "9a9002e1-2f86-4dd6-8340-8697c1837e25", "name": "S2A_MSIL1C_20230405T105031_N0509_R051_T30SYJ_20230405T160934.SAFE", "online": true, "quicklook": "https://catalogue.dataspace.copernicus.eu/odata/v1/Assets(04191c31-23ed-4f6c-8475-ee375e4babaa)/$value", "corners": 5, "cloud_cover": 0.0117351966317298, "days_off": -10}}
```

Once the searching results have been obtained, we can decide what images we want to download by runing the script 'sp_downloading_run.py':

```
python sp_downloading_run.py --sp=s
2023-10-19 16:12:53,973 INFO Starting downloading SAET_pro algorithm...

[0] Scene: S2B_MSIL1C_20230420T104619_N0509_R051_T30SYJ_20230420T125145.SAFE Cloud coverage: 9.19609423989967 5 days
[1] Scene: S2A_MSIL1C_20230405T105031_N0509_R051_T30SYJ_20230405T160934.SAFE Cloud coverage: 0.0117351966317298 -10 days

Number of images to be downloaded (* / 0,2,3 / [2-5])?:
```
**Note:** We can download all images (*), a list of them (0,2,3) or an interval of them ([2-5]). If we type 'ENTER' the scripts will finish. 


Taking into account the metadata information along with the quicklooks images, we can decide which number of images (identifiers) will be downloaded.

Once we have downloaded one or more images, we can process them by using the script 'sp_processing_run.py'. In this example, we will use all the default parameters. As result, the script will display a list with all images previously downloaded and stored in the folder '\data', and it will also allow to select the images to be processed, either S2 or L8-9.

```
python sp_processing_run.py
2023-10-19 17:24:36,247 INFO Starting downloading SAET_pro algorithm...

List of scenes in the data folder:

[0] S2A_MSIL1C_20230823T104631_N0509_R051_T30SXG_20230823T143001
[1] LC08_L1TP_198032_20230825_20230905_02_T1

Number of images to be reprocessed (* / 0,2,3 / [2-5])?:
```
**Note:** We can download all images (*), a list of them (0,2,3) or an interval of them ([2-5]). If we type 'ENTER' the scripts will finish.

**Note:** more examples can be found in the file “examples_of_use.txt”.

## Workflow

Next picture shows the workflow to run SAET in the most convenient way. The recommendation is:
* Select your area of analysis and product of interest. The file map_director.qgz (QGIS) will be very useful to decide which scene (Landsat) or tile (Sentinel-2) will be used.
* Always start with the script 'sp_searching_run.py'.
* Analyse the found images by checking their metadata and quicklooks.
* Select the images to be downloaded based on the previous analysis and run the script "sp_downloding_run.py".
* Process or reprocess any previously downloaded image, changing the needed parameters according to the user's needs by running the script "sp_processing_run.py".

<p align="center">
     <img src="https://github.com/jpalomav/SAET_master/blob/main/doc/images/run_workflow.jpg">
</p>

## 6. OUTPUTS<a name="id6"> </a><small>[(index)](#id0)</small>

After running the tool, a new structure of folders will be created inside the SAET installation folder. Every time SAET is run, new folders are added to the structure. This structure can be similar as follows:

<p align="center">
     <img src="https://github.com/jpalomav/SAET_master/blob/main/doc/images/outputs.jpg">
</p>

- The “ouput_data” folder will be created if it does not exist. Inside, “data”, “sds” and “search_data” folders will be created. “data” folder contains subfolders to download and process every scene. 
- Every type of image (L8, L9 or S2) is stored in its own folder, which is named as the name of the original image (scene folder). The scene folder contains all needed bands and auxiliary files (metadata, cloud mask, etc.). 
- The “temp” folder is where all intermediate output files will be stored. 
- Results (shorelines) will be stored into the “sds” folder, inside of scene folders, and will contain two versions of the shoreline in shapefile format: line format (*_lines.shp) and point format (*_points.shp). Shapefile shorelines are stored in the World GEodetic System 1984 (WGS84 - EPSG:4326). 
- “search_data” folder contains an .html file with the different thumbnails corresponding with the found images and their metadata (.txt files).

<p align="center">
     <img src="https://github.com/jpalomav/SAET_master/blob/main/doc/images/results_html.jpg">
</p>

- The “temp” folder contains some intermediate files that may be interesting review in case we do not obtain the expected results:
    * bb300_r.shp: shapefile containing the beaches file (in WGS84 geographic coordinates) reprojected to the coordinate reference system of the scene.
    * clip_bb300_r.shp: previous shapefile cropped by the scene footprint.
    * bb300_r.tif: previous file converted to binary raster (pixels classified as beach have the code 1).
    * scene_footprint.shp: shapefile containing the footprint polygon of the scene.
    * *_wi.tif: raster file containing the computed water index.
    * *_cmask.tif: raster file containing the binary mask of the cloud coverage (pixels classified as clouds, cirrus or cloud shadows have the code 1).
    * pl.tif: raster file containing the binary mask representing the extracted shoreline at pixel level (pixels classified as shoreline have the code 1).
    * *_B11.shp (for Sentinel-2) or *_B6.shp (for Landsat 8-9): shapefile containing the extracted shoreline in point vector format, without having been processed by the cleaning algorithm.
    * *_cp.shp: shapefile containing the extracted shoreline in point vector format, once it has been processed by the cleaning algorithm. This folder will be copied to the "SDS" folder by changing the prefix "_cp" to "_points", in both shapefile and json format.
    * *_cl.shp: shapefile containing the extracted shoreline in line vector format, once it has been processed by the cleaning algorithm. This folder will be copied to the "SDS" folder by changing the prefix "_cl" to "_lines", in both shapefile and json format.

## 7. CONSIDERATIONS<a name="id7"> </a><small>[(index)](#id0)</small>

-	This tool downloads one or more L8, L9 or S2 scenes, and it downloads the whole scene. Althought it is a reasonably fast process, sometimes downlading process can be a bit slow, depending of the user's bandwidth and the server status. 

-	L8 and L9 products are only available from Collection 2. In USGS servers, Collection 1 are not available anymore as of December 30, 2022 (https://www.usgs.gov/landsat-missions/landsat-collection-1).

-	The algorithm uses the cloud mask information. For L8-9, this information is stored in a .tif file, whereas for S2, it depends on the product (.gml format for the product 1C, and .tif format for the product 2A). This situation can change in the next months and some changes may be needed (see https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming).

-	The shapefiles inside the folder “aux_data” are mandatory to make the tool work. If modifications (removing, updating) in the “beaches.shp” file are needed do not forget to maintain the field “BEACH_CODE” with unique identifiers.

-	The final shoreline is provided in two versions (line and point) and it has the field “BEACH_CODE” to facilitate the subsequent analysis by comparing the same beach section on different dates.

-	One good way to begin using the tool is trying to see what are the L8-9 or S2 scenes that we are interested in. For this goal, we can use the grid shapefiles for L8-9 or S2 ('map_director.qgz' in the folder 'aux_data') and other online viewers, like “OE Browser” (https://apps.sentinel-hub.com/eo-browser). In this website we can see the needed products, their footprints, and their available dates and cloud coverage. Once we know this information, we can use this as parameter in the script 'sp_searching_run.py'. Anyway, the visualization of the quicklooks will help you to decide the best images for your purposes. On the contrary, if we search for images using coordinates, especially in S2 scenes , the algorithm can retrieve more scenes than are needed due to the fact that the AOI can overlap with more than one S2 footprint.

- If we request the most recent Sentinel-2 images, could be possible that we only have access to the 1C product (2A product is not immediately available, and it is needed to spend some time to have access to this product). On the other hand, we also need to consider that 1C product has a cloud mask of lower quality than 2A product.

## 8. INSTALLATION<a name="id8"> </a><small>[(index)](#id0)</small>

SAET has been developed in python and has been tested for the python version 3.9.7 (64 bits). You can install this version from by installing the file “Windows installer (64-bit)” form the link https://www.python.org/downloads/release/python-397. SAET needs some extra libraries to work. In the file “requirements_readme.txt” we can see the minimum versions of these libraries and some notes about the GDAL library:

|     Package    |     Version    |     Description    |
|---|---|---|
|     Python-dateutil    |     2.8.2    |     Functions   to extend the standard datetime module    |
|     Requests    |     2.26.0    |     Library used to manage http requests    |
|     Tqdm    |     4.62.2    |     Library used to manage progress bars    |
|     Numpy    |     1.21.2    |     Numeric   package    |
|     Matplotlib    |     3.4.3    |     Visualization   library    |
|     GDAL    |     3.3.1    |     Geospatial   Data Abstraction Library for raster geospatial data formats.    |
|     Shapely    |     1.7.1    |     Library   to manage shapefiles    |
|     Pyshp    |     2.1.3    |     Library   to manage shapefiles    |
|     Scikit-image    |     0.18.3    |     Image   processing library    |
|     Scikit-learn    |     1.0.2    |     Library   for data analysis and classification    |
|     Scipy    |     1.7.1    |     Scientific   computing library    |
|     Networkx    |     2.6.2    |     Library   for managing and analysing networks    |

The easier way to install SAET to avoid conflicts with other libraries already installed in your python distribution is to create a virtual environment. Virtual environments are used to isolate the installation of the needed libraries for each project, avoiding problems among different versions of the same library. Therefore, is the most recommended method.

## Virtual environment creation and installation of SAET (recommended)

**Note:** You can find a detailed document in this [step-by-step pdf](https://github.com/jpalomav/SAET-Pro/blob/main/doc/tutorials/saet_pro_installation_step_by_step.pdf)

Once you have installed python (for example in “c:\python397_64”), follow the next steps (on Windows):
1. Open a command prompt window.
2. In this window, install the library “virtualenv” by typing 'pip install virtualenv'.
3. Close the command prompt window.
4. Create a new folder called "SAET_installation" (the name does not matter) in whatever location (for example 'c:\SAET_installation').
5. Copy all SAET files into this folder.
6. Open a new command prompt window and change the current folder to the SAET installation folder (type 'cd C:\SAET_installation')
7. In the command prompt type: 'c:\Python397_64\Scripts\virtualenv saet_env' ("saet_env" is the name of a new virtual environment). This will create a new folder named “saet_env”.
8. Activate the new virtual environment by typing: 'saet_env\Scripts\activate'.
9. Install all needed libraries one by one typing 'pip install -r requirements_windows.txt' (for windows), or 'pip install -r requirements_linux.txt' (for linux).
10. **Change your credentials in the file “sp_config.py”.**

To check if SAET has been correctly installed, type the next sentence in the command prompt window:
```
python sp_searching_run.py --h
```

If you have any problems with this way of installation, remove the virtual environment, create it again and try to install the libraries one by one manually. In the file “requirements_readme.txt” we can see the versions of these libraries and some notes about the GDAL library. 
To remove the virtual environment, follow the next steps:

If you have any problems with this way of installation, remove the virtual environment, create it again and try to install the libraries one by one manually. In the file “requirements_readme.txt” we can see the minimum versions of these libraries and some notes about the GDAL library. 
To remove the virtual environment, follow the next steps:
1. Close your command prompt window
2. Delete the folder containing the virtual environment (in this case, the folder “saet_env”)
3. Repeat the steps 7 and 8 to create and activate again the virtual environment.
4. Try the manual installation of each single library typing pip install (library_name)==(library_version). Example:  pip install numpy==1.21.2. **It is recommendable to do the manual installation in the same order as you can see in the table of libraries in the section 8.**

**Important note for manual installation:**

GDAL installation with pip command can be problematic. If errors occur during the installation of GDAL, so try to install the corresponding wheel file, according to your operative system (Windows or Linux).
These files are in the folder "gdal_python_wheels":
- Windows: GDAL-3.3.3-cp39-cp39-win_amd64.whl
- Linux: GDAL-3.4.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl

The installation can be done using the pip command. The example for Windows would be like that: 

```
pip install ./gdal_python_wheels/GDAL-3.3.3-cp39-cp39-win_amd64.whl
```

## Virtual environment creation and installation of SAET on Linux
1. Open a new terminal
2. Type 'pip3 install virtualenv'
3. Close the terminal
4. Go to the SAET installation folder
5. Open a new terminal in this folder
6. Type 'virtualenv saet_env' (“saet_env” is the name of the virtual environment).
7. Activate this new virtual environment typing 'source saet_env/bin/activate'
8. Install the libraries typing 'pip3 install -r requirements_linux.txt'
9. Change your credentials in the file “sp_config.py”
10. Type 'python3 sp_searching_run.py --h'
