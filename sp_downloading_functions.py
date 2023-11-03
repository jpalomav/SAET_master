# *******************************************************************************
# SECTION: IMPORTS
# *******************************************************************************
import os
import pathlib
import logging
import sys
import json
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
import tarfile
import zipfile
import time
import shutil

from sp_parameters_validation import esaAuthentication, usgsAuthentication

# *******************************************************************************
# SECTION: COMMOM FUNCTIONS
# *******************************************************************************


def parseNumberProductsS2(np, scene_list):
    pattern_1 = '*'
    pattern_2 = r'^(?:[1-9]\d*|0)(?:,[1-9]\d*|0)*$'
    pattern_3 = r'^[1-9]\d*-[1-9]\d*$'
    scene_list = list(scene_list.values())
    if np == pattern_1:
        return scene_list
    if re.match(pattern_2, np):
        np_list = [int(n) for n in np.split(',')]
        filtered_scenes = [scene_list[i]
                           for i in np_list if i >= 0 and i <= len(scene_list)]
        return filtered_scenes
    if re.match(pattern_3, np):
        np_tmp = [int(n) for n in np.split('-')]
        if np_tmp[0] > np_tmp[1]:
            np_list = list(range(np_tmp[1], np_tmp[0]+1))
        else:
            np_list = list(range(np_tmp[0], np_tmp[1]+1))
        filtered_scenes = [scene_list[i]
                           for i in np_list if i >= 0 and i <= len(scene_list)]
        return filtered_scenes
    logging.warning('Incorrect format: (* / 0,2,3 / [2-5])'+'\n')
    sys.exit(1)


def untarFile(filepath, outpath):
    '''
    Description:
    ------------
    Untars .tar and .tar.gz files to an specific output folder.
    Removes the .tar or .tar.gz file after uncompress process

    Arguments:
    ------------
    - filepath (string): path to the .zip file
    - outpath (string): path to the output folder

    Returns:
    ------------
    None

    '''
    with tarfile.open(filepath) as file:
        file.extractall(outpath)
    os.remove(filepath, dir_fd=None)


def unzipFile(filepath, outpath):
    '''
    Description:
    ------------
    Unzips .zip files to an specific output folder.
    Removes the .zip file after uncompress process

    Arguments:
    ------------
    - filepath (string): path to the .zip file
    - outpath (string): path to the output folder

    Returns:
    ------------
    None

    '''

    with zipfile.ZipFile(filepath) as file:
        file.extractall(outpath)
    os.remove(filepath, dir_fd=None)


# *******************************************************************************
# SECTION: SENTINEL-2 DONWNLOADING FUNCTIONS
# *******************************************************************************

def filterScenesS2ToDownload(scene_list):
    for i, scene in enumerate(scene_list.values()):
        print(
            f"{[i]} Scene: {scene['name']} Cloud coverage: {scene['cloud_cover']} {scene['days_off']} days")
    print('')
    np = input('Number of images to be downloaded (* / 0,2,3 / [2-5])?: ')
    filtered_scenes = parseNumberProductsS2(np, scene_list)
    return filtered_scenes


def parse_manifest_xml(xml):
    '''
    Description:
    ------------
    Converts manifest xml content to a dictionary structure

    Arguments:
    ------------
    - xml (string): manifest content in xml format

    Returns:
    ------------
    - outputs (list): list with manifest content in a dictionary structure)

    '''

    outputs = []
    root = ET.fromstring(xml)
    for item in root.findall("./dataObjectSection/dataObject"):
        output = {
            'id': item.get('ID'),
            'mimetype': item.find('./byteStream').get('mimeType'),
            'size': int(item.find('./byteStream').get('size')),
            'href': item.find('./byteStream/fileLocation').get('href'),
            'md5sum': item.find('./byteStream/checksum').text
        }
        outputs.append(output)
    return outputs


def downloadSingleBandS2(url, band_name, session, output_folder, product_name):
    # print(url)
    response = session.get(url, allow_redirects=False)
    while response.status_code in (301, 302, 303, 307):
        url = response.headers["Location"]
        response = session.get(url, allow_redirects=False)
    file = session.get(url, verify=False, allow_redirects=True, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    pack_size = 8192  # try 1024, 2048, 4096 and 8192
    # Save the product in home directory
    progress = tqdm(file.iter_content(
        pack_size), f"Downloading {band_name}", total=file_size, unit="B", unit_scale=True, unit_divisor=pack_size)
    folder_path = os.path.join(
        output_folder, product_name.replace('.SAFE', ''))
    os.makedirs(folder_path, exist_ok=True)
    img_path = os.path.join(folder_path, band_name)
    with open(img_path, "wb") as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))


def unZippingS2Safe(zip_path):
    #print('Unzipping... '+zip_path)
    output_folder = os.path.dirname(zip_path)
    if 'MSIL1C' in zip_path:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if ('IMG_DATA' in file_info.filename) and (file_info.filename.endswith('.jp2')):
                    # print(file_info.filename)
                    with zip_ref.open(file_info.filename) as source_file:
                        target_file_path = os.path.join(
                            output_folder, os.path.basename(file_info.filename))
                        with open(target_file_path, 'wb') as target_file:
                            shutil.copyfileobj(source_file, target_file)
                if 'QI_DATA' in file_info.filename:
                    if 'MSK_CLASSI_B00' in file_info.filename:
                        # print(file_info.filename)
                        with zip_ref.open(file_info.filename) as source_file:
                            target_file_path = os.path.join(
                                output_folder, os.path.basename(file_info.filename))
                            with open(target_file_path, 'wb') as target_file:
                                shutil.copyfileobj(source_file, target_file)
                    if 'MSK_CLOUDS_B00' in file_info.filename:
                        # print(file_info.filename)
                        with zip_ref.open(file_info.filename) as source_file:
                            target_file_path = os.path.join(
                                output_folder, os.path.basename(file_info.filename))
                            with open(target_file_path, 'wb') as target_file:
                                shutil.copyfileobj(source_file, target_file)
    if 'MSIL2A' in zip_path:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if ('R20m' in file_info.filename) and (file_info.filename.endswith('.jp2')):
                    # print(file_info.filename)
                    with zip_ref.open(file_info.filename) as source_file:
                        target_file_path = os.path.join(
                            output_folder, os.path.basename(file_info.filename))
                        with open(target_file_path, 'wb') as target_file:
                            shutil.copyfileobj(source_file, target_file)
                if ('R10m' in file_info.filename) and ('_B08_' in file_info.filename):
                    # print(file_info.filename)
                    with zip_ref.open(file_info.filename) as source_file:
                        target_file_path = os.path.join(
                            output_folder, os.path.basename(file_info.filename))
                        with open(target_file_path, 'wb') as target_file:
                            shutil.copyfileobj(source_file, target_file)

    os.remove(zip_path, dir_fd=None)


def downloadWholeImage(url, session, headers, output_folder, product_name):
    # response = session.get(url, allow_redirects=False)
    # while response.status_code in (301, 302, 303, 307):
    #     url = response.headers["Location"]
    #     response = session.get(url, allow_redirects=False)
    # file = session.get(url, verify=False, allow_redirects=True, stream=True)
    response = session.get(url, headers=headers, stream=True)
    pack_size = 8192  # try 1024, 2048, 4096 and 8192
    file_size = int(response.headers.get("Content-Length", 0))
    product_name = product_name.replace('.SAFE', '')
    folder_path = os.path.join(
        output_folder, product_name)
    os.makedirs(folder_path, exist_ok=True)

    zip_path = os.path.join(folder_path, f"{product_name}.zip")
    progress = tqdm(response.iter_content(
        pack_size), f"Downloading {os.path.basename(zip_path)}", total=file_size, unit="B", unit_scale=True, unit_divisor=pack_size)
    with open(zip_path, "wb") as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))
    # uncompress the file
    unZippingS2Safe(zip_path)


def downloadS2CDSE(run_parameters):

    # output data folder and output search folder
    output_data_folder = run_parameters['output_data_folder_s2']
    output_search_folder = run_parameters['output_search_folder']

    # base URL of the product catalogue
    catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    zipper_odata_url = 'https://zipper.dataspace.copernicus.eu/odata/v1'

    # single bands or whole image downloading
    single_bands = False

    # bands dictionaries and band selection

    # 'bands_1c': ['B02', 'B03', 'B08', 'B11', 'B12', 'QA60', 'CPM']
    bands_info = {'bands_1c': ['B02', 'B03', 'B08', 'B11', 'B12', 'QA60', 'CPM'],
                  'band_dict_1c': {'B01': 'IMG_DATA_Band_60m_1_Tile1_Data',
                                   'B02': 'IMG_DATA_Band_10m_1_Tile1_Data',
                                   'B03': 'IMG_DATA_Band_10m_2_Tile1_Data',
                                   'B04': 'IMG_DATA_Band_10m_3_Tile1_Data',
                                   'B05': 'IMG_DATA_Band_20m_1_Tile1_Data',
                                   'B06': 'IMG_DATA_Band_20m_2_Tile1_Data',
                                   'B07': 'IMG_DATA_Band_20m_3_Tile1_Data',
                                   'B08': 'IMG_DATA_Band_10m_4_Tile1_Data',
                                   'B09': 'IMG_DATA_Band_60m_2_Tile1_Data',
                                   'B10': 'IMG_DATA_Band_60m_3_Tile1_Data',
                                   'B11': 'IMG_DATA_Band_20m_5_Tile1_Data',
                                   'B12': 'IMG_DATA_Band_20m_6_Tile1_Data',
                                   'B8A': 'IMG_DATA_Band_20m_4_Tile1_Data',
                                   'TCI': 'IMG_DATA_Band_TCI_Tile1_Data',
                                   'QA60': 'FineCloudMask_Tile1_Data',
                                   'CPM':  'ClassiPixelsMask_Band_00m_0_Tile1_Data'},
                  # 20 m resolution bands
                  'bands_2a': ['B02', 'B03', 'B08', 'B11', 'B12', 'SCL'],
                  'band_dict_2a': {'B01': 'IMG_DATA_Band_AOT_20m_Tile1_Data',
                                   'B02': 'IMG_DATA_Band_B02_20m_Tile1_Data',
                                   'B03': 'IMG_DATA_Band_B03_20m_Tile1_Data',
                                   'B04': 'IMG_DATA_Band_B04_20m_Tile1_Data',
                                   'B05': 'IMG_DATA_Band_B05_20m_Tile1_Data',
                                   'B06': 'IMG_DATA_Band_B06_20m_Tile1_Data',
                                   'B07': 'IMG_DATA_Band_B07_20m_Tile1_Data',
                                   'B08': 'IMG_DATA_Band_B08_10m_Tile1_Data',
                                   'B8A': 'IMG_DATA_Band_B8A_20m_Tile1_Data',
                                   'B11': 'IMG_DATA_Band_B11_20m_Tile1_Data',
                                   'B12': 'IMG_DATA_Band_B12_20m_Tile1_Data',
                                   'SCL': 'IMG_DATA_Band_SCL_20m_Tile1_Data',
                                   'TCI': 'IMG_DATA_Band_TCI_20m_Tile1_Data'}}

    # load text file with metadata for the found images in searching step
    txt_file = str(pathlib.Path(os.path.join(
        os.getcwd(), output_search_folder, 'search_result_s2.txt')))
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as txt:
            scene_list = json.load(txt)

        # authentication
        access_token = esaAuthentication(
            run_parameters['user_esa'], run_parameters['pass_esa'])

        # # Establish session
        session = requests.Session()
        headers = {"Authorization": f"Bearer {access_token}"}
        session.headers.update(headers)
        #session.headers["Authorization"] = f"Bearer {access_token}"

        # selection of scenes to be downloaded
        scene_list = filterScenesS2ToDownload(
            scene_list)  # list of dictionaries

        for scene in scene_list:
            product_identifier = scene['id']
            product_name = scene['name']
            if single_bands:
                url = f"{catalogue_odata_url}/Products({product_identifier})/Nodes({product_name})/Nodes(manifest.safe)/$value"
                #print(f'manifest: {url}')
                response = session.get(url, allow_redirects=False)
                while response.status_code in (301, 302, 303, 307):
                    url = response.headers["Location"]
                    response = session.get(url, allow_redirects=False)

                file = session.get(url, verify=False, allow_redirects=True)
                manifest = parse_manifest_xml(file.content)

                if 'MSIL1C' in product_name:
                    bands = bands_info['bands_1c']
                    band_dict = bands_info['band_dict_1c']
                if 'MSIL2A' in product_name:
                    bands = bands_info['bands_2a']
                    band_dict = bands_info['band_dict_2a']

                for band in bands:
                    # url generation for single band
                    band_id = band_dict[band]
                    file_info = [
                        file_info for file_info in manifest if file_info['id'] == band_id]
                    if len(file_info) == 1:
                        file_info = file_info[0]
                        hrf = file_info['href'].split('/')[1:]
                        url = f"{catalogue_odata_url}/Products({product_identifier})/Nodes({product_name})/Nodes({hrf[0]})/Nodes({hrf[1]})/Nodes({hrf[2]})/Nodes({hrf[3]})/$value"
                        downloadSingleBandS2(
                            url, hrf[3], session, output_data_folder, product_name)
                    time.sleep(3)
            else:
                url = f"{zipper_odata_url}/Products({product_identifier})/$value"
                print(url)
                downloadWholeImage(
                    url, session, headers, output_data_folder, product_name)
                time.sleep(3)

    else:
        logging.warning('Txt file of S2 results not found.'+'\n')
        sys.exit(1)


def downloadS2CDSE2(run_parameters):
    # output data folder and output search folder
    output_data_folder = run_parameters['output_data_folder_s2']
    output_search_folder = run_parameters['output_search_folder']

    # base URL
    zipper_odata_url = 'https://zipper.dataspace.copernicus.eu/odata/v1'
    txt_file = str(pathlib.Path(os.path.join(
        os.getcwd(), output_search_folder, 'search_result_s2.txt')))
    if os.path.exists(txt_file):
        # loading found scenes in searching (sp_searching_run.py)
        with open(txt_file, 'r') as txt:
            scene_list = json.load(txt)

        # authentication
        access_token = esaAuthentication(
            run_parameters['user_esa'], run_parameters['pass_esa'])

        # # Establish session and authentication
        session = requests.Session()
        headers = {"Authorization": f"Bearer {access_token}"}
        session.headers.update(headers)

        # selection of scenes to be downloaded
        scene_list = filterScenesS2ToDownload(
            scene_list)  # list of dictionaries

        for scene in scene_list:
            product_identifier = scene['id']
            product_name = scene['name']
            url = f"{zipper_odata_url}/Products({product_identifier})/$value"
            # print(url)
            downloadWholeImage(
                url, session, headers, output_data_folder, product_name)
            time.sleep(3)
    else:
        logging.warning('Txt file of S2 results not found.'+'\n')
        sys.exit(1)

# *******************************************************************************
# SECTION: LANDSAT DONWNLOADING FUNCTIONS
# *******************************************************************************


def filterScenesLandsatToDownload(scene_list):
    for key, value in scene_list.items():
        print(
            f"{[int(key)]} Scene: {value['display_id']} Cloud coverage: {value['cloud_cover']} {value['days_off']} days")
    print('')
    np = input('Number of images to be downloaded (* / 0,2,3 / [2-5])?: ')
    filtered_scenes = parseNumberProductsS2(np, scene_list)
    return filtered_scenes


def downloadLandsatScene(scene_id, output_data_folder, user_usgs, pass_usgs):
    # USGS authentication
    ee_api_search, ee_api_download = usgsAuthentication(user_usgs, pass_usgs)

    # output folder creation
    if 'LC08' in scene_id:
        output_folder_l = str(pathlib.Path(
            os.path.join(output_data_folder, 'l8')))
    if 'LC09' in scene_id:
        output_folder_l = str(pathlib.Path(
            os.path.join(output_data_folder, 'l9')))
    try:
        # print('Downloading... '+scene_id)
        logging.info('Downloading... '+scene_id)
        # download L8 .tar file to the output folder
        ee_api_download.download(scene_id, output_folder_l)
        folder_path = os.path.join(output_folder_l, scene_id)
        os.makedirs(folder_path, exist_ok=True)
        output_folder_data = str(pathlib.Path(
            os.path.join(output_folder_l, scene_id)))
        # uncompress the file
        # print('Unzipping... '+scene_id)
        logging.info('Unzipping... '+scene_id)
        if os.path.exists(str(pathlib.Path(os.path.join(output_folder_l, scene_id+'.tar')))):
            # be careful -> zip file .zip, .tar, .tar.gz
            untarFile(str(pathlib.Path(os.path.join(output_folder_l,
                                                    scene_id+'.tar'))), output_folder_data)
        if os.path.exists(str(pathlib.Path(os.path.join(output_folder_l, scene_id+'.tar.gz')))):
            untarFile(str(pathlib.Path(os.path.join(output_folder_l,
                                                    scene_id+'.tar.gz'))), output_folder_data)
        if os.path.exists(str(pathlib.Path(os.path.join(output_folder_l, scene_id+'.zip')))):
            unzipFile(str(pathlib.Path(os.path.join(output_folder_l,
                                                    scene_id+'.zip'))), output_folder_data)

        ee_api_search.logout()

    except Exception as e:
        logging.error('Exception: %s', e)
        ee_api_search.logout()
        sys.exit(1)


def downloadLandsat(run_parameters):
    # output data folder and output search folder
    output_data_folder = run_parameters['output_data_folder_landsat']
    output_search_folder = run_parameters['output_search_folder']

    # load text file with metadata for the found images in searching step
    txt_file = str(pathlib.Path(os.path.join(
        os.getcwd(), output_search_folder, 'search_result_landsat.txt')))
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as txt:
            scene_list = json.load(txt)

    # selection of scenes to be downloaded
    scene_list = filterScenesLandsatToDownload(
        scene_list)  # list of dictionaries

    for scene in scene_list:
        downloadLandsatScene(
            scene['display_id'], output_data_folder, run_parameters['user_usgs'], run_parameters['pass_usgs'])
