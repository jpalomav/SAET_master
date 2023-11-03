'''
Functions for searching Landsat and Sentinel-2 images
- Landsat 8 and 9 from USGS
- Sentinel-2 from Copernicus (ESA)
'''

# *******************************************************************************
# SECTION: IMPORTS
# *******************************************************************************
import requests
import logging
import sys
import datetime
import pathlib
import os
import json

from sp_basic_functions import getLandsatQuickLookUrl, getLatLongFromPathRow
from sp_parameters_validation import usgsAuthentication


# *******************************************************************************
# SECTION: VISUALIZATION FUNCTIONS
# *******************************************************************************

def writeHtmlS2CDSE(scene_list, output_search_folder):
    '''
    Description:
    ------------
    HTML file with the quicklook images found in the request are created (for S2 images). This file
    will be launched by the default web browser.

    Arguments:
    ------------
    - image_url_list(list of strings): list of the quicklooks
    - titles (list of strings): titles for each quicklook
    - output_search_folder (string): path to the output html file
    - product_name (string): name of the product for Landsat (to avoid name
      differences on quicklooks).
    - s2_quicklook_ser (string): quicklook source (Copernicus or external)


    Returns:
    ------------
    html_file (object): html file

    '''

    image_url_list = []
    titles = []

    for scene in scene_list.values():
        image_url_list.append(scene['quicklook'])
        titles.append(
            scene['name'].replace('.SAFE', '')+' CC:'+str(round(float(scene['cloud_cover']), 2))+' days:'+str(scene['days_off'])+' Online:'+str(scene['online']))

    html = "<html><head>"
    html += "<style> table, th, td {border: 1px solid black; border-collapse: collapse;}</style>"
    html += "<title>Results</title></head><body>"
    html += '<center><table>'
    columns = 2
    rows = int(len(image_url_list)/columns)+1
    for r in range(1, rows+1):
        html += '<tr>'
        for c in range(1, columns+1):
            i = (r-1)*columns+c
            if i <= len(image_url_list):
                image_url = image_url_list[i-1]
                html += f'<td><center><img src="{image_url}" align="center" width="100%" height="100%"></center></td>'
            else:
                html += f'<td></td>'
        html += '</tr>'
        html += '<tr>'
        for c in range(1, columns+1):
            i = (r-1)*columns+c
            if i <= len(image_url_list):
                title = titles[i-1]
                html += f'<td><center><p><font color="blue">[{i-1}] {title}</font></p></center></td>'
            else:
                html += f'<td></td>'
        html += '</tr>'
    html += '</table></center>'
    html += "</body></html>"
    html_file = str(pathlib.Path(os.path.join(
                    os.getcwd(), output_search_folder, 'search_result_s2.html')))
    with open(html_file, 'w') as outputfile:
        outputfile.write(html)
    return html_file


def writeHtmlLandsat8(scene_list, output_search_folder):
    '''
    Description:
    ------------
    HTML file with the quicklook images found in the request are created. This file
    will be launched by the default web browser.

    Arguments:
    ------------
    - image_url_list(list of strings): list of the quicklooks
    - titles (list of strings): titles for each quicklook
    - output_search_folder (string): path to the output html file
    - product_name (string): name of the product for Landsat (to avoid name
      differences on quicklooks).

    Returns:
    ------------
    html_file (object): html file

    '''

    # image_url_list.append(getLandsatQuickLookUrl(scenes_filtered[key]['display_id']))
    # image_title_list.append(scenes_filtered[key]['display_id']+' CC:' +
    #                                 str(scenes_filtered[key]['scene_cloud_cover']) + '%  Days: '+str(date_difference.days))

    image_url_list = []
    titles = []

    for scene in scene_list.values():
        image_url_list.append(getLandsatQuickLookUrl(scene['display_id']))
        titles.append(scene['display_id']+' CC:' +
                      str(scene['scene_cloud_cover']) + '%  Days: '+str(scene['days_off']))

    html = "<html><head>"
    html += "<style> table, th, td {border: 1px solid black; border-collapse: collapse;}</style>"
    html += "<title>Results</title></head><body>"
    html += '<center><table>'
    columns = 2
    rows = int(len(image_url_list)/columns)+1
    for r in range(1, rows+1):
        html += '<tr>'
        for c in range(1, columns+1):
            i = (r-1)*columns+c
            if i <= len(image_url_list):
                image_url = image_url_list[i-1]
                if 'L2SP' in image_url:
                    image_url = image_url.replace('L2SP', 'L1TP')
                html += f'<td><center><img src="{image_url}" align="center" width="100%" height="100%"></center></td>'
            else:
                html += f'<td></td>'
        html += '</tr>'
        html += '<tr>'
        for c in range(1, columns+1):
            i = (r-1)*columns+c
            if i <= len(image_url_list):
                title = titles[i-1]
                html += f'<td><center><p><font color="blue">[{i-1}] {title}</font></p></center></td>'
            else:
                html += f'<td></td>'
        html += '</tr>'
    html += '</table></center>'
    html += "</body></html>"
    html_file = str(pathlib.Path(os.path.join(
                    os.getcwd(), output_search_folder, 'search_result_landsat.html')))
    with open(html_file, 'w') as outputfile:
        outputfile.write(html)
    return html_file

# *******************************************************************************
# SECTION: TEXT FILE OUTPUT
# *******************************************************************************


def writeTextS2CDSE(scene_list, output_search_folder):
    txt_file = str(pathlib.Path(os.path.join(
        os.getcwd(), output_search_folder, 'search_result_s2.txt')))
    with open(txt_file, 'w') as txt:
        json.dump(scene_list, txt)


def writeTextLandsat(scene_list, output_search_folder):
    scene_metadata = {}
    for key, value in scene_list.items():
        scene_metadata[key] = {'display_id': value['display_id'],
                               'cloud_cover': value['cloud_cover'],
                               'days_off': value['days_off']}

    txt_file = str(pathlib.Path(os.path.join(
        os.getcwd(), output_search_folder, 'search_result_landsat.txt')))

    with open(txt_file, 'w') as txt:
        json.dump(scene_metadata, txt)

# *******************************************************************************
# SECTION: SENTINEL-2 SEARCHING FUNCTIONS
# *******************************************************************************


def searchS2CDSE(run_parameters):
    '''
    Description:
    ------------
    Searches for Sentinel 2 scenes using ESA ODATA api.
    Note: for searching requests there's no need to carry out the authentication
    process

    Arguments:
    ------------
    - run_parameters (list): list of parameters:
        - tiles (list of strings): list of Sentinel 2 tiles ('30TYJ','30TFC')
        - footprint (geometry): in case list of tiles is NONE
        - product_type (string): type of product for S2 images. Can be S2MSI1C or S2MSI2A
        - start_date (string): start date in format YYYYMMDD
        - end_date (string): end date in format YYYYMMDD
        - cloud_cover (int): cloud coverage. Number between 0 and 100.
        - sentinel_overlap (int): exclude tiles with NO DATA values

    Returns:
    ------------
    - scenes (dictionary): list of metadata for every found scene.

    '''
    # run parameters
    # run_parameters['sentinel_overlap'] = sentinel_overlap

    tiles = run_parameters['scene_sentinel_list']
    product_type = run_parameters['s2_product']
    start_date = run_parameters['start_date']
    central_date = run_parameters['central_date']
    end_date = run_parameters['end_date']
    cloud_cover = run_parameters['max_cloud_cover']
    sentinel_overlap = run_parameters['sentinel_overlap']
    aoi = run_parameters['footprint']

    start_date = start_date[0: 4]+'-'+start_date[4: 6]+'-'+start_date[6:]
    central_date = central_date[0: 4]+'-' + \
        central_date[4: 6]+'-'+central_date[6:]
    end_date = end_date[0: 4]+'-'+end_date[4: 6]+'-'+end_date[6:]

    # aoi expression
    if aoi != 'NONE':
        if len(aoi) == 4:
            min_long = aoi[0]
            min_lat = aoi[1]
            max_long = aoi[2]
            max_lat = aoi[3]
            aoi = f"POLYGON(({min_long} {max_lat},{max_long} {max_lat},{max_long} {min_lat},{min_long} {min_lat},{min_long} {max_lat}))"
        if len(aoi) == 2:
            long = aoi[0]
            lat = aoi[1]
            aoi = f"POINT({long} {lat})"

    # base URL of the product catalogue
    catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    # parameters for searching
    cn = "SENTINEL-2"
    pt = product_type
    mcc = cloud_cover
    sps = f"{start_date}T00:00:00.000Z"
    spe = f"{end_date}T00:00:00.000Z"
    tl_list = tiles
    str_tile = ""

    for tl in tl_list:
        str_tile += f"contains(Name,'{tl}') or "
    str_tile = str_tile[0: len(str_tile)-4]

    # query creation in OData protocol format
    search_query = (f"{catalogue_odata_url}/Products?$filter=Collection/Name eq '{cn}' "
                    f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
                    f"and att/OData.CSC.StringAttribute/Value eq '{pt}') "
                    f"and ContentDate/Start gt {sps} "
                    f"and ContentDate/Start lt {spe} "
                    f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
                    f"and att/OData.CSC.DoubleAttribute/Value le {mcc}) ")

    if aoi == 'NONE':
        search_query += f"and ({str_tile})"
    else:
        search_query += f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')"

    search_query += (f"&$expand=Attributes&$expand=Assets"
                     f"&$orderby=ContentDate/Start desc"
                     f"&$top=1000"
                     )

    # print(search_query)
    scenes = {}
    filtered_scenes = {}
    response = requests.get(search_query)
    if response.status_code == 200:
        response = response.json()
        if 'value' in response:
            scene_info = response['value']
            # print(scene_info)
            if len(scene_info) == 0:
                logging.warning('There are no products S2 to download.'+'\n')
                sys.exit(1)
            else:
                for item in scene_info:
                    # print(item['GeoFootprint']['coordinates'][0])
                    scenes[item['Id']] = {'id': item['Id'],
                                          'name': item['Name'],
                                          'online': item['Online'],
                                          'quicklook': item['Assets'][0]['DownloadLink'],
                                          'corners': len(item['GeoFootprint']['coordinates'][0])}
                    for attribute in item["Attributes"]:
                        if attribute['Name'] == 'cloudCover':
                            scenes[item['Id']]['cloud_cover'] = attribute['Value']
                            break
                if sentinel_overlap == 1:
                    for key, value in scenes.items():
                        if scenes[key]['corners'] == 5:
                            filtered_scenes[key] = value
                else:
                    filtered_scenes = scenes.copy()

                flag = 0
                central_date = central_date.replace('-', '')
                for i, scene in enumerate(filtered_scenes.values()):
                    scene_date = scene['name'].split('_')[2].split('T')[0]
                    date_difference = datetime.datetime.strptime(
                        scene_date, "%Y%m%d") - datetime.datetime.strptime(central_date, "%Y%m%d")
                    scene['days_off'] = date_difference.days
                    if central_date < scene_date:
                        print(
                            f"[{i}] Scene: {scene['name'].replace('.SAFE','')} Cloud coverage: {str(round(float(scene['cloud_cover']),2))}% {date_difference.days} days")
                    else:
                        if flag == 0:
                            print('[*******] Central date:'+central_date)
                            print(
                                f"[{i}] Scene: {scene['name'].replace('.SAFE','')} Cloud coverage: {str(round(float(scene['cloud_cover']),2))}% {date_difference.days} days")
                            flag = 1
                        else:
                            print(
                                f"[{i}] Scene: {scene['name'].replace('.SAFE','')} Cloud coverage: {str(round(float(scene['cloud_cover']),2))}% {date_difference.days} days")
                return filtered_scenes

    else:
        logging.warning('The request failed.'+'\n')
        sys.exit(1)


# *******************************************************************************
# SECTION: LANDSAT SEARCHING FUNCTIONS
# *******************************************************************************


def searchLandsat8(run_parameters):
    '''
    Description:
    ------------
    Searches for Landsat 8 scenes using USGS api through landsatxplore module.
    Scenes with the same date will be removed (case of areas of interest that
    overlap the same area in different coordinate reference systems)

    Arguments:
    ------------
    - footprint (tuple of 4 strings): region of interest in format (xmin, ymin, xmax, ymax).
    - product_type (string): type of product for L8 images (landsat_8_c1).
    - start_date (string): start date in format YYYYMMDD
    - end_date (string): end date in format YYYYMMDD
    - max_cloud_cover (int): cloud coverage. Number between 0 and 100.
    - user_usgs (string): user credential for the USGS server
    - pass_usgs (string): password credential for the USGS server

    Returns:
    ------------
    - filtered_scenes (list of dictionaries): list of metadata for every found scene.

    '''

    scenes_id = run_parameters['scene_landsat_list']
    product_type = run_parameters['l8_product']
    start_date = run_parameters['start_date']
    central_date = run_parameters['central_date']
    end_date = run_parameters['end_date']
    max_cloud_cover = run_parameters['max_cloud_cover']
    aoi = run_parameters['footprint']
    user_usgs = run_parameters['user_usgs']
    pass_usgs = run_parameters['pass_usgs']
    l8grid_path = run_parameters['l8grid_path']

    # date adaptation for searching L8 scenes
    start_date = start_date[0:4]+'-'+start_date[4:6]+'-'+start_date[6:]
    end_date = end_date[0:4]+'-'+end_date[4:6]+'-'+end_date[6:]

    # authentication
    ee_api_search, ee_api_download = usgsAuthentication(user_usgs, pass_usgs)

    # query scenes
    logging.info('Searching for Landsat images...'+'\n')
    if aoi != 'NONE':
        if len(aoi) == 4:
            min_long = float(aoi[0])
            min_lat = float(aoi[1])
            max_long = float(aoi[2])
            max_lat = float(aoi[3])
            aoi = (min_long, min_lat, max_long, max_lat)
            try:
                scenes = ee_api_search.search(
                    dataset=product_type,
                    bbox=aoi,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover
                )
            except Exception as e:
                logging.error('Exception: %s', e)
                ee_api_search.logout()
                sys.exit(1)

        if len(aoi) == 2:
            long = float(aoi[0])
            lat = float(aoi[1])
            aoi = (long, lat)
            try:
                scenes = ee_api_search.search(
                    dataset=product_type,
                    longitude=aoi[0],
                    latitude=aoi[1],
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover
                )
            except Exception as e:
                logging.error('Exception: %s', e)
                ee_api_search.logout()
                sys.exit(1)
    else:
        try:
            scenes = []
            for scene_id in scenes_id:
                longi, lat = getLatLongFromPathRow(l8grid_path, scene_id)
                scenes_temp = ee_api_search.search(
                    dataset=product_type,
                    longitude=longi,
                    latitude=lat,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover
                )
                scenes = scenes+scenes_temp
        except Exception as e:
            logging.error('Exception: %s', e)
            ee_api_search.logout()
            sys.exit(1)

    ee_api_search.logout()
    scenes_filtered = {}
    if(len(scenes) > 0):
        # ****************************
        flag = 0
        for i, scene in enumerate(scenes):
            scene = scenes[i]
            if not 'L1GT' in scene['display_id']:
                scenes_filtered[str(i)] = scene
        # print(scenes_filtered)
        for key, value in scenes_filtered.items():
            # print(scenes_filtered[key]['display_id'])
            scene_date = scenes_filtered[key]['display_id'].split('_')[3]
            date_difference = datetime.datetime.strptime(
                scene_date, "%Y%m%d") - datetime.datetime.strptime(central_date, "%Y%m%d")
            scenes_filtered[key]['days_off'] = date_difference.days
            cloud_cover = str(
                round(float(scenes_filtered[key]['scene_cloud_cover']), 2))
            if central_date < scene_date:
                print(
                    f"[{key}] Scene: {scenes_filtered[key]['display_id']} Cloud cover: {cloud_cover}%  {date_difference.days} days.")
            else:
                if flag == 0:
                    print('[*******] Central date: '+central_date)
                    print(
                        f"[{key}] Scene: {scenes_filtered[key]['display_id']} Cloud cover: {cloud_cover}%  {date_difference.days} days.")
                    flag = 1
                else:
                    print(
                        f"[{key}] Scene: {scenes_filtered[key]['display_id']} Cloud cover: {cloud_cover}%  {date_difference.days} days.")

        return scenes_filtered
        # *****************************
    else:
        logging.warning('There are no Landsat images to download'+'\n')
        sys.exit(1)
