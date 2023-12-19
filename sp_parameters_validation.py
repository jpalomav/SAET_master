'''
Validation functions for arguments, including authentication, searching, 
downloading and processing

'''
import argparse
import sys
import datetime
import logging
import re
from landsatxplore2 import api
from landsatxplore2.earthexplorer import EarthExplorer
import requests
import json


def valid_date(s):
    '''
    Description:
    ------------
    Validate date argument for central date (--cd) to ensure
    correct format ('YYYYmmdd')

    Arguments:
    ------------
    - central_date (string): date supposed to be validated

    Returns:
    ------------
    - If test passes: valid date in Int type.
    - If test fails: prints error message and exit

    '''
    try:
        if len(s) != 8:
            print(f'Not a valid date: {s}')
            sys.exit()
        d = datetime.datetime.strptime(s, "%Y%m%d")
        return int(datetime.datetime.strftime(d, "%Y%m%d"))
    except ValueError:
        print(f'Not a valid date: {s}')
        sys.exit()


def parse_args_searching():
    '''
    Function to validate the input parameters. These parameters are obtained from the command-line
    sentence. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp',
                        type=str,
                        help='Coordinates long/lat in these formats: (POINT) fp=long,lat; (AOI) fp=min_long,min_lat,max_long,max_lat. Default: NONE',
                        default='NONE',
                        required=True)
    parser.add_argument('--sd',
                        help='Start date for searching scenes (YYYYMMDD). --sd=20210101. Default:20200101',
                        type=valid_date,
                        default='20200101',
                        required=True)
    parser.add_argument('--cd',
                        help='Central date for storm (YYYYMMDD). --sd=20210101. Default:20200102',
                        type=valid_date,
                        default='20200102',
                        required=True)
    parser.add_argument('--ed',
                        help='End date for searching scenes (YYYYMMDD). --sd=20210101. Default:20200103',
                        type=valid_date,
                        default='20200103',
                        required=True)
    parser.add_argument('--mc',
                        type=int,
                        choices=range(0, 101),
                        metavar='[0-100]',
                        help='maximum cloud coverture for the whole scene [0-100]. --mc=10. Default 100',
                        default='100',
                        required=False)
    parser.add_argument('--lp',
                        type=str,
                        choices=['landsat_ot_c2_l1',
                                 'landsat_ot_c2_l2', 'NONE'],
                        help='Landsat 8 product type. landsat_ot_c2_l1 or landsat_ot_c2_l2 or NONE. Default: landsat_ot_c2_l1',
                        default='landsat_ot_c2_l1',
                        required=True)
    parser.add_argument('--ll',
                        type=str,
                        help='List of scenes for Landsat 8 (number of 6 digits). --ll=198032,199031. Default: NONE',
                        default='NONE',
                        required=True)
    parser.add_argument('--sp',
                        type=str,
                        choices=['S2MSI1C', 'S2MSI2A', 'NONE'],
                        help='Sentinel 2 product type (S2MSI1C / S2MSI2A). --s2=S2MSI1C / --s2=S2MSI2A / NONE. Default: S2MSI1C',
                        default='S2MSI1C',
                        required=True)
    parser.add_argument('--sl',
                        type=str,
                        help='List of scenes for Sentinel 2 (string of 5 characters). --sl=31TCF,30TYK. Default: NONE',
                        default='NONE',
                        required=True)
    parser.add_argument('--so',
                        type=int,
                        choices=range(0, 2),
                        metavar='[0-1]',
                        help='Exclude images with NO DATA values [0-1]. --so=1. Default: 1',
                        default='1',
                        required=False)

    return parser.parse_args()


def validate_footprint(coords):
    pattern = r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?(,-?\d+(\.\d+)?,-?\d+(\.\d+)?)?$'
    if re.match(pattern, coords):
        coords_list = [coord for coord in coords.split(',')]
        return coords_list
    else:
        return []


def parse_args_downloading():
    '''
    Function to validate the input parameters. These parameters are obtained from the command-line
    sentence. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp',
                        type=str,
                        choices=['s', 'l'],
                        help='s -> Sentinel-2; l -> Landsat 8-9. Default: s',
                        default='s',
                        required=True)

    return parser.parse_args()


def parse_args_processing():
    '''
    Function to validate the input parameters. These parameters are obtained from the command-line
    sentence. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--wi',
                        type=str,
                        choices=['aweish', 'aweinsh', 'mndwi', 'kmeans'],
                        help='Water index type (aweish, aweinsh,mndwi,kmeans). --wi=aweinsh. Default: aweinsh',
                        default='aweinsh',
                        required=False)
    parser.add_argument('--th',
                        type=str,
                        choices=['0', '1', '2'],
                        help='Thresholding method (0: standard 0 value, 1: Otsu bimodal, 2: Otsu multimodal 3 classes). --th=0. Default: 0',
                        default='0',
                        required=False)
    parser.add_argument('--mm',
                        type=str,
                        choices=['erosion', 'dilation'],
                        help='Morphological method (erosion, dilation). --mm=dilation, Default: dilation',
                        default='dilation',
                        required=False)
    parser.add_argument('--cl',
                        type=str,
                        choices=['0', '1', '2'],
                        help='Cloud mask level (0: no masking, 1: only opaque clouds, 2: opaque clouds + cirrus + cloud shadows). Default: 0',
                        default='0',
                        required=False)
    parser.add_argument('--ks',
                        type=str,
                        choices=['3', '5'],
                        help='Kernel size for points extraction. Default: 3',
                        default='3',
                        required=False)
    parser.add_argument('--bc',
                        type=str,
                        help='beach code filter list. --bc=520,548 Default: NONE',
                        default='NONE',
                        required=False)

    return parser.parse_args()


def checkCommandArgummentsDonwloading(args):
    # check integrity for parameters
    if '=' in args.sp:
        satellite_platform = args.sp.split('=')[1]
    else:
        satellite_platform = args.sp
    run_parameters = {}
    run_parameters['satellite_platform'] = satellite_platform
    return run_parameters


def checkCommandArgummentsProcessing(args):
    # check integrity for parameters
    if '=' in args.wi:
        water_index = args.wi.split('=')[1]
    else:
        water_index = args.wi

    if '=' in args.th:
        thresholding_method = args.th.split('=')[1]
    else:
        thresholding_method = args.th

    if '=' in args.mm:
        morphology_method = args.mm.split('=')[1]
    else:
        morphology_method = args.mm

    if '=' in args.cl:
        cloud_mask_level = args.cl.split('=')[1]
    else:
        cloud_mask_level = args.cl

    if '=' in args.ks:
        kernel_size = args.ks.split('=')[1]
    else:
        kernel_size = args.ks

    # processing the filter by beach code
    if '=' in args.bc:
        bc = args.sl.split('=')[1]
    else:
        bc = args.bc
    if bc != 'NONE':
        if ',' in bc:
            bc = bc.replace(" ", "")
            res1 = re.findall(r'^\d+(?:[ \t]*,[ \t]*\d+)+$', bc)
            if len(res1) == 1:
                bc = res1[0].split(',')
            else:
                logging.warning('Invalid format for beach code filter.')
                sys.exit(1)
        else:
            res2 = re.findall(r'^\d+$', bc)
            if len(res2) == 1:
                bc = res2
            else:
                logging.warning('Invalid format for beach code filter.')
                sys.exit(1)
    else:
        bc = ['NONE']

    # convert beach code list in string with tuple format
    query = '('
    for i in bc:
        query = query+i+','
    query = query[:-1]
    query = query+')'

    beach_code_filter = query

    run_parameters = {}
    run_parameters['water_index'] = water_index
    run_parameters['thresholding_method'] = thresholding_method
    run_parameters['morphology_method'] = morphology_method
    run_parameters['cloud_mask_level'] = cloud_mask_level
    run_parameters['kernel_size'] = kernel_size
    run_parameters['beach_code_filter'] = beach_code_filter

    return run_parameters


def checkCommandArgummentsSearching(args):
    '''
    Description:
    ------------
    Check the suitable command line parameters format to avoid input errors.
    Converts footprint parameter into suitable geometry in wkt format (needed to
    searching for images).

    Arguments:
    ------------
    - argparse object


    Returns:
    ------------
    - Parameters in suitable format (string or int)

    '''

    # check integrity for parameters
    if '=' in args.lp:
        l8_product = args.lp.split('=')[1]
    else:
        l8_product = args.lp

    if '=' in args.sp:
        s2_product = args.sp.split('=')[1]
    else:
        s2_product = args.sp

    try:
        if '=' in str(args.sd):
            start_date = str(args.sd).split('=')[1]
        else:
            start_date = str(args.sd)
        datetime.datetime.strptime(start_date, '%Y%m%d')
        if '=' in str(args.ed):
            end_date = str(args.ed).split('=')[1]
        else:
            end_date = str(args.ed)
        datetime.datetime.strptime(end_date, '%Y%m%d')
        if '=' in str(args.cd):
            central_date = str(args.cd).split('=')[1]
        else:
            central_date = str(args.cd)
        datetime.datetime.strptime(central_date, '%Y%m%d')
    except:
        # print('Invalid format for start or end date (YYYYMMDD).')
        logging.warning('Invalid format for start or end date (YYYYMMDD).')
        sys.exit(1)

    if not start_date < central_date < end_date:
        # print('Central date must be between start date and end date.')
        logging.warning(
            'Central date must be between start date and end date.')
        sys.exit(1)

    max_cloud_cover = args.mc
    sentinel_overlap = args.so

    # processing footprint for image searching. The final format is different depending on L8 or S2
    scene_landast_list = []
    scene_sentinel_list = []
    if '=' in args.fp:
        footprint_path = args.fp.split('=')[1]
    else:
        footprint_path = args.fp
    if footprint_path != 'NONE':
        res = validate_footprint(footprint_path)
        if len(res) == 0:
            # print('Invalid footprint path or coordinate format.')
            logging.warning('Invalid footprint path or coordinate format.')
            sys.exit(1)
        else:
            if len(res) == 4:
                min_long = res[0]
                min_lat = res[1]
                max_long = res[2]
                max_lat = res[3]
                #footprint = f"POLYGON(({min_long} {max_lat},{max_long} {max_lat},{max_long} {min_lat},{min_long} {min_lat},{min_long} {max_lat}))"
                footprint = [min_long, min_lat, max_long, max_lat]
            if len(res) == 2:
                long = res[0]
                lat = res[1]
                #footprint = f"POINT({long} {lat})"
                footprint = [long, lat]

    else:
        footprint = 'NONE'
        if '=' in args.ll:
            ll = args.ll.split('=')[1]
        else:
            ll = args.ll
        if ll != 'NONE':
            for id_ll in args.ll.split(','):
                res = re.findall('[0-9]{6}', id_ll)
                if len(res) == 0:
                    # print('Invalid Landsat 8 scene code.')
                    logging.warning('Invalid Landsat scene code.')
                    sys.exit(1)

            ll = args.ll.split(',')  # list of Landsat 8 scenes
        else:
            ll = 'NONE'

        if '=' in args.sl:
            sl = args.sl.split('=')[1]
        else:
            sl = args.sl
        if sl != 'NONE':
            for id_sl in sl.split(','):
                res = re.findall('[0-9]{2}[A-Z]{3}', id_sl)
                if len(res) == 0:
                    # print('Invalid Sentinel 2 scene code.')
                    logging.warning('Invalid Sentinel 2 scene code.')
                    sys.exit(1)

            sl = args.sl.split(',')  # list of sentinel 2 scenes
        else:
            sl = 'NONE'

        scene_landast_list = ll
        scene_sentinel_list = sl

    # output dictionary
    run_parameters = {}
    run_parameters['l8_product'] = l8_product
    run_parameters['s2_product'] = s2_product
    run_parameters['start_date'] = start_date
    run_parameters['central_date'] = central_date
    run_parameters['end_date'] = end_date
    run_parameters['max_cloud_cover'] = max_cloud_cover
    run_parameters['footprint'] = footprint
    run_parameters['scene_landsat_list'] = scene_landast_list
    run_parameters['scene_sentinel_list'] = scene_sentinel_list
    run_parameters['sentinel_overlap'] = sentinel_overlap

    return run_parameters


# *******************************************************************************
# SECTION: AUTHENTICATION FUNCTIONS
# *******************************************************************************

def usgsAuthentication(user_usgs, pass_usgs):
    '''
    Description:
    ------------
    Access to the USGS apis (seraching and download) using credentials from
    readAuthenticationParameters() function

    Arguments:
    ------------
    - user_usgs(string): user credential
    - pass_usgs(string): password

    Returns:
    ------------
    - USGS search and download api objects

    '''

    try:
        usgs_api_search = api.API(user_usgs, pass_usgs)
        usgs_api_download = EarthExplorer(user_usgs, pass_usgs)
        return usgs_api_search, usgs_api_download
    except Exception as e:
        logging.error('Exception: %s', e)
        sys.exit(1)


def esaAuthentication(user_esa, pass_esa):
    '''
    Description:
    ------------
    Access to the ESA Odata (for downloading) using credentials from
    readAuthenticationParameters() function

    Arguments:
    ------------
    - user_esa(string): user credential
    - pass_esa(string): password

    Returns:
    ------------
    - token (string): authentication code

    '''
    try:
        auth_server_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": user_esa,
            "password": pass_esa,
        }

        response = requests.post(auth_server_url, data=data,
                                 verify=True, allow_redirects=False)

        access_token = json.loads(response.text)["access_token"]
        return access_token
    except Exception as e:
        logging.error('Exception: %s', e)
        sys.exit(1)
