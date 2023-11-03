'''
Common functions used for the majority of the modules
'''

import os
from osgeo import ogr
import pathlib
import re
import sys
import logging
import fnmatch

# *******************************************************************************
# SECTION: FOLDER FUNCTIONS
# *******************************************************************************


def createFolderCheck(folder_path):
    '''
    Description:
    ------------
    Creates a new folder only in case this folder does not exist

    Arguments:
    ------------
    - folder_path (string): folder to be created

    Returns:
    ------------
    None

    '''

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def recursiveFileSearch(rootdir='.', pattern='*'):
    '''
    Description:
    ------------
    search for files recursively based in pattern strings

    Arguments:
    ------------
    - rootdir (string): path to the base folder
    - pattern (string): pattern to search files

    Returns:
    ------------
    - matches (list of strings): list of absolute paths to each found file

    '''

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(str(pathlib.Path(os.path.join(root, filename))))
    return matches


def createBasicFolderStructure(fs={'output_data': {'data': {'landsat': {'l8': {}, 'l9': {}}, 's2': {}}, 'sds': {}, 'search_data': {}}}, base_path=''):
    '''
    Description:
    ------------
    Creates the folder structure for searching, downloading and processing. 
    The function checks if the folder exists.

    Arguments:
    ------------
    - fs (dictionary): folder structure
    - base_path (string): root folder

    Returns:
    ------------
    None

    '''
    for folder, subfolder in fs.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        if subfolder:
            createBasicFolderStructure(subfolder, folder_path)


def findIntoFolderStructure(fs={'output_data': {'data': {'landsat': {'l8': {}, 'l9': {}}, 's2': {}}, 'sds': {}, 'search_data': {}}}, base_path='', folder_name=''):
    '''
    Description:
    ------------
    Find the path for a specific folder

    Arguments:
    ------------
    - fs (dictionary): folder structure
    - base_path (string): root folder
    - folder_name (string): folder to find

    Returns:
    ------------
    folder_path (string)

    '''

    for folder, subfolder in fs.items():
        folder_path = os.path.join(base_path, folder)
        if folder == folder_name:
            return folder_path
        if subfolder:
            result = findIntoFolderStructure(
                subfolder, folder_path, folder_name)
            if result:
                return result
    return None


def parseNumberProductsFolder(np, scenes):
    '''
    Description:
    ------------
    The function filters the scenes from the output data to be processed 
    acording with a list of numbers (identifiers)

    Arguments:
    ------------
    - np (list of integers): list with the number of products
    - scenes (list of dictionaries): list of scenes to be filtered

    Returns:
    ------------
    - list of dictionaries: list of filtered scenes

    '''
    lista = list(range(0, len(scenes)))
    filtered_scenes = []
    if np != 'NONE':
        if len(np) == 1 and np != '*':
            try:
                return [scenes[int(np)]]
            except:
                logging.warning(f'Incorrect number of product: {np}'+'\n')
                sys.exit(1)
        if np == '*':
            return scenes
        if len(np) > 1:
            #res = re.findall(r'^\d+(?:[ \t]*,[ \t]*\d+)+$', np)
            res = re.findall(r'^\d+(?:,\d+)*$', np)
            res2 = re.findall(r'^[0-9]\d*-[0-9]\d*', np)
            if len(res) != 0:
                n_lista = []
                for n in res[0].split(','):
                    if int(n) in lista:
                        n_lista.append(int(n))
                    else:
                        logging.warning(
                            f'Incorrect number of product: {n}'+'\n')
                        sys.exit(1)
                n_lista_sorted = sorted(list(set(n_lista)))
                for n in n_lista_sorted:
                    filtered_scenes.append(scenes[n])
                return filtered_scenes
            if len(res2) != 0:
                lim_inf = int(res2[0].split('-')[0])
                lim_sup = int(res2[0].split('-')[1])
                if lim_inf > lim_sup:
                    logging.warning(
                        f'Incorrect range of products'+'\n')
                    sys.exit(1)
                else:
                    if lim_inf not in lista or lim_sup not in lista:
                        logging.warning(
                            f'Incorrect range of products'+'\n')
                        sys.exit(1)
                    else:
                        n_lista = list(range(lim_inf, lim_sup+1))
                        for n in n_lista:
                            filtered_scenes.append(scenes[n])
                        return filtered_scenes
            logging.warning(
                f'Incorrect format for list or range of products'+'\n')
            sys.exit(1)
    return []


def filterScenesInfolder(list_of_paths):
    '''
    Description:
    ------------
    Filter the number of scene from the list of scenes. Only for reprocessing mode

    Arguments:
    ------------
    list_of_paths (list[string]): paths to the products folder

    Returns:
    ------------
    - list of filtered scenes

    '''
    if len(list_of_paths) != 0:
        list_of_folders = []
        print('List of scenes in the data folder: ')
        print('')
        for i in range(0, len(list_of_paths)):
            path = pathlib.Path(list_of_paths[i])
            folder = path.name
            print([i], folder)
            list_of_folders.append(folder)
        print('')
        np = input('Number of images to be reprocessed (* / 0,2,3 / [2-5])?: ')
        filtered_scenes = parseNumberProductsFolder(np, list_of_folders)
        return filtered_scenes
    else:
        return []


def recursiveFolderSearch(rootdir='.', pattern='*'):
    '''
    Description:
    ------------
    search for folders recursively based in pattern strings

    Arguments:
    ------------
    - rootdir (string): path to the base folder
    - pattern (string): pattern to search files

    Returns:
    ------------
    - matches (list of strings): list of absolute paths to each found folder

    '''

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for dirname in fnmatch.filter(dirnames, pattern):
            matches.append(str(pathlib.Path(os.path.join(root, dirname))))
    if len(matches) > 0:
        return matches[0]
    else:
        return ''

# *******************************************************************************
# SECTION: IMAGE FUNCTIONS
# *******************************************************************************


def getBandPath(scene_path, band_name):
    '''
    Description:
    ------------
    Get absolute path from a single band or file

    Arguments:
    ------------
    - scene_path (string): path to the target folder
    - band_name (string): band name to search

    Returns:
    ------------
    - band_path (string): path to the band

    '''
    file_list = recursiveFileSearch(scene_path, '*.*')
    band_path = [i for i in file_list if (
        band_name in i) and (not 'xml' in i)]
    if len(band_path) != 0:
        return str(pathlib.Path(band_path[0]))
    else:
        return None


# *******************************************************************************
# SECTION: LANDSAT FUNCTIONS
# *******************************************************************************


def getLatLongFromPathRow(shp_path, scene_id):
    '''
    Description:
    ------------
    Returns long, lat coordinates from path and row code for landsat 8 scenes

    Arguments:
    ------------
    - shp_pat (string): path to the shapefile containing Landsat 8 footprints.
    - scene_id (string): scene code (199032).

    Returns:
    ------------
    - long, lat (float): longitud and latitud of the scene footprint centroid

    '''
    path_id = int(scene_id[0:3])
    row_id = int(scene_id[3:])

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp_path)
    layer = ds.GetLayer()
    layer.SetAttributeFilter("PATH = "+str(path_id)+' and ROW = '+str(row_id))
    for feat in layer:
        geom = feat.GetGeometryRef()
        long = geom.Centroid().GetX()
        lat = geom.Centroid().GetY()
    ds = None
    return long, lat


def getLandsatQuickLookUrl(img_id):
    '''
    Description:
    ------------
    Makes a valid url for dispaying the quiklook image on an standard browser

    Arguments:
    ------------
    - img_id (string): image id

    Returns:
    ------------
    LandsatQuickLookUrl (string): valid url
    '''
    url_base = 'https://earthexplorer.usgs.gov/index/resizeimage?img=https%3A%2F%2Flandsatlook.usgs.gov%2Fgen-browse%3Fsize%3Drrb%26type%3Drefl%26product_id%3Dimg_id&angle=0&size=640'
    LandsatQuickLookUrl = url_base.replace('img_id', img_id)
    return LandsatQuickLookUrl
