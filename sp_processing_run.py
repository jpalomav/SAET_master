import os
import sys
import logging
import logging.handlers
from pathlib import Path
import sp_config  # needded to pass enviromment variables


# import
from sp_parameters_validation import parse_args_processing, checkCommandArgummentsProcessing
from sp_basic_functions import createBasicFolderStructure, findIntoFolderStructure, filterScenesInfolder, recursiveFolderSearch
from sp_processing_functions import processSentinel2SceneFromPath, processLandsatSceneFromPath

# init logging
logger = logging.getLogger("")


def init_logger(level):
    '''
    Init the logging process. The level of logging can be changed from configuration file "saet_config.py"
    '''

    if level == '10':
        log_level = logging.DEBUG
    if level == '20':
        log_level = logging.INFO
    if level == '30':
        log_level = logging.WARNING
    if level == '40':
        log_level = logging.ERROR
    if level == '50':
        log_level = logging.CRITICAL

    logger.setLevel(log_level)

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    if (log_level == logging.INFO) or (log_level == logging.WARNING):
        console_formatting = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s')
    else:
        console_formatting = logging.Formatter(
            '%(asctime)s %(filename)s-%(funcName)s %(levelname)s %(message)s')

    console_handler.setFormatter(console_formatting)
    logger.addHandler(console_handler)


def run_algo(args):
    # check arguments for some parameters
    run_parameters = checkCommandArgummentsProcessing(args)

    # get some configuration parameters
    # these parameters are configured using saet_config.py file
    user_esa = os.getenv('USER_ESA')
    pass_esa = os.getenv('PASS_ESA')
    user_usgs = os.getenv('USER_USGS')
    pass_usgs = os.getenv('PASS_USGS')
    beaches_path = os.getenv('SHP_BEACHES_PATH')
    l8grid_path = os.getenv('SHP_LANDSAT_GRID_PATH')
    s2grid_path = os.getenv('SHP_SENTINEL2_GRID_PATH')
    saet_home_path = os.getenv('SAET_HOME_PATH')

    # adding credentials to run_parameters
    run_parameters['user_esa'] = user_esa
    run_parameters['pass_esa'] = pass_esa
    run_parameters['user_usgs'] = user_usgs
    run_parameters['pass_usgs'] = pass_usgs

    # auxiliar data paths
    if not os.path.isfile(beaches_path):
        logger.warning(
            f'The file {beaches_path} does not exist'+'\n')
        sys.exit(1)
    if not os.path.isfile(l8grid_path):
        logger.warning(
            f'The file {l8grid_path} does not exist'+'\n')
        sys.exit(1)
    if not os.path.isfile(s2grid_path):
        logger.warning(
            f'The file {s2grid_path} does not exist'+'\n')
        sys.exit(1)

    # adding aux_path shp data to run parameters
    run_parameters['l8grid_path'] = l8grid_path
    run_parameters['s2grid_path'] = s2grid_path

    # creating the data folder structure
    fs = createBasicFolderStructure(base_path=saet_home_path)

    # adding output folder for searching to parameters
    run_parameters['output_search_folder'] = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='search_data')

    # adding data folder for downloading S2 scenes to parameters
    run_parameters['output_data_folder_s2'] = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='s2')

    # adding data folder for downloading landsat scenes to parameters
    run_parameters['output_data_folder_landsat'] = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='landsat')

    # PROCESSING IMAGES _______________________________________________________________
    # search for scenes in the data folder
    data_path_s2 = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='s2')
    data_path_l8 = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='l8')
    data_path_l9 = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='l9')
    output_data_folder = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='output_data')

    water_index = run_parameters['water_index']
    thresholding_method = run_parameters['thresholding_method']
    morphology_method = run_parameters['morphology_method']
    cloud_mask_level = run_parameters['cloud_mask_level']
    kernel_size = run_parameters['kernel_size']
    beach_code_filter = run_parameters['beach_code_filter']

    list_of_total_scenes = []

    if data_path_s2 != '':
        list_of_paths_s2 = [Path(f.path)
                            for f in os.scandir(data_path_s2) if f.is_dir()]
        if list_of_paths_s2 != []:
            list_of_total_scenes += list_of_paths_s2

    if data_path_l8 != '':
        list_of_paths_l8 = [Path(f.path)
                            for f in os.scandir(data_path_l8) if f.is_dir()]
        if list_of_paths_l8 != []:
            list_of_total_scenes += list_of_paths_l8

    if data_path_l9 != '':
        list_of_paths_l9 = [Path(f.path)
                            for f in os.scandir(data_path_l9) if f.is_dir()]
        if list_of_paths_l9 != []:
            list_of_total_scenes += list_of_paths_l9

    if list_of_total_scenes != []:
        filtered_scenes = filterScenesInfolder(list_of_total_scenes)
        if filtered_scenes != []:
            print('')
            print('Scenes to be processed: ')
            print('')
            for filtered_scene in filtered_scenes:
                print(filtered_scene)
                print('')
                scene_path = recursiveFolderSearch(
                    output_data_folder, Path(filtered_scene))
                sds_path = recursiveFolderSearch(
                    output_data_folder, Path('sds'))
                # print(data_path)
                if 's2' in scene_path:
                    processSentinel2SceneFromPath(scene_path, filtered_scene, sds_path, beaches_path, water_index,
                                                  thresholding_method, cloud_mask_level, morphology_method, kernel_size, beach_code_filter)
                else:
                    processLandsatSceneFromPath(scene_path, filtered_scene, sds_path, beaches_path, water_index,
                                                thresholding_method, cloud_mask_level, morphology_method, kernel_size, beach_code_filter)


if __name__ == '__main__':
    """ Entrance to the searching algorithm workflow:
        - gets the user arguments
        - inits the logger
        - runs the algorithm
    """
    # getting user parameters
    args = parse_args_processing()
    # inits logger with a required level
    log_level = os.getenv('LOG_LEVEL')
    init_logger(log_level)
    logger.info('Starting downloading SAET_pro algorithm...'+'\n')
    # runs the main algorithm
    run_algo(args)
    print('\n')
    logger.info(
        'SAET_pro downloading algorithm have finished successfully.'+'\n')
    sys.exit(0)
