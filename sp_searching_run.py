# import basic python libraries
import os
import sys
import logging
import logging.handlers
import webbrowser
import sp_config  # needded to pass enviromment variables

# import
from sp_parameters_validation import parse_args_searching
from sp_searching_functions import (searchS2CDSE,
                                    writeHtmlS2CDSE,
                                    searchLandsat8,
                                    writeHtmlLandsat8,
                                    writeTextS2CDSE,
                                    writeTextLandsat)
from sp_parameters_validation import checkCommandArgummentsSearching
from sp_basic_functions import createBasicFolderStructure, findIntoFolderStructure

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
    run_parameters = checkCommandArgummentsSearching(args)

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

    # adding output results mode to run parameters
    run_parameters['output_search_folder'] = findIntoFolderStructure(
        base_path=saet_home_path, folder_name='search_data')

    # SEARCH FOR IMAGES _______________________________________________________________
    if run_parameters['s2_product'] != 'NONE':
        s2_scenes = searchS2CDSE(run_parameters)
        html_file_s2 = writeHtmlS2CDSE(
            s2_scenes, run_parameters['output_search_folder'])
        webbrowser.open(html_file_s2)
        writeTextS2CDSE(s2_scenes, run_parameters['output_search_folder'])

    if run_parameters['l8_product'] != 'NONE':
        l8_scenes = searchLandsat8(run_parameters)
        html_file_l8 = writeHtmlLandsat8(
            l8_scenes, run_parameters['output_search_folder'])
        webbrowser.open(html_file_l8)
        writeTextLandsat(l8_scenes, run_parameters['output_search_folder'])


if __name__ == '__main__':
    """ Entrance to the searching algorithm workflow:
        - gets the user arguments
        - inits the logger
        - runs the algorithm
    """
    # getting user parameters
    args = parse_args_searching()
    # inits logger with a required level
    log_level = os.getenv('LOG_LEVEL')
    init_logger(log_level)
    logger.info('Starting searching SAET_pro algorithm...'+'\n')
    # runs the main algorithm
    run_algo(args)
    print('\n')
    logger.info('SAET_pro searching algorithm have finished successfully.'+'\n')
    sys.exit(0)
