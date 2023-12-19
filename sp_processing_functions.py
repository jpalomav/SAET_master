import logging
import pathlib
import os
import sys
import numpy as np
from osgeo import gdal, osr, ogr
from math import exp, sqrt, degrees, atan2, ceil, sin
import csv
import gc
from tqdm import tqdm
import skimage.morphology as morphology
from skimage.transform import rescale
from skimage.filters import threshold_multiotsu, threshold_otsu
from sklearn.cluster import KMeans
from skimage import exposure
from polynomial import polyfit2d, polyval2d
from matplotlib.pyplot import clf, contour
from scipy.spatial import Delaunay
import networkx as nx
import shapefile
import json

from sp_basic_functions import getBandPath, recursiveFileSearch, createFolderCheck

# *******************************************************************************
# SECTION: COMMON PROCESSING FUNCTIONS
# *******************************************************************************


def createEmptyCloudMaskS2(output_path, raster_template):
    '''
    Description:
    ------------
    Creates an image of 0 values with the same features of the input
    template image (resolution, bounding box and coordinate reference system).

    Arguments:
    ------------
    - output_path (string): path to the output file
    - raster_template (string): path of the image template

    Returns:
    ------------
    None

    '''
    ds_in = gdal.Open(raster_template)
    band_in = ds_in.GetRasterBand(1)
    data_in = band_in.ReadAsArray()
    trs = ds_in.GetGeoTransform()
    rows, cols = data_in.shape
    data_out = np.zeros((rows, cols)).astype(np.uint8)
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(trs)
    ds_out.SetProjection(ds_in.GetProjection())
    ds_out.GetRasterBand(1).WriteArray(data_out)
    ds_out.FlushCache()
    ds_in = None
    band_in = None
    ds_out = None


def getColors(nc):
    '''
    makes one array os nc colors between 0 and 255
    '''
    colors = []
    salto = int(255/nc)
    for i in range(0, 255, salto+1):
        colors.append(i)
    return colors


def getWaterClass(Z, km_classes, colors):
    '''
    returns the water class based on the kmeans class with the minimum mean value
    on the swir1 band
    '''
    means = []
    for c in colors:
        means.append(np.mean(Z[km_classes == c]))
    water_class = np.argmin(means)
    return colors[water_class]


def saveIndex(in_array, out, template_path, dType=gdal.GDT_Float32):
    '''
    Description:
    ------------
    Saves water index to tiff image

    Arguments:
    ------------
    - in_array (numpy matrix): water index data
    - out (string): output path to the tiff image
    - template_path (string): template image to copy resolution, bounding box
    and coordinate reference system.
    - dType: data type format (default: float 32 bits)

    Returns:
    ------------
    None

    '''

    if os.path.exists(out):
        os.remove(out)

    template = gdal.Open(template_path)
    driver = gdal.GetDriverByName('GTiff')
    shape = in_array.shape
    dst_ds = driver.Create(
        out, xsize=shape[1], ysize=shape[0], bands=1, eType=dType)
    proj = template.GetProjection()
    geo = template.GetGeoTransform()
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(in_array)
    dst_ds.FlushCache()
    dst_ds = None


def getIndexMask(index_path, thr_method, tol_area=300):
    '''
    Description:
    ------------
    Computes binary mask from water index using the standar value 0 for
    segmentation

    Arguments:
    ------------
    - index_path (string): path to water index
    - thr_method (string): method to segmentation threshold computation
      {'0': standard zero, '1': otsu bimodal, '2': otsu multimodal with 3 clases}
    - tol_area (int): tolerance to remove small holes. Default: 300

    Returns:
    ------------
    - imgmask (numpy matrix): if area removing is enabled
    - index_copy (numpy matrix): if area removing is disabled

    '''

    index_ds = gdal.Open(index_path)
    band = index_ds.GetRasterBand(1)
    index_data = band.ReadAsArray()
    index_data[index_data == float('-inf')] = 0.
    index_data = np.nan_to_num(index_data)  # replace nan values by 0

    # tolerance for segmentation
    if thr_method == '0':
        tol = 0
    if thr_method == '1':
        cimg = index_data.copy()
        vec_cimg = cimg.reshape(cimg.shape[0]*cimg.shape[1])
        tol = threshold_otsu(vec_cimg)
    if thr_method == '2':
        th_otsu_multi = threshold_multiotsu(index_data, 3)
        if abs(th_otsu_multi[0]) < abs(th_otsu_multi[1]):
            tol = th_otsu_multi[0]
        else:
            tol = th_otsu_multi[1]

    # image binarization according threshold
    index_copy = index_data.copy()
    index_copy[index_data < tol] = 0.  # land
    index_copy[index_data >= tol] = 1.  # water

    if tol_area != 0:  # area removing
        img_mask = removeHolesByArea(index_copy.astype(np.byte), tol_area)
        return img_mask
    else:
        return index_copy


def getBandData(band_path):
    '''
    Returns the data matrix from a band path
    '''
    band = gdal.Open(band_path)
    band_data = band.GetRasterBand(1).ReadAsArray()
    return band_data


def createPixelLine(method, mask, cmask):
    '''
    Description:
    ------------
    Computes binary pixel mask for rough shoreline from water index mask.
    Removes clouds areas from binary pixel mask.

    Arguments:
    ------------
    - method (string): erosion or dilation
    - mask (numpy matrix): binary water index mask
    - cmask (numpy matrix): binary cloud mask

    Returns:
    ------------
    - pixel_line (numpy matrix): binary rough pixel shoreline without cloud areas

    '''

    # getting cloud mask data
    cmask_ds = gdal.Open(cmask)
    cmask_band = cmask_ds.GetRasterBand(1)
    cmask_data = cmask_band.ReadAsArray()

    # kernel for cloud mask buffering
    kernel = np.ones((9, 9), np.uint8)

    # getting pixel shoreline mask
    if method == 'erosion':
        erosion = morphology.binary_erosion(mask)
        pixel_line = mask-erosion
    if method == 'dilation':
        dilation = morphology.binary_dilation(mask)
        pixel_line = dilation-mask

    # cleaning pixel line mask using buffer of cloud areas
    cmask_dilation = morphology.binary_dilation(cmask_data, kernel)
    pixel_line[cmask_dilation == 1] = 0
    pixel_line[pixel_line == 1] = 255
    cmask_data = None

    return pixel_line


def removeHolesByArea(img, area):
    '''
    Description:
    ------------
    Removes litle holes from binary images.

    Arguments:
    ------------
    - img (numpy matrix): input image
    - area (int): area tolerance for connected pixels

    Returns:
    ------------
    - pixel_line (numpy matrix): binary rough pixel shoreline without cloud areas

    '''

    img_closed = morphology.area_closing(img, area, connectivity=1)
    img_closed = morphology.area_closing(~img_closed, area, connectivity=1)
    return ~img_closed


def saveMask(img_out, outFilename, base_path):
    '''
    Description:
    ------------
    Saves binary mask to tiff image (byte data type)

    Arguments:
    ------------
    - img_out(numpy matrix): binary mask data
    - out_filename (string): output path to the tiff image
    - base_path (string): template image to copy resolution, bounding box
    and coordinate reference system.

    Returns:
    ------------
    None

    '''

    g = gdal.Open(base_path)
    geoTransform = g.GetGeoTransform()
    geoProjection = g.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    newDataset = driver.Create(outFilename, g.RasterXSize, g.RasterYSize,
                               1, gdal.GDT_Byte, options=["COMPRESS=Deflate"])
    newDataset.SetGeoTransform(geoTransform)
    newDataset.SetProjection(geoProjection)
    newDataset.GetRasterBand(1).WriteArray(img_out.astype(np.uint8))
    newDataset.FlushCache()
    newDataset = None


def getSourceEpsg():
    '''
    Description:
    ------------
    Gets geopgraphic WGS84 coordinate reference system in EPSG format.
    By default, this crs follows the rule (lat,long). It is needed to change
    this to the rule (long,lat).

    Arguments:
    ------------
    None

    Returns:
    ------------
    - source_epsg (object): osr spatial reference object

    '''
    source_epsg = osr.SpatialReference()
    source_epsg.ImportFromEPSG(4326)
    # be careful -> traditional gis order = (long, lat); default = (lat, long)
    source_epsg.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return source_epsg


def getTargetEpsg(scene_path, band_name):
    '''
    Description:
    ------------
    Gets coordinate reference system in EPSG format for a single band

    Arguments:
    ------------
    - scene_path (string): path to the scene folder
    - band_name (string): name of the target band

    Returns:
    ------------
    - source_epsg (object): osr spatial reference object

    '''

    file_list = recursiveFileSearch(scene_path, '*.*')
    band_path = [i for i in file_list if band_name in i][0]
    ds = gdal.Open(band_path)
    epsg_code = osr.SpatialReference(
        wkt=ds.GetProjection()).GetAttrValue('AUTHORITY', 1)
    target_epsg = osr.SpatialReference()
    target_epsg.ImportFromEPSG(int(epsg_code))
    ds = None
    return target_epsg


def reprojectShp(input_shp, output_shp, inSpatialRef, outSpatialRef):
    '''
    Description:
    ------------
    Reprojects one shapefile between two CRSs

    Arguments:
    ------------
    - input_shp (string): path to the input shapefile
    - output_shp (string): path to the output shapefile
    - inSpatialRef (osr spatial reference object): source crs
    - outSpatialRef (osr spatial reference object): target crs

    Returns:
    ------------
    None

    '''
    # getting transformation matrix between source and target crs
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # create output shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    inDataSet = driver.Open(input_shp)
    inLayer = inDataSet.GetLayer()
    inLayer_geomtype = inLayer.GetGeomType()
    inLayer.ResetReading()

    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    outDataSet = driver.CreateDataSource(output_shp)
    outLayer = outDataSet.CreateLayer(
        " ", geom_type=inLayer_geomtype)  # ogr.wkbPolygon

    # copy input shapefile database structure to the output shapefile
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    outLayerDefn = outLayer.GetLayerDefn()

    # copy attribute values and geometries from input shapefile
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        geom = inFeature.GetGeometryRef()
        if not geom is None:
            geom.Transform(coordTrans)
            outFeature = ogr.Feature(outLayerDefn)
            outFeature.SetGeometry(geom)
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(outLayerDefn.GetFieldDefn(
                    i).GetNameRef(), inFeature.GetField(i))
            outLayer.CreateFeature(outFeature)
            outFeature = None
        inFeature = inLayer.GetNextFeature()

    inLayer.ResetReading()
    inLayer = None
    inDataSet.Destroy()
    outLayer = None
    outDataSet.Destroy()

    # create ESRI.prj file
    outSpatialRef.MorphToESRI()
    file_name = os.path.basename(output_shp)
    dir_name = os.path.dirname(output_shp)
    prj_name = str(pathlib.Path(os.path.join(
        dir_name, file_name.split('.')[0]+'.prj')))
    with open(prj_name, 'w') as prj_file:
        prj_file.write(outSpatialRef.ExportToWkt())


def createShapefileFromRasterFootprint(raster_path, output_shp, target_epsg, geom_type='polygon'):
    '''
    Description:
    ------------
    Creates a shapefile from a raster footprint

    Arguments:
    ------------
    - raster_path (string): path to the input raster
    - output_shp (string): path to the output shapefile
    - target_epsg (osr spatial reference object): crs for the output shp
    - geom_type (string): type of geometry for the output shapefile

    Returns:
    ------------
    None

    '''
    footprint = getRasterFootprint(raster_path)
    dic_geom = {'polygon': ogr.wkbPolygon,
                'point': ogr.wkbPoint, 'line': ogr.wkbLineString}
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(output_shp)
    layer = data_source.CreateLayer(' ', target_epsg, dic_geom[geom_type])
    layer.CreateField(ogr.FieldDefn("Iden", ogr.OFTInteger))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Iden", 1)
    feature.SetGeometry(footprint)
    layer.CreateFeature(feature)
    feature = None
    data_source = None


def getRasterFootprint(raster_path):
    '''
    Description:
    ------------
    Gest raster footprint as polygon geometry in wkb format

    Arguments:
    ------------
    - raster_path (string): path to the input raster

    Returns:
    ------------
    - footprint (string): polygon geometry in wkb format

    '''

    # Get raster geometry
    raster = gdal.Open(raster_path)
    transform = raster.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    xLeft = transform[0]
    yTop = transform[3]
    xRight = xLeft+cols*pixelWidth
    yBottom = yTop+rows*pixelHeight

    # image footprint
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xLeft, yTop)
    ring.AddPoint(xLeft, yBottom)
    ring.AddPoint(xRight, yBottom)
    ring.AddPoint(xRight, yTop)
    ring.AddPoint(xLeft, yTop)
    footprint = ogr.Geometry(ogr.wkbPolygon)
    footprint.AddGeometry(ring)

    return footprint


def clipShapefile(input_shp, output_shp, clip_shp):
    '''
    Description:
    ------------
    Clips one single shapefile using a second shapefile

    Arguments:
    ------------
    - input_shp (string): path to the input shapefile
    - output_shp (string): path to the output shapefile
    - clip_shp (string): path to the shapefile to clip input shapefile

    Returns:
    ------------
    None

    '''

    driver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = driver.Open(input_shp, 0)
    inLayer = inDataSource.GetLayer()

    inClipSource = driver.Open(clip_shp, 0)
    inClipLayer = inClipSource.GetLayer()

    outDataSource = driver.CreateDataSource(output_shp)
    outLayer = outDataSource.CreateLayer('clip', geom_type=ogr.wkbMultiPolygon)

    ogr.Layer.Clip(inLayer, inClipLayer, outLayer)

    # create ESRI.prj file
    outSpatialRef = inLayer.GetSpatialRef()
    outSpatialRef.MorphToESRI()
    file_name = os.path.basename(output_shp)
    dir_name = os.path.dirname(output_shp)
    prj_name = str(pathlib.Path(os.path.join(
        dir_name, file_name.split('.')[0]+'.prj')))
    with open(prj_name, 'w') as prj_file:
        prj_file.write(outSpatialRef.ExportToWkt())

    inDataSource = None
    inClipSource = None
    outDataSource = None


def rasterizeShapefile(input_shp, output_raster, raster_template, bc):
    '''
    Description:
    ------------
    Converts input shapefile to raster TIFF file according to
    the raster template features (spatial resolution, bounding box
    and CRS).
    All geometries in the shapefile will be rasterized to create a binary
    raster with values 0 or 1.

    Arguments:
    ------------
    - input_shp (string): path to the input shapefile
    - output_raster (string): path to the output raster file
    - raster_template (string): path to the raster template file
    - bc (string): code to filter especific beach

    Returns:
    ------------
    NONE

    '''

    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = driver.Open(input_shp, 0)
    template_ds = gdal.Open(raster_template)
    lyr = shp_ds.GetLayer()
    geot = template_ds.GetGeoTransform()
    prj = template_ds.GetProjection()
    driver = gdal.GetDriverByName('GTiff')
    new_raster_ds = driver.Create(
        output_raster, template_ds.RasterXSize, template_ds.RasterYSize, 1, gdal.GDT_Byte)
    new_raster_ds.SetGeoTransform(geot)
    new_raster_ds.SetProjection(prj)
    # filter by beach code if needed
    if bc == '(NONE)':
        gdal.RasterizeLayer(new_raster_ds, [1], lyr)
    else:
        lyr.SetAttributeFilter("BEACH_CODE IN "+bc)
        if lyr.GetFeatureCount() == 0:
            lyr.SetAttributeFilter('')
        gdal.RasterizeLayer(new_raster_ds, [1], lyr)
    new_raster_ds.GetRasterBand(1).SetNoDataValue(2)
    new_raster_data = new_raster_ds.GetRasterBand(1).ReadAsArray()
    new_raster_data[new_raster_data == 255] = 1
    new_raster_data[new_raster_data != 1] = 0
    new_raster_ds.GetRasterBand(1).WriteArray(new_raster_data)
    new_raster_ds.FlushCache()
    new_raster_ds = None
    new_raster_data = None


def maskPixelLine(pixel_line_path, mask_path):
    '''
    Description:
    ------------
    Masks binary rough pixel shoreline with beaches mask. All pixels outside of
    a beach area will be removed (converted to value 0).

    Arguments:
    ------------
    - pixel_line_path (string): path to the pixel line mask
    - mask_path (string): path to the binary beaches mask

    Returns:
    ------------
    None

    '''

    pl_ds = gdal.Open(pixel_line_path, gdal.GA_Update)
    pl_band = pl_ds.GetRasterBand(1)
    pl_data = pl_band.ReadAsArray()

    mask_ds = gdal.Open(mask_path)
    mask_band = mask_ds.GetRasterBand(1)
    mask_data = mask_band.ReadAsArray()

    pl_data[mask_data == 0] = 0
    pl_band.WriteArray(pl_data)
    pl_ds.FlushCache()
    pl_ds = None


# *******************************************************************************
# SECTION: SHORELINE EXTRACTION FUNCTIONS
# *******************************************************************************

def escribeCoords(x, y, xmin, ymax, resol_orig, resol, wm, fil, col, kernel, output_file):
    '''
    Description:
    ------------
    Write extracted contours vertices coordinates to the .txt file.
    The original points are in subpixel image coordinates. They have to be
    converted to world coordinates

    Arguments:
    ------------
    - x (list): list of X coordinates
    - y (list): list of Y coordinates
    - xmin (float): minimum X coordinate of the swir1 image
    - ymin (float): minimum Y coordinate of the swir1 image
    - resol_orig (float): spatial resolution of the swir1 image
    - resol (float): map distance among each extracted point (pixel size / ppp)
    - wm (numpy matrix): weight matrix
    - fil (int): row coordinate for the center pixel in the current kernel
    - col (int): column coordinate for the center pixel in the current kernel
    - kernel (int): kernel size in pixels
    - output_file (string): path to the output file

    Returns:
    ------------
    None

    '''

    for i in range(0, len(x)):
        # coordenadas punto sobre imagen global
        rx = xmin+(col-int(kernel/2.0))*resol_orig+x[i]
        ry = ymax-(fil-int(kernel/2.0))*resol_orig-y[i]
        peso = wm[int(x[i]/resol), int(y[i]/resol)]
        output_file.write(str(rx)+","+str(ry)+","+str(peso)+'\n')


def mejor_curva_pendiente3(v, dx, dy, mp, resol):
    '''
    Description:
    ------------
    Select best contour based on the highest mean slope and centrality criteria

    Arguments:
    ------------
    - v (list): list of contours
    - dx (numpy matrix): first derivative of the fitting function in X axis (slope)
    - dy (numpy matrix): first derivative of the fitting function in Y axis (slope)
    - mp (numpy matrix): weight matrix (centrality criteria)

    Returns:
    ------------
    - candidate (int): index of the selected contour

    '''
    pendientes = []
    p_max = 0
    candidate = -1
    for i, curva in enumerate(v):
        for par in curva:
            x = par[0]
            y = par[1]
            px = abs(polyval2d([x], [y], dx))
            py = abs(polyval2d([x], [y], dy))
            p = np.sqrt(px**2+py**2)
            peso = mp[int(x/resol), int(y/resol)]
            pendientes.append(p*peso)
        p_med = np.average(pendientes)
        if p_med >= p_max:
            p_max = p_med
            candidate = i
        pendientes = []
    return candidate


def get_contour_verts(cn):
    '''
    Description:
    ------------
    Extract vertices from a contour

    Arguments:
    ------------
    - cn (object): matplotlib contour object

    Returns:
    ------------
    - contours (list): list of contours vertices

    '''

    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours


def verticeslaplaciano(x, y, m, kernel, ppp):
    '''
    Description:
    ------------
    Computes contour points from laplacian function = 0.
    Uses matplotlib contour function

    Arguments:
    ------------
    - x, y, m (numpy 1D arrays): x, y , z coordinates for laplacian function
    - axis (string): axis to compute the derivative function
    - kernel (int): kernel size in pixels. Must be an odd number
    - ppp (int): points per pixel. Number of points per pixel extracted.

    Returns:
    ------------
    - v (list): list of contour vertices

    '''
    clf()
    v = []
    zz = polyval2d(x, y, m)
    x = np.reshape(x, (kernel*ppp, kernel*ppp))
    y = np.reshape(y, (kernel*ppp, kernel*ppp))
    zz = np.reshape(zz, (kernel*ppp, kernel*ppp))
    try:  # Prevents errors in contour computing
        CS = contour(x, y, zz, 0, colors='y')
        curvas = get_contour_verts(CS)
        for curva in curvas:
            for parte in curva:
                v.append(parte)
        return v
    except:
        return None


def deriva(m, axis):
    '''
    Description:
    ------------
    Computes derivative function of a matrix in a particular axis

    Arguments:
    ------------
    - m (numpy matrix): input matrix
    - axis (string): axis to compute the derivative function

    Returns:
    ------------
    - nm (numpy array): derivative function

    '''

    f, c = m.shape
    if axis == 'x':
        factores = range(1, c)
        nm = m[:, range(1, c)]
        nm = nm*factores
        ceros = np.zeros((f,), dtype=np.float)
        nm = np.vstack((nm.T, ceros.T)).T
        return nm

    if axis == 'y':
        factores = range(1, f)
        nm = m[range(1, f), :]
        nm = (nm.T*factores).T
        ceros = np.zeros((c,), dtype=np.float)
        nm = np.vstack((nm, ceros))
        return nm


def createData(image, resol):
    '''
    Description:
    ------------
    Creates x, y, z arrays of the resampled kernel

    Arguments:
    ------------
    - image (numpy matrix): resampled kernel
    - resol (float): swir1 spatial resolution

    Returns:
    ------------
    - z, y, z (float arrays)

    '''
    inicio = resol-(resol/2.0)  # pixel center
    z = (np.ravel(image)).astype(float)
    tamdata = int(np.sqrt(len(z)))
    x, y = np.meshgrid(np.arange(inicio, tamdata*resol, resol),
                       np.arange(inicio, tamdata*resol, resol))
    x = (np.ravel(x)).astype(float)
    y = (np.ravel(y)).astype(float)
    return x, y, z


def normcdf(x, mu, sigma):
    '''
    Description:
    ------------
    Computes the normal distribution value

    Arguments:
    ------------
    - x (float): distance from the center of kernel
    - mu: mean of the normal distribution
    - sigma: standar deviation of the normal distribution

    Returns:
    ------------
    - y (float): normal distribution value

    '''

    t = x-mu
    y = 0.5*erfcc(-t/(sigma*sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y


def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196 +
                                                 t*(.09678418+t*(-.18628806+t*(.27886807 +
                                                                               t*(-1.13520398+t*(1.48851587+t*(-.82215223 +
                                                                                                               t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r


def computeWeights(kernel, ppp):
    '''
    Description:
    ------------
    Computes a matrix with values that follows a normal distribution.
    It is used to ponderate the extracted points based on the distance
    of each point to the center of the image kernel

    Arguments:
    ------------
    - kernel (int): kernel size in pixels. Must be an odd number
    - ppp (int): points per pixel. Number of points per pixel extracted.

    Returns:
    ------------
    - p (numpy matrix): weights matrix

    '''

    p = np.zeros((kernel*ppp, kernel*ppp))
    f, c = p.shape
    cont_i = cont_j = 1.0
    for i in range(0, f):
        for j in range(0, c):
            d = np.sqrt((cont_i-(float(f)+1.0)/2.0)**2 +
                        (cont_j-(float(c)+1.0)/2.0)**2)
            p[i, j] = normcdf(-d, 0, 3)*2
            cont_j += 1
        cont_i += 1
        cont_j = 1
    return p


def extractPoints(source_path, pl_path, processing_path, kernel, ppp, degree):
    '''
    Description:
    ------------
    Extract subpixel shoreline points based on kernel analysis over swir1 band, taking as
    template the binary mask of rough shoreline pixel line.
    Values standar used in most of the previous studies with good results are:
    3 (kernel), 4 (ppp), 3 (degree).

    for more information about this algorithm and some results:

    - "Automatic extraction of shorelines from Landsat TM and ETM+ multi-temporal images with subpixel precision". 2012.
      Remote Sensing of Environment. Josep E.Pardo-Pascual, Jaime Almonacid-Caballer, Luis A.Ruiz, Jesus Palomar-Vazquez.

    - "Assessing the Accuracy of Automatically Extracted Shorelines on Microtidal Beaches from Landsat 7,
    Landsat 8 and Sentinel-2 Imagery". 2018. Remote Sensing. Josep E. Pardo-Pascual, Elena Sanchez-Garcia, Jaime Almonacid-Caballer
    Jesus Palomar-Vazquez, Enrique Priego de los Santos, Alfonso Fernández-Sarría, Angel Balaguer-Beser.


    Arguments:
    ------------
    - source_path (string): path to the swir1 band
    - pl_path (string): path to the binary mask of rough shoreline pixel line.
    - processing_path (string): path to the folder to storage results (for each scene,
      this folder is named "temp".
    - kernel (int): kernel size in pixels. Must be an odd number
    - ppp (int): points per pixel. Number of points per pixel extracted. 4 points
      in a 20 m size resolution image means 1 point every 5 meters.
    - degree (int): degree for the mathematical fitting function. Standard values
      are 3 or 5.


    Returns:
    ------------
    - True or False (extraction was success or not)

    '''

    # opens swir1 image
    source_ds = gdal.Open(source_path)
    source_band = source_ds.GetRasterBand(1)
    source_data = source_band.ReadAsArray()

    # opens pixel line mask image
    pl_ds = gdal.Open(pl_path)
    pl_band = pl_ds.GetRasterBand(1)
    pl_data = pl_band.ReadAsArray()

    # creates output text file for coordinate points
    base_name = os.path.basename(source_path).split('.')[0]
    source_data[source_data == float('-inf')] = 0
    if os.path.isfile(str(pathlib.Path(os.path.join(processing_path, base_name+'.d')))):
        os.remove(str(pathlib.Path(os.path.join(processing_path, base_name+'.d'))))
    file_coord = open(str(pathlib.Path(os.path.join(
        processing_path, base_name+'.d'))), 'a')

    # gets swir1 features
    geoTrans = source_ds.GetGeoTransform()
    minXimage = geoTrans[0]
    maxYimage = geoTrans[3]
    dim = source_data.shape
    rows = dim[0]
    columns = dim[1]

    offset = 10  # number of rows and columns preserved to avoid overlapping in adjacent scenes
    c1 = f1 = offset
    c2 = columns - offset
    f2 = rows - offset
    resol_orig = geoTrans[1]  # pixel size
    resol = float(geoTrans[1])/ppp  # point distance
    gap = int(kernel/2)
    points_x = []
    points_y = []
    wm = computeWeights(kernel, ppp)  # weights matrix
    white_pixel = False
    for f in tqdm(range(f1, f2)):
        for c in range(c1, c2):
            valor = pl_data[f, c]
            if valor == 255:  # pixel belongs to the rough pixel line
                white_pixel = True
                nf = f
                nc = c
                # sub-matrix based on kernel size
                sub = source_data[nf-gap:nf+kernel-gap, nc-gap:nc+kernel-gap]
                # sub-matrix resampling based on ppp value
                sub_res = rescale(sub, scale=ppp, order=3, mode='edge')
                cx, cy, cz = createData(sub_res, resol)  # resampled data
                m = polyfit2d(cx, cy, cz, deg=degree)  # fitting data

                # computes laplacian function
                dx = deriva(m, 'x')
                d2x = deriva(dx, 'x')
                dy = deriva(m, 'y')
                d2y = deriva(dy, 'y')
                laplaciano = d2x+d2y

                # get contour points for laplacian = 0
                v = verticeslaplaciano(cx, cy, laplaciano, kernel, ppp)

                if v != None:
                    if len(v) != 0:
                        # if there are more than one contour, we select the contour with highest slope and more centered
                        indice = mejor_curva_pendiente3(v, dx, dy, wm, resol)
                        if indice != -1:
                            linea = v[indice]
                            for i in range(0, len(linea)):
                                par = linea[i]
                                points_x.append(par[0])
                                points_y.append(par[1])

                            # writes the contour points to the text file
                            escribeCoords(points_x, points_y, minXimage, maxYimage,
                                          resol_orig, resol, wm, nf, nc, kernel, file_coord)
                            points_x = []
                            points_y = []
    file_coord.close()

    if white_pixel:
        # variable release
        del source_ds
        del source_band
        del source_data
        del pl_data
        del sub
        del sub_res
        del m
        del cx
        del cy
        del cz
        del wm
        gc.collect()
        return True

    return False

# *******************************************************************************
# SECTION: AVERAGE POINTS FUNCTIONS
# *******************************************************************************


def averagePoints(source_path, processing_path, cluster_distance, min_cluster_size):
    '''
    Description:
    ------------
    Takes the subpixel rough extracted points and computes the average points.
    The algorithm scan in X and Y direction points with the same coordinates and
    makes groups of points by using clustering criteria as maximum distance among
    points and minimum number of points in a cluster.
    Creates a .txt file with the results

    Arguments:
    ------------
    - source_path (string): path to the scene
    - processing_path (string): path to the processing folder
    - cluster_distance (int): maximum distance between two points to be consider a cluster
    - min_cluster_size (int): minimum number of points in a cluster

    Returns:
    ------------
    None

    '''

    # reading coordinates from extracted subpixel points in the kernel analysis
    base_name = os.path.basename(source_path).split('.')[0]
    file_name = str(pathlib.Path(
        os.path.join(processing_path, base_name+'.d')))
    with open(file_name, 'r') as fichero:
        iter1 = csv.reader(fichero, delimiter=',')
        datos = np.asarray([[dato[0], dato[1]]
                           for dato in iter1]).astype(float)
        fichero.seek(0)
        iter2 = csv.reader(fichero, delimiter=',')
        pesos = np.asarray([dato[2] for dato in iter2]).astype(float)

    ejex = np.unique(datos[:, 0])  # unique values on the x-axis
    ejey = np.unique(datos[:, 1])  # unique values on the y-axis

    # computing clusters
    medias = creaCluster(datos, pesos, ejex, ejey,
                         cluster_distance, min_cluster_size)

    # writes results to the output file (average x and average y of every cluster)
    with open(str(pathlib.Path(os.path.join(processing_path, base_name+'.m'))), 'w') as fichero:
        for media in medias:
            fichero.write(str(media[0])+","+str(media[1])+"\n")


def creaCluster(d, p, ex, ey, cluster_distance, min_cluster_size):
    '''
    Description:
    ------------
    Makes groups of points acording clustering criteria (maximum distance among
    points and minimum number of points in a cluster). From each group, the algorithm
    computes a ponderate average value for x and y coordinates based on a weight matrix.

    Arguments:
    ------------
    - d (numpy array): list of X-Y coordinates
    - p (numpy matrix): weight matrix
    - ex (numpy array): unique values on the x-axis
    - ey (numpy array): unique values on the y-axis
    - cluster_distance (int): maximum distance between two points to be consider a cluster
    - min_cluster_size (int): minimum number of points in a cluster

    Returns:
    ------------
    - average_points (list): list of average points for each cluster

    '''

    tol = cluster_distance
    average_points = []

    # clustering in x-axis
    for x in ex:
        id_x = np.nonzero(d[:, 0] == x)
        cy = d[:, 1][id_x]
        pey = p[id_x]
        if len(cy) >= 2:
            orig_coord, pos = getClusters(cy, tol)
            for cp in pos:
                cluster = orig_coord[cp]
                if len(cluster) >= min_cluster_size:
                    p_cluster = pey[cp]
                    media_y = np.average(cluster, weights=p_cluster)
                    average_points.append([x, media_y])

    # clustering in y-axis
    for y in ey:
        id_y = np.nonzero(d[:, 1] == y)
        cx = d[:, 0][id_y]
        pex = p[id_y]
        if len(cx) >= 2:
            orig_coord, pos = getClusters(cx, tol)
            for cp in pos:
                cluster = orig_coord[cp]
                if len(cluster) >= min_cluster_size:
                    p_cluster = pex[cp]
                    media_x = np.average(cluster, weights=p_cluster)
                    average_points.append([media_x, y])
    return average_points


def getClusters(coord, tol):
    '''
    Description:
    ------------
    Makes groups of points based on a maximum distance.

    Arguments:
    ------------
    - coord (list): list of point coordinates with the same x or y value
    - tol (int): cluster distance (maximum distance between two points to
      be consider a cluster)

    Returns:
    ------------
    - orig_coord (list): list of point coordinates with the same x or y value
    - pos (list): index of points that belong tho the same cluster

    '''

    clusters = []
    cluster = []
    orig_coord = coord.copy()
    coord.sort()
    cluster.append(0)
    for i in range(0, len(coord)-1):
        current = coord[i]
        siguiente = coord[i+1]
        dist = siguiente-current
        if dist <= tol:
            cluster.append(i+1)
        else:
            clusters.append(cluster)
            cluster = []
            cluster.append(i+1)
    clusters.append(cluster)
    parcial = []
    pos = []
    for c in clusters:
        for iden in c:
            a, = np.where(orig_coord == coord[iden])
            parcial.append(a[0])
        pos.append(parcial)
        parcial = []
    return orig_coord, pos


# *******************************************************************************
# SECTION: POINT CLEANING FUNCTIONS
# *******************************************************************************

def cleanPoints2(shp_path, tol_rba, level):
    '''
    Description:
    ------------
    Remove outliers points based on two criteria:
    - longest spanning tree algorithm (LST).
    - angle tolerance.

    To improve the performance, the algorithm uses an initial Delaunay triangulation
    to create a direct graph.

    Two versions of cleaned points shapefile is created: point and line versions

    More information:
    "An efficient protocol for accurate and massive shoreline definition from
    mid-resolution satellite imagery". 2020. Coastal Engineering. E. Sanchez-García,
    J.M. Palomar-Vazquez, J.E. Pardo-Pascual, J. Almonacid-Caballer, C. Cabezas-Rabadan,
    L. Gomez-Pujol.

    Arguments:
    ------------
    - shp_path (string): path to the points shapefile
    - tol_rba (int): angle tolerance
    - level (int): takes one point every n (level) points. Speeds the process

    Returns:
    ------------
    None

    '''
    # opens the shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    source_ds = driver.Open(shp_path, 0)
    prj = source_ds.GetLayer().GetSpatialRef()
    base_name = os.path.basename(shp_path).split('.')[0]
    dir_name = os.path.dirname(shp_path)
    layer = source_ds.GetLayer()

    # gest list of unique BEACH_CODE values
    ids = []
    for feature in layer:
        id_feat = feature.GetField('BEACH_CODE')
        if not id_feat is None:
            ids.append(id_feat)
    ids = list(set(ids))
    layer.ResetReading()

    ids.sort()
    # prevents from points with BEACH_CODE 0 (outside of any beach area).
    if ids[0] == 0:
        ids.remove(0)  # removes points

    # creates groups of points with the same BEACH_CODE value
    groups = []
    identifiers = []
    for id_feat in ids:
        geometries = []
        layer.SetAttributeFilter("BEACH_CODE = "+str(id_feat))
        for feature in layer:
            geom = feature.GetGeometryRef()
            if not geom is None:
                geometries.append(geom.Clone())
        groups.append(geometries)
        identifiers.append(id_feat)

    # process each group separately
    clean_geometries = []
    level = 1
    for i in range(0, len(groups)):
        group = groups[i]
        identifier = identifiers[i]
        coords = []
        ng = float(len(group))
        # prevents from too much long numer of points in a group
        # level = ceil(ng/group_size)
        for i in range(0, len(group), level):
            geom = group[i].Clone()
            coords.append([geom.GetX(), geom.GetY()])
        points = np.array(coords)
        if len(points >= 4):  # delaunay triangulation needs 4 or more points
            try:
                tri = Delaunay(points)
                # list of triangles
                lista_tri = tri.simplices
                # list of ids of the connected points wiht LST
                lst = computeLST(lista_tri, points)
                # remove LST point by angle tolerance
                clean_points = cleanPointsByAngle(lst, points, tol_rba)
                # list of cleaned points with its identifier
                clean_geometries.append([clean_points, identifier])
            except:
                pass

    # crates point and line versions of the cleaned points
    makePointShp(str(pathlib.Path(os.path.join(dir_name, base_name +
                 '_cp.shp'))), clean_geometries, prj)
    makeLineShp(str(pathlib.Path(os.path.join(dir_name, base_name +
                '_cl.shp'))), clean_geometries, prj)
    source_ds = None


def computeLST(lista_tri, puntos):
    '''
    Description:
    ------------
    Computes Longest Spanning Tree (LST) of a directed graph.

    Arguments:
    ------------
    - lista_tri (list): list of triangles from Delaunay triangulation
    - puntos (list): list of original points (unclean)

    Returns:
    ------------
    final (list): list of index of points belonging to the LST

    '''

    # creates a graph from triangles
    G = nx.Graph()
    for tri in lista_tri:
        G.add_edge(tri[0], tri[1], weight=dist(puntos[tri[0]], puntos[tri[1]]))
        G.add_edge(tri[1], tri[2], weight=dist(puntos[tri[1]], puntos[tri[2]]))
        G.add_edge(tri[2], tri[0], weight=dist(puntos[tri[2]], puntos[tri[0]]))
    # computes Minimum Spanning tree (MST)
    MST = nx.minimum_spanning_tree(G)
    MST_directed_graph = nx.Graph(MST).to_directed()
    # select the first end point of the grahp from and arbitrary point
    lpath1 = nx.single_source_bellman_ford_path_length(MST_directed_graph, 1)
    indice1 = list(lpath1.keys())[-1]
    # select the second end point of the graph from the fisrt end point
    lpath2 = nx.single_source_bellman_ford_path_length(
        MST_directed_graph, indice1)
    indice2 = list(lpath2.keys())[-1]
    # computes the MST between the first and second end points (LST)
    final = nx.dijkstra_path(MST_directed_graph, indice1, indice2)
    return final


def dist(p1, p2):
    '''
    Description:
    ------------
    Returns distance between two couple of coordinates.

    Arguments:
    ------------
    - p1 (list): first couple of coordinates
    - p2 (list): second couple of coordinates

    Returns:
    ------------
    - (float): distance

    '''
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def cleanPointsByAngle(lst, puntos, tol_rba):
    '''
    Description:
    ------------
    Removes a point if the angle formed between the anterior and posterior
    point is greater than a tolerance.

    Arguments:
    ------------
    - lst (list): list of index in LST graph
    - puntos (list): list of LST points
    - tol_rba (int): angle tolerance

    Returns:
    ------------
    - clean_points (list): list of point ogr geometries

    '''

    clean_points = []
    lineas = []
    for iden in lst:
        lineas.append(puntos[iden])
    if tol_rba != None:
        lineas = removeByAngle(lineas, tol_rba)

    for i in range(0, len(lineas)):
        new_geom = ogr.Geometry(ogr.wkbPoint)
        pt = lineas[i]
        new_geom.AddPoint(pt[0], pt[1])
        clean_points.append(new_geom.Clone())
    return clean_points


def removeByAngle(puntos, tol):
    '''
    Description:
    ------------
    Removes a point if the angle formed between the anterior and posterior
    point is greater than a tolerance. Function included in cleanPointsByAngle()

    Arguments:
    ------------
    - puntos (list): list of LST points
    - tol (int): angle tolerance

    Returns:
    ------------
    - final_points (list): list of cleaned points

    '''

    final_points = []
    for i in range(0, len(puntos)-2):
        a = puntos[i]
        b = puntos[i+1]
        c = puntos[i+2]
        ang = degrees(atan2(c[1]-b[1], c[0]-b[0]) -
                      atan2(a[1]-b[1], a[0]-b[0]))
        if ang < 0:
            ang = ang + 360
        if ang >= tol:
            final_points.append(puntos[i+1])
    final_points.insert(0, puntos[0])
    final_points.append(puntos[len(puntos)-1])
    return final_points


def makePointShp(shp_path, source_list, prj):
    '''
    Description:
    ------------
    Creates a point shapefile from a list of points.

    Arguments:
    ------------
    - shp_path (string): path to the output shapefile
    - source_list (list of list): list of points
    - prj (object): ogr coordinate spatial reference object

    Returns:
    ------------
    None

    '''
    # creates empty point shapefile
    data_source, layer = createEmptyShp(shp_path, 'point', prj)
    layer_defn = layer.GetLayerDefn()

    # adds points to new shapefile along with the BEACH_CODE attribute
    for i in range(0, len(source_list)):
        points = source_list[i][0]
        identifier = source_list[i][1]
        for pt in points:
            feat = ogr.Feature(layer_defn)
            feat.SetGeometry(pt)
            feat.SetField('BEACH_CODE', identifier)
            layer.CreateFeature(feat)
    data_source.FlushCache()
    data_source = None


def makeLineShp(shp_path, source_list, prj):
    '''
    Description:
    ------------
    Creates a polyline shapefile from a list of points.

    Arguments:
    ------------
    - shp_path (string): path to the output shapefile
    - source_list (list of list): list of points
    - prj (object): ogr coordinate spatial reference object

    Returns:
    ------------
    None

    '''
    # creates empty point shapefile
    data_source, layer = createEmptyShp(shp_path, 'line', prj)
    layer_defn = layer.GetLayerDefn()

    # crates lines from  each point sublist to new shapefile along with
    # the BEACH_CODE attribute
    for i in range(0, len(source_list)):
        points = source_list[i][0]
        identifier = source_list[i][1]
        feat = ogr.Feature(layer_defn)
        new_geom = ogr.Geometry(ogr.wkbLineString)
        for point_geom in points:
            new_geom.AddPoint(point_geom.GetX(), point_geom.GetY())
        feat.SetGeometry(new_geom)
        feat.SetField('BEACH_CODE', identifier)
        layer.CreateFeature(feat)
    data_source.FlushCache()
    data_source = None


def createEmptyShp(shp_path, geom_type, prj):
    '''
    Description:
    ------------
    Creates an empty shapefile of a specific geometry and projection.
    Adds the field 'BEACH_CODE'.

    Arguments:
    ------------
    - shp_path (string): path to the output shapefile
    - geom_type (string): type of geometry ('point', 'line' or 'polygon')
    - prj (object): ogr coordinate spatial reference object

    Returns:
    ------------
    - data_source (object): ogr data source object
    - layer (object): ogr layer object

    '''

    dict_geom = {'point': ogr.wkbPoint,
                 'line': ogr.wkbLineString, 'polygon': ogr.wkbPolygon}
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)
    data_source = driver.CreateDataSource(shp_path)
    layer = data_source.CreateLayer(
        ' ', geom_type=dict_geom[geom_type], srs=prj)
    id_field = ogr.FieldDefn("BEACH_CODE", ogr.OFTInteger)
    id_field.SetWidth(10)
    layer.CreateField(id_field)

    return data_source, layer

# *******************************************************************************
# SECTION: SHAPEFILE CREATION FUNCTIONS
# *******************************************************************************


def createShpFromAverageFile(source_path, processing_path):
    '''
    Description:
    ------------
    Converts average point coordinates from .txt file to .shp file.
    The name of the shapefile is based on the name of the .xtx file.
    In order to copy the beach code to each average point, an attribute
    field call "BEACH_CODE" is added.

    Arguments:
    ------------
    - source_path (string): path to the template swir1 image
    - processing_path (string): path to the processing output path.

    Returns:
    ------------
    - shp_path (string): path to the .shp file

    '''

    # gets projection from template image
    source_ds = gdal.Open(source_path)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(source_ds.GetProjectionRef())

    # reads average point coordinates from .txt file
    base_name = os.path.basename(source_path).split('.')[0]
    file_name = str(pathlib.Path(
        os.path.join(processing_path, base_name+'.m')))
    with open(file_name, 'r') as f:
        data = csv.reader(f, delimiter=',')
        coords = np.asarray([[dato[0], dato[1]]
                            for dato in data]).astype(float)

    # creates a new shapefile and adds a database structure
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_path = str(pathlib.Path(os.path.join(
        processing_path, base_name+'.shp')))
    if os.path.exists(shp_path):
        driver.DeleteDataSource(shp_path)
    data_source = driver.CreateDataSource(shp_path)
    layer = data_source.CreateLayer(' ', geom_type=ogr.wkbPoint, srs=prj)
    id_field = ogr.FieldDefn("Id_pnt", ogr.OFTInteger)
    id_field.SetWidth(10)
    layer.CreateField(id_field)
    id_field2 = ogr.FieldDefn("BEACH_CODE", ogr.OFTInteger)
    id_field2.SetWidth(10)
    layer.CreateField(id_field2)
    layer_defn = layer.GetLayerDefn()

    # creates shapefile features
    for i in range(0, len(coords)):
        values = coords[i]
        feat = ogr.Feature(layer_defn)
        pnt = ogr.Geometry(ogr.wkbPoint)
        pnt.AddPoint(values[0], values[1])
        feat.SetGeometry(pnt)
        feat.SetField('Id_pnt', i)
        feat.SetField('BEACH_CODE', 0)
        layer.CreateFeature(feat)
    data_source.FlushCache()
    source_ds = None
    return shp_path


def copyShpIdentifiers(shp_polygons, shp_points):
    '''
    Description:
    ------------
    Copies the beach code from the beach shapefile to each average point using
    two geometry intersection test.

    Arguments:
    ------------
    - shp_polygons (string): path to beaches shapefile
    - shp_points (string): path to the points shapefile.

    Returns:
    ------------
    None

    '''

    # opens both polygons and points shapefiles
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds_pol = driver.Open(shp_polygons, 0)
    ds_point = driver.Open(shp_points, 1)

    layer_pol = ds_pol.GetLayer()
    layer_point = ds_point.GetLayer()

    layer_point_defn = layer_point.GetLayerDefn()
    field_names = [layer_point_defn.GetFieldDefn(
        i).GetName() for i in range(layer_point_defn.GetFieldCount())]

    # adds beach code field
    if not 'BEACH_CODE' in field_names:  # to ensure that attribute "BEACH_CODE" exists
        new_field = ogr.FieldDefn('BEACH_CODE', ogr.OFTInteger)
        layer_point.CreateField(new_field)

    # populates beach code field
    for feat_pol in layer_pol:
        id_feat_pol = feat_pol.GetField('BEACH_CODE')
        geom_pol = feat_pol.GetGeometryRef()
        geom_envelope = envelopeToGeom(geom_pol)
        for feat_point in layer_point:
            geom_point = feat_point.GetGeometryRef()
            if geom_point.Intersect(geom_envelope):  # first intersection test
                if geom_point.Intersect(geom_pol):  # second intersection test
                    feat_point.SetField('BEACH_CODE', id_feat_pol)
                    layer_point.SetFeature(feat_point)

    ds_point.FlushCache()
    ds_point = None
    ds_pol = None


def envelopeToGeom(geom):
    '''
    Description:
    ------------
    Returns the bounding box of a polygon geometry.

    Arguments:
    ------------
    - geom (objetc): ogr geometry object

    Returns:
    ------------
    poly_envelope (object): ogr geometry object

    '''

    # gets bounding box from geometry
    (minX, maxX, minY, maxY) = geom.GetEnvelope()

    # creates ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minX, minY)
    ring.AddPoint(maxX, minY)
    ring.AddPoint(maxX, maxY)
    ring.AddPoint(minX, maxY)
    ring.AddPoint(minX, minY)

    # creates polygon
    poly_envelope = ogr.Geometry(ogr.wkbPolygon)
    poly_envelope.AddGeometry(ring)
    return poly_envelope


def copyShpToFolder(input_path, output_path, source_epsg):
    '''
    Description:
    ------------
    Copies final point and line shapefiles of the extracted shoreline to the
    SDS folder. The copied file is previously reprojected to the WGS84 lat-long
    spatial reference system

    Arguments:
    ------------
    - source_epsg(object): spatial reference system for input shapefile
    - input_path (string): path to the input shapefiles
    - output_path (string): path to the output shapefiles

    Returns:
    ------------
    None

    '''
    cp_files = recursiveFileSearch(input_path, '*_cp.shp')
    cl_files = recursiveFileSearch(input_path, '*_cl.shp')

    root_name = os.path.basename(cp_files[0]).split('_cp.')[0]
    output_folder = str(pathlib.Path(os.path.join(output_path, root_name)))
    createFolderCheck(output_folder)
    target_epsg = getSourceEpsg()

    for cp_file in cp_files:
        addIdField(cp_file)
        reprojectShp(cp_file, str(pathlib.Path(
            os.path.join(output_folder, root_name+'_points.shp'))), source_epsg, target_epsg)
        exportToGeojson(str(pathlib.Path(
            os.path.join(output_folder, root_name+'_points.shp'))))

    for cl_file in cl_files:
        addIdField(cl_file)
        reprojectShp(cl_file, str(pathlib.Path(
            os.path.join(output_folder, root_name+'_lines.shp'))), source_epsg, target_epsg)
        exportToGeojson(str(pathlib.Path(
            os.path.join(output_folder, root_name+'_lines.shp'))))


def addIdField(input_path):
    '''
    Description:
    ------------
    Adds the field "ID_FEAT" to the input shapefile

    Arguments:
    ------------
    - input_path (string): path to the input shapefile

    Returns:
    ------------
    None

    '''
    # open shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds_shp = driver.Open(input_path, 1)
    layer_shp = ds_shp.GetLayer()

    layer_shp_defn = layer_shp.GetLayerDefn()
    field_names = [layer_shp_defn.GetFieldDefn(
        i).GetName() for i in range(layer_shp_defn.GetFieldCount())]

    # adds id field
    if not 'ID_FEAT' in field_names:  # to ensure that attribute "ID_FEAT" exists
        new_field = ogr.FieldDefn('ID_FEAT', ogr.OFTInteger)
        layer_shp.CreateField(new_field)

    # populates id field
    id_feat_shp = 0
    for feat_shp in layer_shp:
        feat_shp.SetField('ID_FEAT', id_feat_shp)
        layer_shp.SetFeature(feat_shp)
        id_feat_shp += 1

    ds_shp.FlushCache()
    ds_shp = None


def exportToGeojson(shp_path):
    '''
    Description:
    ------------
    Exports shapefile to GeoJson format

    Arguments:
    ------------
    - shp_path (string): path to the input shapefile

    Returns:
    ------------
    None

    '''
    base_name = os.path.basename(shp_path).split('.')[0]
    dir_name = os.path.dirname(shp_path)
    geojson_path = str(pathlib.Path(
        os.path.join(dir_name, base_name+'.json')))
    with shapefile.Reader(shp_path) as shp:
        geojson_data = shp.__geo_interface__
        with open(geojson_path, 'w') as geojson_file:
            geojson_file.write(json.dumps(geojson_data))


def gmlToShp(gml_path, shp_path):
    '''
    Description:
    ------------
    Converts .gml file to .shp file.
    .gml files contains cloud mask for S2MSI1C product

    Arguments:
    ------------
    - gml_path (string): path to the input .gml file
    - shp_path (string): path to the output .shp file

    Returns:
    ------------
    NONE

    '''

    gml_ds = ogr.Open(gml_path)
    gml_layer = gml_ds.GetLayer()
    if gml_layer is None:
        # prevents empty .gml files (no clouds)
        gml_ds.Destroy()
        return False
    else:
        # get projection and database definition from .gml file
        gml_projection = gml_layer.GetSpatialRef()
        gml_layer_defn = gml_layer.GetLayerDefn()

        # creates shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(shp_path):
            driver.DeleteDataSource(shp_path)
        shp_ds = driver.CreateDataSource(shp_path)
        shp_layer = shp_ds.CreateLayer(
            ' ', geom_type=gml_layer_defn.GetGeomType(), srs=gml_projection)
        in_field_count = gml_layer_defn.GetFieldCount()

        # clones database definition
        for fld_index in range(in_field_count):
            src_fd = gml_layer_defn.GetFieldDefn(fld_index)
            fd = ogr.FieldDefn(src_fd.GetName(), src_fd.GetType())
            fd.SetWidth(src_fd.GetWidth())
            fd.SetPrecision(src_fd.GetPrecision())
            shp_layer.CreateField(fd)

        # copy attributte values and geometries
        in_feat = gml_layer.GetNextFeature()
        while in_feat is not None:
            geom = in_feat.GetGeometryRef().Clone()
            out_feat = ogr.Feature(feature_def=shp_layer.GetLayerDefn())

            for fld_index in range(in_field_count):
                src_fd = gml_layer_defn.GetFieldDefn(fld_index)
                name = src_fd.GetName()
                value = in_feat.GetField(fld_index)
                out_feat.SetField(name, value)

            out_feat.SetGeometry(geom)
            shp_layer.CreateFeature(out_feat)
            out_feat.Destroy()
            in_feat.Destroy()
            in_feat = gml_layer.GetNextFeature()

        gml_ds.Destroy()
        shp_ds.Destroy()
        return True

# *******************************************************************************
# SECTION: SENTINEL-2 PROCESSING FUNCTIONS
# *******************************************************************************


def processSentinel2SceneFromPath(scene_path, title, sds_path, shp_path, wi_type, thr_method, cloud_mask_level, morphology_method, kernel_size, bc):
    '''
    Description:
    ------------
    Run the whole workflow reprocessing Sentinel-2 scenes from a path.

    Arguments:
    ------------

    - scene_path (string): path to the folder of the Sentinel-2 image
    - title (string): identifier of the image (coincident with the name of the folder)
    - shp_path (string): path to shapefile containing beaches areas
    - wi_type (string): type of water index
    - thr_method (string); type of thresholding method
    - cloud_mask_level (string): cloud masking severity
    - morphology_method (string): type of morphology method
    - kernel_size (int): kernel size for the extraction algorithm
    - bc (list of string): list of beach polygons to be processed

    Returns:
    ------------
    None

    '''
    #print('Processing '+title+' ...')
    logging.info('Processing '+title+' ...')
    processing_path = str(pathlib.Path(
        os.path.join(scene_path, 'temp')))
    #print('Computing water index band...')
    logging.info('Computing water index band...')
    if wi_type == 'aweinsh':
        aweinshS2(scene_path)
    if wi_type == 'aweish':
        aweishS2(scene_path)
    if wi_type == 'mndwi':
        mndwiS2(scene_path)
    if wi_type == 'kmeans':
        computeKmeansS2(scene_path)
    #print('Computing cloud mask...')
    logging.info('Computing cloud mask...')
    createCloudMaskS2(scene_path, cloud_mask_level)
    #print('Computing water index mask...')
    logging.info('Computing water index mask...')
    if wi_type != 'kmeans':
        wi_path = getBandPath(processing_path, 'wi.tif')
        wi_mask = getIndexMask(wi_path, thr_method)
    else:
        wi_path = getBandPath(processing_path, 'kmeans_mask.tif')
        wi_mask = getBandData(wi_path)
    cmask_band = getBandPath(processing_path, 'cmask.tif')
    #print('Computing rough pixel line...')
    logging.info('Computing rough pixel line...')
    pixel_line = createPixelLine(
        morphology_method, wi_mask, cmask_band)
    saveMask(pixel_line, str(pathlib.Path(os.path.join(
        processing_path, 'pl.tif'))), cmask_band)
    source_epsg = getSourceEpsg()
    target_epsg = getTargetEpsg(scene_path, 'B11')
    #print('Reprojecting shp of beaches...')
    logging.info('Reprojecting shp of beaches...')
    reprojectShp(shp_path, str(pathlib.Path(os.path.join(processing_path,
                                                         'bb300_r.shp'))), source_epsg, target_epsg)
    #print('Computing footprint band...')
    logging.info('Computing footprint band...')
    createShapefileFromRasterFootprint(getBandPath(scene_path, 'B11'), str(pathlib.Path(os.path.join(
        processing_path, 'scene_footprint.shp'))), target_epsg, geom_type='polygon')
    #print('Clipping shp of beaches by scene footprint...')
    logging.info('Clipping shp of beaches by scene footprint...')
    clipShapefile(str(pathlib.Path(os.path.join(processing_path, 'bb300_r.shp'))), str(pathlib.Path(os.path.join(
        processing_path, 'clip_bb300_r.shp'))), str(pathlib.Path(os.path.join(processing_path, 'scene_footprint.shp'))))
    #print('Rasterizing beaches subset...')
    logging.info('Rasterizing beaches subset...')
    rasterizeShapefile(str(pathlib.Path(os.path.join(processing_path, 'bb300_r.shp'))), str(pathlib.Path(os.path.join(
        processing_path, 'bb300_r.tif'))), getBandPath(scene_path, 'B11'), bc)
    #print('Masking rough pixel line with beaches subset...')
    logging.info('Masking rough pixel line with beaches subset...')
    maskPixelLine(str(pathlib.Path(os.path.join(processing_path, 'pl.tif'))),
                  str(pathlib.Path(os.path.join(processing_path, 'bb300_r.tif'))))
    #print('Extracting points...')
    logging.info('Extracting points...')
    res = extractPoints(getBandPath(scene_path, 'B11'), str(pathlib.Path(os.path.join(
        processing_path, 'pl.tif'))), processing_path, int(kernel_size), 4, 3)
    if res:
        #print('Computing average points...')
        logging.info('Computing average points...')
        averagePoints(getBandPath(scene_path, 'B11'),
                      processing_path, 50, 3)
        #print('Making point shp...')
        logging.info('Making point shp...')
        shp_path_average = createShpFromAverageFile(
            getBandPath(scene_path, 'B11'), processing_path)
        #print('Transfering beaches identifiers...')
        logging.info('Transfering beaches identifiers...')
        copyShpIdentifiers(str(pathlib.Path(os.path.join(
            processing_path, 'clip_bb300_r.shp'))), shp_path_average)
        #print('Cleaning points and making final shoreline in line vector format...')
        logging.info(
            'Cleaning points and making final shoreline in line vector format...')
        cleanPoints2(shp_path_average, 150, 1)
        #print('Export final shoreline shapefiles to SDS folder...')
        logging.info(
            'Export final shoreline shapefiles to SDS folder...')
        copyShpToFolder(processing_path, sds_path, target_epsg)
    else:
        logging.warning('No results in extraction points process.')
        sys.exit(1)


def downScaling(input_file):
    '''
    Description:
    ------------
    Reduces the image spatial resolution from 10 m. to 20 m.
    Uses gdal.RegenerateOverviews() function.

    Arguments:
    ------------
    - input_file (string): path to image input file

    Returns:
    ------------
    - final (numpy matrix): output image downscaled

    '''

    factor = 2  # ratio to reduce resolution from 10 m to 20 m.
    input_name = os.path.basename(input_file)
    output_path = os.path.dirname(input_file)
    output_name = input_name.split('.')[0]+'_20.tif'
    output_file = str(pathlib.Path(os.path.join(output_path, output_name)))
    logging.info('Downscaling '+input_name+' ...')
    # print('Downscaling '+input_name+' ...')
    g = gdal.Open(input_file, gdal.GA_ReadOnly)
    total_obs = g.RasterCount
    drv = gdal.GetDriverByName("MEM")
    dst_ds = drv.Create("", g.RasterXSize, g.RasterYSize, 1, gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(g.GetGeoTransform())
    dst_ds.SetProjection(g.GetProjectionRef())
    dst_ds.GetRasterBand(1).WriteArray(g.ReadAsArray())

    geoT = g.GetGeoTransform()
    drv = gdal.GetDriverByName("GTiff")
    resampled = drv.Create(output_file, int(
        g.RasterXSize/factor), int(g.RasterYSize/factor), 1, gdal.GDT_UInt16)

    this_geoT = (geoT[0], geoT[1]*factor, geoT[2],
                 geoT[3], geoT[4], geoT[5]*factor)
    resampled.SetGeoTransform(this_geoT)
    resampled.SetProjection(g.GetProjectionRef())
    resampled.SetMetadata({"TotalNObs": "%d" % total_obs})
    gdal.RegenerateOverviews(dst_ds.GetRasterBand(
        1), [resampled.GetRasterBand(1)], 'average')
    resampled.GetRasterBand(1).SetNoDataValue(0)
    resampled.FlushCache()
    final = resampled.GetRasterBand(1).ReadAsArray()
    resampled = None
    return final


def aweinshS2(scene_path):
    '''
    Description:
    ------------
    Computes S2 aweinsh water index for the analysis area. Involved bands: B03(green)
    B08(nir), B11(swir1), B12(swir2).
    Downscales some bands if it is needed to homogenize spatial resolutions

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''

    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # prevents numpy errors for invalid values or divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_wi.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B11')

    # getting bands data and downscaling if it is needed
    band_green = gdal.Open(getBandPath(scene_path, 'B03'))
    pix_size = band_green.GetGeoTransform()[1]
    if pix_size == 10.0:
        data_green = downScaling(getBandPath(
            scene_path, 'B03')).astype(np.float32)
    else:
        data_green = band_green.GetRasterBand(
            1).ReadAsArray().astype(np.float32)

    band_nir = gdal.Open(getBandPath(scene_path, 'B08'))
    pix_size = band_nir.GetGeoTransform()[1]
    if pix_size == 10.0:
        data_nir = downScaling(getBandPath(
            scene_path, 'B08')).astype(np.float32)
    else:
        data_nir = band_nir.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir1 = gdal.Open(getBandPath(scene_path, 'B11'))
    data_swir1 = band_swir1.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir2 = gdal.Open(getBandPath(scene_path, 'B12'))
    data_swir2 = band_swir2.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # computing water index
    aweinsh = 4 * (data_green - data_swir1) - \
        (0.25 * data_nir + 2.75 * data_swir2)

    # saving water index
    saveIndex(aweinsh, outFileName, base_path)


def aweishS2(scene_path):
    '''
    Description:
    ------------
    Computes S2 aweish water index for the analysis area. Involved bands: B02(blue),
    B03(green), B08(nir), B11(swir1), B12(swir2).
    Downscales some bands if it is needed to homogenize spatial resolutions

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''

    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # prevents numpy errors for invalid values or divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_wi.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B11')

    # getting bands data and downscaling if it is needed
    band_blue = gdal.Open(getBandPath(scene_path, 'B02'))
    pix_size = band_blue.GetGeoTransform()[1]
    if pix_size == 10.0:
        data_blue = downScaling(getBandPath(
            scene_path, 'B02')).astype(np.float32)
    else:
        data_blue = band_blue.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_green = gdal.Open(getBandPath(scene_path, 'B03'))
    pix_size = band_green.GetGeoTransform()[1]
    if pix_size == 10.0:
        data_green = downScaling(getBandPath(
            scene_path, 'B03')).astype(np.float32)
    else:
        data_green = band_green.GetRasterBand(
            1).ReadAsArray().astype(np.float32)

    band_nir = gdal.Open(getBandPath(scene_path, 'B08'))
    pix_size = band_nir.GetGeoTransform()[1]
    if pix_size == 10.0:
        data_nir = downScaling(getBandPath(
            scene_path, 'B08')).astype(np.float32)
    else:
        data_nir = band_nir.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir1 = gdal.Open(getBandPath(scene_path, 'B11'))
    data_swir1 = band_swir1.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir2 = gdal.Open(getBandPath(scene_path, 'B12'))
    data_swir2 = band_swir2.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # computing water index
    aweish = data_blue + (2.5 * data_green) - \
        (1.5 * (data_nir + data_swir1)) - (0.25 * data_swir2)

    # saving water index
    saveIndex(aweish, outFileName, base_path)


def mndwiS2(scene_path):
    '''
    Description:
    ------------
    Computes S2 mndwi water index for the analysis area. Involved bands: B03(green),
    B11(swir1).
    Downscales some bands if it is needed to homogenize spatial resolutions

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''

    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # prevents numpy errors for invalid values or divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_wi.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B11')

    # getting bands data and downscaling if it is needed
    band_green = gdal.Open(getBandPath(scene_path, 'B03'))
    pix_size = band_green.GetGeoTransform()[1]
    if pix_size == 10.0:
        data_green = downScaling(getBandPath(
            scene_path, 'B03')).astype(np.float32)
    else:
        data_green = band_green.GetRasterBand(
            1).ReadAsArray().astype(np.float32)

    band_swir1 = gdal.Open(getBandPath(scene_path, 'B11'))
    data_swir1 = band_swir1.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # computing water index
    mndwi = (data_green - data_swir1) / (data_green + data_swir1)

    # saving water index
    saveIndex(mndwi, outFileName, base_path)


def computeKmeansS2(scene_path):
    '''
    Description:
    ------------
    Computes kmeans clasterization method. Involved bands: Swir1 (B11)

    The number of classes by default are 3

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''
    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_kmeans_mask.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B11')

    # open image
    img = gdal.Open(base_path)
    img_data = img.GetRasterBand(1).ReadAsArray()
    img_data[img_data == float('-inf')] = 0.0

    # compute percentiles for image contrast enhancing
    p1, p2 = np.percentile(img_data, (0.5, 99.5))
    img_data = exposure.rescale_intensity(img_data, in_range=(p1, p2))

    # convert image to only one row array
    w, h = img_data.shape
    Z = img_data.reshape((-1, 1))
    Z = np.float32(Z)

    # compute kmeans classification
    number_clusters = 3
    km = KMeans(number_clusters)
    km.fit(Z)
    labels = km.labels_

    # assign codes (colors) to each clase
    colors = getColors(number_clusters)
    km_classes = np.zeros(w*h, dtype='uint8')
    for ix in range(km_classes.shape[0]):
        km_classes[ix] = colors[labels[ix]]

    # get water class
    water_class = getWaterClass(Z, km_classes, colors)

    # compute and save mask
    km_classes = km_classes.reshape((w, h))
    binary_mask = np.where(km_classes == water_class, 1, 0)
    saveMask(binary_mask, outFileName, base_path)


def createCloudMaskS2(scene_path, cloud_mask_level):
    '''
    Description:
    ------------
    Creates binary cloud mask image for S2 according the image of
    cloud classification (MSK_CLASSI_B00.gml or MSK_CLASSI_B00.jp2 file for S2MSI1C and SCL band for S2MSI2A).
    Saves the cloud mask to the processing folder (folder "temp"
    relative to the scene path).

    WARNING: The cloud classification for 2A product is more much accurate and
    confident than the cloud classification for the product 1C. It seems that
    this situation can change on 26th of october of 2021.
    More information:
    https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming

    Arguments:
    ------------
    - scene_path (string): path to scene folder
    - cloud_mask_level (string): 0, 1 or 2. Level of cloud masking

    Returns:
    ------------
    None

    '''

    # create temp folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)
    mask_values = []

    # compute cloud mask for S2MSI1C product
    if 'MSIL1C' in scene_path:
        # in this case, the .gml file (vector) has to be converted to .shp file
        # and then to a binary raster image.
        gml_classi_path = getBandPath(scene_path, 'MSK_CLOUDS_B00.gml')
        jp2_classi_path = getBandPath(scene_path, 'MSK_CLOUDS_B00.jp2')
        if gml_classi_path is None:  # cloud mask in raster format
            path = pathlib.PurePath(scene_path)
            raster_path = path.name+'_cmask.tif'
            output_path = str(pathlib.Path(
                os.path.join(output_folder, raster_path)))
            raster_template = getBandPath(scene_path, 'B11')

            if cloud_mask_level == '1' or cloud_mask_level == '2':
                mask_values = [1]
                #msk_classi_path = getBandPath(scene_path, 'CPM.jp2')
                msk_ds = gdal.Open(jp2_classi_path)
                msk_data = msk_ds.GetRasterBand(1).ReadAsArray()
                cloud_mask = np.isin(msk_data, mask_values)
                name_cloud_mask = path.name+'_cmask.tif'
                # image template to copy resolution, bounding box and coordinate reference system.
                base_path = getBandPath(scene_path, 'B11')
                outFileName = str(pathlib.Path(
                    os.path.join(output_folder, name_cloud_mask)))
                saveMask(cloud_mask, outFileName, base_path)
            else:
                createEmptyCloudMaskS2(output_path, raster_template)
        else:  # cloud mask in gml format
            path = pathlib.PurePath(scene_path)
            shp_path = path.name+'_cmask.shp'
            raster_path = path.name+'_cmask.tif'
            output_path = str(pathlib.Path(
                os.path.join(output_folder, raster_path)))
            raster_template = getBandPath(scene_path, 'B11')
            output_shp = str(pathlib.Path(
                os.path.join(output_folder, shp_path)))

            if cloud_mask_level == '1' or cloud_mask_level == '2':
                # prevents .gml files without clouds
                res = gmlToShp(gml_classi_path, output_shp)
                if res:
                    rasterizeShapefile(
                        output_shp, output_path, raster_template)
                else:
                    createEmptyCloudMaskS2(output_path, raster_template)
            else:
                createEmptyCloudMaskS2(output_path, raster_template)

    # compute cloud mask for S2MSI2A product
    if 'MSIL2A' in scene_path:
        # cloud codes: 2 -> dark area; 3 -> cloud shadow; 8 -> cloud medium; 9 -> cloud high; 10 -> cirrus

        if cloud_mask_level == '0':
            mask_values = [-1]
        if cloud_mask_level == '1':
            mask_values = [9]
        if cloud_mask_level == '2':
            mask_values = [3, 8, 9, 10]

        scl_band_path = getBandPath(scene_path, 'SCL')
        scl_ds = gdal.Open(scl_band_path)
        scl_data = scl_ds.GetRasterBand(1).ReadAsArray()

        cloud_mask = np.isin(scl_data, mask_values)
        # remove objects with less than 10 connected pixels
        morphology.remove_small_objects(
            cloud_mask, min_size=10, connectivity=1, in_place=True)

        path = pathlib.PurePath(scene_path)
        # name of the binary cloud mask
        name_cloud_mask = path.name+'_cmask.tif'
        # image template to copy resolution, bounding box and coordinate reference system.
        base_path = getBandPath(scene_path, 'B11')
        outFileName = str(pathlib.Path(
            os.path.join(output_folder, name_cloud_mask)))
        saveMask(cloud_mask, outFileName, base_path)

# *******************************************************************************
# SECTION: LANDSAT PROCESSING FUNCTIONS
# *******************************************************************************


def processLandsatSceneFromPath(scene_path, scene_id, sds_path, shp_path, wi_type, thr_method, cloud_mask_level, morphology_method, kernel_size, bc):
    '''
    Description:
    ------------
    Run the whole workflow reprocessing Landsat-8 scenes from the data folder.

    Arguments:
    ------------
    - scene_path (string): folder that contains the image to be processed
    - scene_id (string): name of the image (id)
    - sds_path (string): folder for store every extracted shoreline. Obtained from createFolderTree() function
    - shp_path (string): path to shapefile containing beaches areas
    - wi_type (string): type of water index
    - thr_method (string); type of thresholding method
    - cloud_mask_level (string): cloud masking severity
    - morphology_method (string): type of morphology method
    - kernel_size (int): kernel size for the extraction algorithm
    - bc (list of string): list of beach polygons to be processed

    Returns:
    ------------
    None

    '''

    # print('Processing '+scene_path+' ...')
    logging.info('Processing '+scene_id+' ...')
    processing_path = str(pathlib.Path(
        os.path.join(scene_path, 'temp')))
    # print('Computing water index band...')
    logging.info('Computing water index band...')
    if wi_type == 'aweinsh':
        aweinshLandsat(scene_path)
    if wi_type == 'aweish':
        aweishLandsat(scene_path)
    if wi_type == 'mndwi':
        mndwiLandsat(scene_path)
    if wi_type == 'kmeans':
        computeKmeansLandsat(scene_path)
    # print('Computing cloud mask...')
    logging.info('Computing cloud mask...')
    createCloudMaskL8(scene_path, cloud_mask_level)
    logging.info('Computing water index mask...')
    if wi_type != 'kmeans':
        wi_path = getBandPath(processing_path, 'wi.tif')
        wi_mask = getIndexMask(wi_path, thr_method)
    else:
        wi_path = getBandPath(processing_path, 'kmeans_mask.tif')
        wi_mask = getBandData(wi_path)
    cmask_band = getBandPath(processing_path, 'cmask')
    # print('Computing rough pixel line...')
    logging.info('Computing rough pixel line...')
    pixel_line = createPixelLine(
        morphology_method, wi_mask, cmask_band)
    saveMask(pixel_line, str(pathlib.Path(os.path.join(
        processing_path, 'pl.tif'))), cmask_band)
    source_epsg = getSourceEpsg()
    target_epsg = getTargetEpsg(scene_path, 'B6')
    # print('Reprojecting shp of beaches...')
    logging.info('Reprojecting shp of beaches...')
    reprojectShp(shp_path, str(pathlib.Path(os.path.join(processing_path,
                                                         'bb300_r.shp'))), source_epsg, target_epsg)
    # print('Computing footprint band...')
    logging.info('Computing footprint band...')
    createShapefileFromRasterFootprint(getBandPath(scene_path, 'B6'), str(pathlib.Path(os.path.join(
        processing_path, 'scene_footprint.shp'))), target_epsg, geom_type='polygon')
    # print('Clipping shp of beaches by scene footprint...')
    logging.info('Clipping shp of beaches by scene footprint...')
    clipShapefile(str(pathlib.Path(os.path.join(processing_path, 'bb300_r.shp'))), str(pathlib.Path(os.path.join(
        processing_path, 'clip_bb300_r.shp'))), str(pathlib.Path(os.path.join(processing_path, 'scene_footprint.shp'))))
    # print('Rasterizing beaches subset...')
    logging.info('Rasterizing beaches subset...')
    rasterizeShapefile(str(pathlib.Path(os.path.join(processing_path, 'bb300_r.shp'))), str(pathlib.Path(os.path.join(
        processing_path, 'bb300_r.tif'))), getBandPath(scene_path, 'B6'), bc)
    # print('Masking rough pixel line with beaches subset...')
    logging.info('Masking rough pixel line with beaches subset...')
    maskPixelLine(str(pathlib.Path(os.path.join(processing_path, 'pl.tif'))),
                  str(pathlib.Path(os.path.join(processing_path, 'bb300_r.tif'))))
    # print('Extracting points...')
    logging.info('Extracting points...')
    res = extractPoints(getBandPath(scene_path, 'B6'), str(pathlib.Path(os.path.join(
        processing_path, 'pl.tif'))), processing_path, int(kernel_size), 4, 3)
    if res:
        # print('Computing average points...')
        logging.info('Computing average points...')
        averagePoints(getBandPath(scene_path, 'B6'),
                      processing_path, 50, 3)
        # print('Making point shp...')
        logging.info('Making point shp...')
        shp_path_average = createShpFromAverageFile(
            getBandPath(scene_path, 'B6'), processing_path)
        # print('Transfering beaches identifiers...')
        logging.info('Transfering beaches identifiers...')
        copyShpIdentifiers(str(pathlib.Path(os.path.join(
            processing_path, 'clip_bb300_r.shp'))), shp_path_average)
        # print('Cleaning points and making final shoreline in line vector format...')
        logging.info(
            'Cleaning points and making final shoreline in line vector format...')
        cleanPoints2(shp_path_average, 150, 1)
        # print('Export final shoreline shapefiles to SDS folder...')
        logging.info(
            'Export final shoreline shapefiles to SDS folder...')
        copyShpToFolder(processing_path, sds_path, target_epsg)
    else:
        logging.warning('No results in extraction points process.')
        sys.exit(1)


def aweinshLandsat(scene_path):
    '''
    Description:
    ------------
    Computes L8 aweinsh water index for the analysis area. Involved bands: B3(green)
    B5(nir), B6(swir1), B7(swir2)

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''

    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # prevents numpy errors for invalid values or divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_wi.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B6')

    # getting bands data
    band_green = gdal.Open(getBandPath(scene_path, 'B3'))
    data_green = band_green.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_nir = gdal.Open(getBandPath(scene_path, 'B5'))
    data_nir = band_nir.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir1 = gdal.Open(getBandPath(scene_path, 'B6'))
    data_swir1 = band_swir1.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir2 = gdal.Open(getBandPath(scene_path, 'B7'))
    data_swir2 = band_swir2.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # getting parameters to convert Dns to TOA values from MTL file
    rmb, rab, se = getTOAParameters(getBandPath(scene_path, 'MTL.txt'), '1')

    # DNs to TOA conversion
    se_factor = sin(se*(np.pi/180.0))
    dg_toa = (data_green*rmb+rab) / se_factor
    dn_toa = (data_nir*rmb+rab) / se_factor
    ds1_toa = (data_swir1*rmb+rab) / se_factor
    ds2_toa = (data_swir2*rmb+rab) / se_factor

    # computing water index
    aweinsh = 4 * (dg_toa - ds1_toa) - (0.25 * dn_toa + 2.75 * ds2_toa)

    # saving water index
    saveIndex(aweinsh, outFileName, base_path)


def mndwiLandsat(scene_path):
    '''
    Description:
    ------------
    Computes L8 mndwi water index for the analysis area. Involved bands: B3(green)
    B6(swir1)

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''
    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # prevents numpy errors for invalid values or divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_wi.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B6')

    # getting bands data
    band_green = gdal.Open(getBandPath(scene_path, 'B3'))
    data_green = band_green.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir1 = gdal.Open(getBandPath(scene_path, 'B6'))
    data_swir1 = band_swir1.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # getting parameters to convert Dns to TOA values from MTL file
    rmb, rab, se = getTOAParameters(getBandPath(scene_path, 'MTL.txt'), '1')

    # DNs to TOA conversion
    se_factor = sin(se*(np.pi/180.0))
    dg_toa = (data_green*rmb+rab) / se_factor
    ds1_toa = (data_swir1*rmb+rab) / se_factor

    # computing water index
    mndwi = (dg_toa - ds1_toa) / (dg_toa + ds1_toa)

    # saving water index
    saveIndex(mndwi, outFileName, base_path)


def getTOAParameters(mtl, band):
    '''
    Description:
    ------------
    Get TOA parameters from MTL metadata file

    Arguments:
    ------------
    - mtl (string): path to the MTL file
    - band (string): band name

    Returns:
    ------------
    - rmb (float): reflectance value to multiply for a single band
    - rab (float): reflectance value to add for a single band
    - se (float): sun elevation for the image scene

    '''

    with open(mtl, "r") as mtl:
        for line in mtl:
            if "REFLECTANCE_MULT_BAND_"+band in line:
                rmb = float(line.strip().split("=")[1].strip())
            if "REFLECTANCE_ADD_BAND_"+band in line:
                rab = float(line.strip().split("=")[1].strip())
            if "SUN_ELEVATION" in line:
                se = float(line.strip().split("=")[1].strip())
    return rmb, rab, se


def aweishLandsat(scene_path):
    '''
    Description:
    ------------
    Computes L8 aweish water index for the analysis area. Involved bands: B2(blue)
    B3(green), B5(nir), B6(swir1), B7(swir2)

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''

    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # prevents numpy errors for invalid values or divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_wi.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B6')

    # getting bands data
    band_blue = gdal.Open(getBandPath(scene_path, 'B2'))
    data_blue = band_blue.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_green = gdal.Open(getBandPath(scene_path, 'B3'))
    data_green = band_green.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_nir = gdal.Open(getBandPath(scene_path, 'B5'))
    data_nir = band_nir.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir1 = gdal.Open(getBandPath(scene_path, 'B6'))
    data_swir1 = band_swir1.GetRasterBand(1).ReadAsArray().astype(np.float32)

    band_swir2 = gdal.Open(getBandPath(scene_path, 'B7'))
    data_swir2 = band_swir2.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # getting parameters to convert Dns to TOA values from MTL file
    rmb, rab, se = getTOAParameters(getBandPath(scene_path, 'MTL.txt'), '1')

    # DNs to TOA conversion
    se_factor = sin(se*(np.pi/180.0))
    db_toa = (data_blue*rmb+rab) / se_factor
    dg_toa = (data_green*rmb+rab) / se_factor
    dn_toa = (data_nir*rmb+rab) / se_factor
    ds1_toa = (data_swir1*rmb+rab) / se_factor
    ds2_toa = (data_swir2*rmb+rab) / se_factor

    # computing water index
    aweish = db_toa + (2.5 * dg_toa) - \
        (1.5 * (dn_toa + ds1_toa)) - (0.25 * ds2_toa)

    # saving water index
    saveIndex(aweish, outFileName, base_path)


def computeKmeansLandsat(scene_path):
    '''
    Description:
    ------------
    Computes kmeans clasterization method. Involved bands: Swir1 (B6)

    The number of classes by default are 3

    Arguments:
    ------------
    - scene_path (string): path to the scene folder

    Returns:
    ------------
    None

    '''

    # create output folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)

    # output file name setting
    path = pathlib.PurePath(scene_path)
    name = path.name+'_kmeans_mask.tif'
    outFileName = str(pathlib.Path(os.path.join(output_folder, name)))

    # template image to copy resolution, bounding box and coordinate reference system
    base_path = getBandPath(scene_path, 'B6')

    # open image
    img = gdal.Open(base_path)
    img_data = img.GetRasterBand(1).ReadAsArray()
    img_data[img_data == float('-inf')] = 0.0

    # compute percentiles for image contrast enhancing
    p1, p2 = np.percentile(img_data, (0.5, 99.5))
    img_data = exposure.rescale_intensity(img_data, in_range=(p1, p2))

    # convert image to only one row array
    w, h = img_data.shape
    Z = img_data.reshape((-1, 1))
    Z = np.float32(Z)

    # compute kmeans classification
    number_clusters = 3
    km = KMeans(number_clusters)
    km.fit(Z)
    labels = km.labels_

    # assign codes (colors) to each clase
    colors = getColors(number_clusters)
    km_classes = np.zeros(w*h, dtype='uint8')
    for ix in range(km_classes.shape[0]):
        km_classes[ix] = colors[labels[ix]]

    # get water class
    water_class = getWaterClass(Z, km_classes, colors)

    # compute and save mask
    km_classes = km_classes.reshape((w, h))
    binary_mask = np.where(km_classes == water_class, 1, 0)
    saveMask(binary_mask, outFileName, base_path)


def createCloudMaskL8(scene_path, cloud_mask_level):
    '''
    Description:
    ------------
    Creates binary cloud mask image for L8 according the image of
    cloud classification (band BQA).
    Saves the cloud mask to the processing folder (folder "temp"
    relative to the scene path).

    Arguments:
    ------------
    - scene_path (string): path to scene folder
    - cloud_mask_level (string): 0, 1 or 2. Level of cloud masking

    Returns:
    ------------
    None

    '''
    # create temp folder if it is needed
    output_folder = str(pathlib.Path(os.path.join(scene_path, 'temp')))
    createFolderCheck(output_folder)
    mask_values = []

    # landsat from collection 1
    if '_01_' in scene_path:
        # list of values related with medium or high cloud confidence and cirrus values
        cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
        cloud_shadow_values = [2976, 2980, 2984, 2988, 3008, 3012,
                               3016, 3020, 7072, 7076, 7080, 7084, 7104, 7108, 7112, 7116]
        cirrus_values = [6816, 6820, 6824, 6828, 6848, 6852, 6856, 6860, 6896, 6900, 6904, 6908, 7072, 7076, 7080, 7084, 7104,
                         7108, 7112, 7116, 7840, 7844, 7848, 7852, 7872, 7876, 7880, 7884]

        if cloud_mask_level == '0':
            mask_values = [-1]
        if cloud_mask_level == '1':
            mask_values = mask_values+cloud_values
        if cloud_mask_level == '2':
            mask_values = cloud_values+cirrus_values+cloud_shadow_values

        qa_band_path = getBandPath(scene_path, 'BQA')

    if '_02_' in scene_path:
        # list of values related with medium or high cloud confidence and cirrus values
        cloud_values = [22280, 24082, 22080]
        cloud_shadow_values = [23888, 23826, 24144]
        cirrus_values = [55052, 56854]

        if cloud_mask_level == '0':
            mask_values = [-1]
        if cloud_mask_level == '1':
            mask_values = mask_values+cloud_values
        if cloud_mask_level == '2':
            mask_values = cloud_values+cirrus_values+cloud_shadow_values

        qa_band_path = getBandPath(scene_path, 'QA_PIXEL')

    qa_ds = gdal.Open(qa_band_path)
    qa_data = qa_ds.GetRasterBand(1).ReadAsArray()

    cloud_mask = np.isin(qa_data, mask_values)
    # remove objects with less than 10 connected pixels
    morphology.remove_small_objects(
        cloud_mask, min_size=10, connectivity=1, in_place=True)

    path = pathlib.PurePath(scene_path)
    # name of the binary cloud mask
    name_cloud_mask = path.name+'_cmask.tif'
    # image template to copy resolution, bounding box and coordinate reference system.
    base_path = getBandPath(scene_path, 'B6')
    outFileName = str(pathlib.Path(
        os.path.join(output_folder, name_cloud_mask)))
    saveMask(cloud_mask, outFileName, base_path)
