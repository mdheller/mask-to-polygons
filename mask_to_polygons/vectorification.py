import json

import cv2
import geojson
import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.geometry
from geojson import FeatureCollection
from mask_to_polygons.processing import buildings, polygons


def polygons_from_binary(contours, hierarchy):
    pass


def buildings_from_binary(contours, hierarchy):
    pass


def geometries_from_mask(mask, thresh, transform, mode, open_kernel, close_kernel):

    # transform
    if isinstance(transform, rasterio.transform.Affine):
        pass
    elif isinstance(transform, str):
        with rasterio.open(transform, 'r') as dataset:
            transform = dataset.transform
    elif callable(transform):
        # Transform can be function of form f(x, y) which is assumed to convert from
        # pixel coordinates to (lat, lng)
        transform_fn = transform
        transform = rasterio.transform.IDENTITY

    # opening kernel
    if isinstance(open_kernel, int):
        open_kernel = np.ones((open_kernel, open_kernel), np.uint8)
    elif not open_kernel:
        open_kernel = None
    elif isinstance(open_kernel, np.ndarray):
        pass
    else:
        raise Exception()

    # closing kernel
    if isinstance(close_kernel, int):
        close_kernel = np.ones((close_kernel, close_kernel), np.uint8)
    elif not close_kernel:
        close_kernel = None
    elif isinstance(close_kernel, np.ndarray):
        pass
    else:
        raise Exception()

    # https://docs.opencv.org/3.4.1/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    _, binary = cv2.threshold(mask, thresh, 0xff, cv2.THRESH_BINARY)

    # https://docs.opencv.org/3.4.1/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
    if open_kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    if close_kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # https://docs.opencv.org/3.4.1/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # vectorize
    if mode == 'buildings':
        polys = buildings_from_binary(contours, hierarchy)
    else:
        polys = buildings_from_binary(contours, hierarchy)

    return polys


def geojson_from_mask(mask, thresh, transform, mode, open_kernel=3, close_kernel=3):
    features = []
    polys = geometries_from_mask(mask, thresh, transform, mode)
    for poly in polys:
        features.append({
            'type': 'Feature',
            'properties': {},
            'geometry': poly
        })
    return geojson.dumps(FeatureCollection(features))


def shapley_from_mask(mask, thresh, transform, mode, open_kernel=3, close_kernel=3):
    polys = geometries_from_mask(mask, thresh, transform, mode)
    return [shapely.geometry.shape(poly) for poly in polys]
