import json

import cv2
import numpy as np
import rasterio
import shapely.affinity
import shapely.geometry
from shapely_geojson import Feature, FeatureCollection, dumps


def polygons_from_binary(contours, hierarchy, pixel_tolerence):
    polys = []
    for contour in contours:
        poly = shapely.geometry.Polygon([p for p in contour])
        if pixel_tolerence > 0:
            poly = poly.simplify(pixel_tolerence, preserve_topology=True)
        polys.append(poly)
    return polys


def buildings_from_binary(contours, hierarchy):
    pass


def geometries_from_mask(mask, thresh, transform, mode, x_offset, y_offset, open_kernel, close_kernel, pixel_tolerence):

    # Ensure transform
    if isinstance(transform, rasterio.transform.Affine):
        pass
    elif isinstance(transform, str):
        with rasterio.open(transform, 'r') as dataset:
            transform = dataset.transform
    else:
        raise Exception('Bad transform')

    # Ensure opening kernel
    if isinstance(open_kernel, int):
        open_kernel = np.ones((open_kernel, open_kernel), np.uint8)
    elif not open_kernel:
        open_kernel = None
    elif isinstance(open_kernel, np.ndarray):
        pass
    else:
        raise Exception('Bad opening kernel')

    # Ensure closing kernel
    if isinstance(close_kernel, int):
        close_kernel = np.ones((close_kernel, close_kernel), np.uint8)
    elif not close_kernel:
        close_kernel = None
    elif isinstance(close_kernel, np.ndarray):
        pass
    else:
        raise Exception('Bad closing kernel')

    # Produce binary image
    # https://docs.opencv.org/3.4.1/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    _, binary = cv2.threshold(mask, thresh, 0xff, cv2.THRESH_BINARY)

    # Filter
    # https://docs.opencv.org/3.4.1/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
    if open_kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    if close_kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # Extract contours
    # https://docs.opencv.org/3.4.1/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Offset the contours (apply correct screen space coordinates)
    offset_contours = []
    for contour in contours:
        offset_contour = contour[:, 0, :]
        offset_contour = offset_contour + np.array([x_offset, y_offset])
        offset_contours.append(offset_contour)

    # Vectorize according to mode
    if mode == 'buildings':
        polys = buildings_from_binary(offset_contours, hierarchy)
    else:
        polys = polygons_from_binary(offset_contours, hierarchy, pixel_tolerence)

    # Convert from screen space to world space
    a = transform[0]
    b = transform[1]
    xoff = transform[2]
    d = transform[3]
    e = transform[4]
    yoff = transform[5]
    matrix = [a, b, d, e, xoff, yoff]

    return [shapely.affinity.affine_transform(p, matrix) for p in polys]


def geojson_from_mask(mask, thresh, transform, mode, x_offset=0, y_offset=0, open_kernel=3, close_kernel=3, pixel_tolerence=0):
    features = []
    polys = geometries_from_mask(mask, thresh, transform, mode, x_offset, y_offset, open_kernel, close_kernel, pixel_tolerence)
    features = [Feature(p) for p in polys]
    feature_collection = FeatureCollection(features)
    return dumps(feature_collection, indent=2)


def shapley_from_mask(mask, thresh, transform, mode, x_offset=0, y_offset=0, open_kernel=3, close_kernel=3, pixel_tolerence=0):
    polys = geometries_from_mask(mask, thresh, transform, mode, x_offset, y_offset, open_kernel, close_kernel, pixel_tolerence)
    return polys
