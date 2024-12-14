#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import rasterio             # Reproject raster samples
from rasterio import windows
import fiona 

from shapely.geometry import Point, MultiPoint
import numpy as np               # numerical array manipulation


from scipy.spatial import KDTree

epsilon = 1e-07
# In[ ]:



SEARCH_DISTANCE = 0.72


def CrsToPixel(points, src):
    '''
    transform the geometry points with a coordinate reference system to pixel coordinates
    '''
    transform = src.meta['transform']
    x,y = rasterio.transform.rowcol(transform, points.x, points.y)
    
    return x, y
    
def CrsToPixel_ratio(points, src):
    '''
    transform the geometry points with a coordinate reference system to pixel coordinates
    '''
    transform = src.meta['transform']
    x,y = rasterio.transform.rowcol(transform, points.x, points.y)
    
    return [i  /src.height for i in x], [j/src.width for j in y]
    
def nearest_neighbor_within(others, point, max_distance):
    """Find nearest point among others up to a maximum distance.

    Args:
        others: a list of Points or a MultiPoint
        point: a Point
        max_distance: maximum distance to search for the nearest neighbor

    Returns:
        A shapely Point if one is within max_distance, None otherwise
    """
    search_region = point.buffer(max_distance)
    interesting_points = search_region.intersection(MultiPoint(others))

    if not interesting_points:
        closest_point = None
    elif isinstance(interesting_points, Point):
        closest_point = interesting_points
    else:
        distances = [point.distance(ip) for ip in interesting_points.geoms
                     if point.distance(ip) > 0]
        closest_point = interesting_points.geoms[distances.index(min(distances))]

    return closest_point 

import math

def evaluation_pixel(true_points, predict_points, threshold=2):
    
    len_pred = len(predict_points)
    
    TP, FP, FN = 0, 0, 0
    t = threshold*math.sqrt(2)

    
    for p in true_points:
        
        if len(predict_points) == 0:
            continue
        p = np.asarray(p).reshape(1,2)

        tree = KDTree(p)

        dist, idx = tree.query(predict_points, k=1, distance_upper_bound=t)

        dist_index = np.where(dist != math.inf)
 
        no_point = len(dist_index[0])
        if no_point == 0:
            FN+=1


        else:
            dist = dist[dist_index][0]
            if dist <= t:
                idx = dist_index[0][0]
                TP+=1
                predict_points.pop(idx)

    FP = len_pred-TP

    if TP == 0 and FP == 0:
        Precision = 1
    else:
        Precision = float(TP/(TP+FP))
    if TP == 0 and FN == 0:
        Recall = 1
    else:
        Recall = float(TP/(TP+FN))
    F1 = 2*(Precision*Recall)/(Precision+Recall+epsilon)
    accuracy = {
      "TP": TP,
      "FP": FP,
      "FN": FN,
      "Precision":Precision,
      "Recall":Recall,
      "F1":F1
    }
    return accuracy 

def createShapefileObject(points, crs, wfile):
    schema = {
        'geometry': 'Point',
        'properties': {'id': 'str'},
        }
    #with fiona.open(wfile, 'w', crs=meta.get('crs').to_dict(), driver='ESRI Shapefile', schema=schema) as sink:
    with fiona.open(wfile, 'w', crs=crs, driver='ESRI Shapefile', schema=schema) as sink:
        for idx, point in enumerate(points):
            sink.write({
                'geometry': mapping(point),
                'properties': {'id': str(idx)},
                })
            #print(mapping(point))   
    
def evaluation(true_points, predict_points, threshold = SEARCH_DISTANCE, index = 'wildebeest', ShapefilePath = None, meta = None):
  True_Positives = []
  False_Positives = []
  False_Negatives = []
  positives = predict_points.copy()
  for true_point in true_points:
    true_positive = nearest_neighbor_within(positives, true_point, threshold)
    if true_positive == None:
     False_Negatives.append(true_point)
    else:
     True_Positives.append(true_positive)
     positives.remove(true_positive)
  False_Positives = positives

  if ShapefilePath != None:
    createShapefileObject(True_Positives, meta, wfile =  os.path.join(ShapefilePath, "patch"+index+"_tp.shp"))
    createShapefileObject(False_Positives, meta, wfile = os.path.join(ShapefilePath, "patch"+index+"_fp.shp"))
    createShapefileObject(False_Negatives, meta, wfile = os.path.join(ShapefilePath, "patch"+index+"_fn.shp"))

  TP = len(True_Positives)
  FP = len(False_Positives)
  FN = len(False_Negatives)
  if TP == 0 and FP == 0:
    Precision = 1
  else:
    Precision = float(TP/(TP+FP))
  if TP == 0 and FN == 0:
    Recall = 1
  else:
    Recall = float(TP/(TP+FN))
  F1 = 2*(Precision*Recall)/(Precision+Recall+epsilon)
  accuracy = {
      "TP": TP,
      "FP": FP,
      "FN": FN,
      "Precision":Precision,
      "Recall":Recall,
      "F1":F1
  }
  return accuracy




def writeTiff(image, meta, wfile):
  transform = meta.meta['transform']
  crs = meta.meta['crs']

  new_dataset = rasterio.open(wfile, 'w', driver='GTiff',
                            height = image.shape[0], width = image.shape[1],
                            count=1, dtype=str(image.dtype),
                            crs=crs,
                            transform=transform)

  new_dataset.write(image,1)
  new_dataset.close()

