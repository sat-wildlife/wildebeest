import geopandas as gpd


def clip_points_by_polygon(point_shapefile, polygon_shapefile, output_shapefile):
    """
    Clips points from a point shapefile based on the boundaries of a polygon shapefile.

    Parameters:
    - point_shapefile: Path to the input point shapefile.
    - polygon_shapefile: Path to the input polygon shapefile.
    - output_shapefile: Path to save the resulting clipped point shapefile.
    """

    # Load the point shapefile into a GeoDataFrame
    points = gpd.read_file(point_shapefile)

    # Load the polygon shapefile into a GeoDataFrame
    polygons = gpd.read_file(polygon_shapefile)

    # Ensure the coordinate reference systems (CRS) match between the two GeoDataFrames
    if points.crs != polygons.crs:
        polygons = polygons.to_crs(points.crs)  # Reproject polygons to match the points CRS

    # Perform a spatial join to find points that fall within the polygons
    clipped_points = gpd.sjoin(points, polygons, how="inner", predicate="within")

    # Save the resulting clipped points to the specified output shapefile
    clipped_points.to_file(output_shapefile)
