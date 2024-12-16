import geopandas as gpd

polygon_shapefile_path = r'D:\predict_box.shp'  # Path to the polygon shapefile
output_shapefile_path = r'E:\predict_center.shp'  # Path to save the centroid shapefile

polygons_gdf = gpd.read_file(polygon_shapefile_path)

centroids = polygons_gdf.geometry.centroid  # Extract the geometric center of each polygon

# Create a new GeoDataFrame with the centroids as the geometry
centroids_gdf = gpd.GeoDataFrame(
    polygons_gdf.drop(columns='geometry'),  # Keep all original attributes except the geometry
    geometry=centroids  # Replace the geometry with the centroids
)

# Save the new GeoDataFrame with centroids to the output shapefile
centroids_gdf.to_file(output_shapefile_path)
