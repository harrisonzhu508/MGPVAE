import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from shapely.geometry import Polygon

def create_pixel_square(lonlat_centroid, lonlat_sep):
    x1 = [lonlat_centroid[0] - lonlat_sep / 2, lonlat_centroid[1] + lonlat_sep / 2]
    x2 = [lonlat_centroid[0] + lonlat_sep / 2, lonlat_centroid[1] + lonlat_sep / 2]
    x3 = [lonlat_centroid[0] + lonlat_sep / 2, lonlat_centroid[1] - lonlat_sep / 2]
    x4 = [lonlat_centroid[0] - lonlat_sep / 2, lonlat_centroid[1] - lonlat_sep / 2]
    return Polygon([x1, x2, x3, x4])

class Figures:
    def __init__(self) -> None:
        self.figures = []

    def append(self, x: Figure):
        self.figures.append(x)