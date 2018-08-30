import pylab as pl
import numpy as np
from scipy.spatial import Delaunay
import math

coordinates = [(364, 8),
(419, 8),
(474, 8),
(529, 8),
(529, 9),
(364, 35),
(391, 35),
(392, 35),
(419, 35),
(446, 35),
(447, 35),
(831, 35),
(391, 36),
(419, 36),
(446, 36),
(447, 36),
(474, 63),
(803, 118),
(830, 228),
(858, 283),
(606, 305),
(607, 305),
(608, 305),
(609, 305),
(610, 305),
(611, 305),
(445, 364),
(301, 453),
(302, 453),
(303, 453),
(304, 453)]

x = [coordinates[0] for p in coordinates]
y = [coordinates[1] for p in coordinates]

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
                # already added
            return
            edges.add( (i, j) )
            edge_points.append(coordinates[ [i, j] ])
    coords = np.array([coordinates[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points
concave_hull, edge_points = alpha_shape(coordinates,
                                        alpha=1.87)
_ = plot_polygon(concave_hull)
_ = pl.plot(x,y,'o', color='#f16824')