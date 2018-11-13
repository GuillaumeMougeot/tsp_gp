# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:24:42 2018

@author: Guillaume
"""

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import os



def norm(vect):
    return np.sqrt(vect[0]**2+vect[1]**2)

def dot(v1, v2):
    return (v1[0]*v2[0] + v1[1]*v2[1])

def cross(v1, v2):
    return np.sin(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

def distance_pt_line(point, line):
    """Distance between point and line. Line is defined with two points"""
    cb = np.array(line[1]) - np.array(point)
    ab = np.array(line[1]) - np.array(line[0])
    ch = cb - dot(cb, ab)/(norm(ab)**2) * ab
    ah = (ab - cb) + ch

    test = dot(ah, ab)
    if np.abs(dot(ch,ah))>0.0001:
        print("erreur" + str(cb) + "t" + str(ab))
    if test<0:
        return norm(ab - cb)
    elif test>norm(ab)**2:
        return norm(cb)
    else:
        return norm(ch)



def non_hull(vertices, size):
    """Return list of vertices which are not in the hull"""
    output = []
    for i in range(size):
        if not (i in vertices):
            output += [i]
    return np.array(output)

def closer_edge(point, edges):
    """Return the closer edge to a point"""
    
    m = distance_pt_line(point, edges[0])
    m_i = 0
    for i in range(1,len(edges)):
        if distance_pt_line(point, edges[i]) < m:
            m = distance_pt_line(point, edges[i])
            m_i = i
        
    return m_i


def check_hull(vertices):
    """Return True if a hull can be computed"""
    if len(vertices) < 3:
        return False
    else:
        # Do we have colinear points?
        count = 2
        a = np.array(vertices[0]) - np.array(vertices[1])
        cross_p = 0
        while count < len(vertices) and cross_p == 0:
            cross_p += cross(a, np.array(vertices[0]) - np.array(vertices[count]))
            count += 1
        return (cross_p != 0)
            

def closer(vertices, border, in_pts):
    """Return the index of the closer vertex (in in_pts, which sould not be
    empty) from the edges defined by hull"""
    hull_vertices = vertices[border]
    n = len(border)
    
    edges = np.array([[hull_vertices[i-1], hull_vertices[i]] for i in range(n)])
    point = vertices[in_pts[0]]

    index_in = 0
    index_border = closer_edge(point, edges)
    
    result = distance_pt_line(point, edges[index_border])
    for i in range(1, len(in_pts)):
        point = vertices[in_pts[i]]
        closer_edge_i = closer_edge(point, edges)
        if distance_pt_line(point, edges[closer_edge_i]) < result:
            result = distance_pt_line(point, edges[closer_edge_i])
            index_in = i
            index_border = closer_edge_i
    return index_in, index_border
    

def tsp_solver(vertices):
    if check_hull(vertices):
        hull = ConvexHull(vertices)
        in_pts = non_hull(hull.vertices, len(vertices))
    else:
        return vertices
    
    border = hull.vertices.copy()
    while len(in_pts) > 0:
        index_in, index_border = closer(vertices, border, in_pts)
        if index_border == 0:
            border = np.insert(border, index_border, in_pts[index_in])
        else:
            border = np.insert(border, index_border, in_pts[index_in])
        in_pts = np.delete(in_pts, index_in)
    
    return vertices[border]
        


if __name__ == "__main__":
    
    filename = 'save.npy'
    
    old_cities = True
    nb_cities = 20
    range_cities = 10
    
    if os.path.isfile(filename) and old_cities:
        print('Read file from: ' + filename)
        cities = np.load(filename)
    else:
        cities = []
        while len(cities) < nb_cities:
            city = list(np.random.randint(range_cities, size=2))
            if not (city in cities):
                cities += [city]
        cities = np.array(cities)
        np.save(filename, cities)
    
    hull = ConvexHull(cities)
    
    out = tsp_solver(cities)
    print(out)
    print(cities)
    plt.plot(out[:,0], out[:,1], '-o')
    
    
    plt.plot(cities[hull.vertices,0], cities[hull.vertices,1], 'r--', lw=2)
    plt.show()













