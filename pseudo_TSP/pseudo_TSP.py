# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 18:51:19 2018

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
    return norm(cb - dot(cb, ab)/(norm(ab)**2) * ab)


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


def reorganize(vertices, in_pts, hull, it):
    
    if it == 0:
        hull_vertices = vertices[hull.vertices]
        
        n = len(hull_vertices)
        edges = np.array([[hull_vertices[i-1], hull_vertices[i]] for i in range(n)])
    else:
        hull_vertices = vertices[circular_permutation(hull.vertices, len(vertices))]

        n = len(hull_vertices)
        edges = np.array([[hull_vertices[i], hull_vertices[i+1]] for i in range(n-1)])
        
    points_edges = []
    
    for i in range(len(in_pts)):
        points_edges += [[in_pts[i], closer_edge(vertices[in_pts[i]], edges)]]
    
    edges_points = []
    
    for i in range(len(edges)):
        edge_points = []
        edge_points += [edges[i][0]]
        for j in range(len(points_edges)):
            if i == points_edges[j][1]:
                edge_points += [vertices[points_edges[j][0]]]
        edge_points += [edges[i][1]]
        edges_points += [np.array(edge_points)]
    
    return np.array(edges_points)


def circular_permutation(hull_vertices, n):
    new_hull_vertices = hull_vertices.copy()
    safety = 1
    nb = len(hull_vertices)
    # Circular permutation
    while (not (new_hull_vertices[0] == 0 and new_hull_vertices[-1] == n-1) and
           not (new_hull_vertices[0] == n-1 and new_hull_vertices[-1] == 0) and
           safety <= nb):
        for i in range(nb):
            new_hull_vertices[i] = hull_vertices[i-safety]
        safety += 1
    if safety > nb:
        print("Error: vertex problem: " + str(safety))
    if new_hull_vertices[0] != n-1 and new_hull_vertices[0] != 0:
        print("Error: new hull problem")
    
    # If the list is flipped
    if (new_hull_vertices[0] == n-1 and new_hull_vertices[-1] == 0):
        new_hull_vertices_2 = new_hull_vertices.copy()
        for i in range(nb):
            new_hull_vertices_2[nb - i - 1] = new_hull_vertices[i]
        return new_hull_vertices_2
    else:
        return new_hull_vertices


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


def algo(vertices, it=0):
    """Recursively iterate a convex hull"""
    if check_hull(vertices):
        hull = ConvexHull(vertices)
        in_pts = non_hull(hull.vertices, len(vertices))
    else:
        return vertices
    if len(in_pts) == 0:
        if it==0:
            return vertices[hull.vertices]
        else:
            new_hull_vertices = circular_permutation(hull.vertices, len(vertices))
            return vertices[new_hull_vertices]
    else:
        new_set = reorganize(vertices, in_pts, hull, it)
        new_vertices = []
        for i in range(len(new_set)):
            if i == len(new_set)-1:
                new_vertices += list(algo(new_set[i], it+1))
            else:
                new_vertices += list(algo(new_set[i], it+1))[:-1]
        return np.array(new_vertices)
            
        


if __name__ == "__main__":
    
    filename = 'save.npy'
    
    old_cities = False
    nb_cities = 16
    range_cities = 4
    
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
    
    out = algo(cities)
    print(out)
    print(cities)
    plt.plot(out[:,0], out[:,1], '-o')
    
    
    plt.plot(cities[hull.vertices,0], cities[hull.vertices,1], 'r--', lw=2)
    plt.show()













