# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:54:13 2018

@author: Guillaume
"""

"""
This code presents a genetic programming approach of the travelling salesman 
problem (TSP).
It is mainly inspired from the introduction of the Julian F. Miller's book:
Cartesian Genetic Programming.

The function descibed in pseudo_TSP are an implementation of a personnal
approach of this problem. They are inspired from convex hull finding. I
then input these solutions, which are not found in O(exp(n)) but less,
into the GP algorithm to see if it can improve the result. Conclusion:
it can for high dimension (number of cities > 30)... I'm not perfect...
"""

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import os

from pseudo_TSP.pseudo_TSP import algo
from pseudo_TSP.pseudo_TSP_2 import tsp_solver

# filename is the file where the cities are stored
# filename_best_tsp stores the best solution
filename = 'save\\save.npy'
filename_best_tsp = 'save\\best_tsp.npy'

# Parameters of the cities:
## old_cities: should we train our model on the old dataset or should we
## renew the dataset
## nb_cities: number of cities
## range_cities: cities are randomly placed on a grid of size range_cities²
old_cities = True
nb_cities = 20
range_cities = 10


# Load the cities if old_cities is set to True
# Create a new bunch of cities otherwise using the parameters above and save
# this new set in filename file. To generate the cities it suppose that two
# cities cannot be placed at the same place. And more important: the number
# of cities should be lower than the square of range_cities otherwise you'll
# finish in an endless loop!
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



def permutation(c):
    """Random permutation of cities. Return the permutated list and the index
    Input: list of cities
    Output: tuple, tuple[0] contains the flipped cities coordinates,
    tuple[1] contains the flipped indices.
    """
    n = len(c)
    index = [i for i in range(n)]
    permutated_i = np.random.permutation(index)
    output = []
    for i in permutated_i:
        output += [c[i]]
    return np.array(output), permutated_i

    
    
# Output of the TSP solver implemented by my tiny hands :P
# It contains a list of tuple which are the coordinate of the cities in order
# corresponding to the TS positions
out = tsp_solver(cities)

# The code below change the format from a list of tuple to a simple list,
# simply because this part needs it to work... The element of the list
# correspond to the index of the cities in the initial list. That is to say
# that it computes the permutation indices
# Compute permutation (supposed there is no double) = brute force...
per_out = []
for i in range(nb_cities):
    j = 0
    while j < nb_cities and list(out[i]) != list(cities[j]):
        j += 1
    per_out += [j]

first_tsp = (out, np.array(per_out))

# Contain a saved instance of the previous best solution
if old_cities and os.path.isfile(filename_best_tsp):
    saved = np.load(filename_best_tsp)
    second_tsp = (cities[saved], saved)
else:
    second_tsp = permutation(cities)
    np.save(filename_best_tsp, second_tsp[1])

# Third solution
out = algo(cities)[:-1]

per_out = []
for i in range(nb_cities):
    j = 0
    while j < nb_cities and list(out[i]) != list(cities[j]):
        j += 1
    per_out += [j]

third_tsp = (out, np.array(per_out))

# Parameter of the genetic algorithm
# size: number of individuals in the population
# first_tsp, second_tsp, third_tsp: are tuples containing the cities computed
# with the other method we will feed these solutions to the network to see if
# it can improve it.
# pop: contains the population which is a list of tuple
# epochs: number of generation. We will apply genetic algorithm for each
# generation: selection, crossover, mutation
# selection_rate: selection rate define for genetic selection
size = 1000
pop = [first_tsp, second_tsp, third_tsp]
pop += [permutation(cities) for i in range(size-3)]
epochs = 100
selection_rate = 0.2



def distance(c1, c2):
    """Return distance between two cities"""
    return np.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)


def fitness(tsp):
    """Compute the fitness of a cities list. It is simple the travelling
    salesman total travelling distance"""
    cities = tsp[0]
    output = 0
    n = len(cities)
    for i in range(n):
        output += distance(cities[i-1], cities[i])
    return output


def crossover(tsp1, tsp2):
    """Crossover between two parents. Return a child"""
    assert len(tsp1[0])==len(tsp2[0])
    n = len(tsp1[0])
    
    output = []
    output_i = []
    
    count1 = 0
    count2 = 0
    
    alea = [0]*n + [1]*n
    np.random.shuffle(alea)
    
    for i in range(2*n):
        if alea[i] == 0:
            if not tsp1[1][count1] in output_i:
                output_i += [tsp1[1][count1]]
                output += [tsp1[0][count1]]
            count1 += 1
        else:
            if not tsp2[1][count2] in output_i:
                output_i += [tsp2[1][count2]]
                output += [tsp2[0][count2]]
            count2 += 1
    
    return np.array(output), np.array(output_i)



def crossover_pop(pop, target_size):
    """Create a crossover on a population until reaching the target_size"""
    
    n = len(pop)
    
    assert n>1 # More than one individual
    assert n<target_size # Less individuals in the input pop than in the output one
    
    alea = [0,0]
    
    while len(pop) < target_size:
        # Choose two parents
        while alea[0]==alea[1]:
            alea = np.random.randint(0,n,size=2)
            
        # Add the child to the population after a mutation
        pop += [mutation(crossover(pop[alea[0]], pop[alea[1]]))]
    #check_pop(pop,cities)
    return pop
    
    

def mutation(tsp):
    """Flip two elements in the list"""
    alea = np.random.randint(0,10,size=2)
    tsp[0][alea[0]], tsp[0][alea[1]] = tsp[0][alea[1]].copy(), tsp[0][alea[0]].copy()
    tsp[1][alea[0]], tsp[1][alea[1]] = tsp[1][alea[1]], tsp[1][alea[0]]
    return tsp



def selection(pop, prop=0.2):
    """Return a sublist of pop of size = 0.2 * size(pop) selected on the
    fitness criteria"""
    n = len(pop)
    
    dtype = [('index',int), ('fitness',float)]
    values = []
    for i in range(n):
        values += [(i,fitness(pop[i]))]
    fit = np.array(values, dtype=dtype)
    sorted_fit = np.sort(fit, order='fitness')
    selected_size = int(n*prop)
    
    output = []
    for i in range(selected_size):
        output += [pop[sorted_fit[i][0]]]
        
    return output

def display(tsp, ax):
    """Display a solution"""
    print("Best fitness: " + str(fitness(tsp)))
    tsp_i = tsp[0]
    x = []
    y = []
    for i in range(-1, len(tsp_i)):
        x += [tsp_i[i][0]]
        y += [tsp_i[i][1]]
    ax.plot(x, y, marker='o')


def display_best(pop, ax):
    """Display the best solution"""
    m = fitness(pop[0])
    m_i = 0
    for i in range(len(pop)):
        fit = fitness(pop[i])
        if fit < m:
            m_i = i
            m = fit
    np.save(filename_best_tsp, np.array(pop[m_i][1]))
    display(pop[m_i], ax)

def check_index(tsp, cities):
    """Check if a solution is acceptable"""
    for i in range(len(tsp)):
        if list(tsp[0][i]) != list(cities[tsp[1][i]]):
            print("f*ck!!! mistakes!!!" + str(i))


def check_pop(pop, cities):
    """Check if a population is acceptable"""
    for i in range(len(pop)):
        print("index" + str(i))
        check_index(pop[i], cities)
        

# Display the best solution before training
fig, axs = plt.subplots(1, 4, figsize=(12,4.8))

axs[0].set_title("Previously saved best solution")
display(pop[1], axs[0])
axs[1].set_title("Solution " + str(1) + " before training")
display(pop[0], axs[1])
axs[2].set_title("Solution " + str(2) + " before training")
display(pop[2], axs[2])


for g in tqdm(range(epochs)):
    pop = crossover_pop(selection(pop,selection_rate), size)

# Display the best solution after training
axs[3].set_title("Best solution after training")
display_best(pop, axs[3])
plt.show()