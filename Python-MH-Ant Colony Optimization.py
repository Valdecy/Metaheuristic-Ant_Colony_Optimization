############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Ant Colony Optimization

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Ant_Colony_Optimization, File: Python-MH-Ant Colony Optimization.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Ant_Colony_Optimization>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import math
import os

# Function: Probability Matrix 
def city_probability (attractiveness, thau, city = 0, alpha = 1, beta = 2, city_list = []):
    probability = pd.DataFrame(0, index = Xdata.index, columns = ['atraction','probability','cumulative_probability'])
    for i in range(0, probability.shape[0]):
        if (i+1 not in city_list):
            print(i)
            probability.iloc[i, 0] = (thau.iloc[i, city]**alpha)*(attractiveness.iloc[i, city]**beta)
    for i in range(0, probability.shape[0]):
        if (i+1 not in city_list and probability['atraction'].sum() != 0):
            probability.iloc[i, 1] = probability.iloc[i, 0]/probability['atraction'].sum()
        if (i == 0):
            probability.iloc[i, 2] =  probability.iloc[i, 1] 
        else:
            probability.iloc[i, 2] = probability.iloc[i, 1] + probability.iloc[i - 1, 2] 
    
    if (len(city_list) > 0):
        for i in range(0, len(city_list)):
            probability.iloc[city_list[i]-1, 2] = 0.0
            
    return probability

# Function: Select Next City
def city_selection(probability_matrix, city_list = []):
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    city = 0
    for i in range(0, probability_matrix.shape[0]):
        if (random <= probability_matrix.iloc[i, 2] and i+1 not in city_list):
          city = i+1
          break
     
    return city

# Function: Update Thau
def update_thau(Xdata, thau, decay = 0.5, accumulate = 0, city_list = [1,2,1]):
    if (accumulate == 0):
        thau = thau*decay
    distance = 0
    for i in range(0, len(city_list)-1):
        j = i + 1
        distance = distance + Xdata.iloc[city_list[i]-1,city_list[j]-1]
    
    pheromone = 1/distance
    
    for i in range(0, len(city_list)-1):
        j = i + 1 
        thau.iloc[city_list[i]-1,city_list[j]-1] = thau.iloc[city_list[i]-1,city_list[j]-1] + pheromone
        
    return thau, distance

# ACO Function
def ant_colony_optimization(Xdata, ants = 5, iterations = 50, alpha = 1, beta = 2, decay = 0.5):
    
    h = pd.DataFrame(np.nan, index = Xdata.index, columns = list(Xdata.columns.values))
    distance = Xdata.values.sum()
    best_routes = []
    
    # h matrix 
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i == j or Xdata.iloc[i,j] == 0):
                h.iloc[i,j] = 0.000001
            else:
                h.iloc[i,j] = 1/Xdata.iloc[i,j]
    count = 0
    while (count <= iterations):
        print("Iteration = ", count)
        if (count == 0):
            thau = pd.DataFrame(1, index = Xdata.index, columns = list(Xdata.columns.values))
        
        for ant in range(0, ants):
            city_list = []
            initial = int((ant + 1) - Xdata.shape[0]*math.floor((ant +1)/Xdata.shape[0]))
            if ((ant + 1) % Xdata.shape[0] == 0):
                initial = Xdata.shape[0]
            city_list.append(initial)
            
            for i in range(0, Xdata.shape[0] - 1):
                probability  = city_probability(h, thau, city = i, alpha = alpha, beta = beta, city_list = city_list)
                path_point = city_selection(probability, city_list = city_list)
                city_list.append(path_point)
            city_list.append(city_list[0])
            thau, path_distance = update_thau(Xdata, thau, decay = decay, accumulate = ant, city_list = city_list)
            if (path_distance < distance):
                best_routes.append([city_list, path_distance])
                distance = path_distance
        count = count + 1
    
    return  best_routes

######################## Part 1 - Usage ####################################

df = pd.read_csv('Python-MH-Ant Colony Optimization-Dataset.txt', sep = '\t')
X = df

ant_colony_optimization(X, ants = 5, iterations = 50, alpha = 1, beta = 2, decay = 0.5)
