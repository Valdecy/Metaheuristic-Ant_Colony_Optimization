# Metaheuristic-Ant_Colony_Optimization
Ant Colony Optimization Function for TSP problems. The function returns: 1) A list with the order of the cities to visit, and the total distance for visiting this same list order.

* X = Distance Matrix.

* buid_distance_matrix (HELPER FUNCTION) = Tranforms coordinates in a distance matrix (euclidean distance).

* ants = Initial number of ants. The Default Value is 5.

* alpha = Pheromone update value. The Default Value is 1.

* beta = Pheromone update value. The Default Value is 2.

* decay = Pheromone evaporation rate. The Default Value is 0.05.

* iterations = Total number of iterations. The Default Value is 5.

* plot_tour_distance_matrix (HELPER FUNCTION) = A projection is generated based on the distance matrix. The estimated projection may present a plot with path crosses, even for the 2-opt optimal solution (Red Point = Initial city; Orange Point = Second City).

* plot_tour_coordinates (HELPER FUNCTION) = Plots the 2-opt optimal solution (Red Point = Initial city; Orange Point = Second City).

![alt text](https://github.com/Valdecy/Metaheuristic-Ant_Colony_Optimization/blob/master/Python-MH-Ant%20Colony%20Optimization.gif)

<p align="center"> 
<img src="https://github.com/Valdecy/Metaheuristic-Ant_Colony_Optimization/blob/master/Python-MH-Ant%20Colony%20Optimization.gif">
</p>
