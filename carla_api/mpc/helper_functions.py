import math
import carla
import casadi as ca
import numpy as np
from carla_api.mpc.config import d_safe

epsilon = 0.001

def calculate_terminal_deviation(predicted_destination, destination):

    distance = (predicted_destination[0] - destination[0])**2 + (predicted_destination[1] - destination[1])**2

    return distance #**0.5


def calculate_min_distance_so(current_position, obstacles): 
    # current_position = (ego_vehicle_x, ego_vehicle_y)
    # obstacles = np.array(number_of_objects, 3) where 3 is for x/y/z coordinates

    dist_list = []

    for i in range(obstacles.shape[0]):
        temp_dist = (current_position[0] -  obstacles[i,0])**2 + (current_position[1] - obstacles[i,1])**2 + epsilon
        temp_dist = (1/temp_dist)
        dist_list.append(temp_dist)

    return sum(dist_list)  



def calculate_min_distance_do(current_position, predicted_trajs, k):
    # current_position = (ego_vehicle_x, ego_vehicle_y)
    # predicted_trajs = list of np.array(10, 2) where 10 is horizon and 2 is for x/y coordinates

    dist_list = []

    for i in range(len(predicted_trajs)):
        temp_dist = (current_position[0] -  predicted_trajs[i][k,0])**2 + (current_position[1] - predicted_trajs[i][k,1])**2 + epsilon
        temp_dist = (1/temp_dist)
        dist_list.append(temp_dist)

    return sum(dist_list)


def find_closest_waypoint(current_position, cx, cy):
        distances = np.sum(( np.array([[current_position[0]], [current_position[1]]]) -
                             np.stack((cx, cy)) )**2, axis=0)
        # idx = np.argmin(distances)
        two_smallest_indices = np.argsort(distances)[:2]

        return cx[two_smallest_indices[0]], cy[two_smallest_indices[0]], cx[two_smallest_indices[1]], cy[two_smallest_indices[1]] 


def calculate_lateral_deviation(current_position, x_ref1, y_ref1, x_ref2, y_ref2):
    num = (y_ref2 - y_ref1) * current_position[0] - (x_ref2 - x_ref1) * current_position[1] + x_ref2 * y_ref1 - y_ref2 * x_ref1
    denom = np.sqrt((y_ref2 - y_ref1) ** 2 + (x_ref2 - x_ref1) ** 2)
    if denom > 0.01:
        return num / denom
    else:
        return 0
    

def distance_to_closest_waypoint(current_position, cx, cy):

    x_ref1, y_ref1, x_ref2, y_ref2 = find_closest_waypoint(current_position, cx, cy)
    
    return calculate_lateral_deviation(current_position, x_ref1, y_ref1, x_ref2, y_ref2)



def distance_to_next_closest_waypoint(current_position, cx, cy):

    # distances = np.sum(( np.array([[current_position[0]], [current_position[1]]]) -
    #                          np.stack((cx, cy)) )**2, axis=0)
       
    # three_smallest_indices = np.argsort(distances)[:3]
    # next_closest_idx = np.max(three_smallest_indices)

    # waypt_x = cx[next_closest_idx]
    # waypt_y = cy[next_closest_idx]

    # dist_to_next_waypoint = (current_position[0] -  waypt_x)**2 + (current_position[1] - waypt_y)**2

    # return dist_to_next_waypoint

    return ((current_position[0] - cx)**2 + (current_position[1] - cy)**2) ** 0.5

