# https://github.com/Slimercorp/MPC-Carla

import numpy as np
import time
import casadi as ca
import carla

from carla_api.mpc.helper_functions import calculate_terminal_deviation, calculate_min_distance_so, calculate_min_distance_do, \
    distance_to_closest_waypoint, distance_to_next_closest_waypoint
from carla_api.mpc.kinematic_bicycle import bicycle_model

from carla_api.mpc.config import N, dt, d_safe, d_safe_soft, d_tolerance
from carla_api.mpc.config import DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD
from carla_api.mpc.config import MAX_CONTROL_WHEEL_ANGLE, MAX_CONTROL_ACCELERATION, MAX_CONTROL_BRAKING, MIN_SPEED

from carla_api.mpc.config import FINE_STEER_COEF, FINE_ACC_COEF, FINE_STEER_DOT_COEF, \
    FINE_ACC_DOT_COEF, TERMINAL_COST_COEF, SOC_COST_COEF, DOC_COST_COEF, SOR_COST_COEF



class MpcController:
    def __init__(self, world, ego_vehicle, horizon=N, dt=dt):
        self.world = world
        self.carla_map = world.get_map()
        self.horizon = horizon
        self.opti = None
        self.dt = dt
        self.sol = None
        self.cost = None
        self.is_success = False
        self.X = None
        self.U = None
        self.control_buffer = {"acceleration": [0] * self.horizon, "wheel_angle": [0] * self.horizon}
        self.buffer_index = 0
        self.ego_vehicle = ego_vehicle
        #self.red_light = [None]*N  # if red_light then True, else False
        #self.stop_sign = [None]*N  # if stop_sign then True, else False
      

        self.static_obstacles = [
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Fences, 
            carla.CityObjectLabel.Poles,
            carla.CityObjectLabel.TrafficSigns,
            carla.CityObjectLabel.TrafficLight,
            carla.CityObjectLabel.Walls,
            carla.CityObjectLabel.Static,
            carla.CityObjectLabel.Pedestrians,
         
        ]
        
        self.static_obstacles_soft = [
            carla.CityObjectLabel.GuardRail, 
            carla.CityObjectLabel.Sidewalks,
            carla.CityObjectLabel.Car,
            carla.CityObjectLabel.Truck,
            carla.CityObjectLabel.Bus,
            carla.CityObjectLabel.Motorcycle,
            carla.CityObjectLabel.Bicycle,
            carla.CityObjectLabel.Rider,
            carla.CityObjectLabel.Vegetation, 
            ]


    def reset_solver(self, x0, y0, yaw0, v0, obstacles, obstacles_soft, waypoints): 
        self.opti = ca.Opti()
        
        s_opts = {"max_cpu_time": 0.8, 
				  "print_level": 0, 
			      #"constr_viol_tol": 1e-2,
				  #"acceptable_constr_viol_tol": 0.01, 
                  #"nlp_scaling_method": "gradient-based",
                  } # "none" "equilibration-based" "gradient-based"
                  
        p_opts = {"expand": True, "print_time": False}
      
        self.opti.solver('ipopt', p_opts, s_opts)

        self.X = self.opti.variable(4, self.horizon + 1)  # State: (x, y, yaw, v)
        self.U = self.opti.variable(2, self.horizon)  # Control (delta, acc)
        
        
        
        self.opti.set_initial(self.X, np.tile([x0, y0, yaw0, v0], (self.horizon + 1,1)).T)
        
        self.set_init_vehicle_state(x0, y0, yaw0, v0)
        


        for k in range(self.horizon):

            # C1
            self.opti.subject_to(
                self.X[:, k + 1] == bicycle_model(self.X[:, k], self.U[:, k]))

            # C2
            self.opti.subject_to(self.U[0, k] <= MAX_CONTROL_WHEEL_ANGLE)
            self.opti.subject_to(self.U[0, k] >= -MAX_CONTROL_WHEEL_ANGLE)
            self.opti.subject_to(self.U[1, k] <= MAX_CONTROL_ACCELERATION)
            self.opti.subject_to(self.U[1, k] >= MAX_CONTROL_BRAKING)
            
            # C NEW : STATIC OBS AVOIDANCE
            
            obstacles_k = obstacles[k]
  
            if obstacles_k.size:
                for i in range(obstacles_k.shape[0]):
                    self.opti.subject_to((self.X[0, k] -  obstacles_k[i,0])**2 + (self.X[1, k] - obstacles_k[i,1])**2 >= d_safe**2)
                    
            obstacles_k_soft = obstacles_soft[k]
  
            if obstacles_k_soft.size:
                for i in range(obstacles_k_soft.shape[0]):
                    self.opti.subject_to((self.X[0, k] -  obstacles_k_soft[i,0])**2 + (self.X[1, k] - obstacles_k_soft[i,1])**2 >= d_safe_soft**2)
                
                
            # C NEW : STAY ON THE ROAD
             
            if k==0:     
                self.opti.subject_to((self.X[0, k] - waypoints[k][0])**2 + (self.X[1, k] - waypoints[k][1])**2 <= d_tolerance**2)
                
            if k==(self.horizon-1):
                self.opti.subject_to((self.X[0, k] - waypoints[-1][0])**2 + (self.X[1, k] - waypoints[-1][1])**2 <= d_tolerance**2)
            
            """if k>1:
               self.opti.subject_to(self.X[3, k] > MIN_SPEED)"""
           
            

    def set_init_vehicle_state(self, x, y, yaw, v):
        self.opti.subject_to(self.X[:, 0] == ca.vertcat(x, y, yaw, v))

    def update_cost_function(self, destination, dyn_predicted_traj): 
        self.cost = 0

        terminal_cost = TERMINAL_COST_COEF * \
            		 calculate_terminal_deviation(self.X[:2, self.horizon-1], destination)
            		 
        self.cost += terminal_cost 

        for k in range(self.horizon):

            fine_steer_dot = 0
            fine_acc_dot = 0
            if k > 0:
                fine_steer_dot += FINE_STEER_DOT_COEF * \
                    (self.U[0, k] - self.U[0, k - 1]) ** 2
                #fine_acc_dot += FINE_ACC_DOT_COEF * \
                 #   (self.U[1, k] - self.U[1, k - 1]) ** 2

	    

            dyn_obs_collision_cost = DOC_COST_COEF * \
                                   calculate_min_distance_do(
                                   self.X[:2, k], dyn_predicted_traj, k)

           
       
            self.cost += fine_steer_dot + dyn_obs_collision_cost
            
         
        self.opti.minimize(self.cost)

    def solve(self):
        try:
            self.sol = self.opti.solve()
            self.is_success = True

            wheel_angle_, acceleration_ = self.get_controls_value()
            self.control_buffer["acceleration"] = self.control_buffer["acceleration"][1:] + [
                acceleration_]
            self.control_buffer["wheel_angle"] = self.control_buffer["wheel_angle"][1:] + [
                wheel_angle_]
            print(f"Control buffer updated: {self.control_buffer}")
            self.buffer_index = 0
        except Exception as e:
            print(f"Error in MPC solver: {e}")
            print(
                "Error or delay upon MPC solution calculation. Previous calculated control value will be used")
                
            
            self.is_success = False

    def get_controls_value(self):
        if self.is_success:
            wheel_angle_ = self.sol.value(self.U[0, 0])
            acceleration_ = self.sol.value(self.U[1, 0])
            print("X-values: ", self.sol.value(self.X))
      
        else:
            if self.buffer_index < self.horizon:
                acceleration_ = self.control_buffer["acceleration"][self.buffer_index]
                wheel_angle_ = self.control_buffer["wheel_angle"][self.buffer_index]
                self.buffer_index += 1
            else:
                raise Exception("Control buffer is empty.")

        return wheel_angle_, acceleration_

    def get_optimized_cost(self):
        return self.sol.value(self.cost)

   
    def get_static_obstacles(self, ego_predicted_traj):
        bboxs = self.get_static_obstacle_bbox(ego_predicted_traj)
        return bboxs

    def get_static_obstacle_bbox(self, ego_predicted_traj):
        bounding_boxes = []
        for k in range(ego_predicted_traj.shape[0]):
            bounding_boxes_at_k = []
            for object_type in self.static_obstacles:
                type_bbs = self.world.get_level_bbs(object_type)
                for bb in type_bbs:
                    distance = np.linalg.norm(np.array(
                        [ego_predicted_traj[k, 0] - bb.location.x, ego_predicted_traj[k, 1] - bb.location.y]))
                    if distance <= DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD:
                        bounding_boxes_at_k.append(
                            [bb.location.x, bb.location.y, bb.location.z])
            bounding_boxes.append(np.array(bounding_boxes_at_k))
        # list of np np arrays of size (n_static_obstacles, 3) for each timestep (k)
        return bounding_boxes
        
    def get_static_obstacles_soft(self, ego_predicted_traj):
        bboxs = self.get_static_obstacle_bbox_soft(ego_predicted_traj)
        return bboxs

    def get_static_obstacle_bbox_soft(self, ego_predicted_traj):
        bounding_boxes = []
        for k in range(ego_predicted_traj.shape[0]):
            bounding_boxes_at_k = []
            for object_type in self.static_obstacles_soft:
                type_bbs = self.world.get_level_bbs(object_type)
                for bb in type_bbs:
                    distance = np.linalg.norm(np.array(
                        [ego_predicted_traj[k, 0] - bb.location.x, ego_predicted_traj[k, 1] - bb.location.y]))
                    if distance <= DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD:
                        bounding_boxes_at_k.append(
                            [bb.location.x, bb.location.y, bb.location.z])
            bounding_boxes.append(np.array(bounding_boxes_at_k))
        # list of np np arrays of size (n_static_obstacles, 3) for each timestep (k)
        return bounding_boxes


    @staticmethod
    def process_control_inputs(wheel_angle_rad, acceleration_m_s_2):
        if acceleration_m_s_2 == 0:
            throttle = 0
            brake = 0
        elif acceleration_m_s_2 < 0:
            throttle = 0
            brake = acceleration_m_s_2 / MAX_CONTROL_BRAKING
        else:
            throttle = acceleration_m_s_2 / MAX_CONTROL_ACCELERATION
            brake = 0
        steer = wheel_angle_rad / MAX_CONTROL_WHEEL_ANGLE
        return throttle, brake, steer

        
