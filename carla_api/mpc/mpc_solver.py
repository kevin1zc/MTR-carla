# https://github.com/Slimercorp/MPC-Carla

import numpy as np
import time
import casadi as ca
import carla

from carla_api.mpc.helper_functions import calculate_terminal_deviation, calculate_min_distance_so, calculate_min_distance_do, \
                                           distance_to_closest_waypoint, distance_to_next_closest_waypoint
from carla_api.mpc.kinematic_bicycle import bicycle_model

from carla_api.mpc.config import N, dt
from carla_api.mpc.config import EPSILON, DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD
from carla_api.mpc.config import MAX_CONTROL_WHEEL_ANGLE, MAX_CONTROL_ACCELERATION, MAX_CONTROL_BRAKING, MIN_SPEED

from carla_api.mpc.config import FINE_STEER_COEF, FINE_ACC_COEF, FINE_STEER_DOT_COEF, \
                               FINE_ACC_DOT_COEF, TERMINAL_COST_COEF, SOC_COST_COEF, DOC_COST_COEF, SOR_COST_COEF

from carla_api.mpc.config import MAX_BRAKING_M_S_2, MAX_WHEEL_ANGLE_RAD, MAX_ACCELERATION_M_S_2


class MpcController:
    def __init__(self, world, ego_vehicle, predicted_trajs, waypoints, horizon = N, dt = dt):
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

        # predicted_traj = ego vehicle + 7 closest agents next 8 seconds (80 timesteps)
        # ego
        self.ego_predicted_traj = np.array(predicted_trajs[: 10]) # (10, 2)
        # 7 closest agents predicted traj
        self.dyn_predicted_traj = np.array(predicted_trajs[10:]).reshape(-1, 10, 2) # (7, 10, 2)
        self.ego_vehicle = ego_vehicle
        self.red_light = [None]*N # if red_light then True, else False
        self.stop_sign = [None]*N # if stop_sign then True, else False
        self.waypoints = waypoints if len(waypoints) >= 10 else waypoints + [waypoints[-1]] * (10 - len(waypoints)) # next k closest waypoints

        self.epsilon = 1e-6

        self.static_obstacles = [
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Fences,
            carla.CityObjectLabel.Poles,
            carla.CityObjectLabel.TrafficSigns,
            carla.CityObjectLabel.TrafficLight,
            carla.CityObjectLabel.Walls,
            carla.CityObjectLabel.GuardRail,
            carla.CityObjectLabel.Static,
        ]

    def reset_solver(self):
        self.opti = ca.Opti()
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_cpu_time': self.dt
        }
        self.opti.solver('ipopt', opts)

        self.X = self.opti.variable(4, self.horizon + 1)  # State: (x, y, yaw, v)
        self.U = self.opti.variable(2, self.horizon)  # Control (delta, acc)

        for k in range(self.horizon):

            # C1
            self.opti.subject_to(self.X[:, k + 1] == bicycle_model(self.X[:, k], self.U[:, k], self.dt))


            # C2
            self.opti.subject_to(self.U[0, k] <= MAX_CONTROL_WHEEL_ANGLE)
            self.opti.subject_to(self.U[0, k] >= -MAX_CONTROL_WHEEL_ANGLE)
            self.opti.subject_to(self.U[1, k] <= MAX_CONTROL_ACCELERATION)
            self.opti.subject_to(self.U[1, k] >= MAX_CONTROL_BRAKING)

            # # C3 & C4
            # if self.red_light[k] or self.stop_sign[k]:
            #     self.opti.subject_to(self.X[3, k] == 0)  # velocity is zero
            # else:
            #     self.opti.subject_to(self.X[3, k] == MIN_SPEED) 


    def set_init_vehicle_state(self, x, y, yaw, v): 
        self.opti.subject_to(self.X[:, 0] == ca.vertcat(x, y, yaw, v))

    def update_cost_function(self, destination=None, obstacles=None):  
        self.cost = 0

        # CF1
        terminal_cost = TERMINAL_COST_COEF * calculate_terminal_deviation(self.X[:2, self.horizon-1], destination)

        for k in range(self.horizon):

            """ fine_x = ca.if_else(ca.fabs(self.X[0, k] - x_ref[k]) > PATH_TOLERANCE_M,
                                FINE_X_COEF * (self.X[0, k] - x_ref[k]) ** 2, 0)
            fine_y = ca.if_else(ca.fabs(self.X[1, k] - y_ref[k]) > PATH_TOLERANCE_M,
                                FINE_Y_COEF * (self.X[1, k] - y_ref[k]) ** 2, 0)
            fine_v = FINE_V_COEF * (self.X[3, k] - v_ref[k]) ** 2 """

            # CF2
            fine_steer = FINE_STEER_COEF * self.U[0, k] ** 2
            fine_acc = FINE_ACC_COEF * self.U[1, k] ** 2

            fine_steer_dot = 0
            fine_acc_dot = 0
            if k > 0:
                fine_steer_dot += FINE_STEER_DOT_COEF * (self.U[0, k] - self.U[0, k - 1]) ** 2
                fine_acc_dot += FINE_ACC_DOT_COEF * (self.U[1, k] - self.U[1, k - 1]) ** 2

            # CF3
            if obstacles[k].size > 0:
                static_obs_collision_cost = SOC_COST_COEF * calculate_min_distance_so(self.X[:2, k], obstacles[k])
            else:
                static_obs_collision_cost = 0
            
            # CF4
            dyn_obs_collision_cost = DOC_COST_COEF * calculate_min_distance_do(self.X[:2, k], self.dyn_predicted_traj[:, k, :])

            # CF5
            stay_on_road_cost = SOR_COST_COEF * distance_to_next_closest_waypoint(self.X[:2, k], self.waypoints[k][0], self.waypoints[k][1])
            # stay_on_road_cost = 0

            self.cost += terminal_cost + fine_steer + fine_acc + fine_steer_dot + fine_acc_dot \
                        + static_obs_collision_cost + dyn_obs_collision_cost + stay_on_road_cost

        self.opti.minimize(self.cost)

    def solve(self):
        try:
            self.sol = self.opti.solve()
            self.is_success = True

            wheel_angle_, acceleration_ = self.get_controls_value()
            self.control_buffer["acceleration"] = self.control_buffer["acceleration"][1:] + [acceleration_]
            self.control_buffer["wheel_angle"] = self.control_buffer["wheel_angle"][1:] + [wheel_angle_]
            print(f"Control buffer updated: {self.control_buffer}")
            self.buffer_index = 0
        except Exception as e:
            print(f"Error in MPC solver: {e}")
            print("Error or delay upon MPC solution calculation. Previous calculated control value will be used")
            self.is_success = False

    def get_controls_value(self):
        if self.is_success:
            wheel_angle_= self.sol.value(self.U[0, 0])
            acceleration_ = self.sol.value(self.U[1, 0])
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
    
    def run_mpc(self):
        start_time = time.time()

        self.reset_solver()

        x0, y0, yaw0, v0 = self.get_ego_vehicle_state()  
        self.set_init_vehicle_state(x0, y0, yaw0, v0)

        # predicted_traj = ego vehicle + 7 closest agents next 8 seconds (80 timesteps)
        # logger.log_controller_input(x0, y0, v0, theta0, x_ref[0], y_ref[0], V_REF, theta_ref[0])

        self.update_cost_function(self.get_destination(), self.get_static_obstacles())
        self.solve()

        wheel_angle, acceleration = self.get_controls_value()  # optimal_solution.control_sequence[0]
        print(f"Wheel angle: {wheel_angle}, Acceleration: {acceleration}")
        throttle, brake, steer = self.process_control_inputs(wheel_angle, acceleration)

        end_time = time.time()
        mpc_calculation_time = end_time - start_time
        print(f"Calculation time of MPC controller: {mpc_calculation_time:.6f} seconds")

        time.sleep(max(self.dt - mpc_calculation_time, 0))
        
        return throttle, brake, steer

    def get_ego_vehicle_state(self):
        transform = self.ego_vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = np.deg2rad(transform.rotation.yaw)
        v = np.sqrt(self.ego_vehicle.get_velocity().x**2 + self.ego_vehicle.get_velocity().y**2)
        
        return x, y, yaw, v 
    
    def get_destination(self):
        # location = self.carla_map.get_waypoint(
        #     carla.Location(self.ego_predicted_traj[-1][0].item(), self.ego_predicted_traj[-1][1].item(), 0), 
        #     project_to_road=True
        # ).transform.location
        return (self.waypoints[-1][0], self.waypoints[-1][1]) # (x, y)

    def get_static_obstacles(self):
        bboxs = self.get_static_obstacle_bbox()
        return bboxs

    def get_static_obstacle_bbox(self):
        bounding_boxes = []
        for k in range(self.ego_predicted_traj.shape[0]):
            bounding_boxes_at_k = []
            for object_type in self.static_obstacles:
                type_bbs = self.world.get_level_bbs(object_type)
                for bb in type_bbs:
                    distance = np.linalg.norm(np.array([self.ego_predicted_traj[k, 0] - bb.location.x, self.ego_predicted_traj[k, 1] - bb.location.y]))
                    if distance <= DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD:
                        bounding_boxes_at_k.append([bb.location.x, bb.location.y, bb.location.z])
            bounding_boxes.append(np.array(bounding_boxes_at_k))
        return bounding_boxes # list of np np arrays of size (n_static_obstacles, 3) for each timestep (k)

    def apply_control(self, steer, throttle, brake):  # We will not apply control values here. Zhuochen will add this to his code. 
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
    
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
    
