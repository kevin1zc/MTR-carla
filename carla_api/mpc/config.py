import numpy as np

EPSILON = 1e-6
DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD = 10

# Vehicle parameters
L = 2.8        # Wheelbase, e.g. 2.8 m for a typical car
# v_ref = 10.0         # Desired reference speed (m/s), ~36 km/h
d_safe = 2.0         # Minimum safe distance from obstacles (meters)  try values from 2 to 5


# MPC horizon
N = 10               # Number of steps in the prediction horizon
dt = 0.1           # Timestep (seconds)


# Vehicle characteristics (from CARLA simulator, see get_physics_control())
MAX_CONTROL_WHEEL_ANGLE = 70 / 180 * np.pi               # 1.22 Radian
MAX_CONTROL_ACCELERATION = 10                            # m/s^2
MAX_CONTROL_BRAKING = -4.1                               # m/s^2
MIN_SPEED = 10 

TERMINAL_COST_COEF = 1
FINE_STEER_COEF = 1
FINE_ACC_COEF = 1
FINE_STEER_DOT_COEF = 1
FINE_ACC_DOT_COEF = 1
SOC_COST_COEF = 1
DOC_COST_COEF = 1
SOR_COST_COEF = 1

MAX_BRAKING_M_S_2 = MAX_CONTROL_BRAKING
MAX_WHEEL_ANGLE_RAD = MAX_CONTROL_ACCELERATION
MAX_ACCELERATION_M_S_2 = MAX_CONTROL_WHEEL_ANGLE

