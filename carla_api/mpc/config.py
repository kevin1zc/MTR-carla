import numpy as np

EPSILON = 1e-6
DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD = 20

# Vehicle parameters
L = 2.8        # Wheelbase, e.g. 2.8 m for a typical car
d_safe = 2.0 # 2.0       # Minimum safe distance from obstacles (meters)  
d_safe_soft = 0.005 # 0.04
d_tolerance = 3.0 # 5.0

# MPC horizon
N = 5            # Number of steps in the prediction horizon
dt = 0.1          # Timestep (seconds)


# Vehicle characteristics (from CARLA simulator, see get_physics_control())
MAX_CONTROL_WHEEL_ANGLE = (70 / 180) * np.pi               # 1.22 Radian
MAX_CONTROL_ACCELERATION = 8.5                            # m/s^2
MAX_CONTROL_BRAKING = -4.1                               # m/s^2
MIN_SPEED = 1.0 # we do not use this

TERMINAL_COST_COEF = 2.0
FINE_STEER_COEF = 0.0 #1.0
FINE_ACC_COEF = 0.0 #0.5
FINE_STEER_DOT_COEF = 300.0 #20.0
FINE_ACC_DOT_COEF = 0.0 #5.0
SOC_COST_COEF = 0.0 # 1000.0
DOC_COST_COEF = 100000.0
SOR_COST_COEF = 0.0 #200.0

Follow_Agent = False
