#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import logging
import os
import sys
from pathlib import Path

import time
import carla
import lanelet2
import numpy as np
import numpy.random as random
import pygame
import torch
import matplotlib.pyplot as plt
from lanelet2.routing import RoutingGraph

from carla_api.agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from carla_api.agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from carla_api.agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from carla_api.mpc.mpc_solver import MpcController
from carla_api.mpc.helper_functions import precompute_parking_exit_path, choose_ahead_waypoint
from carla_api.mpc.config import N, dt, Follow_Agent
from carla_api.utils.carla_utils import HUD, KeyboardControl
from carla_api.utils.mtr_data_utils import create_scene_level_data, decode_map_features, generate_prediction_dicts
from carla_api.utils.world import World
from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils
from mtr.utils import common_utils


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    args.sync = True
    delta_t = 0.02

    if delta_t > 0.1:
        sys.exit(
            'Error: Delta_t must be no greater than 0.1s in accordance with MTR.')

    import sys
    print(sys.path, os.getcwd())
    map = lanelet2.io.load(
        "./carla_maps/OSM/Town10HD.osm", lanelet2.io.Origin(0, 0))
    traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                  lanelet2.traffic_rules.Participants.Vehicle)
    routing_graph = RoutingGraph(map, traffic_rules)

    #routing_graph.previous(map.laneletLayer[6135])  # actually next lane
    #routing_graph.following(map.laneletLayer[6135])  # actually previous lane
    # also have left and right relations (in correct order)

    map_infos, dynamic_map_infos = decode_map_features(map, routing_graph)

    cfg_path = "./tools/cfgs/waymo/mtr+100_percent_data.yaml"
    cfg_from_yaml_file(cfg_path, cfg)
    cfg.TAG = Path(cfg_path).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_path.split('/')[1:-1])
    logger = common_utils.create_logger(None, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(
    ) else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    model = model_utils.MotionTransformer(config=cfg.MODEL)

    model_path = "./carla_api/model/latest_model.pth"
    model.load_params_from_file(model_path, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)

        map_name = "Town10HD_Opt"

        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = delta_t
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, delta_t, args)
        # world = World(client.load_world(map_name), hud, args)
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(
                world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(
                    ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior, opt_dict={
                                  'sampling_resolution': 0.12})

        f_start_location_ = world.player.get_location()
        
        if Follow_Agent:
            wp0 = world.map.get_waypoint(f_start_location_)
            wp50  = wp0.next(300.0)[0]  
                     
            goal  = wp50.transform.location
            agent.set_destination(carla.Location(goal.x, goal.y, 0))
            trace = agent.trace_route(wp0, wp50)
            route = []
            for wp in trace:
                route.append([wp[0].transform.location.x,
                             wp[0].transform.location.y])
            route = np.array(route)
            
        else:
            route = precompute_parking_exit_path(carla_map=world.map, start_location=f_start_location_)
            goal = route[-1]
            agent.set_destination(carla.Location(goal[0], goal[1], 0))
        
        
        
        """trace = agent.trace_route(start_waypoint, end_waypoint)
        route = []
        for wp in trace:
            route.append([wp[0].transform.location.x,
                         wp[0].transform.location.y])
        route = np.array(route)"""
        # plt.plot(start_location.x, start_location.y, 'go')
        # plt.plot(destination.x, destination.y, 'ro')
        # plt.plot(route[:, 0], route[:, 1], 'b-')
        # plt.show()

        clock = pygame.time.Clock()

        throttle, brake, steer = 0, 0, 0
        
        time_step = -1
        
        mpc = MpcController(world.world, world.player, horizon=N, dt=dt)
        
    
        run_flag = True
        
        while run_flag:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            info = world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if info is not None:
                info['map_infos'] = map_infos
               # info['dynamic_map_infos'] = dynamic_map_infos
                ret_infos = create_scene_level_data(info, cfg.DATA_CONFIG)
                batch_dict = {
                    'batch_size': 1,
                    'input_dict': ret_infos,
                    'batch_sample_count': [len(info['vehicle_ids'])]
                }
                with torch.no_grad():
                    batch_pred_dicts = model(batch_dict)
                    final_pred_dicts = generate_prediction_dicts(batch_pred_dicts)[
                        0]

                for pred_dict in final_pred_dicts:  # TODO: Loop through predictions of each object
                    pred_trajs = pred_dict['pred_trajs']
                    pred_scores = pred_dict['pred_scores']
                    object_id = pred_dict['object_id']
                    object_type = pred_dict['object_type']

                """print("final_pred_dicts :", final_pred_dicts)
                print("final_pred_dicts shape :", len(final_pred_dicts))
                print("pred_ego :", final_pred_dicts[0])
                print("pred_ego shape :", len(final_pred_dicts[0]))"""
                
                
                pred_ego = final_pred_dicts[0]
                traj_index = np.argmax(pred_ego['pred_scores'])
                #destination = pred_ego['pred_trajs'][traj_index][10]
                # plt.plot(f_start_location.x, f_start_location.y, 'go')
                # plt.plot(f_destination.x, f_destination.y, 'ro')
                # plt.plot(route[:, 0], route[:, 1], 'b-')
                # plt.plot(pred_ego['pred_trajs'][traj_index][:11, 0], pred_ego['pred_trajs'][traj_index][:11, 1], 'r*-')
                # plt.show()
                
                dyn_vehic_list = []
                for i in range(1,len(final_pred_dicts)):
                    temp_vehic_index = np.argmax(final_pred_dicts[i]['pred_scores'])
                    temp_traj = final_pred_dicts[i]['pred_trajs'][temp_vehic_index][: N]
                    dyn_vehic_list.append(temp_traj)

                # Get waypoints
                current_location = np.array([world.player.get_location().x, world.player.get_location().y])
                current_velocity = np.sqrt(world.player.get_velocity().x**2 + world.player.get_velocity().y**2)
                
                if current_velocity <= 5:
                    multiplier = 1.2
                elif current_velocity <= 10:
                    multiplier = 3.6
                elif current_velocity <= 20:
                    multiplier = 6.2
                elif current_velocity <= 30:
                    multiplier = 8
                elif current_velocity <= 40:
                    multiplier = 10
                elif current_velocity <= 50:
                    multiplier = 11
                elif current_velocity <= 60:
                    multiplier = 12
                elif current_velocity <= 70:
                    multiplier = 14
                else:
                    multiplier = 14
                    
                
                transform = world.player.get_transform()
                x0 = transform.location.x
                y0 = transform.location.y
                yaw0 = np.deg2rad(transform.rotation.yaw)
                v0 = np.sqrt(world.player.get_velocity().x**2 +
                    world.player.get_velocity().y**2)
                
            
                fwd = transform.get_forward_vector()  
                heading = np.array([fwd.x, fwd.y])
                
                
                route_new, closest_index = choose_ahead_waypoint(waypoints=route, pos=current_location, heading=heading)
                
                
                if route_new is not False:
                    termi = int(N * multiplier)
                    closest_k_waypoint = list(route_new[closest_index + 1: closest_index + 1 + termi])
                else:
                    closest_k_waypoint = list(route[-1])
                    
                route = route_new
                
                del route_new
                

                
                time_step += 1
                
                """if time_step < 2:
                    world.player.apply_control(carla.VehicleControl())
                    continue"""
                	
                
                start_time = time.time()

     
                
                temp_list = []
                
                if isinstance(closest_k_waypoint[-1], np.float64):
                    temp_list.append(closest_k_waypoint)
                    waypoints = temp_list * N
                else:
                    waypoints = closest_k_waypoint if len(closest_k_waypoint) >= N else closest_k_waypoint + \
                            [closest_k_waypoint[-1]] * (N - len(closest_k_waypoint)) 
     
                mpc.reset_solver(x0, y0, yaw0, v0, mpc.get_static_obstacles(np.array(pred_ego['pred_trajs'][traj_index][: N])), \
                		         mpc.get_static_obstacles_soft(np.array(pred_ego['pred_trajs'][traj_index][: N])), waypoints)
               
                mpc.update_cost_function(goal, dyn_vehic_list)
                mpc.solve()
                
                
                if mpc.is_success:
                    print("is_success :", mpc.is_success)
                    wheel_angle, acceleration = mpc.get_controls_value()
                    throttle, brake, steer = mpc.process_control_inputs(wheel_angle, acceleration)
                    print("throttle", throttle)
                    print("brake", brake)
                    control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                    #mpc.opti.debug.show_infeasibilities()
                else:
                    print("is_success : False")
                    agent.set_destination(carla.Location(goal[0], goal[1], 0))
                    control = agent.run_step()
                    control.manual_gear_shift = False 
                
                #end_time = time.time()
                #mpc_calculation_time = end_time - start_time
                #print(f"Calculation time of MPC controller: {mpc_calculation_time:.6f} seconds")
           
                
                world.player.apply_control(control)
                #mpc.opti.debug.show_infeasibilities()
             
                
                transform = world.player.get_transform()
                x0 = transform.location.x
                y0 = transform.location.y
   
                dest = (goal[0], goal[1])
                
                final_distance = (x0 - dest[0])**2 + (y0 - dest[1])**2
                
                if final_distance < 5.0:
                
                    run_flag = False
                    print("Finished Successfully!")
		
		

                

    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='720x420',
        help='Window resolution (default: 720x420)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
