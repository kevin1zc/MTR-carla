#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import logging
import math
import os
import numpy.random as random
import sys
import weakref
import numpy as np
import torch
import pygame
import lanelet2
from lanelet2.routing import RoutingGraph

import carla

from carla_api.agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from carla_api.agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from carla_api.agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

from carla_api.utils.carla_utils import find_weather_presets, get_actor_display_name, get_actor_blueprints, HUD, \
    KeyboardControl
from carla_api.utils.world import World
from carla_api.utils.mtr_data_utils import create_scene_level_data, parse_carla_data, decode_map_features

from pathlib import Path
from mtr.datasets.waymo.waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, \
    polyline_type
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
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
    map = lanelet2.io.load("../../carla_maps/OSM/Town10HD.osm", lanelet2.io.Origin(0, 0))
    traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                  lanelet2.traffic_rules.Participants.Vehicle)
    routing_graph = RoutingGraph(map, traffic_rules)

    routing_graph.previous(map.laneletLayer[6135])  # actually next lane
    routing_graph.following(map.laneletLayer[6135])  # actually previous lane
    # also have left and right relations (in correct order)

    info = parse_carla_data()
    map_infos, dynamic_map_infos = decode_map_features(map, routing_graph)

    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     batch_size=args.batch_size,
    #     dist=dist_test, workers=args.workers, logger=logger, training=False
    # )

    cfg_path = "../../tools/cfgs/waymo/mtr+100_percent_data.yaml"
    cfg_from_yaml_file(cfg_path, cfg)
    cfg.TAG = Path(cfg_path).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_path.split('/')[1:-1])
    logger = common_utils.create_logger(None, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    model = model_utils.MotionTransformer(config=cfg.MODEL)

    model_path = "../model/latest_model.pth"
    it, epoch = model.load_params_from_file(model_path, logger=logger, to_cpu=False)
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

        args.sync = True

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        # world = World(client.load_world(map_name), hud, args)
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        clock = pygame.time.Clock()

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            track_infos = world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if track_infos is not None:
                info['track_infos'] = track_infos
                info['map_infos'] = map_infos
                info['dynamic_map_infos'] = dynamic_map_infos
                ret_infos = create_scene_level_data(info, cfg.DATA_CONFIG)
                batch_dict = {
                    'batch_size': 1,
                    'input_dict': ret_infos,

                }
                with torch.no_grad():
                    batch_pred_dicts = model(batch_dict)

                pred_traj = batch_pred_dicts['pred_trajs'].cpu().numpy()

            if agent.done():
                # if agent.done() or update:
                if args.loop or True:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

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
        default='1280x720',
        help='Window resolution (default: 1280x720)')
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
