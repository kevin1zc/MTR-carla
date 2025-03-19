import sys
import carla
import math
import numpy as np
import numpy.random as random
from .carla_utils import find_weather_presets, get_actor_display_name, get_actor_blueprints
from .sensors import CollisionSensor, LaneInvasionSensor, GnssSensor, CameraManager
from collections import defaultdict, deque
from functools import partial


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, delta_t, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self._delta_t = delta_t
        self._trajectories = defaultdict(partial(deque, maxlen=11))
        self._simulation_steps = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        ego_transform = self.player.get_transform()
        vehicles = self.world.get_actors().filter('vehicle.*')

        def dist(l):
            return math.sqrt((l.x - ego_transform.location.x) ** 2 + (l.y - ego_transform.location.y)
                             ** 2 + (l.z - ego_transform.location.z) ** 2)

        vehicle_dist = []

        for vehicle in vehicles:
            transform = vehicle.get_transform()
            loc = transform.location
            yaw = transform.rotation.yaw
            vel = vehicle.get_velocity()
            if vehicle.id != self.player.id:
                vehicle_dist.append((dist(loc), vehicle.id, vehicle))
            dim = vehicle.bounding_box.extent * 2
            traj = np.array([loc.x, loc.y, loc.z, dim.x, dim.y, dim.z, math.radians(yaw), vel.x, vel.y, 1])
            if self._simulation_steps % int(0.1 / self._delta_t) == 0:
                self._trajectories[vehicle.id].append(traj)

        vehicle_dist.sort()

        nearby_vehicles = []

        for dist, _, vehicle in vehicle_dist:
            if dist > 100.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            nearby_vehicles.append((dist, vehicle_type))

        track_ids = [self.player.id]
        for i in range(min(7, len(vehicle_dist))):  # find the 7 closest vehicles
            track_ids.append(vehicle_dist[i][1])

        self._simulation_steps += 1
        info = None
        if self._simulation_steps == int(0.5 / self._delta_t):
            info = self.parse_carla_data(track_ids)
            info['vehicle_ids'] = track_ids
            self._simulation_steps = 0

        self.hud.tick(self, clock, nearby_vehicles)
        return info

    def parse_carla_data(self, track_ids):
        info = {}
        info['scenario_id'] = "scenario_0"
        info['timestamps_seconds'] = np.linspace(0.0, 9.0, 91)  # list of int of shape (91)
        info['current_time_index'] = 10  # int, 10
        info['sdc_track_index'] = 0  # set ego vehicle index to be 0
        info['objects_of_interest'] = []  # list, could be empty list

        info['tracks_to_predict'] = {
            'track_index': list(range(len(track_ids))),
            'difficulty': [0] * len(track_ids)
        }

        info['tracks_to_predict']['object_type'] = ['TYPE_VEHICLE'] * len(track_ids)

        info['track_infos'] = self.decode_tracks(track_ids)
        return info

    def decode_tracks(self, track_ids):
        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }
        for i, object_id in enumerate(track_ids):
            trajs = self._trajectories[object_id]
            full_traj = np.zeros((11, 10))
            cur_traj = np.stack(trajs, axis=0)
            full_traj[-len(cur_traj):] = cur_traj
            track_infos['object_id'].append(object_id)
            track_infos['object_type'].append('TYPE_VEHICLE')
            track_infos['trajs'].append(full_traj)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)

        return track_infos

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def get_player_id(self):
        return self.player.id

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
