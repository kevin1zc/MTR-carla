import numpy as np
import torch

from mtr.utils import common_utils
from mtr.datasets.waymo.waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, \
    polyline_type


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features(map, routing_graph):
    lanes = map.laneletLayer
    lines = map.lineStringLayer
    points = map.pointLayer
    regulatories = map.regulatoryElementLayer

    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    polylines = []

    data_cnt = 0
    polyline_cnt = 0
    for lane in lanes:
        if 'subtype' not in lane.attributes:  # TODO: skip for now. If next lane is set, then it's a bike lane; otherwise, it's road edge
            continue
        if lane.attributes['subtype'] == 'walkway':  # skip walkway
            continue

        cur_info = {'id': int(data_cnt)}

        if lane.attributes['subtype'] == 'crosswalk':
            global_type = 18
            map_infos['crosswalk'].append(cur_info)

        else:  # TODO: add stop sign, speed bump
            cur_info['speed_limit_mph'] = 30.0  # TODO: need to check the parsed value in the map
            cur_info['type'] = lane_type[2]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane
            cur_info['interpolating'] = False  # TODO: after parsing all lanes, check "Next Lanelet" of each lane.
            # TODO: If more than one, set interpolating to True for both

            # routing_graph.previous(lane): actually next lane
            # routing_graph.following(lane): actually previous lane
            cur_info['entry_lanes'] = [entry_lane.id for entry_lane in routing_graph.following(lane)]
            cur_info['exit_lanes'] = [exit_lane.id for exit_lane in routing_graph.previous(lane)]

            left_boundary = lines[lane.leftBound.id]
            right_boundary = lines[lane.rightBound.id]

            cur_info['left_boundary'] = [{
                'start_index': left_boundary[0].id, 'end_index': left_boundary[len(left_boundary) - 1].id,
                'feature_id': 0,  # TODO: check ids in road_line or road_edge, one of the two
                'boundary_type': 1  # roadline type
            }]

            cur_info['right_boundary'] = [{
                'start_index': right_boundary[0].id, 'end_index': right_boundary[len(right_boundary) - 1].id,
                'feature_id': 0,  # TODO: check ids in road_line or road_edge, one of the two
                'boundary_type': 1  # roadline type
            }]

            global_type = 2  # TODO: check
            map_infos['lane'].append(cur_info)

        left_points = np.array(
            [(float(pt.attributes['local_x']), float(pt.attributes['local_y']), pt.z, global_type) for pt in
             lane.leftBound])
        right_points = np.array(
            [(float(pt.attributes['local_x']), float(pt.attributes['local_y']), pt.z, global_type) for pt in
             lane.rightBound])

        cur_polyline = (left_points + right_points) / 2

        cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
        cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (polyline_cnt, polyline_cnt + len(cur_polyline))
        polyline_cnt += len(cur_polyline)
        data_cnt += 1

    for line in lines:  # TODO: need to check line or edge
        if 'type' not in line.attributes or 'subtype' not in line.attributes:
            continue
        if line.attributes['type'] == 'traffic_light':  # TODO: TBD
            continue
        if line.attributes['type'] == 'traffic_sign':  # TODO: Check what the sign is
            continue

        cur_info = {'id': str(data_cnt)}
        osm_line_type = line.attributes['subtype']
        if osm_line_type == 'dashed':
            cur_info['type'] = road_line_type[1]
            global_type = 6
        elif osm_line_type == 'solid_solid':
            cur_info['type'] = road_line_type[2]
            global_type = 7
        elif osm_line_type == 'solid_solid_solid_solid':
            cur_info['type'] = road_line_type[6]
            global_type = 11
        else:
            cur_info['type'] = road_line_type[0]
            global_type = -1

        left_points = np.array(
            [(float(pt.attributes['local_x']), float(pt.attributes['local_y']), pt.z, global_type) for pt in
             lane.leftBound])
        right_points = np.array(
            [(float(pt.attributes['local_x']), float(pt.attributes['local_y']), pt.z, global_type) for pt in
             lane.rightBound])

        cur_polyline = (left_points + right_points) / 2

        cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
        cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

        map_infos['road_line'].append(cur_info)

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (polyline_cnt, polyline_cnt + len(cur_polyline))
        polyline_cnt += len(cur_polyline)
        data_cnt += 1

        # TODO: road edge is skipped for now
        # cur_info['type'] = road_edge_type[cur_data.road_edge.type]
        #
        # global_type = polyline_type[cur_info['type']]
        # cur_polyline = np.stack(
        #     [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
        # cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
        # cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
        #
        # map_infos['road_edge'].append(cur_info)

    for reg in regulatories:
        lane_id, state, stop_point = [], [], []
        if reg.attributes['subtype'] == 'traffic_light':  # TODO: else: speed_limit, skipped
            for traffic_light in reg.trafficLights:
                a = 0

        # dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        # dynamic_map_infos['state'].append(np.array([state]))
        # dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    # for cur_data in dynamic_map_states:  # (num_timestamp)
    #     lane_id, state, stop_point = [], [], []
    #     for cur_signal in cur_data.lane_states:  # (num_observed_signals)
    #         lane_id.append(cur_signal.lane)
    #         state.append(signal_state[cur_signal.state])
    #         stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])
    #
    #     dynamic_map_infos['lane_id'].append(np.array([lane_id]))
    #     dynamic_map_infos['state'].append(np.array([state]))
    #     dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines
    return map_infos, dynamic_map_infos


def parse_carla_data():
    info = {}
    info['scenario_id'] = "scenario_0"
    info['timestamps_seconds'] = np.linspace(0.0, 9.0, 91)  # list of int of shape (91)
    info['current_time_index'] = 10  # int, 10
    info['sdc_track_index'] = 0  # set ego vehicle index to be 0
    info['objects_of_interest'] = []  # list, could be empty list

    info['tracks_to_predict'] = {
        'track_index': [0],
        'difficulty': [0]
    }

    info['tracks_to_predict']['object_type'] = ['TYPE_VEHICLE']



    # TODO: track_infos = decode_tracks_info(scenario.tracks)

    return info


def create_scene_level_data(info, dataset_cfg):
    """
    Args:
        index (index):

    Returns:

    """
    scene_id = info['scenario_id']

    sdc_track_index = info['sdc_track_index']
    current_time_index = info['current_time_index']
    timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

    track_infos = info['track_infos']

    track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
    obj_types = np.array(track_infos['object_type'])
    obj_ids = np.array(track_infos['object_id'])
    obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
    obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
    obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

    center_objects, track_index_to_predict = get_interested_agents(
        track_index_to_predict=track_index_to_predict,
        obj_trajs_full=obj_trajs_full,
        current_time_index=current_time_index,
        obj_types=obj_types, scene_id=scene_id
    )

    (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask,
     center_gt_trajs,
     center_gt_trajs_mask, center_gt_final_valid_idx,
     track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = create_agent_data_for_center_objects(
        center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
        track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
        timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
    )

    ret_dict = {
        'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
        'obj_trajs': torch.from_numpy(obj_trajs_data),
        'obj_trajs_mask': torch.from_numpy(obj_trajs_mask),
        'track_index_to_predict': torch.from_numpy(track_index_to_predict_new),  # used to select center-features
        'obj_trajs_pos': torch.from_numpy(obj_trajs_pos),
        'obj_trajs_last_pos': torch.from_numpy(obj_trajs_last_pos),
        'obj_types': obj_types,
        'obj_ids': obj_ids,

        'center_objects_world': torch.from_numpy(center_objects),
        'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
        'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

        'obj_trajs_future_state': torch.from_numpy(obj_trajs_future_state),
        'obj_trajs_future_mask': torch.from_numpy(obj_trajs_future_mask),
        'center_gt_trajs': torch.from_numpy(center_gt_trajs),
        'center_gt_trajs_mask': torch.from_numpy(center_gt_trajs_mask),
        'center_gt_final_valid_idx': torch.from_numpy(center_gt_final_valid_idx),
        'center_gt_trajs_src': torch.from_numpy(obj_trajs_full[track_index_to_predict]),
    }

    if not dataset_cfg.get('WITHOUT_HDMAP', False):
        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
            center_objects=center_objects, map_infos=info['map_infos'],
            center_offset=dataset_cfg.get('CENTER_OFFSET_OF_MAP', (30.0, 0)),
            dataset_cfg=dataset_cfg,
        )  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

        ret_dict['map_polylines'] = torch.from_numpy(map_polylines_data)
        ret_dict['map_polylines_mask'] = torch.from_numpy((map_polylines_mask > 0))
        ret_dict['map_polylines_center'] = torch.from_numpy(map_polylines_center)

    return ret_dict


def get_interested_agents(track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
    center_objects_list = []
    track_index_to_predict_selected = []

    for k in range(len(track_index_to_predict)):
        obj_idx = track_index_to_predict[k]

        assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

        center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
        track_index_to_predict_selected.append(obj_idx)

    center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
    track_index_to_predict = np.array(track_index_to_predict_selected)
    return center_objects, track_index_to_predict


def create_agent_data_for_center_objects(
        center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
        obj_types, obj_ids
):
    obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = generate_centered_trajs_for_agents(
        center_objects=center_objects, obj_trajs_past=obj_trajs_past,
        obj_types=obj_types, center_indices=track_index_to_predict,
        sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future
    )

    # generate the labels of track_objects for training
    center_obj_idxs = np.arange(len(track_index_to_predict))
    center_gt_trajs = obj_trajs_future_state[
        center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 4)
    center_gt_trajs_mask = obj_trajs_future_mask[
        center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
    center_gt_trajs[center_gt_trajs_mask == 0] = 0

    # filter invalid past trajs
    assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
    valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

    obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
    obj_trajs_data = obj_trajs_data[:,
                     valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
    obj_trajs_future_state = obj_trajs_future_state[:,
                             valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
    obj_trajs_future_mask = obj_trajs_future_mask[:,
                            valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
    obj_types = obj_types[valid_past_mask]
    obj_ids = obj_ids[valid_past_mask]

    valid_index_cnt = valid_past_mask.cumsum(axis=0)
    track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
    sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

    assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
    assert len(obj_types) == obj_trajs_future_mask.shape[1]
    assert len(obj_ids) == obj_trajs_future_mask.shape[1]

    # generate the final valid position of each object
    obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
    obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
    for k in range(num_timestamps):
        cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
        obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

    center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
    for k in range(center_gt_trajs_mask.shape[1]):
        cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
        center_gt_final_valid_idx[cur_valid_mask] = k

    return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
            center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)


def generate_centered_trajs_for_agents(center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps,
                                       obj_trajs_future):
    """[summary]

    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_trajs_past (num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_types (num_objects):
        center_indices (num_center_objects): the index of center objects in obj_trajs_past
        centered_valid_time_indices (num_center_objects), the last valid time index of center objects
        timestamps ([type]): [description]
        obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    Returns:
        ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
        ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
        ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
    """
    assert obj_trajs_past.shape[-1] == 10
    assert center_objects.shape[-1] == 10
    num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, box_dim = obj_trajs_past.shape
    # transform to cpu torch tensor
    center_objects = torch.from_numpy(center_objects).float()
    obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
    timestamps = torch.from_numpy(timestamps)

    # transform coordinates to the centered objects
    obj_trajs = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )

    ## generate the attributes for each object
    object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
    object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
    object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
    object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
    object_onehot_mask[:, sdc_index, :, 4] = 1

    object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

    object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = torch.cat((
        obj_trajs[:, :, :, 0:6],
        object_onehot_mask,
        object_time_embedding,
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9],
        acce,
    ), dim=-1)

    ret_obj_valid_mask = obj_trajs[:, :, :,
                         -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    ##  generate label for future trajectories
    obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
    obj_trajs_future = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_future,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )
    ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
    ret_obj_valid_mask_future = obj_trajs_future[:, :, :,
                                -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

    return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()


def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
    obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
        angle=-center_heading
    ).view(num_center_objects, num_objects, num_timestamps, 2)

    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    # rotate direction of velocity
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

    return obj_trajs


def create_map_data_for_center_objects(center_objects, map_infos, center_offset, dataset_cfg):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """
    num_center_objects = center_objects.shape[0]

    # transform object coordinates by center objects
    def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
        xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    polylines = torch.from_numpy(map_infos['all_polylines'].copy())
    center_objects = torch.from_numpy(center_objects)

    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines.numpy(), point_sampled_interval=dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
        vector_break_dist_thresh=dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
        num_points_each_polyline=dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
    )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = dataset_cfg.NUM_OF_SRC_POLYLINES

    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(
            batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
        center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(
            num_center_objects, 1)
        center_offset_rot = common_utils.rotate_points_along_z(
            points=center_offset_rot.view(num_center_objects, 1, 2),
            angle=center_objects[:, 6]
        ).view(num_center_objects, 2)

        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

        dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(
            dim=-1)  # (num_center_objects, num_polylines)
        topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
        map_polylines = batch_polylines[
            topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[
            topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
        map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )

    temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(
        dim=-2)  # (num_center_objects, num_polylines, 3)
    map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None],
                                                      min=1.0)  # (num_center_objects, num_polylines, 3)

    map_polylines = map_polylines.numpy()
    map_polylines_mask = map_polylines_mask.numpy()
    map_polylines_center = map_polylines_center.numpy()

    return map_polylines, map_polylines_mask, map_polylines_center


def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0,
                                      num_points_each_polyline=20):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                   axis=-1)  # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = \
        (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

    # # CHECK the results
    # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
    # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
    # assert center_dist.max() < 10
    return ret_polylines, ret_polylines_mask
