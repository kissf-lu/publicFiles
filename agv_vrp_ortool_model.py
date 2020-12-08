# -*- coding: utf-8 -*-
"""
"""
import time
from functools import wraps

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from .conf import PathData, TaskDispatch
from utils.logger import sim_logger

default_fir_solution_strategy = \
    routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC

log = sim_logger(user=__name__)


def time_func(func):
    """
    Decorator that reports the execution time.
    """""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(f"{func.__name__} run: {(end - start) * 1000:.0f}ms")
        return result

    return wrapper


def int_comb(arr, length: int = 2, shuffle: bool = False):
    """
    """
    if shuffle:
        np.random.shuffle(arr)

    _len = len(arr)
    for i in range(0, _len, length):
        if _len < 2 or i == _len - 1:
            break
        else:
            yield arr[i], arr[i + 1]


def distance(from_node, to_node, order: int = 1):
    _loc_check_instance = (
        isinstance(from_node, (tuple, list))
        and isinstance(to_node, (tuple, list))
    )
    _dis_check_instance = (
        isinstance(from_node, (int, float))
        and isinstance(to_node, (int, float))
    )
    if _loc_check_instance:
        delta_x = np.array(to_node) - np.array(from_node)
        dis = np.linalg.norm(delta_x, ord=order)
    elif _dis_check_instance:
        dis = np.abs(to_node - from_node)
    else:
        msg = (
            f"{__name__} distance func get nodes val error:"
            f"<from: {from_node}, to: {to_node}>"
        )
        raise ValueError(msg)
    return dis


def locations_distance_callback(data, manager, distances_: dict):
    """Creates callback to return distance between points."""
    # distances_ = {}
    index_manager_ = manager
    pd_map = data['pickups_deliveries']
    penalty = 5000
    # precompute distance between location to have distance callback in O(1)
    end = len(data['locations']) - 1
    for from_counter, from_node in enumerate(data['locations']):
        distances_[from_counter] = {}
        for to_counter, to_node in enumerate(data['locations']):
            if from_counter == to_counter:
                distances_[from_counter][to_counter] = 0
            elif from_counter == end or to_counter == end:
                distances_[from_counter][to_counter] = 0
            elif from_counter in pd_map and to_counter in pd_map:
                distances_[from_counter][to_counter] = penalty
            else:
                distances_[from_counter][to_counter] = distance(
                    from_node=from_node,
                    to_node=to_node,
                    order=1
                )

    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        _from_node = index_manager_.IndexToNode(from_index)
        _to_node = index_manager_.IndexToNode(to_index)
        return distances_[_from_node][_to_node]

    return distance_callback


def search_parameters_cfg(fir_solution_strategy=default_fir_solution_strategy):
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = fir_solution_strategy

    search_parameters.time_limit.seconds = 2

    search_parameters.log_search = True

    return search_parameters


def add_pickup_deliver(
        manager, routing, distance_dimension, p2d: dict):
    for pickup, delivery in p2d.items():
        pickup_index = manager.NodeToIndex(pickup)
        delivery_index = manager.NodeToIndex(delivery)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))


def _start2end_solution(data, manager, routing, solution):
    """Prints solution on console."""

    routes_sol_data = []

    for vehicle_id in range(data['num_vehicles']):
        # 从路径中获取车的起始index
        index = routing.Start(vehicle_id)
        route_distance = 0
        routes = []
        while not routing.IsEnd(index):
            _node_ = manager.IndexToNode(index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            _step_ = routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            route_distance += _step_

            routes.append(_node_)
        routes.append(manager.IndexToNode(index))

        routes_sol_data.append(PathData(routes, route_distance))

        log.info(f'Route for vehicle {vehicle_id}')
        _agv_path = [f"agv{r}" if i == 0 else f"task{r}" for i, r in
                     enumerate(routes)]
        log.info(' -> '.join(_agv_path))
        log.info(f'Distance of the route: {route_distance}m')

    max_route_distance = max(routes_sol_data, key=lambda x: x.dis)
    min_route_distance = min(routes_sol_data, key=lambda x: x.dis)
    sum_route_distance = sum(map(lambda x: x.dis, routes_sol_data))
    msg = (
        f"the route distances: "
        f"<sum:{sum_route_distance}, "
        f"max:{max_route_distance.dis}, "
        f"min: {min_route_distance.dis}>"
    )
    log.info(msg)
    return routes_sol_data


@time_func
def mdvrp_pd_tsp(data):
    """
    """
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['locations']),
        data['num_vehicles'],
        data['starts'],
        data['ends']
    )
    # ------------------ create routing model -------------------
    routing = pywrapcp.RoutingModel(manager)
    #
    distance_dict = {}
    transit_callback_index = routing.RegisterTransitCallback(
        locations_distance_callback(data, manager, distance_dict))
    # ------------------ Define cost of each arc ----------------
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # ------------------ Add Distance constraint ----------------
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        2000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(2000)
    # ------------------- 添加拣选派送模式 --------------------------
    add_pickup_deliver(
        manager=manager,
        routing=routing,
        distance_dimension=distance_dimension,
        p2d=data['pickups_deliveries']
    )
    # ------------------- 求解 -------------------------------------
    parameters = search_parameters_cfg(
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(parameters)
    res = None
    if solution:
        res = _start2end_solution(data, manager, routing, solution)
    else:
        code_status = {
            0: 'Problem not solved yet',
            1: 'Problem solved successfully',
            2: 'No solution found to the problem',
            3: 'Time limit reached before finding a solution',
            4: 'Model, model parameters, or flags are not valid',
        }
        log.info(f"Solver status: {code_status.get(routing.status())}")

    return res


def init_data(agv_loc, task_loc):
    # agv位置
    agv_locations = agv_loc
    # 任务位置：默认按照[(pickup1,delivery1), ...]
    task_locations = task_loc
    # 求解成本的最后收敛点位，根据坐标系情况定义
    end_locations = TaskDispatch.TASK_END_LOC
    # ortools 求解器的agv位置，任务位置
    locations = agv_locations + task_locations + end_locations
    #
    agv_num = len(agv_locations)
    task_num = len(task_locations)

    agv_idx = [i for i in range(agv_num)]
    task_idx = [agv_num + i for i in range(task_num)]
    # 模拟pickup delivery的匹配
    pickup_deliveries = task_idx[:]

    pickups_deliveries = {
        i[0]: i[1] for i in int_comb(pickup_deliveries, shuffle=False)}

    end_idx = [len(locations) - 1] * agv_num

    if len(agv_idx) != len(end_idx):
        msg = (f"start agv num must equal to end nodes: "
               f"<{len(end_idx)}, {len(end_idx)}>")
        raise ValueError(msg)

    log.info(
        f"agv_task_info: <agv_num: {agv_num} , "
        f"task_num: {task_num}, "
        f"end_locations: {end_idx}>")
    log.info(
        f"location_info: <agv_loc_idx: {agv_idx} , "
        f"task_loc_idx: {task_idx}>")
    log.info(f"p2d constrain: {pickups_deliveries}")
    _config_data = {
        'locations': locations, 'num_vehicles': agv_num, 'starts': agv_idx,
        'ends': end_idx, 'pickups_deliveries': pickups_deliveries}

    return _config_data
