import numpy as np
import cplex
import time
import pickle
import numpy as np
import numba
from numba.typed import List
from enum import Enum
import json
import copy

import os
from os import path as opath
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


from math import pi, sin, cos, sqrt, atan2, radians
import re
import pathlib
import random
import pickle


all_termination_event = threading.Event()
sp1_termination_event = threading.Event()


is_frac = lambda x: abs(x-round(x)) > 0.0001
is_zero = lambda x: x <= 0.0001
is_one = lambda x: abs(x-1) <= 0.0001

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    


class Prob:
    def __init__(self, datafile, T_Gamma, t_range, D_Delta, d_range, num_drones, drone_time_multiplier, drone_cost_multiplier, flight_limit, demand_ratio):

        if datafile.split('.')[-1] == 'txt':
            prob_name, n, num_vehicles, capacity, pos_x, pos_y, demand, ready_time, due_time, service_time, truck_distance_mat, truck_time_mat, drone_distance_mat, drone_time_mat, round_mult = self.read_solomon(datafile)
        if datafile.split('.')[-1] == 'vrp':
            prob_name, n, num_vehicles, capacity, pos_x, pos_y, demand, ready_time, due_time, service_time, truck_distance_mat, truck_time_mat, drone_distance_mat, drone_time_mat, round_mult = self.read_vrp(datafile)

        num_nodes = n + 2

        node_id = n + 1

        a_i = ready_time.astype(np.int64)
        b_i = due_time.astype(np.int64)
        s_i = service_time.astype(np.int64)
        demand =  demand.astype(np.int64)

        N = [i for i in range(1, n+1)]
        N_s = [0] + N
        N_t = N + [n+1]
        N_st = [0] + N + [n+1]

        l_tT = truck_time_mat.astype(np.int64)
        l_CT = truck_distance_mat.astype(np.int64)
        l_T_delta = np.around(l_tT * t_range, 0).astype(np.int64)
        l_tD = np.around(drone_time_mat * drone_time_multiplier, 0).astype(np.int64)
        l_cD = np.around(drone_distance_mat * drone_cost_multiplier * 2, 0).astype(np.int64)
        l_D_delta = np.around(l_T_delta * d_range, 0).astype(np.int64)

        demand_quantile = np.quantile(demand[:num_nodes], demand_ratio)

        truck_feasible = lambda i,j: a_i[i]+s_i[i]+l_tT[i,j] <= b_i[j]
        drone_feasible = lambda i,j: drone_distance_mat[i,j] <= flight_limit * round_mult and demand[j] <= demand_quantile

        A_T = [(i,j) for i in N_s for j in N_t if i != j and (i,j) != (0,n+1) and truck_feasible(i,j)]
        A_D = [(i,j) for i in N for j in N if i != j and drone_feasible(i,j)]


        drone_avail_nodes = [[a[1] for a in A_D if a[0] == i] for i in N_st]
        
        Out_i = [[a[1] for a in A_T if a[0] == i] for i in N_st]

        D = list(range(num_drones))




        L = [
            np.array(range(len(drone_arcs))) for drone_arcs in drone_avail_nodes
        ]

        self.datafile = datafile
        self.prob_name = prob_name

        self.T_Gamma = T_Gamma
        self.t_range = t_range
        self.D_Delta = D_Delta
        self.d_range = d_range
        self.n = n
        self.num_nodes = num_nodes
        self.a_i = a_i
        self.b_i = b_i
        self.s_i = s_i
        self.demand = demand
        self.demand_ratio = demand_ratio
        self.t_service_time = service_time
        
        self.flight_limit = flight_limit


        self.drone_time_multiplier = drone_time_multiplier
        self.drone_cost_multiplier = drone_cost_multiplier
        
        self.N = N
        self.N_s = N_s
        self.N_t = N_t
        self.N_st = N_st

        self.A_T = A_T
        self.A_D = A_D

        self.Out_i = to_list_of_array_int(Out_i)
        self.drone_avail_nodes = to_list_of_array_int(drone_avail_nodes)
        
        self.D = D
        self.L = L

        self.pos_x = pos_x
        self.pos_y = pos_y

        self.Drone_dist_ij = drone_distance_mat
        
        self.D = D

        self.l_CT = l_CT
        self.l_tD = l_tD
        self.l_cD = l_cD
        self.l_tT = l_tT
        self.l_T_delta = l_T_delta
        self.l_D_delta = l_D_delta
        
        self.capacity = capacity
        self.num_vehicles = num_vehicles

        self.round_mult = round_mult

    def read_solomon(self, datafile):
        with open(datafile, 'r') as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]
        prob_name = lines[0]

        n = int([l.split()[0] for l in lines if len(l.split()) > 0][-1])
        num_nodes = n + 2

        num_vehicles, capacity = int(lines[4].split()[0]), int(lines[4].split()[1])
        
        pos_x = np.zeros(num_nodes)
        pos_y = np.zeros(num_nodes)
        demand = np.zeros(num_nodes)
        ready_time = np.zeros(num_nodes)
        due_time = np.zeros(num_nodes)
        service_time = np.zeros(num_nodes)


        round_mult = 100


        for l in lines[9:n+10]:
            toks = l.split()

            node_id = int(toks[0])
            pos_x[node_id] = int(toks[1])
            pos_y[node_id] = int(toks[2])
            demand[node_id] = int(toks[3])
            ready_time[node_id] = int(toks[4]) * round_mult
            due_time[node_id] = int(toks[5]) * round_mult
            service_time[node_id] = int(toks[6]) * round_mult

        t = n + 1
        pos_x[t] = pos_x[0]
        pos_y[t] = pos_y[0]
        demand[t] = demand[0]
        ready_time[t] = ready_time[0]
        due_time[t] = due_time[0]
        service_time[t] = service_time[0]


        

        truck_distance_mat = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in  range(num_nodes):
                truck_distance_mat[i,j] = round(round_mult * np.hypot(pos_x[i] - pos_x[j], pos_y[i] - pos_y[j]))

        truck_time_mat = truck_distance_mat.copy()

        drone_distance_mat = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in  range(num_nodes):
                drone_distance_mat[i,j] = round(round_mult * np.hypot(pos_x[i] - pos_x[j], pos_y[i] - pos_y[j]))

        drone_time_mat = drone_distance_mat.copy()
        
        

        return prob_name, n, num_vehicles, capacity, pos_x, pos_y, demand, ready_time, due_time, service_time, truck_distance_mat, truck_time_mat, drone_distance_mat, drone_time_mat, round_mult
    
    def read_vrp(self, datafile):

        f = open(datafile, 'r')

        truck_dist = []
        coord = {}


        all_lines = f.readlines()


        line_idx = 0
        while True:
            line = all_lines[line_idx].strip()
            if line.startswith('EOF'):
                break

            if line.startswith('NAME'):
                prob_name = line.split(':')[1].strip()

            if line.startswith('DIMENSION'):
                n = int(line.split(':')[1]) - 1
                
            if line.startswith('CAPACITY'):
                capacity = int(line.split(':')[1])

            if line.startswith('NODE_COORD_SECTION'):
                for i in range(n+1):
                    line_idx += 1
                    line = all_lines[line_idx].strip()
                    node_coord = re.split(" ", line)
                    coord[int(node_coord[0])] = (float(node_coord[1]), float(node_coord[2]))

            if line.startswith('TIME_WINDOW_SECTION'):
                ready_time = np.zeros(n+2, dtype=np.int64)
                due_time = np.zeros(n+2, dtype=np.int64)
                for i in range(n+1):
                    line_idx += 1
                    line = all_lines[line_idx].strip()
                    tw = line.split()
                    ready_time[i] = int(tw[1])
                    due_time[i] = int(tw[2])

                ready_time[n+1] = ready_time[0]
                due_time[n+1] = due_time[0]
                
            if line.startswith('DEMAND_SECTION'):
                demand = np.zeros(n+2, dtype=np.int64)
                for i in range(n+1):
                    line_idx += 1
                    line = all_lines[line_idx].strip()
                    toks = line.split()
                    demand[i] = int(toks[1])
                    
            if line.startswith('SERVICE_TIME_SECTION'):
                service_time = np.zeros(n+2, dtype=np.int64)
                for i in range(n+1):
                    line_idx += 1
                    line = all_lines[line_idx].strip()
                    toks = line.split()
                    service_time[i] = int(toks[1])
                
            if line.startswith('TRUCK_EDGE_WEIGHT_SECTION'):
                truck_dist = []
                for i in range(n+1):
                    line_idx += 1
                    line = all_lines[line_idx].strip()
                    dist = re.split("\W+", line.strip())
                    dist = [int(i) for i in dist]
                    dist.append(dist[0])
                    truck_dist.append(dist)

                truck_dist_mat = np.array(truck_dist, dtype=np.int64)
                truck_dist_mat = np.vstack([truck_dist_mat, truck_dist_mat[0]])

            if line.startswith('DRONE_EDGE_WEIGHT_SECTION'):
                drone_dist = []
                for i in range(n+1):
                    line_idx += 1
                    line = all_lines[line_idx].strip()
                    dist = re.split("\W+", line.strip())
                    dist = [int(i) for i in dist]
                    dist.append(dist[0])
                    drone_dist.append(dist)

                drone_dist_mat = np.array(drone_dist, dtype=np.int64)
                drone_dist_mat = np.vstack([drone_dist_mat, drone_dist_mat[0]])


            line_idx += 1

        f.close()

        num_vehicles = n

        pos_x = np.array([x for x,y in coord.values()])
        pos_y = np.array([y for x,y in coord.values()])

        truck_distance_mat = truck_dist_mat
        truck_time_mat = truck_dist_mat
        drone_distance_mat = drone_dist_mat
        drone_time_mat = drone_dist_mat

        round_mult = 1


        return prob_name, n, num_vehicles, capacity, pos_x, pos_y, demand, ready_time, due_time, service_time, truck_distance_mat, truck_time_mat, drone_distance_mat, drone_time_mat, round_mult

    def __repr__(self):
        return f'Prob("{self.datafile}")'

    def prob_info(self):
        return {
                'name': self.prob_name,
                'datafile': self.datafile,
                'num_cust_nodes': self.n,
                'num_tot_nodes': self.num_nodes,
                'Gamma' : self.T_Gamma,
                't_range' : self.t_range,
                'Delta' : self.D_Delta,
                'd_range' : self.d_range,
                'num_drones': len(self.D),
                'drone_time_multiplier': self.drone_time_multiplier,
                'drone_cost_multiplier': self.drone_cost_multiplier,
                'flight_limit': self.flight_limit,
                'demand_ratio': self.demand_ratio
            }

    def get_file_name(self):
        return f'{self.prob_name}_N[{self.n}]_𝚪[{self.T_Gamma}]_𝛄[{self.t_range}]_𝚫[{self.D_Delta}]_𝛅[{self.d_range}]_D[{len(self.D)}]_DT[{self.drone_time_multiplier}]_DC[{self.drone_cost_multiplier}]_FL[{self.flight_limit}]_DR[{self.demand_ratio}]'
    
    

class MIPSolver:
    def __init__(self, prob):
        self.prob = prob
        self.formulate()

    def formulate(self):
        prob = self.prob

        # Cplex Python API does not accept Numpy int...
        l_CT = prob.l_CT.astype(np.float64)
        l_cD = prob.l_cD.astype(np.float64)
        a_i = prob.a_i.astype(np.float64)
        b_i = prob.b_i.astype(np.float64)
        s_i = prob.s_i.astype(np.float64)
        demand = prob.demand.astype(np.float64)
        l_tT = prob.l_tT.astype(np.float64)
        l_tD = prob.l_tD.astype(np.float64)
        l_T_delta = prob.l_T_delta.astype(np.float64)
        l_D_delta = prob.l_D_delta.astype(np.float64)



        pcpx = cplex.Cplex()

        pcpx.variables.add(
            names = [f'x_{i}_{j}' for i,j in prob.A_T],
            types = ['B'] * len(prob.A_T),
            obj = [l_CT[i,j] for i,j in prob.A_T]
        )

        pcpx.variables.add(
            names = [f'y_{d}_{i}_{j}' for d in prob.D for i,j in prob.A_D],
            types = ['B'] * len(prob.D) * len(prob.A_D),
            obj = [l_cD[i,j] for d in prob.D for i,j in prob.A_D]
        )

        pcpx.variables.add(
            names = [f'f_{i}' for i in prob.N_t]
        )

        pcpx.variables.add(
            names = [f'e_{i}' for i in prob.N_st],
            lb = [demand[i] for i in prob.N_st],
            ub = [demand[0]] + [prob.capacity] * len(prob.N_t)
        )

        pcpx.variables.add(
            names = [f'u_{i}_{gamma}' for i in prob.N_st for gamma in range(prob.T_Gamma + 1)],
            lb = [a_i[i] if i == 0 else 0 for i in prob.N_st for gamma in range(prob.T_Gamma + 1)],
            ub = [a_i[i] if i == 0 else b_i[i] for i in prob.N_st for gamma in range(prob.T_Gamma + 1)]
        )

        pcpx.variables.add(
            names = [f'v_{i}_{gamma}' for i in prob.N_st for gamma in range(prob.T_Gamma + 1)],
            lb = [a_i[i] + prob.s_i[i] for i in prob.N_st for gamma in range(prob.T_Gamma + 1)],
            ub = [b_i[0]] * len(prob.N_st) * (prob.T_Gamma + 1)
        )



        pcpx.variables.add(
            names = [f'p_{d}_{l}_{i}_{j}' for d in prob.D for i,j in prob.A_D for l in prob.L[i]],
            types = ['B' for d in prob.D for i,j in prob.A_D for l in prob.L[i] ]
        )

        pcpx.variables.add(
            names = [f'q_{delta}_{gamma}_{d}_{l}_{i}' for delta in range(prob.D_Delta+1) for gamma in range(prob.T_Gamma+1) for d in prob.D for i in prob.N for l in prob.L[i]],
        )

        pcpx.variables.add(
            names = [f'r_{delta}_{gamma}_{d}_{l}_{i}' for delta in range(prob.D_Delta+1) for gamma in range(prob.T_Gamma+1) for d in prob.D for i in prob.N for l in prob.L[i]],
        )


        pcpx.variables.add(
            names = [f'w_{i}_{gamma}' for i in prob.N for gamma in range(prob.T_Gamma + 1)]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'x_{i}_{j}' for i in prob.N_s if (i,j) in prob.A_T] + [f'y_{d}_{i}_{j}' for d in prob.D for i in prob.N if (i,j) in prob.A_D],
                    val = [1 for i in prob.N_s if (i,j) in prob.A_T] + [1 for d in prob.D for i in prob.N if (i,j) in prob.A_D])
                for j in prob.N
            ],
            rhs = [1] * len(prob.N),
            senses = ['E'] * len(prob.N),
            names = [f'visit_{j}' for j in prob.N]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'x_{i}_{j}' for j in prob.N_t if (i,j) in prob.A_T] + [f'x_{j}_{i}' for j in prob.N_s if (j,i) in prob.A_T],
                    val = [1 for j in prob.N_t if (i,j) in prob.A_T] + [-1 for j in prob.N_s if (j,i) in prob.A_T])
                for i in prob.N
            ],
            rhs = [0] * len(prob.N),
            senses = ['E'] * len(prob.N),
            names = [f'flow_balance_at_{i}' for i in prob.N]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'x_{0}_{j}' for j in prob.Out_i[0]] + [f'x_{i}_{prob.n + 1}' for i in prob.N if (i,prob.n+1) in prob.A_T],
                    val = [1 for j in prob.Out_i[0]] + [-1 for i in prob.N if (i,prob.n+1) in prob.A_T])
            ],
            rhs = [0],
            senses = ['E'],
            names = ['num_depot_in_out']
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'x_{k}_{i}' for k in prob.N_s if (k,i) in prob.A_T] + [f'y_{d}_{i}_{j}' for d in prob.D],
                    val = [1 for k in prob.N_s if (k,i) in prob.A_T] + [-1] * len(prob.D))
                for i,j in prob.A_D
            ],
            rhs = [0] * len(prob.A_D),
            senses = ['G'] * len(prob.A_D),
            names = [f'drone_dispatch_at_{i}_to_{j}' for i,j in prob.A_D]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'f_{i}'] + [f'y_{d}_{i}_{j}' for d in prob.D for j in prob.N if (i,j) in prob.A_D],
                    val = [1] + [-demand[j] for d in prob.D for j in prob.N if (i,j) in prob.A_D])
                for i in prob.N
            ],
            rhs = [0] * len(prob.N),
            senses = ['E'] * len(prob.N),
            names = [f'drone_demands_at_{i}' for i in prob.N]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'e_{j}'] + [f'e_{i}'] + [f'f_{j}'] + [f'x_{i}_{j}'],
                    val = [1] + [-1] + [-1] + [-(demand[j] + prob.capacity)])
                for i,j in prob.A_T
            ],
            rhs = [-prob.capacity] * len(prob.A_T),
            senses = ['G'] * len(prob.A_T),
            names = [f'MTZ_demand_{i}_{j}' for i,j in prob.A_T]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'u_{j}_{gamma}'] + [f'v_{i}_{gamma}'] + [f'x_{i}_{j}'],
                    val = [1] + [-1] + [-(l_tT[i,j] + b_i[0])])
                for i,j in prob.A_T
                for gamma in range(prob.T_Gamma + 1)
            ],
            rhs = [-b_i[0]] * len(prob.A_T) * (prob.T_Gamma + 1),
            senses = ['G'] * len(prob.A_T) * (prob.T_Gamma + 1),
            names = [f'MTZ_truck_time_recursive_1_{i}_{j}_{gamma}' for i,j in prob.A_T for gamma in range(prob.T_Gamma + 1)]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'u_{j}_{gamma}'] + [f'v_{i}_{gamma - 1}'] + [f'x_{i}_{j}'],
                    val = [1] + [-1] + [-(l_tT[i,j] + l_T_delta[i,j] + b_i[0])])
                for i,j in prob.A_T
                for gamma in range(1, prob.T_Gamma + 1)
            ],
            rhs = [-b_i[0]] * len(prob.A_T) * (prob.T_Gamma),
            senses = ['G'] * len(prob.A_T) * (prob.T_Gamma),
            names = [f'MTZ_truck_time_recursive_2_{i}_{j}_{gamma}' for i,j in prob.A_T for gamma in range(1, prob.T_Gamma + 1)]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'v_{i}_{gamma}'] + [f'u_{i}_{gamma}'],
                    val = [1] + [-1])
                for i in prob.N_st
                for gamma in range(prob.T_Gamma + 1)
            ],
            rhs = [s_i[i] for i in prob.N_st for gamma in range(prob.T_Gamma + 1)],
            senses = ['G'] * len(prob.N_st) * (prob.T_Gamma + 1),
            names = [f'leaving_time_at_{i}_{gamma}_1' for i in prob.N_st for gamma in range(prob.T_Gamma + 1)]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'y_{d}_{i}_{j}'] + [f'p_{d}_{l}_{i}_{j}' for l in prob.L[i]],
                    val = [1] + [-1  for l in prob.L[i]])
                for d in prob.D
                for i,j in prob.A_D
            ],
            rhs = [0] * len(prob.D) * len(prob.A_D),
            senses = ['E'] * len(prob.D) * len(prob.A_D),
            names = [f'drone_{d}_dispatch_{i}_{j}_sequence' for d in prob.D for i,j in prob.A_D]
        )


        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'p_{d}_{0}_{k}_{j}' for (k,j) in prob.A_D if k==i],
                    val = [1 for (k,j) in prob.A_D if k==i])
                for d in prob.D
                for i in prob.N
                if len(prob.L[i])>0
            ],
            rhs = [1 for d in prob.D for i in prob.N if len(prob.L[i])>0],
            senses = ['L' for d in prob.D for i in prob.N if len(prob.L[i])>0],
            names = [f'drone_{d}_dispatch_{i}_first_dispatch'  for d in prob.D for i in prob.N if len(prob.L[i])>0]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'p_{d}_{l+1}_{k}_{j}' for (k,j) in prob.A_D if k==i] + [f'p_{d}_{l}_{k}_{j}' for (k,j) in prob.A_D if k==i],
                    val = [1 for (k,j) in prob.A_D if k==i] + [-1 for (k,j) in prob.A_D if k==i])
                for d in prob.D
                for i in prob.N
                for l in prob.L[i][:-1]
                if len(prob.L[i])>1
            ],
            rhs = [0 for d in prob.D for i in prob.N for l in prob.L[i][:-1] if len(prob.L[i])>1],
            senses = ['L' for d in prob.D for i in prob.N for l in prob.L[i][:-1] if len(prob.L[i])>1],
            names = [f'drone_{d}_dispatch_{i}_force_sequence_{l}th'  for d in prob.D for i in prob.N for l in prob.L[i][:-1] if len(prob.L[i])>1]
        )


        M = b_i[0] * 2

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'q_{0}_{gamma}_{d}_{0}_{i}', f'u_{i}_{gamma}', f'p_{d}_{0}_{i}_{j}'],
                    val = [1, -1, -M-l_tD[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
            ],
            rhs = [-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1)],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1)],
            names = [f'drone_{d}_dispatch_({i}_{j})_first_arrivaltime_{gamma}_{0}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1)]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'q_{delta}_{gamma}_{d}_{0}_{i}', f'u_{i}_{gamma}', f'p_{d}_{0}_{i}_{j}'],
                    val = [1, -1, -M-l_tD[i,j]-l_D_delta[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for delta in range(1, prob.D_Delta+1)
            ],
            rhs = [-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(1, prob.D_Delta+1)],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(1, prob.D_Delta+1)],
            names = [f'drone_{d}_dispatch_({i}_{j})_first_arrivaltime_{gamma}_{delta}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(1, prob.D_Delta+1)]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'q_{delta}_{gamma}_{d}_{l+1}_{i}', f'r_{delta}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l+1}_{i}_{j}'],
                    val = [1, -1, -M-l_tD[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for delta in range(prob.D_Delta+1)
                for l in prob.L[i][:-1]
            ],
            rhs = [-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta+1) for l in prob.L[i][:-1]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta+1) for l in prob.L[i][:-1]],
            names = [f'drone_{d}_dispatch_({i}_{j})_{l}th_arrivaltime_{gamma}_{delta}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta+1) for l in prob.L[i][:-1]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'q_{delta+1}_{gamma}_{d}_{l+1}_{i}', f'r_{delta}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l+1}_{i}_{j}'],
                    val = [1, -1, -M-l_tD[i,j]-l_D_delta[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for delta in range(prob.D_Delta)
                for l in prob.L[i][:-1]
            ],
            rhs = [-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta) for l in prob.L[i][:-1]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta) for l in prob.L[i][:-1]],
            names = [f'drone_{d}_dispatch_({i}_{j})_{l}th_arrivaltime_{gamma}_{delta}_plus_one' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta) for l in prob.L[i][:-1]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'r_{delta}_{gamma}_{d}_{l}_{i}', f'q_{delta}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l}_{i}_{j}'],
                    val = [1, -1, -M-l_tD[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for delta in range(prob.D_Delta+1)
                for l in prob.L[i]
            ],
            rhs = [s_i[j]-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta+1) for l in prob.L[i]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta+1) for l in prob.L[i]],
            names = [f'drone_{d}_dispatch_({i}_{j})_{l}th_returntime_{gamma}_{delta}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta+1) for l in prob.L[i]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'r_{delta+1}_{gamma}_{d}_{l}_{i}', f'q_{delta}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l}_{i}_{j}'],
                    val = [1, -1, -M-l_tD[i,j]-l_D_delta[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for delta in range(prob.D_Delta)
                for l in prob.L[i]
            ],
            rhs = [s_i[j]-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta) for l in prob.L[i]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta) for l in prob.L[i]],
            names = [f'drone_{d}_dispatch_({i}_{j})_{l}th_returntime_{gamma}_{delta}_plus_one' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(prob.D_Delta) for l in prob.L[i]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'r_{0}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l}_{i}_{j}'],
                    val = [1, -M-l_tD[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for l in prob.L[i]
            ],
            rhs = [a_i[j] + s_i[j] - M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for l in prob.L[i]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for l in prob.L[i]],
            names = [f'drone_{d}_dispatch_({i}_{j})_{l}th_returntime_after_TW_{gamma}_{0}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for l in prob.L[i]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'r_{delta}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l}_{i}_{j}'],
                    val = [1, -M-l_tD[i,j]])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for delta in range(1,prob.D_Delta+1)
                for l in prob.L[i]
            ],
            rhs = [a_i[j] + s_i[j] - M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(1,prob.D_Delta+1) for l in prob.L[i]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(1,prob.D_Delta+1) for l in prob.L[i]],
            names = [f'drone_{d}_dispatch_({i}_{j})_{l}th_returntime_after_TW_{gamma}_{delta}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for delta in range(1,prob.D_Delta+1) for l in prob.L[i]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'u_{j}_{gamma}', f'q_{prob.D_Delta}_{gamma}_{d}_{l}_{i}', f'p_{d}_{l}_{i}_{j}'],
                    val = [1, -1, -M])
                for d in prob.D
                for (i,j) in prob.A_D
                for gamma in range(prob.T_Gamma+1)
                for l in prob.L[i]
            ],
            rhs = [-M for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for l in prob.L[i]],
            senses = ['G' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for l in prob.L[i]],
            names = [f'drone_{d}_arrivaltime_{j}_from_{i}_{l}th_{gamma}' for d in prob.D for (i,j) in prob.A_D for gamma in range(prob.T_Gamma+1) for l in prob.L[i]]
        )

        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'u_{i}_{gamma}', f'w_{i}_{gamma}', f'r_{prob.D_Delta}_{gamma}_{d}_{l}_{i}'],
                    val = [1, 1, -1])
                for d in prob.D
                for i in prob.N
                for gamma in range(prob.T_Gamma+1)
                for l in prob.L[i]
            ],
            rhs = [0 for d in prob.D for i in prob.N for gamma in range(prob.T_Gamma+1) for l in prob.L[i]],
            senses = ['G' for d in prob.D for i in prob.N for gamma in range(prob.T_Gamma+1) for l in prob.L[i]],
            names = [f'waitingtime_at_{i}_for_drone_{d}_{l}th_returntime_{gamma}' for d in prob.D for i in prob.N for gamma in range(prob.T_Gamma+1) for l in prob.L[i]]
        )




        pcpx.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(
                    ind = [f'v_{i}_{gamma}'] + [f'u_{i}_{gamma}'] + [f'w_{i}_{gamma}'],
                    val = [1] + [-1] + [-1])
                for i in prob.N
                for gamma in range(prob.T_Gamma + 1)
            ],
            rhs = [0 for i in prob.N for gamma in range(prob.T_Gamma + 1)],
            senses = ['G'] * len(prob.N) * (prob.T_Gamma + 1),
            names = [f'leaving_time_at_{i}_{gamma}_2' for i in prob.N for gamma in range(prob.T_Gamma + 1)]
        )

    
        self.pcpx = pcpx


    def solve(self, num_cpus=2, time_limit=3600, bnb_memory_limit=48000, show_log=False, truck_routes=None):
        pcpx = self.pcpx

        start_time = time.time()

        pcpx.parameters.threads.set(num_cpus)
        pcpx.parameters.timelimit.set(time_limit)
        pcpx.parameters.mip.limits.treememory.set(bnb_memory_limit) # in megabytes

        if not show_log:
            pcpx.parameters.mip.display.set(0)
            pcpx.set_log_stream(None)
            pcpx.set_results_stream(None)

        cb = MIPCallback()
        contextmask = 0
        contextmask |= cplex.callbacks.ContextType.relaxation

        pcpx.set_callback(cb, contextmask)


        if truck_routes is not None:
            # Fix truck routes
            for tr in truck_routes:
                for i,j in zip(tr[:-1], tr[1:]):
                    pcpx.variables.set_lower_bounds(f'x_{i}_{j}', 1)


        pcpx.solve()

        solve_time = time.time() - start_time
        sol = pcpx.solution

        self.sol = sol
        self.solve_time = solve_time

        results = {
            'problem': self.prob.prob_info(),
            'timelimit': time_limit,
            'num_cpus': num_cpus,
            'alg': 'MIP',
            'time': solve_time,
            'root_LP': cb.root_lp,
            'obj': pcpx.solution.get_objective_value(),
            'final_gap': pcpx.solution.MIP.get_mip_relative_gap(),
            'global_lb': pcpx.solution.MIP.get_best_objective(),
            'num_bnb_nodes': pcpx.solution.progress.get_num_nodes_processed(), #cplex._internal._procedural.getnodecnt(pcpx._env._e, pcpx._lp),
            'num_bnb_remaining_nodes': pcpx.solution.progress.get_num_nodes_remaining(),
            'solution': self.get_solution(),
            'termination_status': pcpx.solution.get_status_string()
        }

        return results
    
    def solve_lp_rlx(self):
        prob = self.prob
        pcpx = self.pcpx

        pcpx.parameters.mip.display.set(0)

        for i,j in prob.A_T:
            pcpx.variables.set_types(f'x_{i}_{j}', pcpx.variables.type.continuous)

        for d in prob.D:
            for i,j in prob.A_D:
                pcpx.variables.set_types(f'y_{d}_{i}_{j}', pcpx.variables.type.continuous)
                pcpx.variables.set_types(f'h_{d}_{i}_{j}', pcpx.variables.type.continuous)

        for d in prob.D:
            for i,j,k in prob.A_R:
                pcpx.variables.set_types(f'p_{d}_{i}_{j}_{k}', pcpx.variables.type.continuous)
        
        pcpx.solve()

        sol = pcpx.solution

        self.sol = sol

    def get_solution(self):
        pcpx = self.pcpx
        prob = self.prob

        positive_arcs = [(i,j) for i,j in prob.A_T if pcpx.solution.get_values(f'x_{i}_{j}') > 0.9]

        truck_routes = []

        s = 0
        t = prob.n+1

        for idx,(i,j) in enumerate(positive_arcs):
            if i == s:
                route = [i,j]
                cur_node = j
                while True:
                    for (k,l) in positive_arcs:
                        if k == cur_node:
                            route.append(l)
                            cur_node = l
                            break
                    if cur_node == t:
                        break

                truck_routes.append(route)

        drone_solution = []
        for d in prob.D:
            drone_arcs = [(pcpx.solution.get_values(f'v_{j}_{0}'), (i,j)) for i,j in prob.A_D if pcpx.solution.get_values(f'y_{d}_{i}_{j}') > 0.9]
            drone_arcs = sorted(drone_arcs)
            drone_solution.append([(i,j) for v,(i,j) in drone_arcs])


        sol = []
        for tr in truck_routes:
            drone_dispatches = []
            for d in prob.D:
                for i in tr[1:-1]:
                    this_drone_dispatches = []
                    for l in prob.L[i]:
                        for j in prob.drone_avail_nodes[i]:
                            if pcpx.solution.get_values(f'p_{d}_{l}_{i}_{j}') > 0.9:
                                this_drone_dispatches.append((i,j))

                drone_dispatches.append(this_drone_dispatches)
            
            sol.append((tr, drone_dispatches))

   
        return sol



class MIPCallback():
    def __init__(self):
        self.root_lp = None

    def invoke(self, context):
        if context.get_long_info(cplex.callbacks.Context.info.node_uid) == 0:
            # Update root node relaxation value
            self.root_lp = context.get_relaxation_objective()
        # print(f' ==> {context.get_relaxation_objective()}, {context.get_long_info(cplex.callbacks.Context.info.node_uid)}')


def to_list_of_array_int(list_of_list):
    return [np.array(l, dtype=np.int64) for l in list_of_list]


def to_numba_list_of_list_int(list_of_list):
    numba_l_of_l = List()
    for idx, j_list in enumerate(list_of_list):
        numba_l = List()

        for j in j_list:
            numba_l.append(j)
        
        if len(numba_l) == 0:
            numba_l_of_l.append(numba.typed.List.empty_list(numba.int64))
        else:
            numba_l_of_l.append(numba_l)

    return numba_l_of_l

def to_numba_list_of_list_float(list_of_list):
    numba_l_of_l = List()
    for idx, j_list in enumerate(list_of_list):
        numba_l = List()

        for j in j_list:
            numba_l.append(j)
        
        if len(numba_l) == 0:
            numba_l_of_l.append(numba.typed.List.empty_list(numba.float64))
        else:
            numba_l_of_l.append(numba_l)

    return numba_l_of_l


@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def solve_labeling_and_backtracking(
    num_nodes, 
    num_drones,
    T_Gamma, 
    Out_i, 
    phi_i, 
    np_alpha_ij, 
    np_p_ij,
    np_argsorted_alpha_i,
    a_i, 
    b_i, 
    s_i, 
    demand_data, 
    capacity, 
    l_tT, 
    l_T_delta,
    l_CT, 
    l_tD, 
    l_D_delta,
    Drone_Out_i,
    np_cT_ij,
    preallocated_labels=None,
    aggressive_dominance=False,
    aggressive_extension=False
    ):
    
    COST, DEMAND, _, PREV, VALID, LB, _, _, LID, PREV_LID, NUM_LABELS = solve_labeling(num_nodes, num_drones, T_Gamma, Out_i, phi_i, np_alpha_ij, np_p_ij, np_argsorted_alpha_i, a_i, b_i, s_i, demand_data, capacity, l_tT, l_T_delta, l_CT, l_tD, l_D_delta, Drone_Out_i, preallocated_labels=preallocated_labels, aggressive_dominance=aggressive_dominance, aggressive_extension=aggressive_extension)

    # Sort labels by the lower bounds of reduced costs
    sorted_args = np.argsort(COST[-1,:NUM_LABELS[-1]]-LB[-1,:NUM_LABELS[-1]])
    COST[-1,:NUM_LABELS[-1]] = COST[-1][sorted_args]
    DEMAND[-1,:NUM_LABELS[-1]] = DEMAND[-1][sorted_args]
    PREV[-1,:NUM_LABELS[-1]] = PREV[-1][sorted_args]
    VALID[-1,:NUM_LABELS[-1]] = VALID[-1][sorted_args]
    LID[-1,:NUM_LABELS[-1]] = LID[-1][sorted_args]
    PREV_LID[-1,:NUM_LABELS[-1]] = PREV_LID[-1][sorted_args]

    truck_routes, truck_route_objs, truck_route_demands, truck_route_rcs = path_packtracking(COST, DEMAND, PREV, VALID, LID, PREV_LID, NUM_LABELS, num_nodes-2, np_cT_ij)

    return truck_routes, truck_route_objs, truck_route_demands, truck_route_rcs 




@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def allocate_label_memory(num_nodes, T_Gamma, size):
    MASTER_LABELS = (
        np.zeros((num_nodes,size), dtype=np.float32), # COST
        np.zeros((num_nodes,size, T_Gamma + 1), dtype=np.int32), # MOVETIME
        np.zeros((num_nodes,size), dtype=np.int32), # DEMAND
        np.zeros((num_nodes,size), dtype=np.int64), # VISIT
        np.zeros((num_nodes,size), dtype=np.int8), # PREV
        np.zeros((num_nodes,size), dtype=np.int32), # PREV_TIME
        np.zeros((num_nodes,size), dtype=np.bool8), # VALID
        np.zeros((num_nodes,size), dtype=np.bool8), # CHECK_TREATED

        np.zeros((num_nodes,size), dtype=np.float32), # LB
        np.zeros((num_nodes,size), dtype=np.int16), # LB_DEMAND
        np.zeros((num_nodes,size), dtype=np.float32), # UB
        np.zeros((num_nodes,size), dtype=np.int32), # LID
        np.zeros((num_nodes,size), dtype=np.int32) # PREV_LID
    )

    return MASTER_LABELS




@numba.njit(cache=True, parallel=False, nogil=True, fastmath=True)
def solve_labeling(
    num_nodes, 
    num_drones,
    T_Gamma, 
    Out_i, 
    phi_i, 
    np_alpha_ij, 
    np_p_ij,
    np_argsorted_alpha_i,
    a_i, 
    b_i, 
    s_i, 
    demand_data, 
    capacity, 
    l_tT, 
    l_T_delta,
    l_CT, 
    l_tD, 
    l_D_delta,
    Drone_Out_i,
    preallocated_labels = None,
    max_num_labels=10000,
    aggressive_dominance=False,
    aggressive_extension=False
    ):


    t = num_nodes - 1


    if preallocated_labels is None:
        COST, MOVETIME, DEMAND, VISIT, PREV, PREV_TIME, VALID, CHECK_TREATED, LB, LB_DEMAND, UB, LID, PREV_LID = allocate_label_memory(num_nodes, T_Gamma, max_num_labels)
    else:
        COST, MOVETIME, DEMAND, VISIT, PREV, PREV_TIME, VALID, CHECK_TREATED, LB, LB_DEMAND, UB, LID, PREV_LID = preallocated_labels


    NUM_LABELS = np.zeros(num_nodes, dtype=np.int32)

    max_num_labels = 0


    num_created_labels = 0
    num_removed_labels = 0


    all_bits_less = lambda l1, l2: (l1 & l2) == l1
    check_visited = lambda visit, i: (visit >> (i - 1)) & 1 == 1
    set_visit = lambda visit, i: visit | (1 << (i - 1))

    drone_arc = np.zeros((num_nodes,num_nodes), dtype=np.bool8)
    for i in range(num_nodes):
        for j in Drone_Out_i[i]:
            drone_arc[i,j] = True

    np_out_i = np.ones((num_nodes,num_nodes), dtype=np.int32) * -1
    for i in range(num_nodes):
        for idx, j in enumerate(Out_i[i]):
            np_out_i[i,idx] = j


    def add_label(i, cost, movetime, demand, visit, prev, prev_time, prev_lid, lid, treat, lb, lb_demand, ub):

        num_labels = NUM_LABELS[i]

        VALID_i = VALID[i]

        for idx in range(num_labels):

            if not VALID_i[idx]:
                COST[i][idx] = cost
                MOVETIME[i][idx] = movetime
                DEMAND[i][idx] = demand
                VISIT[i][idx] = visit
                PREV[i][idx] = prev
                PREV_TIME[i][idx] = prev_time
                VALID[i][idx] = True
                CHECK_TREATED[i][idx] = treat

                LB[i][idx] = lb
                LB_DEMAND[i][idx] = lb_demand
                UB[i][idx] = ub

                LID[i][idx] = lid
                PREV_LID[i][idx] = prev_lid

                return

        COST[i][num_labels] = cost
        MOVETIME[i][num_labels] = movetime
        DEMAND[i][num_labels] = demand
        VISIT[i][num_labels] = visit
        PREV[i][num_labels] = prev
        PREV_TIME[i][num_labels] = prev_time
        VALID[i][num_labels] = True
        CHECK_TREATED[i][num_labels] = treat

        LB[i][num_labels] = lb
        LB_DEMAND[i][num_labels] = lb_demand
        UB[i][num_labels] = ub

        LID[i][num_labels] = lid
        PREV_LID[i][num_labels] = prev_lid

        NUM_LABELS[i] += 1

 

    def remove_label(i, k):
        VALID[i][k] = False    

    def get_label(i, k):
        return COST[i][k], MOVETIME[i][k], DEMAND[i][k], VISIT[i][k], PREV[i][k], PREV_TIME[i][k], PREV_LID[i][k], LID[i][k], CHECK_TREATED[i][k], LB[i][k], LB_DEMAND[i][k], UB[i][k]

    def get_num_labels(i):
        return NUM_LABELS[i]

    def drone_deliverable_bestcase(i, j, l, u):
        t1 = l + l_tD[i,j]
        t2 = max(t1, a_i[j]) + s_i[j] + l_tD[j,i]
        return (t1 <= b_i[j]) and (t2 <= u), t2

    def drone_deliverable_worstcase(i, j, l, u):
        t1 = l + l_tD[i,j] + l_D_delta[i,j]
        t2 = max(t1, a_i[j]) + s_i[j] + l_tD[j,i] + l_D_delta[j,i]
        return (t1 <= b_i[j]) and (t2 <= u), t2

    def calculate_lb(i, visit, l, u, u_max, remamined_capa):
        if aggressive_dominance or num_drones == 0:
            return 0, 0
        
        lb_sum_alpha = 0
        lb_demand_sum = 0
        drone_operation_times = np.array([l] * num_drones)
        for k in np_argsorted_alpha_i[i]:
            alpha = np_alpha_ij[i,k]
            if alpha == 0:
                break
            if not check_visited(visit, k) and b_i[k] <= u_max:
                selected_drone = drone_operation_times.argmin()
                dispatch_time = drone_operation_times[selected_drone]

                if dispatch_time >= u:
                    break

                deliverable, return_time = drone_deliverable_worstcase(i, k, dispatch_time, u)

                if deliverable and (lb_demand_sum + demand_data[k]) <= remamined_capa:
                    drone_operation_times[selected_drone] = return_time
                    lb_sum_alpha += alpha
                    lb_demand_sum += demand_data[k]
                    
        return lb_sum_alpha, lb_demand_sum


    def calculate_ub(i, l, u):
        if aggressive_dominance or num_drones == 0:
            return 0

        ub_sum_alpha = 0
        drone_operation_time = 0
        drone_operation_total_time = (u-l) * num_drones

        for k in np_argsorted_alpha_i[i]:
            alpha = np_alpha_ij[i,k]
            if alpha > 0:
                deliverable, return_time = drone_deliverable_bestcase(i, k, l, u)

                if deliverable:
                    if drone_operation_time + np_p_ij[i,k] <= drone_operation_total_time:
                        drone_operation_time += np_p_ij[i,k]
                        ub_sum_alpha += alpha
                    else:
                        ub_sum_alpha += alpha * (drone_operation_total_time - drone_operation_time)/np_p_ij[i,k]
                        break
            else:
                break

        return ub_sum_alpha


    E = set()

    if num_nodes > 0:
        add_label(
            0, # node
            0, # COST
            np.zeros(T_Gamma + 1), # MOVETIME
            0, # DEMAND
            0, # VISIT
            0, # PREV
            0, # PREV_TIME
            0, # PREV_LID
            0, # LID
            False, # CHECK_TREATED,
            0, # LB
            0, # LB_DEMAND
            0 # UB
            )
    # i, cost, movetime, demand, visit, prev, prev_time, prev_lid, lid, treat, lb, lb_demand, ub)
        E.add(0)

    terminate_now = False

    while len(E) > 0:
        i = E.pop()

        for idx_i in range(get_num_labels(i)): 
            if VALID[i][idx_i] and not CHECK_TREATED[i][idx_i]:
                
                cost, movetime, demand, visit, prev, prev_time, prev_lid, lid, treat, lb, lb_demand, ub = get_label(i, idx_i)
                CHECK_TREATED[i][idx_i] = True

                for j in np_out_i[i]:
                    if j == -1:
                        break
                    new_move_times = np.zeros_like(movetime)

                    new_move_times[0] = max(movetime[0], a_i[i]) + s_i[i] + l_tT[i,j]

                    if T_Gamma != 0:
                        for gamma in range(1, T_Gamma + 1):
                            new_move_times[gamma] = max(max(movetime[gamma], a_i[i]) + s_i[i] + l_tT[i][j], max(movetime[gamma-1], a_i[i]) + s_i[i] + l_tT[i,j] + l_T_delta[i,j])

                    

                    if aggressive_extension:
                        if drone_arc[prev, i] and prev != 0:
                            using_drone_time = max(max(prev_time+l_tD[prev,i]+l_D_delta[prev,i], a_i[i])+s_i[i]+l_tD[prev,i]+l_D_delta[prev,i], a_i[prev])+s_i[prev] + l_tT[prev,j] + l_T_delta[prev,j]
                            only_truck_time = new_move_times[T_Gamma]

                            using_drone_cost = l_CT[prev,j] - np_alpha_ij[prev,i]
                            only_truck_cost = l_CT[prev,i] - phi_i[i] + l_CT[i,j]

                            if using_drone_time < only_truck_time:
                                continue
                    else:
                        if drone_arc[prev, i] and prev != 0:
                            using_drone_time = max(max(prev_time+l_tD[prev,i]+l_D_delta[prev,i], a_i[i])+s_i[i]+l_tD[prev,i]+l_D_delta[prev,i], a_i[prev])+s_i[prev] + l_tT[prev,j] + l_T_delta[prev,j]
                            only_truck_time = new_move_times[T_Gamma]

                            using_drone_cost = l_CT[prev,j] - np_alpha_ij[prev,i]

                            slack_j = b_i[j] - new_move_times[T_Gamma]

                            if slack_j > 0:
                                only_truck_cost = l_CT[prev,i] - phi_i[i] + l_CT[i,j] - sum([np_alpha_ij[i,k] for k in Drone_Out_i[i] if k!=j and np_alpha_ij[i,k] > 0 and max(movetime[T_Gamma]+l_tD[i,k], a_i[k])+s_i[k]+l_tD[k,i]+l_tT[i,j] <= b_i[j]])

                            if using_drone_time < only_truck_time and using_drone_cost < only_truck_cost:
                                continue


                    
                    new_demand = demand + demand_data[j]

                    if (not check_visited(visit, j)) and np.all(new_move_times <= b_i[j]) and new_demand <= capacity:
                        # Extend the label
                        new_cost = cost + l_CT[i,j] - phi_i[j]
                        new_prev = i
                        new_prev_time = movetime[T_Gamma]
                        new_prev_lid = lid
                        new_visit = set_visit(visit, j)
                        new_treat = False

                        dominated = False

                        # Bounds calculation
                        new_lb, new_ub = 0, 0


                        u_i = movetime[T_Gamma] # earlist arrival time at i

                        min_drone_delivery_time_i = max(0, a_i[i] + s_i[i] - u_i)
                        max_drone_delivery_time_i = max(0, b_i[j] - l_tT[i,j] - u_i)

                        if min_drone_delivery_time_i > 0:
                            cal_lb, cal_lb_demand = calculate_lb(i, new_visit, u_i,  a_i[i] + s_i[i], a_i[i] + s_i[i] + l_tT[i,j], capacity - new_demand)

                            if cal_lb > lb:
                                new_lb = cal_lb
                                new_lb_demand = cal_lb_demand
                            else:
                                new_lb = lb
                                new_lb_demand = lb_demand
                        else:
                            new_lb = lb
                            new_lb_demand = lb_demand
                        
                        if max_drone_delivery_time_i > 0:
                            new_ub = ub + calculate_ub(i, u_i, b_i[j] - l_tT[i,j])
                        else:
                            new_ub = ub

                        if j == t:
                            if new_cost - new_ub >= 0:
                                continue

                        for idx_j in range(get_num_labels(j)):
                            if VALID[j][idx_j]:
                                cost2, movetime2, demand2, visit2, prev2, prev_time2, prev_lid2, lid2, treat2, lb2, lb_demand2, ub2 = get_label(j, idx_j)

                                if j != t:
                                    if ((aggressive_dominance and cost2 <= new_cost) or (cost2 - lb2 <= new_cost - new_ub and all_bits_less(visit2, new_visit))) and (demand2 + lb_demand2) <= new_demand and movetime2[T_Gamma] <= new_move_times[T_Gamma]:
                                        dominated = True
                                        num_removed_labels += 1
                                        break
                                else:
                                    if cost2 - lb2 <= new_cost - new_ub:
                                        dominated = True
                                        num_removed_labels += 1
                                        break


                        if not dominated:
                            dominated_labels = []

                            for idx_j in range(get_num_labels(j)):
                                if VALID[j][idx_j]:
                                    cost2, movetime2, demand2, visit2, prev2, prev_time2, prev_lid2, lid2, treat2, lb2, lb_demand2, ub2 = get_label(j, idx_j)

                                    if j != t:
                                        if ((aggressive_dominance and new_cost <= cost2) or (new_cost - new_lb <= cost2 - ub2 and all_bits_less(new_visit, visit2)) ) and (new_demand+new_lb_demand) <= demand2 and  new_move_times[T_Gamma] <= movetime2[T_Gamma]:
                                            dominated_labels.append(idx_j)
                                    else:
                                        if new_cost - new_lb <= cost2 - ub2:
                                            dominated_labels.append(idx_j)


                            for idx in dominated_labels:
                                remove_label(j, idx)
                                num_removed_labels += 1


                            num_created_labels += 1
                            add_label(j, new_cost, new_move_times, new_demand, new_visit, new_prev, new_prev_time, new_prev_lid, num_created_labels, new_treat, new_lb, new_lb_demand, new_ub)

                            if j != t:
                                E.add(j)



            if (num_created_labels + num_removed_labels) % 1000 == 0:
                with numba.objmode(terminate_now='b1'):
                    terminate_now = check_early_termination()

                if terminate_now:
                    # print('ET')
                    break

        if terminate_now:
            break

    return COST, DEMAND, VISIT, PREV, VALID, LB, LB_DEMAND, UB, LID, PREV_LID, NUM_LABELS


def check_early_termination():
    return sp1_termination_event.is_set() or all_termination_event.is_set()



@numba.jit(nopython=True, cache=True, nogil=True, fastmath=True)
def path_packtracking(COST, DEMAND, PREV, VALID, LID, PREV_LID, NUM_LABELS, n, np_cT_ij):

    add_truck_routes = []

    add_truck_route_objs = []
    add_truck_route_demands = []
    add_truck_route_rcs = []

    s = 0
    t = n + 1

    for idx_l in range(NUM_LABELS[t]):        
        if VALID[t][idx_l]:
            path = []
            path.append(t)
            cost = COST[t][idx_l]
            demand = DEMAND[t][idx_l]
            obj_cof = 0
            
            j = t

            i, i_lid = PREV[j][idx_l], PREV_LID[j][idx_l]

            while True:
                path.append(i)
                obj_cof += np_cT_ij[i,j]
                
                if i == s:
                    path = np.array(path)[::-1]
                    add_truck_routes.append(path)
                    add_truck_route_objs.append(obj_cof)
                    add_truck_route_demands.append(demand)
                    add_truck_route_rcs.append(cost)
                    break
            
                found_idx = -1

                for idx_i in range(NUM_LABELS[i]):
                    lid = LID[i][idx_i]
                    if lid == i_lid:
                        found_idx = idx_i
                        break

                if found_idx >= 0 and VALID[i][found_idx]:
                    j = i
                    i, i_lid = PREV[i][found_idx], PREV_LID[i][found_idx]
                else:
                    break

            
        # print(path)

    return add_truck_routes, add_truck_route_objs, add_truck_route_demands, add_truck_route_rcs



@numba.jit(nopython=True, cache=True, nogil=True, fastmath=True)
def solve_dronepattern(
        truck_route, 
        truck_rc, 
        truck_demands, 
        capacity, 
        alpha_ij,
        l_tD, 
        l_tT, 
        num_drones, 
        DroneOut_i, 
        a_i, 
        b_i, 
        s_i, 
        l_T_delta,
        l_D_delta,
        demands_i,
        T_Gamma,
        D_Delta,
        aggressive=False
    ):

    truck_route_i_to_idx = {i:idx for (idx, i) in zip(range(len(truck_route)), truck_route)}

    remaining_capacity = capacity - truck_demands

    drone_arc_set = [(i,j) for i in truck_route[1:-1] for j in DroneOut_i[i] if alpha_ij[i,j]>0.1 and j not in truck_route and (demands_i[j] <= remaining_capacity)]



    min_truck_travel_time = np.zeros((len(truck_route), len(truck_route)))


    for idx_i in range(len(truck_route)-1):
        i = truck_route[idx_i]
        travel_time = 0
        for idx_j in range(idx_i+1, len(truck_route)):
            j = truck_route[idx_j]
            travel_time += l_tT[truck_route[idx_j-1],j] 
            min_truck_travel_time[idx_i,idx_j] = travel_time



    def check_feasible_arc_pair(a1, a2):
        (i,j), (k,l) = a1, a2

        if j == l: # delivery for the same node
            return False
        
        idx_i = truck_route_i_to_idx[i]
        idx_k = truck_route_i_to_idx[k]
        if idx_i < idx_k:
            earliest_arrival_at_k = a_i[j] + s_i[j] + l_tD[i,j] + min_truck_travel_time[idx_i, idx_k]
            return (earliest_arrival_at_k <= b_i[k]) and (earliest_arrival_at_k + l_tD[k,l] <= b_i[l])
        elif idx_i > idx_k:
            earliest_arrival_at_i = a_i[l] + s_i[l] + l_tD[k,l] + min_truck_travel_time[idx_k, idx_i]
            # print(f'{min_truck_travel_time[idx_k, idx_i]=}, {earliest_arrival_at_i <= b_i[i]=}, , {earliest_arrival_at_i=}, {l_tD[i,j]=}, {earliest_arrival_at_i + l_tD[i,j] <= b_i[j]=}')
            return (earliest_arrival_at_i <= b_i[i]) and (earliest_arrival_at_i + l_tD[i,j] <= b_i[j])
        else:
            # print(f'{max(a_i[j]+s_i[j]+l_tD[i,j], a_i[l]+s_i[l]+l_tD[k,l]) + l_tT[i,truck_route[idx_k+1]]=}, {b_i[truck_route[idx_k+1]]=}')
            return max(a_i[j]+s_i[j]+l_tD[i,j], a_i[l]+s_i[l]+l_tD[k,l]) + l_tT[i,truck_route[idx_k+1]] <= b_i[truck_route[idx_k+1]]



    def get_not_serviceable_arcs(candidate_arc_set):
        not_servable_arcs = []
        for a1 in candidate_arc_set:
            sum_alpha = alpha_ij[a1]
            for a2 in candidate_arc_set:
                if a1 != a2:
                    feasible = check_feasible_arc_pair(a1, a2)
                    if feasible:
                        sum_alpha += alpha_ij[a2]

                    # print(f'{a1}-{a2}: {feasible}')

            if sum_alpha <= truck_rc:
                not_servable_arcs.append(a1)
                
            # print(f'{a1}: {sum_alpha=}, {truck_rc=}')
        return not_servable_arcs


    def get_must_serve_arcs(candidate_arc_set):
        must_serve_arcs = []
        total_alpha = sum([alpha_ij[a] for a in candidate_arc_set])
        for a1 in candidate_arc_set:
            if total_alpha - alpha_ij[a1] < truck_rc:
                must_serve_arcs.append(a1)
        return must_serve_arcs


    def get_max_drone_dispatches(candidate_arc_set):

        max_drone_dispatches = 0
        for i in truck_route:
            num_drone_dispatches = 0
            for (k,l) in candidate_arc_set:
                if k == i:
                    num_drone_dispatches += 1

            max_drone_dispatches = max(max_drone_dispatches, num_drone_dispatches)

        return max_drone_dispatches


    def permutations(A, k):
        if not aggressive:
            r = [[(i,i) for i in range(0)]]
            for i in range(k):
                r = [[a] + b for a in A for b in r if (a in b)==False]
            return r
        else:
        
            r = np.array([a_i[j]+b_i[j] for (i,j) in A]).argsort()
            return [[A[k] for k in r]]



    def calculate_waiting_time(drone_dispatch_sequence, arrival_time):
        L = list(range(len(drone_dispatch_sequence)))
        𝚫 = list(range(D_Delta+1))

        q_𝛅_l = np.zeros((len(𝚫), len(L)), dtype=np.int32)
        r_𝛅_l = np.zeros((len(𝚫), len(L)), dtype=np.int32)

        (i,j) = drone_dispatch_sequence[0]
        q_𝛅_l[0,0] = arrival_time + l_tD[i,j]
        if q_𝛅_l[0,0] > b_i[j]:
            return None, None
        r_𝛅_l[0,0] = max(q_𝛅_l[0,0] + s_i[j] + l_tD[i,j], a_i[j] + s_i[j] + l_tD[i,j])

        for 𝛅 in range(1, D_Delta+1):
            q_𝛅_l[𝛅,0] = arrival_time + l_tD[i,j] + l_D_delta[i,j]
            if q_𝛅_l[𝛅,0] > b_i[j]:
                return None, None
            r_𝛅_l[𝛅,0] = max(
                q_𝛅_l[𝛅,0] + s_i[j] + l_tD[i,j], 
                q_𝛅_l[𝛅-1,0] + s_i[j] + l_tD[i,j] + l_D_delta[i,j], 
                a_i[j] + s_i[j] + l_tD[i,j]
            )


        # Get the earlest available time for the next drone delivery
        r_hat_𝛅_l = lambda 𝛅,l: arrival_time if l+1 < num_drones else min(r_𝛅_l[𝛅][:l+1]) if l+1==num_drones else min(-np.partition(-r_𝛅_l[𝛅][:l+1], num_drones)[:num_drones])

        for l in L[1:]:
            (i,j) = drone_dispatch_sequence[l]
            q_𝛅_l[0,l] = r_hat_𝛅_l(0,l-1) + l_tD[i,j]
            if q_𝛅_l[0,l] > b_i[j]:
                return None, None
            r_𝛅_l[0,l] = max(
                q_𝛅_l[0,l] + s_i[j] + l_tD[i,j], 
                a_i[j] + s_i[j] + l_tD[i,j]
            )

        for 𝛅 in range(1,D_Delta+1):
            for l in L[1:]:
                (i,j) = drone_dispatch_sequence[l]
                q_𝛅_l[𝛅,l] = max(
                    r_hat_𝛅_l(𝛅,l-1) + l_tD[i,j],
                    r_hat_𝛅_l(𝛅-1,l-1) + l_tD[i,j] + l_D_delta[i,j]
                )

                if q_𝛅_l[𝛅,l] > b_i[j]:
                    return None, None

                r_𝛅_l[𝛅,l] = max(
                    q_𝛅_l[𝛅,l] + s_i[j] + l_tD[i,j], 
                    q_𝛅_l[𝛅-1,l] + s_i[j] + l_tD[i,j] + l_D_delta[i,j], 
                    a_i[j] + s_i[j] + l_tD[i,j]
                )

        # Backtracking for finding drone assignments
        drone_numbers = []
        for l, (i,j) in enumerate(drone_dispatch_sequence):
            if l < num_drones:
                drone_numbers.append(l)
            else:
                arrivaltime1 = q_𝛅_l[D_Delta][l]
                for prev_l, prev_dn in zip(range(l), drone_numbers):
                    arrivaltime2 = r_𝛅_l[D_Delta][prev_l] + l_tD[i,j]
                    if abs(arrivaltime1-arrivaltime2) < 0.001:
                        drone_numbers.append(prev_dn)
                        break
                    if D_Delta > 0:
                        arrivaltime2 = r_𝛅_l[D_Delta-1][prev_l] + l_tD[i,j] + l_D_delta[i,j]
                        # print(arrivaltime1, arrivaltime2)
                        if abs(arrivaltime1-arrivaltime2) < 0.001:
                            drone_numbers.append(prev_dn)
                            break

        return drone_numbers, r_𝛅_l[D_Delta].max()



    def try_solution(try_arc_set):

        drone_arcs = []
        drone_numbers = []

        # Drone scheduling at a given node is optimal if the number of drones >= drone dispatches

        sum_demand = 0
        for (i,j) in try_arc_set:
            sum_demand += demands_i[j]

            if sum_demand + truck_demands > capacity:
                return None # Over capacity
                # print('Bad truck route! (over capacity)')

        u = np.zeros(T_Gamma + 1) # arrival time
        v = np.zeros(T_Gamma + 1) # leaving time

        for idx_i in range(1, len(truck_route)):
            i = truck_route[idx_i]
            
            prev = truck_route[idx_i-1]

            u[0] = v[0] + l_tT[prev, i]
            for gamma in range(1, T_Gamma+1):
                u[gamma] = max(
                    v[gamma] + l_tT[prev, i], 
                    v[gamma-1] + l_tT[prev, i] + l_T_delta[prev, i]
                )

            # print(f'{i=}, {u=}, {b_i[i]=}')

            for gamma in range(T_Gamma+1):
                if u[gamma] > b_i[i]:
                    # print(f'Time window: {i=}')
                    return None

            
                    
            drone_dispatches_i = [(k,l) for (k,l) in try_arc_set if k==i]

            # print(f'{drone_dispatches_i=}')

            if len(drone_dispatches_i) == 0:
                # No drone dispatches
                for gamma in range(T_Gamma+1):
                    v[gamma] = max(
                        u[gamma] + s_i[i], 
                        a_i[i] + s_i[i]
                    )
            elif len(drone_dispatches_i) <= num_drones:
                # All drone dispaches can be started at the same time
                drone_numbers_i, leaving_time = calculate_waiting_time(drone_dispatches_i, u[T_Gamma])
                if drone_numbers_i is None:
                    return None
                
                drone_arcs.extend(drone_dispatches_i)
                drone_numbers.extend(drone_numbers_i)

                v[T_Gamma] = max(
                    u[T_Gamma] + s_i[i], 
                    a_i[i] + s_i[i],
                    leaving_time
                )

                for gamma in range(T_Gamma):
                    _, leaving_time = calculate_waiting_time(drone_dispatches_i, u[gamma])
                    v[gamma] = max(
                        u[gamma] + s_i[i], 
                        a_i[i] + s_i[i],
                        leaving_time
                    )

            else:
                # Drone dispatching sequence should be determined
                # Try all sequence permutations
                min_leaving_time = 10000000
                min_drone_dispatch_sequence_i = None
                min_drone_numbers_i = None
                all_sequence_permutations = permutations(drone_dispatches_i, len(drone_dispatches_i))

                lb_leaving_time = max([a_i[j] + s_i[j] + l_tD[i,j] + l_D_delta[i,j] for (i,j) in drone_dispatches_i])

                # print(lb_leaving_time)
                
                for drone_dispatch_sequence_i in all_sequence_permutations:
                    drone_numbers_i, leaving_time = calculate_waiting_time(drone_dispatch_sequence_i, u[T_Gamma])
                    if drone_numbers_i is not None:
                        if leaving_time < min_leaving_time:
                            min_leaving_time = leaving_time
                            min_drone_dispatch_sequence_i = drone_dispatch_sequence_i
                            min_drone_numbers_i = drone_numbers_i
                            # print(min_drone_dispatch_sequence_i, min_drone_numbers_i)

                            if lb_leaving_time == leaving_time:
                                # This is optimal!
                                break

                if min_drone_dispatch_sequence_i is not None:
                    drone_arcs.extend(min_drone_dispatch_sequence_i)
                    drone_numbers.extend(min_drone_numbers_i)

                    v[T_Gamma] = max(
                        u[T_Gamma] + s_i[i], 
                        a_i[i] + s_i[i],
                        min_leaving_time
                    )

                    for gamma in range(T_Gamma):
                        _, leaving_time = calculate_waiting_time(min_drone_dispatch_sequence_i, u[gamma])
                        v[gamma] = max(
                            u[gamma] + s_i[i], 
                            a_i[i] + s_i[i],
                            leaving_time
                        )
                else:
                    return None
                

        if len(drone_arcs)==0 or len(drone_numbers)==0:
            return None
        else:
            drone_pattern = [((i,j),d) for ((i,j),d) in zip(drone_arcs, drone_numbers)]
            return drone_pattern



    # Find a set of drone arcs that make the this truck route unbeneficial
    not_servable_arcs = get_not_serviceable_arcs(drone_arc_set)
    drone_arc_set = [a for a in drone_arc_set if a not in not_servable_arcs]

    # print(drone_arc_set)

    # print(f'{truck_route=}')

    # truck_rc - total_alpha = lowerbound of reduced cost
    total_alpha = 0
    for a in drone_arc_set:
        total_alpha += alpha_ij[a]

    # print(f'{total_alpha=}')

    if truck_rc - total_alpha > -0.01 : # reduced cost cannot be negative
        # print(f'bad truck route! {not_servable_arcs=}: {total_alpha=} < {truck_rc=}')
        return -1, None
    else:

        # print(not_servable_arcs, [(a,alpha_ij[a]) for a in drone_arc_set])

        # Find a set of drone arcs that must be served to be the truck route beneficial
        must_serve_arcs = get_must_serve_arcs(drone_arc_set)

        # print(must_serve_arcs, truck_rc)
        
        # If there are more than two must-serve drone arcs, check compatability of them
        if len(must_serve_arcs) >= 2 and get_max_drone_dispatches(must_serve_arcs) <= num_drones:
            drone_pattern = try_solution(must_serve_arcs)

            if drone_pattern is None:
                # print('Infeasible must_serve_arcs', drone_pattern)
                return -1, None

        # Here we try various drone patterns for campatability
        # Things to consider:
        # 1. At most one arc for a node can be selected for drone delivery, i.e., (1,10) and (3,10) cannot be selected at the same time
        # 2. We use the earliest due date rule for the drone dispatches at the same waiting node
        # 3. If a given drone pattern is infeasible, we remove an arc with smallest alpha_ij and try again until the sum of alpha is less than truck_rc => Not anymore. We try all drone nodes combinations with not greater demands sums than the spare capacity


        # Nodes that drones visit
        all_drone_nodes_arcs = {}
        for idx, (i,j) in enumerate(drone_arc_set):
            if j not in all_drone_nodes_arcs:
                all_drone_nodes_arcs[j] = np.zeros(0, dtype=np.int8)
            
            all_drone_nodes_arcs[j] = np.concatenate((all_drone_nodes_arcs[j], np.array([idx], dtype=np.int8)))

        # print(all_drone_nodes_arcs, drone_arc_set)

        best_sum_feasible_alpha = 0
        best_feasible_drone_pattern = None

        all_feasible_drone_pattern = []

        all_drone_nodes = List(all_drone_nodes_arcs.keys())

        if aggressive: # Preventing full emneration of subsets
            all_drone_nodes = all_drone_nodes[:6] if len(all_drone_nodes) != 0 else all_drone_nodes


        # Get subsets of drone nodes whose demand sum is not greater than the remaining capacity
        for idx_subset in range(2**len(all_drone_nodes)-1,0,-1):

            sum_demands = 0
            max_sum_alpha = 0
            idx_node = 0
            drone_nodes = {}
            skip = False
            while idx_subset > 0:
                if idx_subset & 1 == 1:
                    drone_node = all_drone_nodes[idx_node]
                    if sum_demands + demands_i[drone_node] > remaining_capacity:
                        skip = True
                        break
                    sum_demands += demands_i[drone_node]
                    max_sum_alpha += max([alpha_ij[drone_arc_set[arc_idx]] for arc_idx in all_drone_nodes_arcs[drone_node]])
                    drone_nodes[drone_node] = all_drone_nodes_arcs[drone_node]
                idx_subset = idx_subset >> 1
                idx_node += 1

            # print(f'{drone_nodes}, {truck_rc - max_sum_alpha}')
            
            if not skip and truck_rc - max_sum_alpha < -0.1 and max_sum_alpha > best_sum_feasible_alpha:

                # Satisfying capaciaty constraint and havng sufficient dual values
                drone_nodes_carry_num = {}

                num_tries = 1
                for j, arcs in drone_nodes.items():
                    if len(arcs) > 1:
                        drone_nodes_carry_num[j] = num_tries
                        num_tries *= len(arcs)


                for idx_try in range(num_tries):
                    try_drone_arc_set = []
                    for j, arcs in drone_nodes.items():
                        arc_idx = 0
                        if len(arcs) > 1:
                            # Rotate arcs
                            arc_idx = (idx_try // drone_nodes_carry_num[j]) % len(arcs)

                        try_drone_arc_set.append(drone_arc_set[arcs[arc_idx]])

                    # print(try_drone_arc_set)

                    try_drone_arc_alpha = [alpha_ij[i,j] for (i,j) in try_drone_arc_set]
                    sum_alpha = sum(try_drone_arc_alpha)

                    if truck_rc - sum_alpha < -0.1 and sum_alpha > best_sum_feasible_alpha:                        
                        drone_pattern = try_solution(try_drone_arc_set)

                        if drone_pattern is not None:
                            if len(try_drone_arc_set) != len(drone_pattern):
                                print(truck_rc, sum_alpha, truck_rc-sum_alpha, try_drone_arc_alpha, try_drone_arc_set, drone_pattern)

                            all_feasible_drone_pattern.append(drone_pattern)
                            if best_sum_feasible_alpha < sum_alpha:
                                best_sum_feasible_alpha = sum_alpha
                                best_feasible_drone_pattern = drone_pattern



        if best_feasible_drone_pattern is not None:
            # Solved the problem!
            return 1, all_feasible_drone_pattern
        elif truck_rc < -0.1:            
            # truck route has a negative reduced cost
            return 1, all_feasible_drone_pattern



    # No negative reduced cost drone patterns
    return -1, None




class MP:
    def __init__(self, prob, num_cpus=8):
        self.prob = prob

        self.num_cpus = num_cpus

        self.route_arc_list = {(i,j) : set() for i,j in prob.A_T}

        self.obj_value = cplex.infinity

        self.formulate()
    
    def formulate(self):
        prob = self.prob

        cpx = cplex.Cplex()

        cpx.set_log_stream(None)
        cpx.set_results_stream(None)
        cpx.parameters.threads.set(self.num_cpus)

        cpx.variables.add(
            names = [f'y_{i}_{j}' for i,j in prob.A_D],
            obj = [float(prob.l_cD[i,j]) for i,j in prob.A_D]
        )

        self.col_begin = cpx.variables.get_num()
        
        const_visit_all_nodes_start = cpx.linear_constraints.get_num()

        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=[f'y_{i}_{k}' for i,k in prob.A_D if j == k],
                                 val=[1 for i,k in prob.A_D if j == k])
                for j in prob.N
            ],
            rhs = [1] * len(prob.N),
            senses = ['E'] * len(prob.N),
            names = [f'visit_{j}_at_least' for j in prob.N]
        )

        const_visit_all_nodes_end = cpx.linear_constraints.get_num() - 1
        const_drone_avail_start = cpx.linear_constraints.get_num()

        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=[f'y_{i}_{j}'],
                                 val=[-1])
                for i,j in prob.A_D
            ],
            rhs = [0] * len(prob.A_D),
            senses = ['G'] * len(prob.A_D),
            names = [f'drone_avail_{i}_{j}' for i,j in prob.A_D]
        )

        const_drone_avail_end = cpx.linear_constraints.get_num() - 1
        
        self.cpx = cpx

        self.routes = []
        self.route_objs = []
        self.drone_patterns = []
        self.R = []
        

        self.const_visit_all_nodes_start = const_visit_all_nodes_start
        self.const_visit_all_nodes_end = const_visit_all_nodes_end
        self.const_drone_avail_start = const_drone_avail_start
        self.const_drone_avail_end = const_drone_avail_end
        
    def add_columns(self, cols):
        num_cols = len(self.R)
        cpx = self.cpx
        prob = self.prob

        r = num_cols
        for col in cols:
            truck_cost = col['truck_cost']
            truck_route = col['truck_route']
            drone_pattern = col['drone_pattern']
            
            cpx.variables.add(
                obj = [float(truck_cost)],
                names = [f'x_{r}'],
                columns = [
                    cplex.SparsePair(
                        ind = [f'visit_{i}_at_least' for i in truck_route[1:-1]] + [f'drone_avail_{i}_{j}' for (i,j),d in drone_pattern],
                        val = [1 for i in truck_route[1:-1]] + [1 for (i,j),d in drone_pattern]
                    )
                ]
            )

            for (i,j) in zip(truck_route[:-1], truck_route[1:]):
                self.route_arc_list[i,j].add(r)

            self.routes.append(truck_route)
            self.route_objs.append(truck_cost)
            self.drone_patterns.append(drone_pattern)
            self.R.append(r)

            r += 1


    def solve(self):
        cpx = self.cpx

        cpx.solve()

        if cpx.solution.get_status() == 1:
            self.obj_value = cpx.solution.get_objective_value()

        sol = cpx.solution

        self.sol = sol


    def get_objective_value(self):
        return self.obj_value

    def get_dual(self):
        all_duals = self.sol.get_dual_values()

        phi_i = all_duals[self.const_visit_all_nodes_start:self.const_visit_all_nodes_end + 1]
        tmp_alpha_ij = all_duals[self.const_drone_avail_start:self.const_drone_avail_end + 1]
        alpha_ij = {(i,j): k for (i,j),k in zip(self.prob.A_D, tmp_alpha_ij)}
        
        self.phi_i = phi_i
        self.alpha_ij = alpha_ij
        
        return phi_i, alpha_ij
    
    def is_integer_solution(self):
        nz_cols = self.nonzero_cols()
        for val in nz_cols:
            if is_frac(val):
                return False
            
        return True


    def nonzero_cols(self):
        return [val for val in self.sol.get_values(self.col_begin, self.cpx.variables.get_num()-1) if not is_zero(val)]

    def get_solution(self):
        route_sol = [(self.sol.get_values(f'x_{r}'), list(self.routes[r]), list(self.drone_patterns[r])) for r in self.R if self.sol.get_values(f'x_{r}') > 0.00001]
        pattern_sol = [(self.sol.get_values(f'y_{i}_{j}'), (i,j)) for (i,j) in self.prob.A_D if self.sol.get_values(f'y_{i}_{j}') > 0.00001]

        return route_sol, pattern_sol
    
    
    def solve_ip(self):
        prob = self.prob
        cpx = self.cpx

        cpx.parameters.mip.display.set(0)

        for r in self.R:
            cpx.variables.set_types(f'x_{r}', cpx.variables.type.integer)
        
        cpx.solve()




class SP:
    def __init__(self, prob, num_cpus=8, num_cols_early_termination=100, sp_time_limit=-1, max_num_labels=1000000):
        self.prob = prob
        
        num_nodes = prob.n + 2

        self.np_cT_ij = prob.l_CT

        self.np_a_i = np.array(prob.a_i)
        self.np_b_i = np.array(prob.b_i)
        self.np_s_i = np.array(prob.s_i)
        self.np_demand = np.array(prob.demand)     

        np_p_ij = np.ones((num_nodes, num_nodes)) * 100000.0

        for (i,j) in prob.A_D:
            np_p_ij[i,j] = prob.l_tD[i,j] + self.np_s_i[j] + prob.l_tD[j,i]

        self.np_p_ij = np_p_ij.astype(np.int64)

        self.num_cpus = num_cpus

        # This only affects sp1 with threads
        self.num_cols_early_termination = num_cols_early_termination
        
        self.sp_time_limit = sp_time_limit

        self.decomposition_round_counter = 0

        # Pre-allocate memory for labels for all threads
        self.preallocated_labels = [
            allocate_label_memory(num_nodes, prob.T_Gamma, max_num_labels)
            for _ in range(num_cpus)
        ]
        


    def solve(self, phi_i, np_alpha_ij, Out_i, DroneOut_i, max_num_labels=-1, decomposition_level=1, ensure_exact_solving=True):

        def solve_sp1_sp2_with_decomposition(executor, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, sp2_check_num_col_found=False, aggressive_dominance=False, aggressive_extension=False):
            futures = []
            a_0, b_0 = self.prob.a_i[0], self.prob.b_i[0]
            for i1 in Out_i[0]:
                par_Out_0 = np.array([i1], dtype=np.int64)

                if decomposition_level == 1:
                    par_Out_i = Out_i.copy()
                    par_Out_i[0] = par_Out_0
                    future = executor.submit(self.solve_sp1_sp2, phi_i, np_alpha_ij, List(par_Out_i), nb_DroneOut_i,  max_num_labels=max_num_labels, sp2_check_num_col_found=sp2_check_num_col_found, aggressive_dominance=aggressive_dominance, aggressive_extension=aggressive_extension)
                    futures.append(future)

                elif decomposition_level == 2:
                    for i2 in Out_i[i1]:
                        par_Out_i = Out_i.copy()
                        par_Out_i[0] = par_Out_0
                        par_Out_i[i1] = np.array([i2], dtype=np.int64)

                        future = executor.submit(self.solve_sp1_sp2, phi_i, np_alpha_ij, List(par_Out_i), nb_DroneOut_i, max_num_labels=max_num_labels, sp2_check_num_col_found=sp2_check_num_col_found, aggressive_dominance=aggressive_dominance, aggressive_extension=aggressive_extension)
                        futures.append(future)


            return futures
        

        def solve_sp1_sp2_parallelly(executor, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, sp2_check_num_col_found=True, aggressive_dominance=False, aggressive_extension=False):
            nonlocal all_num_feasible_truck_routes, num_labels, tot_sp1_time, tot_sp2_time, all_sp2_col_found
            

            futures = solve_sp1_sp2_with_decomposition(executor, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, sp2_check_num_col_found=sp2_check_num_col_found, aggressive_dominance=aggressive_dominance, aggressive_extension=aggressive_extension)

            # print(f'{len(futures)}')

            for future in as_completed(futures):

                if future.cancelled():
                    # print(f'canceled: {future}')
                    continue


                truck_route_list, sp2_col_found, num_feasible_truck_routes, sp1_time, sp2_time = future.result()

                all_num_feasible_truck_routes += num_feasible_truck_routes
                tot_sp1_time += sp1_time
                tot_sp2_time += sp2_time
                num_labels += len(truck_route_list)

                all_sp2_col_found.extend(sp2_col_found)


                if self.num_cols_early_termination > 0 and len(all_sp2_col_found) > self.num_cols_early_termination and not sp1_termination_event.is_set():
                    for pending_f in futures:
                        pending_f.cancel()

                    # print('Signal!')
                    sp1_termination_event.set()

                if self.sp_time_limit > 0 and len(all_sp2_col_found) > 0 and (time.time() - sp_start_time) > self.sp_time_limit and not sp1_termination_event.is_set():
                    for pending_f in futures:
                        pending_f.cancel()

                    # print('Signal!(Time limit)')
                    sp1_termination_event.set()


                


        sp_time = 0
        tot_sp1_time = 0
        tot_sp2_time = 0
        num_labels = 0

        nb_DroneOut_i = List(DroneOut_i)

        sp_start_time = time.time()


        all_sp2_col_found = []
        self.num_col_found_this_iter = 0

        sp1_termination_event.clear()


        all_num_feasible_truck_routes = 0

        exact_solving = False



        tot_sp_heu1_time = 0
        tot_sp_heu2_time = 0
        tot_sp_exact_time = 0


        with ThreadPoolExecutor(max_workers=self.num_cpus) as executor:

            # Aggressive solving (not exact)
            sp_heu1_start_time = time.time()
            solve_sp1_sp2_parallelly(executor, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, sp2_check_num_col_found=True, aggressive_dominance=True, aggressive_extension=True)
            
            tot_sp_heu1_time = time.time() - sp_heu1_start_time
            print('✅', end='')


            # Solve again exactly if no columns found
            if len(all_sp2_col_found) == 0 and ensure_exact_solving:

                sp_heu2_start_time = time.time()
                solve_sp1_sp2_parallelly(executor, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, sp2_check_num_col_found=True, aggressive_dominance=False, aggressive_extension=True)

                tot_sp_heu2_time = time.time() - sp_heu2_start_time
                print('✅', end='')

                if len(all_sp2_col_found) == 0:
                    sp_exact_start_time = time.time()
                    solve_sp1_sp2_parallelly(executor, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, sp2_check_num_col_found=True, aggressive_dominance=False, aggressive_extension=False)

                    tot_sp_exact_time = time.time() - sp_exact_start_time
                    print('✅', end='')


                    exact_solving = True

            print(' ', end='')
        
        sp_time = time.time() - sp_start_time


        # print(f'SP1 end... # labels={num_labels}')


        min_reduced_cost = min(ci['reduced_cost'] for ci in all_sp2_col_found) if len(all_sp2_col_found) > 0 else 0

        return {
            'phi': phi_i,
            'alpha': np_alpha_ij,
            'min_reduced_cost': min_reduced_cost,
            'sp_time': sp_time,
            'tot_sp_heu1_time': tot_sp_heu1_time,
            'tot_sp_heu2_time': tot_sp_heu2_time,
            'tot_sp_exact_time': tot_sp_exact_time,
            'tot_sp1_time': tot_sp1_time, # summed across all threads
            'tot_sp2_time': tot_sp2_time, # summed across all threads
            'num_labels': num_labels,
            'num_truck_routes': all_num_feasible_truck_routes,
            'num_gen_cols': len(all_sp2_col_found),
            'early_terminated': sp1_termination_event.is_set() or not exact_solving, 
            'gen_cols': all_sp2_col_found # list of (rc, truck_rc, truck_cost, truck_route, drone_pattern, solving_time)
        }
    
    def solve_sp1_sp2(self, phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, max_num_labels=-1, sp2_check_num_col_found=False, aggressive_dominance=False, aggressive_extension=False):
        start_time = time.time()

        # thread = threading.current_thread()
        # print(f'Worker thread: name={thread.name}, idnet={threading.get_ident()}, id={threading.get_native_id()}')


        truck_routes, truck_route_objs, truck_route_demands, truck_route_rcs = self.solve_sp1(phi_i, np_alpha_ij, Out_i, nb_DroneOut_i, max_num_labels=max_num_labels, aggressive_dominance=aggressive_dominance, aggressive_extension=aggressive_extension)

        sp1_time = time.time() - start_time

        truck_route_list = tuple(zip(truck_routes, truck_route_objs, truck_route_demands, truck_route_rcs))

        

        start_time = time.time()

        sp2_col_found, num_feasible_truck_routes = self.solve_sp2(truck_route_list, np_alpha_ij, nb_DroneOut_i, check_num_col_found=sp2_check_num_col_found, aggressive=False)

        sp2_time = time.time() - start_time

        

        return truck_route_list, sp2_col_found, num_feasible_truck_routes, sp1_time, sp2_time




    def solve_sp1(self, phi_i, np_alpha_ij, Out_i, DroneOut_i, max_num_labels=10000, aggressive_dominance=False, aggressive_extension=False):
        prob = self.prob

        num_nodes = prob.n + 2
        num_drones = len(prob.D)
        np_argsorted_alpha_i = np.array([np.argsort(np_alpha_ij[i]/self.np_p_ij[i])[::-1] for i in range(num_nodes)])

        thread = threading.current_thread()

        if thread.name.startswith('ThreadPoolExecutor'):
            thread_no = int(thread.name.split('_')[1])
            # print(thread.name)
        else:
            thread_no = None

        truck_routes, truck_route_objs, truck_route_demands, truck_route_rcs = solve_labeling_and_backtracking(prob.num_nodes, num_drones, prob.T_Gamma, Out_i, phi_i, np_alpha_ij, self.np_p_ij, np_argsorted_alpha_i, self.np_a_i, self.np_b_i, self.np_s_i, self.np_demand, prob.capacity, prob.l_tT, prob.l_T_delta, prob.l_CT, prob.l_tD, prob.l_D_delta, DroneOut_i, self.np_cT_ij, preallocated_labels=self.preallocated_labels[thread_no] if thread_no is not None else None, aggressive_dominance=aggressive_dominance, aggressive_extension=aggressive_extension)


        return truck_routes, truck_route_objs, truck_route_demands, truck_route_rcs 


    def solve_sp2(self, truck_routes_list, np_alpha_ij, DroneOut_i, check_num_col_found=False, aggressive=False):
        # this becomes exact when drone_dispatch_sequence_permutation=True

        prob = self.prob

        l_tD = prob.l_tD
        l_tT = prob.l_tT
        num_drones = len(prob.D)
        a_i = prob.a_i
        b_i = prob.b_i
        s_i = prob.s_i

        l_T_delta = prob.l_T_delta
        l_D_delta = prob.l_D_delta
        demands = prob.demand
        T_Gamma = prob.T_Gamma
        D_Delta = prob.D_Delta
        capacity = prob.capacity


        num_infeasibles = 0

        num_feasible_truck_routes = 0
        col_found = []


        for idx, truck_info in enumerate(truck_routes_list):

            if check_early_termination():
                break

            if check_num_col_found and self.num_cols_early_termination > 0 and self.num_col_found_this_iter > self.num_cols_early_termination:
                break

            truck_route, truck_cost, truck_demands, truck_rc = truck_info

            start_time = time.time()

            if len(prob.D) > 0:
                ret_code, drone_patterns = solve_dronepattern(
                        truck_route, 
                        truck_rc, 
                        truck_demands, 
                        capacity, 
                        np_alpha_ij,
                        l_tD, 
                        l_tT, 
                        num_drones, 
                        DroneOut_i, 
                        a_i, 
                        b_i, 
                        s_i, 
                        l_T_delta,
                        l_D_delta,
                        demands,
                        T_Gamma,
                        D_Delta,
                        aggressive=aggressive
                    )
            else:
                if truck_rc < -0.01:
                    ret_code, drone_patterns = 1, []
                else:
                    ret_code, drone_patterns = -1, []


            if ret_code == -1:
                num_infeasibles += 1
            elif ret_code == 1:
                if len(drone_patterns) > 0:
                    for drone_pattern in drone_patterns:
                        rc = truck_rc - sum([np_alpha_ij[i,j] for ((i,j),_) in drone_pattern])
                        col_info = {
                                'reduced_cost': rc,
                                'truck_rc': truck_rc,
                                'truck_cost': truck_cost,
                                'truck_demands': truck_demands,
                                'truck_route': truck_route,
                                'drone_pattern': drone_pattern,
                                'status': 'SP2 Exact' 
                            }

                        col_found.append(col_info)

                else:
                    col_info = {
                            'reduced_cost': truck_rc,
                            'truck_rc': truck_rc,
                            'truck_cost': truck_cost,
                            'truck_demands': truck_demands,
                            'truck_route': truck_route,
                            'drone_pattern': [],
                            'status': 'SP2 Exact' 
                        }
                    col_found.append(col_info)
                        
                num_feasible_truck_routes += 1
                if check_num_col_found:
                    self.num_col_found_this_iter += 1
            else:
                # Something is wrong...
                print(idx)
                print(truck_info)
                # print(rc, ret_code, not_servable_arcs, must_serve_arcs, drone_pattern)

        return col_found, num_feasible_truck_routes

    
    def calculate_reduced_cost(self, truck_route, drone_pattern, phi_i, alpha_ij):
        rc = sum([self.prob.l_CT[i,j] for (i,j) in zip(truck_route[:-1], truck_route[1:])]) - sum([phi_i[i] for i in truck_route[1:-1]]) - sum([alpha_ij[i,j] for ((i,j),d) in drone_pattern])

        return rc




class BnBNode:

    NODE_STATUS = Enum('NodeStatus', 'INFEASIBLE FRACTIONAL INTEGER UNSOLVED')
    id = 0


    def __init__(self, prob, mp, sp, parent = None):

        self.ub = cplex.infinity
        self.lb = -cplex.infinity

        self.node_status = BnBNode.NODE_STATUS.UNSOLVED

        self.id = BnBNode.id
        BnBNode.id += 1

        if parent:
            self.parent_id = parent.id
            self.Out_arc_i = copy.deepcopy(parent.Out_arc_i)
            self.fbd_arcs = copy.deepcopy(parent.fbd_arcs)
            self.DroneOut_i = copy.deepcopy(parent.DroneOut_i)
            self.depth = parent.depth + 1
            self.lb = parent.lb

        else:
            self.parent_id = None
            self.Out_arc_i = copy.deepcopy(prob.Out_i)
            self.fbd_arcs = set()
            self.DroneOut_i = copy.deepcopy(prob.drone_avail_nodes)
            self.depth = 0

        self.prob = prob
        
        
        
        self.mp = mp
        self.sp = sp

        self.node_stats = {
            'obj_val': self.ub
        }


    def solve(self, incumbent=cplex.infinity, decomposition_level=1):
        mp = self.mp
        sp = self.sp

        prob = self.prob
        Out_i = self.Out_arc_i
        DroneOut_i = self.DroneOut_i


        self.restore()
        self.prepare()

        node_stats = self.node_stats

        node_stats['mp_time'] = 0
        node_stats['sp_time'] = 0
        node_stats['tot_sp_heu1_time'] = 0
        node_stats['tot_sp_heu2_time'] = 0
        node_stats['tot_sp_exact_time'] = 0
        node_stats['num_sp_heu1'] = 0
        node_stats['num_sp_heu2'] = 0
        node_stats['num_sp_exact'] = 0
        node_stats['tot_sp1_time'] = 0
        node_stats['tot_sp2_time'] = 0
        node_stats['cg_iterations'] = 0
        node_stats['num_truck_routes'] = 0
        node_stats['num_gen_cols'] = 0
        node_stats['num_labels'] = 0
        
        node_stats['incumbent'] = incumbent

        node_stats['cg_info'] = []

        iter = 0


        while True:
            iter += 1

            mp_start_time = time.time()
            mp.solve()
            mp_obj_val = mp.sol.get_objective_value()

            cur_iter_int_sol = False

            # Check if current solution is integer, and upddate incumbent if it is better
            self.check_node_status()

            cur_iter_int_sol = self.node_status == self.NODE_STATUS.INTEGER
            if cur_iter_int_sol and mp_obj_val < incumbent:
                node_stats['incumbent'] = mp_obj_val
                node_stats['incumbent_solution'] = self.get_solution()
                # print(mp_obj_val)

            print(f'   {"*" if cur_iter_int_sol else " "} {iter:3d} ==> obj:{mp_obj_val:.2f}', end=' | ')
            mp_end_time = time.time()



            node_stats['mp_time'] += mp_end_time - mp_start_time

            if mp.cpx.solution.get_status() == mp.cpx.solution.status.infeasible:
                break

            if all_termination_event.is_set():
                break

            phi_i, alpha_ij = mp.get_dual()

            phi_i = np.array([0] + phi_i + [0])
            np_alpha_ij = np.zeros((prob.n + 2, prob.n + 2))
            for (i,j), alpha in alpha_ij.items():
                if alpha > 0.001:
                    np_alpha_ij[i,j] = alpha


            sp_result = sp.solve(phi_i, np_alpha_ij, Out_i, DroneOut_i, decomposition_level=decomposition_level)

            gen_cols = sp_result['gen_cols']

            cg_info = {
                'phi': phi_i,
                'alpha': np_alpha_ij,
                'sp_time': sp_result['sp_time'],
                'tot_sp_heu1_time': sp_result['tot_sp_heu1_time'],
                'tot_sp_heu2_time': sp_result['tot_sp_heu2_time'],
                'tot_sp_exact_time': sp_result['tot_sp_exact_time'],
                'tot_sp1_time': sp_result['tot_sp1_time'],
                'tot_sp2_time': sp_result['tot_sp2_time'],
                'num_truck_routes': sp_result['num_truck_routes'],
                'num_gen_cols': len(gen_cols),
                'num_labels': sp_result['num_labels']
            }

            node_stats['cg_info'].append(cg_info)

            node_stats['sp_time'] += sp_result['sp_time']
            node_stats['tot_sp_heu1_time'] += sp_result['tot_sp_heu1_time']
            node_stats['tot_sp_heu2_time'] += sp_result['tot_sp_heu2_time']
            node_stats['tot_sp_exact_time'] += sp_result['tot_sp_exact_time']
            node_stats['num_sp_heu1'] += 1 if sp_result['tot_sp_heu1_time'] > 0 else 0
            node_stats['num_sp_heu2'] += 1 if sp_result['tot_sp_heu2_time'] > 0 else 0
            node_stats['num_sp_exact'] += 1 if sp_result['tot_sp_exact_time'] > 0 else 0
            node_stats['tot_sp1_time'] += sp_result['tot_sp1_time']
            node_stats['tot_sp2_time'] += sp_result['tot_sp2_time']
            node_stats['num_truck_routes'] += sp_result['num_truck_routes']
            node_stats['num_gen_cols'] += len(gen_cols)
            node_stats['num_labels'] += sp_result['num_labels']

            print(f'generated cols:{len(gen_cols):5d} | sp:{sp_result["sp_time"]:5.3f} | sp1:{sp_result["tot_sp1_time"]/(sp_result["tot_sp1_time"]+sp_result["tot_sp2_time"])*100:5.1f}% | sp2:{sp_result["tot_sp2_time"]/(sp_result["tot_sp1_time"]+sp_result["tot_sp2_time"])*100:5.1f}% | min. reduced cost:{sp_result["min_reduced_cost"]:.2f} ')

            if not sp_result['early_terminated'] and mp_obj_val + sp_result['min_reduced_cost'] > incumbent:
                break

            if len(gen_cols) > 0:
                # print(gen_cols)
                mp.add_columns(gen_cols)
            else:
                break


        # self.check_node_status()

        if self.node_status != self.NODE_STATUS.INFEASIBLE:
            self.lb = mp.get_objective_value()

        node_stats['cg_iterations'] = iter
        node_stats['node_status'] = self.node_status
        node_stats['obj_val'] = self.lb
        
        

        return

    
    def prepare(self):
        if len(self.fbd_arcs) > 0:
            for i in self.fbd_arcs:
                self.mp.cpx.variables.set_upper_bounds([(f'x_{r}', 0) for r in self.mp.route_arc_list[i]])

        self.mp.cpx.variables.set_upper_bounds([(f'y_{i}_{j}', 0) for (i,j) in self.prob.A_D if j not in self.DroneOut_i[i]])

    def restore(self):

        self.mp.cpx.variables.set_upper_bounds([(f'y_{i}_{j}', 1) for (i,j) in self.prob.A_D])

        fixed_cols = [idx for idx, ub in enumerate(self.mp.cpx.variables.get_upper_bounds(self.mp.col_begin, self.mp.cpx.variables.get_num()-1)) if ub < 1]
        if len(fixed_cols) > 0:
            self.mp.cpx.variables.set_upper_bounds([(col + self.mp.col_begin, cplex.infinity) for col in fixed_cols])

    
    def check_node_status(self):
        if self.mp.cpx.solution.get_status() == self.mp.cpx.solution.status.infeasible:
            self.node_status = self.NODE_STATUS.INFEASIBLE
        else:

            if self.mp.is_integer_solution():
                self.node_status = self.NODE_STATUS.INTEGER
            else:
                self.node_status = self.NODE_STATUS.FRACTIONAL


    def nonzero_cols(self):
        return self.mp.nonzero_cols()

    def get_solution(self):
        return self.mp.get_solution()


    def solve_heuristic(self):


        heu_obj = None
        heu_sol = None

        self.mp.solve_ip()
        if self.mp.cpx.solution.get_status() == 101:
            heu_obj = self.mp.cpx.solution.get_objective_value()
            heu_sol = self.get_solution()
        

        # Restore the master relaxation solution  
        self.mp.cpx.set_problem_type(self.mp.cpx.problem_type.LP)
        self.mp.cpx.solve()

        return heu_obj, heu_sol

    
    def create_fractional_route_nodes(self):
        mp = self.mp
        sp = self.sp
        prob = self.prob

        route_sol = [(mp.sol.get_values(f'x_{r}'), mp.routes[r])
                        for r in mp.R
                        if mp.sol.get_values(f'x_{r}') > 0.001]

        drone_sol = [(mp.sol.get_values(f'y_{i}_{j}'), (i,j)) for (i,j) in prob.A_D if mp.sol.get_values(f'y_{i}_{j}')>0.01]
        # print(route_sol)

        def drone_truck_branching():
            # Drone-Truck branching
            drone_truck_nodes = []

            for idx_r1 in range(len(route_sol)):
                v1, tr1 = route_sol[idx_r1]
                for i in tr1[1:-1]:
                    sum_flow = v1
                    for idx_r2 in range(len(route_sol)):
                        if idx_r1 != idx_r2:
                            v2, tr2 = route_sol[idx_r2]
                            for k in tr2[1:-1]:
                                if i == k:
                                    sum_flow += v2

                    if sum_flow < 0.99:
                        drone_truck_nodes.append((sum_flow, i))


            if len(drone_truck_nodes) > 0:

                # Pick the node with the most fractional value
                drone_truck_nodes =  sorted(drone_truck_nodes, key=lambda x:abs(x[0]-0.5))

                v, drone_truck_node = drone_truck_nodes[0]

                print(f' 🚚 <=> 🛩️ Truck-drone branching: {drone_truck_node=}')

                Out_arc_i1 = copy.deepcopy(self.Out_arc_i)

                # Forbid all arcs departing from drone_truck_node so that this node should be visited by a drone
                fbd_arcs_1 = [(drone_truck_node, j) for j in Out_arc_i1[drone_truck_node]]
                Out_arc_i1[drone_truck_node] = np.array([], dtype=np.int64)
                n1 = BnBNode(prob, mp, sp, self)
                n1.Out_arc_i = Out_arc_i1

                n1.fbd_arcs.update(fbd_arcs_1)

                # print(fbd_arcs_1)

                # Make this node being not availble for drones
                DroneOut_i2 = copy.deepcopy(self.DroneOut_i)

                for i, delta_i in enumerate(DroneOut_i2):
                    if drone_truck_node in delta_i:
                        DroneOut_i2[i] = np.array([j for j in delta_i if j != drone_truck_node], dtype=np.int64)


                n2 = BnBNode(prob, mp, sp, self)
                n2.DroneOut_i = DroneOut_i2

                return n2, n1
            else:
                return None, None


        def fractional_truck_flow_divergence_branching():
            divergence_nodes = []

            for idx_r1 in range(len(route_sol)):
                v1, tr1 = route_sol[idx_r1]
                for i,j in zip(tr1[1:-1], tr1[2:]):
                    for idx_r2 in range(idx_r1+1, len(route_sol)):
                        v2, tr2 = route_sol[idx_r2]
                        for k,l in zip(tr2[1:-1], tr2[2:]):
                            if i == k and j != l:
                                # Found divergence node
                                divergence_nodes.append((v1+v2, i, j, l))


            if len(divergence_nodes) > 0:
                # Fractional truck flow branching

                # Sort by sum of values
                divergence_nodes = sorted(divergence_nodes, key=lambda x:x[0], reverse=True)
                divergence_node, j1, j2 = divergence_nodes[0][1:]

                print(f' 🚚 -< 🚚 Truck flow branching: {divergence_node=}, {j1=}, {j2=}')


                delta_divergence_node = list(self.Out_arc_i[divergence_node].copy())
                # print(self.Out_arc_i[divergence_node])

                delta_divergence_node.remove(j1)
                delta_divergence_node.remove(j2)

                fbd_arcs_1 = [(divergence_node, j1)]
                fbd_arcs_2 = [(divergence_node, j2)]

                Out_arc_i1 = copy.deepcopy(self.Out_arc_i)
                Out_arc_i2 = copy.deepcopy(self.Out_arc_i)

                for idx_j, j in enumerate(delta_divergence_node):
                    if idx_j % 2 == 0:
                        fbd_arcs_1.append((divergence_node, j))
                    else:
                        fbd_arcs_2.append((divergence_node, j))

                Out_arc_i1[divergence_node] = np.array([arc[1] for arc in fbd_arcs_2], dtype=np.int64)
                Out_arc_i2[divergence_node] = np.array([arc[1] for arc in fbd_arcs_1], dtype=np.int64)

                n1 = BnBNode(prob, mp, sp, self)
                n1.Out_arc_i = Out_arc_i1

                n2 = BnBNode(prob, mp, sp, self)
                n2.Out_arc_i = Out_arc_i2

                # print(Out_arc_i1[divergence_node])
                # print(Out_arc_i2[divergence_node])

                n1.fbd_arcs.update(fbd_arcs_1)
                n2.fbd_arcs.update(fbd_arcs_2)

                # print(fbd_arcs_1)
                # print(fbd_arcs_2)

                return n2, n1
            else:
                return None, None


        def fractional_truck_flow_convergence_branching():
            convergence_nodes = []

            for idx_r1 in range(len(route_sol)):
                v1, tr1 = route_sol[idx_r1]
                for i,j in zip(tr1[:-2], tr1[1:-1]):
                    for idx_r2 in range(idx_r1+1, len(route_sol)):
                        v2, tr2 = route_sol[idx_r2]
                        for k,l in zip(tr2[:-2], tr2[1:-1]):
                            if i != k and j == l:
                                # Found divergence node
                                convergence_nodes.append((v1+v2, i, k, j))

            if len(convergence_nodes) > 0:

                # Sort by sum of values
                convergence_nodes = sorted(convergence_nodes, key=lambda x:x[0], reverse=True)
                i1, i2, convergence_node = convergence_nodes[0][1:]

                print(f' 🚚 >- 🚚 Truck flow branching: {convergence_node=}, {i1=}, {i2=}')

                fbd_arcs_1 = [(i1, convergence_node)]
                fbd_arcs_2 = [(i2, convergence_node)]

                Out_arc_i1 = copy.deepcopy(self.Out_arc_i)
                Out_arc_i2 = copy.deepcopy(self.Out_arc_i)

                Out_arc_i1[i1] = np.array([j for j in Out_arc_i1[i1] if j!=convergence_node], dtype=np.int64)
                Out_arc_i2[i2] = np.array([j for j in Out_arc_i2[i2] if j!=convergence_node], dtype=np.int64)

                delta_convergence_node = list([i for (i, adj_i) in enumerate(self.Out_arc_i) if convergence_node in adj_i])
                # print(self.Out_arc_i[divergence_node])

                delta_convergence_node.remove(i1)
                delta_convergence_node.remove(i2)

                for idx_i, i in enumerate(delta_convergence_node):
                    if idx_i % 2 == 0:
                        fbd_arcs_1.append((i, convergence_node))
                        Out_arc_i1[i] = np.array([j for j in Out_arc_i1[i] if j!=convergence_node], dtype=np.int64)
                    else:
                        fbd_arcs_2.append((i, convergence_node))
                        Out_arc_i2[i] = np.array([j for j in Out_arc_i2[i] if j!=convergence_node], dtype=np.int64)

                
                n1 = BnBNode(prob, mp, sp, self)
                n1.Out_arc_i = Out_arc_i1

                n2 = BnBNode(prob, mp, sp, self)
                n2.Out_arc_i = Out_arc_i2

                # print(Out_arc_i1[divergence_node])
                # print(Out_arc_i2[divergence_node])

                n1.fbd_arcs.update(fbd_arcs_1)
                n2.fbd_arcs.update(fbd_arcs_2)

                # print(fbd_arcs_1)
                # print(fbd_arcs_2)

                return n2, n1
            else:
                return None, None
        

        def drone_branching():
            # Drone branching

            multiple_drone_nodes = []

            # Find positive drone arcs to serve the same node
            for idx1 in range(len(drone_sol)-1):
                v1, (i,j) = drone_sol[idx1]
                for idx2 in range(idx1+1, len(drone_sol)):
                    v2, (k,l) = drone_sol[idx2]
                    if j == l:
                        multiple_drone_nodes.append((i,k,l))

            if len(multiple_drone_nodes) > 0:

                i1, i2, drone_node = multiple_drone_nodes[0]

                print(f' 🛩️ >-< 🛩️ Drone node branching: {drone_node=}, {i1=}, {i2=}')

                # Make two partitions of drones arcs
                DroneOut_i1 = copy.deepcopy(self.DroneOut_i)
                DroneOut_i2 = copy.deepcopy(self.DroneOut_i)

                DroneOut_i1[i1] = np.array([j for j in DroneOut_i1[i1] if j!=drone_node], dtype=np.int64)
                DroneOut_i2[i2] = np.array([j for j in DroneOut_i2[i2] if j!=drone_node], dtype=np.int64)

                delta_drone_node = list([i for (i, adj_i) in enumerate(self.DroneOut_i) if drone_node in adj_i])

                delta_drone_node.remove(i1)
                delta_drone_node.remove(i2)

                for idx_i, i in enumerate(delta_drone_node):
                    if idx_i % 2 == 0:
                        DroneOut_i1[i] = np.array([j for j in DroneOut_i1[i] if j!=drone_node], dtype=np.int64)
                    else:
                        DroneOut_i2[i] = np.array([j for j in DroneOut_i2[i] if j!=drone_node], dtype=np.int64)

                n1 = BnBNode(prob, mp, sp, self)
                n1.DroneOut_i = DroneOut_i1

                n2 = BnBNode(prob, mp, sp, self)
                n2.DroneOut_i = DroneOut_i2

                return n2, n1
            else:
                return None, None


        n1, n2 = fractional_truck_flow_divergence_branching()
        if n1 is None or n2 is None:
            n1, n2 = fractional_truck_flow_convergence_branching()
            if n1 is None or n2 is None:
                n1, n2 = drone_truck_branching()
                if n1 is None or n2 is None:
                    n1, n2 = drone_branching()
                    if n1 is None or n2 is None:

                        # This should not happen!
                        print('@@@@@@')
                        
                        print(drone_sol)

                        route_sol = [(mp.sol.get_values(f'x_{r}'), mp.routes[r], mp.drone_patterns[r])
                            for r in mp.R
                            if mp.sol.get_values(f'x_{r}') > 0.001]
                        
                        print(route_sol)

                        return None, None

        return n2, n1    


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)


class BnPSolver:
    def __init__(self, prob, num_cpus=8, num_cols_early_termination=-1, sp_time_limit=-1, max_num_labels=1000000):

        self.prob = prob

        self.num_cpus = num_cpus

        self.mp = MP(prob)
        self.sp = SP(prob, num_cpus=num_cpus, sp_time_limit=sp_time_limit, num_cols_early_termination=num_cols_early_termination if num_cols_early_termination>0 else prob.num_nodes, max_num_labels=max_num_labels)

        sp1_termination_event.clear()
        all_termination_event.clear()

        # Generate initial columns
        self.gen_init_cols()


    def gen_init_cols(self):
        prob = self.prob
        mp = self.mp
        sp = self.sp

        s = 0
        t = prob.n+1


        init_truck_routes = []

        for i in prob.N:
            init_truck_route = np.array([t,i,s])[::-1] # To prevent numba from compile different function signature...
            init_truck_route_obj = sum([prob.l_CT[i,j] for (i,j) in zip(init_truck_route[:-1], init_truck_route[1:])]) 
            init_drone_pattern = ()

            init_truck_routes.append((init_truck_route, init_truck_route_obj, prob.demand[i], -1.0))



        nb_DroneOut_i = List(prob.drone_avail_nodes)

        np_alpha_ij = np.zeros((prob.n + 2, prob.n + 2))

        for (i,j) in prob.A_D:
            np_alpha_ij[i,j] = 10000.0





        init_cols, _ = sp.solve_sp2(tuple(init_truck_routes), np_alpha_ij, nb_DroneOut_i, check_num_col_found=False, aggressive=True)
        mp.add_columns(init_cols)


        # Make sure njit functions compiled
        sp.solve(np.zeros(prob.n+2), np.zeros((prob.n+2,prob.n+2)), List(prob.Out_i), nb_DroneOut_i)

    def solve(self, time_limit=3600, decomposition_level=1):
        prob = self.prob
        mp = self.mp
        sp = self.sp



        BnBNode.id = 0
        nodes = PriorityQueue()
        incumbent_value = cplex.infinity
        best_solution = None
        best_node = None

        elapsed_time = 0

        tot_cg_iterations = 0
        tot_heu_time = 0
        tot_mp_time = 0
        tot_sp_time = 0
        tot_sp_heu1_time = 0
        tot_sp_heu2_time = 0
        tot_sp_exact_time = 0
        tot_num_sp_heu1 = 0
        tot_num_sp_heu2 = 0
        tot_num_sp_exact = 0
        tot_sp1_time = 0
        tot_sp2_time = 0
        tot_num_gen_cols = 0
        tot_num_labels = 0

        results = {
            'problem': prob.prob_info(),
            'timelimit': time_limit,
            'num_cpus': sp.num_cpus,
            'alg': 'BnP'
        }



        root_node = BnBNode(prob=prob, mp=mp, sp=sp, parent=None)

        global_lb = -cplex.infinity

        nodes.put(PrioritizedItem(root_node.lb, root_node))
        num_nodes = 1


        def timer_limit_reached():  
            print("Time limit!!!")
            all_termination_event.set()
            S.cancel()

        S = threading.Timer(time_limit, timer_limit_reached)
        S.start()

        start_time = time.time()


        while not nodes.empty() and elapsed_time <= time_limit:
            best_node = nodes.get()

            self.cur_node = best_node

            node = best_node.item

            if node.lb >= incumbent_value:
                global_lb = incumbent_value
                continue

            global_lb = best_node.priority

            now_time = time.time()
            elapsed_time = now_time - start_time

            # if not detail_log and (now_time - last_log_time) > 10:
            last_log_time = now_time
            if incumbent_value < cplex.infinity and global_lb > -cplex.infinity:
                gap = ((incumbent_value - global_lb) / incumbent_value) * 100
            else:
                gap = -1.0
            print(f"👉 Node {node.id:3d} [{nodes.qsize():3d} remained] ------------------ {elapsed_time:.2f} sec. incumbent: {incumbent_value:.1f}, GAP: {gap:.2f}")

            node.solve(incumbent=incumbent_value, decomposition_level=decomposition_level)
            node_stats = node.node_stats

            tot_cg_iterations += node_stats['cg_iterations']
            tot_mp_time += node_stats['mp_time']
            tot_sp_time += node_stats['sp_time']
            tot_sp_heu1_time += node_stats['tot_sp_heu1_time']
            tot_sp_heu2_time += node_stats['tot_sp_heu2_time']
            tot_sp_exact_time += node_stats['tot_sp_exact_time']
            tot_num_sp_heu1 += node_stats['num_sp_heu1']
            tot_num_sp_heu2 += node_stats['num_sp_heu2']
            tot_num_sp_exact += node_stats['num_sp_exact']
            tot_sp1_time += node_stats['tot_sp1_time']
            tot_sp2_time += node_stats['tot_sp2_time']
            tot_num_gen_cols += node_stats['num_gen_cols']
            tot_num_labels += node_stats['num_labels']




            if node.id == 0:
                results['root_LP'] = node.lb
                global_lb = node.lb

            if node_stats['incumbent'] < incumbent_value:
                incumbent_value = node_stats['incumbent']
                best_node = node
                best_solution = node_stats['incumbent_solution']

            heu_time = time.time()
            heu_obj, heu_sol = node.solve_heuristic()
            # print(heu_obj, heu_sol)
            if heu_obj is not None and heu_obj < incumbent_value - 0.001:
                print(f'  👍 Heuristic found a new incumbent! obj:{heu_obj:.2f}')
                incumbent_value = heu_obj
                best_node = node
                best_solution = heu_sol
            tot_heu_time += time.time() - heu_time


            if all_termination_event.is_set():
                break

            if node.node_status is node.NODE_STATUS.INTEGER:
                obj_val = node.lb

                print(f'  👍 Integer solution!!! obj:{obj_val:.2f}', )

                if obj_val < incumbent_value:
                    incumbent_value = obj_val
                    best_node = node
                    best_solution = node.get_solution()

            elif node.node_status is node.NODE_STATUS.FRACTIONAL:

                obj_val = node.lb

                if obj_val - incumbent_value <= -1.0:
                    print('  😿  Fractional solution...', end='')
                    
                    n1, n2 = node.create_fractional_route_nodes()
                    # print("@@@@@@@@@@@@")
                    # ray.shutdown()
                    # break
                                        
                    if n1 is not None and n2 is not None:
                        nodes.put(PrioritizedItem(n1.lb, n1))
                        nodes.put(PrioritizedItem(n2.lb, n2))
                        num_nodes += 2
                    else:
                        print('?????????')
                        node.mp.cpx.write('error.lp')
                        pickle.dump(node, open('error_node.dmp', 'wb'))

                else:
                    print('  👍  Pruned!')

            elif node.node_status is node.NODE_STATUS.INFEASIBLE:
                print('  👍 Infeasible...')

        end_time = time.time()

        S.cancel()

        results['tot_cg_iterations'] = tot_cg_iterations
        results['global_lb'] = global_lb
        results['time'] = end_time - start_time
        results['obj'] = incumbent_value
        results['tot_mp_time'] = tot_mp_time
        results['tot_sp_time'] = tot_sp_time
        results['tot_sp_heu1_time'] = tot_sp_heu1_time
        results['tot_sp_heu2_time'] = tot_sp_heu2_time
        results['tot_sp_exact_time'] = tot_sp_exact_time
        results['tot_num_sp_heu1'] = tot_num_sp_heu1
        results['tot_num_sp_heu2'] = tot_num_sp_heu2
        results['tot_num_sp_exact'] = tot_num_sp_exact
        results['tot_sp1_time'] = tot_sp1_time
        results['tot_sp2_time'] = tot_sp2_time
        results['tot_heu_time'] = tot_heu_time
        results['raw_solution'] = best_solution
        if best_solution is not None:
            results['solution'] = self.get_refined_solution(best_solution)
        else:
            results['solution'] = None
        results['num_bnb_nodes'] = num_nodes
        results['num_bnb_remaining_nodes'] = nodes.qsize()
        results['tot_num_gen_cols'] = tot_num_gen_cols
        results['tot_num_labels'] = tot_num_labels
        results['termination_status'] = 'Time limit' if all_termination_event.is_set() else 'Optimal'


        return results
    
    def get_refined_solution(self, raw_solution):

        route_pattern_sol, drone_sol = raw_solution

        drone_sol_set = {(i,j) for v,(i,j) in drone_sol}

        sol = []

        for v,tr,dp in route_pattern_sol:
            drone_dispatches = []
            for d in self.prob.D:
                this_drone_dispatches = []
                drone_dispatches.append(this_drone_dispatches)
                for (i,j),d2 in dp:
                    if (i,j) in drone_sol_set and d == d2:
                        this_drone_dispatches.append((i,j))


            sol.append((tr, drone_dispatches))

        return sol



class HeuristicSolver:
    def __init__(self, prob, num_cpus=8, num_cols_early_termination=-1, sp_time_limit=-1, max_num_labels=1000000):

        self.prob = prob

        self.num_cpus = num_cpus

        self.mp = MP(prob)
        self.sp = SP(prob, num_cpus=num_cpus, sp_time_limit=sp_time_limit, num_cols_early_termination=num_cols_early_termination if num_cols_early_termination>0 else prob.num_nodes, max_num_labels=max_num_labels)

        sp1_termination_event.clear()
        all_termination_event.clear()

        # Generate initial columns
        self.gen_init_cols()


    def gen_init_cols(self):
        prob = self.prob
        mp = self.mp
        sp = self.sp

        s = 0
        t = prob.n+1


        init_truck_routes = []

        for i in prob.N:
            init_truck_route = np.array([t,i,s])[::-1] # To prevent numba from compile different function signature...
            init_truck_route_obj = sum([prob.l_CT[i,j] for (i,j) in zip(init_truck_route[:-1], init_truck_route[1:])]) 
            init_drone_pattern = ()

            init_truck_routes.append((init_truck_route, init_truck_route_obj, prob.demand[i], -1.0))

        nb_DroneOut_i = List(prob.drone_avail_nodes)

        np_alpha_ij = np.zeros((prob.n + 2, prob.n + 2))

        for (i,j) in prob.A_D:
            np_alpha_ij[i,j] = 10000.0


        init_cols, _ = sp.solve_sp2(tuple(init_truck_routes), np_alpha_ij, nb_DroneOut_i, check_num_col_found=False, aggressive=True)
        mp.add_columns(init_cols)


        # Make sure njit functions compiled
        sp.solve(np.zeros(prob.n+2), np.zeros((prob.n+2,prob.n+2)), List(prob.Out_i), nb_DroneOut_i)

    def get_refined_solution(self, raw_solution):

        route_pattern_sol, drone_sol = raw_solution

        drone_sol_set = {(i,j) for v,(i,j) in drone_sol}

        sol = []

        for v,tr,dp in route_pattern_sol:
            drone_dispatches = []
            for d in self.prob.D:
                this_drone_dispatches = []
                drone_dispatches.append(this_drone_dispatches)
                for (i,j),d2 in dp:
                    if (i,j) in drone_sol_set and d == d2:
                        this_drone_dispatches.append((i,j))


            sol.append((tr, drone_dispatches))

        return sol


    def solve(self, time_limit=600, decomposition_level=1):

        prob = self.prob
        sp = self.sp
        mp = self.mp

        Out_i = prob.Out_i
        DroneOut_i = prob.drone_avail_nodes

        results = {
            'problem': prob.prob_info(),
            'timelimit': time_limit,
            'num_cpus': sp.num_cpus,
            'alg': 'CG Heuristic'
        }
        tot_heu_time = 0
        tot_mp_time = 0
        tot_sp_time = 0
        tot_sp1_time = 0
        tot_sp2_time = 0
        tot_num_gen_cols = 0
        tot_num_labels = 0

        iter = 0

        fixed_routes = set()

        best_heu_obj = cplex.infinity
        best_heu_sol = None

        def timer_limit_reached():  
            print("Time limit!!!")
            sp1_termination_event.set()
            all_termination_event.set()
            S.cancel()

        S = threading.Timer(time_limit, timer_limit_reached)
        S.start()

        start_time = time.time()
        elapsed_time = 0

        while elapsed_time <= time_limit:

            iter += 1

            mp_start_time = time.time()
            mp.solve()
            mp_obj_val = mp.sol.get_objective_value()

            tot_mp_time += time.time() - mp_start_time

            cur_iter_int_sol = False

            cur_iter_int_sol = mp.is_integer_solution()

            if cur_iter_int_sol and mp_obj_val < best_heu_obj - 0.1:
                best_heu_obj = mp_obj_val
                best_heu_sol = mp.get_solution()


            print(f'   {"*" if cur_iter_int_sol else " "} {iter:3d} ==> obj:{mp_obj_val:9.2f}', end=' | ')
            mp_end_time = time.time()

            phi_i, alpha_ij = mp.get_dual()

            phi_i = np.array([0] + phi_i + [0])
            np_alpha_ij = np.zeros((prob.n + 2, prob.n + 2))
            for (i,j), alpha in alpha_ij.items():
                if alpha > 0.001:
                    np_alpha_ij[i,j] = alpha


            sp_result = sp.solve(phi_i, np_alpha_ij, Out_i, DroneOut_i, decomposition_level=decomposition_level, ensure_exact_solving=False)

            gen_cols = sp_result['gen_cols']

            tot_sp_time += sp_result['sp_time']
            tot_sp1_time += sp_result['tot_sp1_time']
            tot_sp2_time += sp_result['tot_sp2_time']
            tot_num_gen_cols += len(sp_result['gen_cols'])
            tot_num_labels += sp_result['num_labels']

            now_time = time.time()
            elapsed_time = now_time - start_time

            print(f'generated cols:{len(gen_cols):5d} | sp:{sp_result["sp_time"]:5.3f} | sp1:{sp_result["tot_sp1_time"]/(sp_result["tot_sp1_time"]+sp_result["tot_sp2_time"])*100:5.1f}% | sp2:{sp_result["tot_sp2_time"]/(sp_result["tot_sp1_time"]+sp_result["tot_sp2_time"])*100:5.1f}% | min. reduced cost:{sp_result["min_reduced_cost"]:9.2f} | Els. time:{elapsed_time:6.2f} | Incumbent: {best_heu_obj:9.2f}')


            gen_cols = sp_result['gen_cols']
            if len(gen_cols) > 0:
                mp.add_columns(gen_cols)
            else:

                frac_route_sol = [(mp.sol.get_values(f'x_{r}'), r, mp.routes[r])
                                for r in mp.R
                                if is_frac(mp.sol.get_values(f'x_{r}')) and r not in fixed_routes]

                if len(frac_route_sol) > 0:

                    heu_time = time.time()
                    mp.solve_ip()
                    heu_obj = mp.cpx.solution.get_objective_value()

                    if heu_obj < best_heu_obj - 0.1:
                        best_heu_obj = heu_obj
                        best_heu_sol = mp.get_solution()

                        print(f'👍 Best Heuristic obj!! ==> {heu_obj:2f}')
                    tot_heu_time += time.time() - heu_time

                    for r in mp.R:
                        mp.cpx.variables.set_types(f'x_{r}', mp.cpx.variables.type.continuous)

                    mp.cpx.set_problem_type(cplex.Cplex.problem_type.LP)



                    for r in fixed_routes:
                        mp.cpx.variables.set_lower_bounds(f'x_{r}', 0)

                    frac_route_sol = sorted(frac_route_sol, key=lambda x:-x[0])
                    fix_r = frac_route_sol[0][1]
                    mp.cpx.variables.set_lower_bounds(f'x_{fix_r}', 1)

                    print(f'      🔒 Fix route {fix_r}: {frac_route_sol[0]}')

                    fixed_routes.add(fix_r)
                else:
                    break

            
        heu_time = time.time()
        mp.solve_ip()
        heu_obj = mp.cpx.solution.get_objective_value()

        if heu_obj < best_heu_obj - 0.1:
            best_heu_obj = heu_obj
            best_heu_sol = mp.get_solution()

            print(f'👍 Best Heuristic obj!! ==> {heu_obj:2f}')
        tot_heu_time += time.time() - heu_time
                    
        S.cancel()

        end_time = time.time()

        results['tot_cg_iterations'] = iter
        results['time'] = end_time - start_time
        results['obj'] = best_heu_obj
        results['tot_heu_time'] = tot_heu_time
        results['tot_mp_time'] = tot_mp_time
        results['tot_sp_time'] = tot_sp_time
        results['tot_sp1_time'] = tot_sp1_time
        results['tot_sp2_time'] = tot_sp2_time
        results['raw_solution'] = best_heu_sol
        if best_heu_sol is not None:
            results['solution'] = self.get_refined_solution(best_heu_sol)
        else:
            results['solution'] = None
        results['tot_num_gen_cols'] = tot_num_gen_cols
        results['tot_num_labels'] = tot_num_labels
        results['tot_num_route_fixes'] = len(fixed_routes)

        return results
            




if __name__ == '__main__':


    datafile = 'problems/solomon/25/R101.txt'

    prob = Prob(datafile=datafile,
        T_Gamma = 2, t_range = 0.25,
        D_Delta = 2, d_range = 0.25,
        num_drones = 2,
        drone_time_multiplier = 0.25,
        drone_cost_multiplier = 0.25,
        flight_limit = 25,
        demand_ratio = 0.75,

    )


    bnp_solver = BnPSolver(prob, num_cpus=8, sp_time_limit=10)
    bnp_results = bnp_solver.solve(time_limit=3600, decomposition_level=1)
    