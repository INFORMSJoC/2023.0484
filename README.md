[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Branch-and-Price Algorithm for Robust Drone-Vehicle Routing Problem with Time Windows

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper [A Branch-and-Price Algorithm for Robust Drone-Vehicle Routing Problem with Time Windows](https://doi.org/10.1287/ijoc.2023.0484) by Jaegwan Joo and Chungmok Lee.

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0484

https://doi.org/10.1287/ijoc.2024.0484.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{Joo2024,
  author =        {Joo, Jaegwan and Lee, Chungmok},
  publisher =     {INFORMS Journal on Computing},
  title =         {A Branch-and-Price Algorithm for Robust Drone-Vehicle Routing Problem with Time Windows},
  year =          {2024},
  doi =           {10.1287/ijoc.2024.0484.cd},
  url =           {https://github.com/INFORMSJoC/2024.0484},
  note =          {Available for download at \url{https://github.com/INFORMSJoC/2024.0484}},
}
```

## Description

The goal of this repository is to share data and results related to the Robust Drone-Vehicle Routing Problem with Time Windows (RDVRRPTW), solved using our solution approaches.

## Building

We develop exact and heuristic algorithms to solve the Robust Drone-Vehicle Routing Problem with Time Windows (RDVRRPTW) through a Branch-and-Price framework. Additionally, we decompose the column generation subproblem into two optimization problems, resulting in a two-phase column generation algorithm.

### Prerequisites

- Python 3.8 or higher
- Required Libraries: Cplex, NumPy, Numba

### Structure

The source code is available in the [src](src) directory and includes the following components:

- `main.py`: 
  - Defines the `Prob` class, which manages the problem data.
  - Contains the `MIPSolver` class, which solve MIP model.
  - Contains the `BnPSolver` class, which solve Branch-and-Price algorithm.
  - Contains the `HeuristicSolver` class, which solve CG-based Heuristic algorithm.
- `mp_root.py`: Define the `MP` class, which solve the extended master formulation (MP).
- `mpsp_root.py`: Define the `MP` class, which solve the set-partitioning master formulation (MP-SP).

### Run

At the bottom of each python file, set the path and required parameter values for the instance as follows:
```
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

```

Then, excute the program as follows:

```bash
python main.py
```

#### Parameter lists

- "T_Gamma": *𝚪*, 
- "t_range": *coefficient for truck's travel time uncertainty*, 
- "D_Delta": *𝛥*, 
- "d_range": *coefficient for drone's flight time uncertainty*, 
- "num_drones": *number of drones*, 
- "drone_time_multiplier": *a constant to be multiplied to drone's flight time*, 
- "drone_cost_multiplier": *a constant to be multiplied to drone's cost*, 
- "flight_limit": *drone's maximum flight range*, 
- "demand_ratio": *drone's demand limit ratio*

## Results

The results present the obtained solutions for all instances, solved using our solution approach (MIP, BnP and CG-based based Heuristic).

Please see the [results](results) directory to view the solutions and detailed descriptions.

## Data

- UPS Instances: Randomly generated from a real-life use case of UPS in the area of NC, US. Refer to Kang and Lee (2020) for details.
- Solomon Instances: Well-known VRPTW instances by Solomon (1987).

### References

- Kang, Munjeong, and Chungmok Lee. "An exact algorithm for heterogeneous drone-truck routing problem." Transportation Science 55, no. 5 (2021): 1088-1112.
- Solomon,  Marius M. "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints." Operations Research 35, no, 2 (1987): 254–265.

## Results

The results present the solutions for all instances, solved using our solution approach.

Please see the [results](results) directory to view the solutions and detailed descriptions.
