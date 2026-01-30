# Multiobjective Multidimensional Optimization Algorithms Analysis

# Multi-Objective 0–1 Knapsack Optimization

This repository contains Python implementations of multiple exact and algorithmic methods for solving the **multi-objective, multi-constraint 0–1 knapsack problem**, with the goal of computing the complete set of **non-dominated (Pareto-optimal) solutions**.

The project explores brute-force enumeration, geometric partitioning, and region-based optimization techniques, integrating **Gurobi** for lexicographic and weighted-sum optimization.

---

## Problem Overview

We consider the multi-objective 0–1 knapsack problem with:

- Binary decision variables  
- Multiple linear constraints  
- Multiple conflicting objective functions  

The objective is to enumerate all **non-dominated points (NDPs)** in objective space.  
This problem is NP-hard and becomes computationally challenging as problem size and dimensionality increase.

---

## Implemented Methods

### 1. Brute Force Enumeration (BF)
- Enumerates all feasible binary solutions
- Computes objective values and removes dominated points
- Guarantees correctness but scales exponentially

### 2. Rectangle Division Method (RDM)
- Exact geometric method for bi-objective problems
- Recursively partitions the objective space into rectangles
- Uses lexicographic optimization to locate extreme points

### 3. Supernal Point Method (SPM)
- Region-based algorithm for multi-objective optimization
- Iteratively explores regions defined by supernal points
- Uses weighted-sum optimization to discover new NDPs

### 4. COMP_2D (Improved SPM – 2 Objectives)
- Enhanced supernal-point approach for bi-objective problems
- Dynamically adjusts objective weights
- Randomized region selection and dominance pruning

### 5. COMP_3D (Improved SPM – 3+ Objectives)
- Extension of COMP_2D to higher-dimensional objective spaces
- Generalized region pruning and dominance filtering

---

## Input Format

Input files follow this structure:

1. Number of items `n`
2. Constraint bounds `b`
3. Objective coefficient vectors (negative values)
4. Constraint coefficient vectors

Example:
5

10

-3 -2 -4 -1 -5

-2 -3 -1 -4 -2

2 1 3 2 1

---

## Usage

### Requirements
- Python 3.x
- NumPy
- SciPy
- Pandas
- Gurobi Optimizer (licensed)

### Running the Solver

```python
SolveKnapsack("input.txt", method=2)
```

### Method options

1 – Brute Force (BF)
2 – Rectangle Division Method (RDM)
3 – Supernal Point Method (SPM)
4 – COMP_2D
5 – COMP_3D

### Output

Each run generates:
- A file containing lexicographically sorted non-dominated points
- A summary file containing:
  - Runtime (seconds)
  - Number of NDPs found
  - Number of regions or rectangles explored

### Technologies Used:
- Python
- Gurobi Optimizer
- NumPy
- SciPy
- Pandas
- Integer Linear Programming (ILP)

---

## Author
Victoria Piroian

University of Toronto

Faculty of Applied Science & Engineering, 2023



