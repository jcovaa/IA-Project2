# IA - Assignment 2

The goal of the project is to simulate a real machine learning consulting project by identifying a practical problem, building a proof of concept with artificial data, and presenting it through a web app. The goal is to understand both the technical and practical challenges of delevering machine learning solutions in a real-world context.

## Problem Description

### Client

Amkor Technology is one of the world's largest providers of semiconductor packaging and test services. The contact person for this project is a Volume Planner within the company's supply chain planning department.

### Context

The volume planner department is responsible for forecasting and managing the quantity of materials and production capacity required to fulfill client orders. Volume planners work closely with two other departments:

- **Material Planners** - Responsible for procuring and managing raw materials and components.
- **Capacity Planners** - Responsible for ensuring that production capacity is available to meet demand.

Clients served by the volume planning team fall into two categories:

- **High-volume clients** - Large accounts with significant and relatively predictable demand.
- **Low-volume clients** - Smaller accounts with more variable and harder-to-predict demand.

Additionally, clients follow one of two planning approaches:

- **Weekly budget planning** - Clients submit a demand forecast on a weekly basis, allowing for more frequent adjustments.
- **Annual budget planning** - Clients submit a single yearly forecast upfront, which remains largely fixed throughout the year.

### Problem Statement

A recurring challenge in volume planning is the **gap between a client's planned volume and their actual volume consuption.**
Clients frequently deviate from their submitted forecasts (ordering more or less than planned), which create problems across the supply chain:

- Stock shortages
- Excess inventory

### Proposed ML Solution

The goal is to develop a **predictive model** that estimates the expected deviation between a client's planned volume and their actual volume consumption for a given material and time period.

Given features such as client type, planning method, current stock, material characteristics, and available production capacity, the model will classify each client-material-period combination into one of three risk categories: **under_budget**, **on_track**, or **over_budget**.

This classification gives volume planners an immediate, actionable risk signal, allowing them to adjust orders, flag high-risk clients, and coordinate earlier with material and capacity planners, improving overall supply chain stability.

### Target Variable

`deviation_class` - A 3-class label indicating whether the client's actual consumption will fall **below**, or **above** their planned budget for a given period.

| Class          | Meaning                           | Business Impact                            |
| -------------- | --------------------------------- | ------------------------------------------ |
| `under_budget` | Client consumes less than planned | Excess stock, wasted capacity reservation  |
| `on_track`     | Client consumes close to planned  | No action needed                           |
| `over_budget`  | Client consumes more than planned | Stock shortage risk, urgent reorder needed |

### Features

| Feature              | Type        | Description                                     |
| -------------------- | ----------- | ----------------------------------------------- |
| `client_type`        | Categorical | High-volume or small-volume client              |
| `planning_type`      | Categorical | Weekly or annual budget planning                |
| `material`           | Categorical | Type of material being planned                  |
| `planned_volume`     | Numerical   | Volume commited by the client for the period    |
| `stock_level`        | Numerical   | Current available stock for the material        |
| `week_of_year`       | Numerical   | Week number, capturing seasonal demand patterns |
| `buy_price`          | Numerical   | Unit purchase cost of the material              |
| `capacity_available` | Numerical   | Production capacity available for the period    |
