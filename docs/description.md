# IA - Assignment 2

The goal of the project is to simulate a real machine learning consulting project by identifying a practical problem, building a proof of concept with artificial data, and presenting it through a web app. The goal is to understand both the technical and practical challenges of delevering machine learning solutions in a real-world context.

## Problem Description

### Client

The client for this project is Grupo Brisa, one of the largest higway operators in Portugal, responsible for managing motorway infrastructure, traffic monitoring, and emergency response coordination.

### Context

Highway operators such as Grupo Brisa must continuously monitor traffic and weather conditions to ensure road safety and efficient emergency response management.
Weather conditions can significantly influence accident occurence on highways. Heayy rain, low visibility, strong winds, and other adverse meteorological conditions increase the probability of road incidents and directly impact operational planning.

The objective is to combine meteorological information with historical road accident records to identify patterns that help anticipate critical situations and support faster, more efficient responses from highway operators.

### Problem Statement

Grupo Brisa must efficiently allocate emergency response vehicles accros the motorway network to guarantee rapid intervention during road accidents.

Each accident requires the deployent of three emergency vehicles. As a result, accurately forecasting the expected number of daily accidents is essential for operational planning, vehicle allocation, and emergency readiness.

The system should help highway operators:

- Anticipate periods of elevated accident risk
- Improve emergency vehicle allocation
- Reduce response times during critical situations
- Optimize operational resource management

### Proposed ML Solution

The goal is to develop a machine learning model capable of forecasting the daily number of road accidents on Portuguese highways.

The dataset contains 10,000 generated rows designed to simulate realistic motorway conditions.
The model combines:

- Temporal features
- Meteorological conditions
- Infrastructure-related variables
- Traffic density patterns

### Target Variable

`accident_count` - Numerical target representing the number of accidents expected under a specific set of road, traffic, weather, and infrastructure conditions.

| Variable         | Meaning                                                 |
| ---------------- | ------------------------------------------------------- |
| `accident_count` | Expected number of accidents for the simulated scenario |

### Features

The final dataset contains 12 input feature grouped into four categories:

**Temporal Features**

| Feature   | Type      | Description                                   |
| --------- | --------- | --------------------------------------------- |
| `hour`    | Numerical | Hour of the day                               |
| `weekday` | Numerical | Day of the week                               |
| `month`   | Numerical | Month of the year                             |
| `holiday` | Binary    | Indicates whether the day is a public holiday |

**Meteorological Conditions**

| Feature            | Type      | Description                        |
| ------------------ | --------- | ---------------------------------- |
| `precipitation_mm` | Numerical | Total precipitation in millimeters |
| `visibility_km`    | Numerical | Visibility in kilometers           |
| `temperature_c`    | Numerical | Temperature in degrees Celsius     |
| `wind_speed_kmh`   | Numerical | Wind speed in kilometers per hour  |

**Traffic Features**

| Feature           | Type      | Description                    |
| ----------------- | --------- | ------------------------------ |
| `traffic_density` | Numerical | Traffic density in the highway |

**Infrastructure Features**

| Feature       | Type      | Description                                         |
| ------------- | --------- | --------------------------------------------------- |
| `num_lanes`   | Numerical | Number of motorway lanes                            |
| `ilumination` | Binary    | Indicates whether the road section has ilumnination |
| `work_zone `  | Binary    | Indicates active road works                         |
