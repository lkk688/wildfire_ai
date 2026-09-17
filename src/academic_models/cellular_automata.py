"""
2D Stochastic Cellular Automata (CA) Wildfire Simulator
Academic Discrete Spatial Modeling Baseline (Alexandridis et al., 2008).

Cell states:
0: Unburned (available fuel)
1: Burning (flaming front)
2: Burned (consumed ash / inactive)

Probability of ignition from burning neighbor (k, l) to unburned cell (i, j):
p_burn = p_0 * (1 + p_veg) * (1 + p_den) * p_w(V, theta) * p_s(slope)
p_spot = alpha_spot * exp(-lambda_spot * distance)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class CellularAutomataSimulator:
    """
    Simulates stochastic wildfire perimeter growth using an 8-neighbor
    Cellular Automata (CA) lattice with wind vectoring, topography, and ember spotting.
    """

    # 8-neighbor offsets and base distances
    NEIGHBORS = [
        (-1, 0, 1.0),   # North
        (1, 0, 1.0),    # South
        (0, -1, 1.0),   # West
        (0, 1, 1.0),    # East
        (-1, -1, 1.414),# NW
        (-1, 1, 1.414), # NE
        (1, -1, 1.414), # SW
        (1, 1, 1.414),  # SE
    ]

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (100, 100),
        cell_size_m: float = 10.0,
        burn_duration_steps: int = 3,  # How many steps a cell remains burning before becoming burned
    ):
        self.ny, self.nx = grid_shape
        self.cell_size_m = cell_size_m
        self.burn_duration = burn_duration_steps

    def simulate(
        self,
        ignition_coords: List[Tuple[int, int]],
        total_steps: int = 60,
        wind_speed_ms: float = 8.0,
        wind_dir_deg: float = 45.0,  # 0=North, 90=East, 180=South, 270=West
        fuel_moisture: float = 0.08,
        elevation_grid: Optional[np.ndarray] = None,
        spotting_enabled: bool = True,
        random_seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Run the Cellular Automata simulation.
        Returns burned mask, arrival step grid, and progression metrics.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # State matrix: 0=Unburned, 1=Burning, 2=Burned
        state = np.zeros((self.ny, self.nx), dtype=np.int8)
        burn_timer = np.zeros((self.ny, self.nx), dtype=np.int8)
        arrival_step = np.full((self.ny, self.nx), -1, dtype=np.int32)

        # Set initial ignitions
        for iy, ix in ignition_coords:
            if 0 <= iy < self.ny and 0 <= ix < self.nx:
                state[iy, ix] = 1
                burn_timer[iy, ix] = self.burn_duration
                arrival_step[iy, ix] = 0

        # Precompute wind factors
        # Convert wind direction to unit vector (dx, dy)
        wind_rad = np.radians(wind_dir_deg)
        wind_u = np.sin(wind_rad)  # Easting component
        wind_v = np.cos(wind_rad)  # Northing component
        c1 = 0.045
        c2 = 0.131

        # Base ignition probability (damped by fuel moisture)
        p0 = max(0.58 * (1.0 - 2.8 * fuel_moisture), 0.05)

        time_history = []
        area_history = []

        for step in range(1, total_steps + 1):
            burning_cells = np.argwhere(state == 1)
            if len(burning_cells) == 0:
                break

            # Find unburned cells adjacent to burning cells
            new_ignitions = set()

            for by, bx in burning_cells:
                for dy, dx, dist in self.NEIGHBORS:
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < self.ny and 0 <= nx < self.nx and state[ny, nx] == 0:
                        # 1. Neighbor direction vector
                        dir_x = dx / dist
                        dir_y = -dy / dist  # -dy because row 0 is top (North)

                        # Angle between neighbor direction and wind direction
                        cos_theta = dir_x * wind_u + dir_y * wind_v
                        
                        # Alexandridis wind factor: pw = exp(c1 * V) * exp(V * c2 * (cos(theta) - 1))
                        p_w = np.exp(c1 * wind_speed_ms) * np.exp(wind_speed_ms * c2 * (cos_theta - 1.0))

                        # 2. Slope factor: ps = exp(a * slope)
                        if elevation_grid is not None:
                            diff_elev = elevation_grid[ny, nx] - elevation_grid[by, bx]
                            slope_rad = np.arctan(diff_elev / (dist * self.cell_size_m))
                            p_s = np.exp(3.533 * (np.tan(slope_rad) ** 1.2)) if slope_rad > 0 else np.exp(2.0 * np.tan(slope_rad))
                        else:
                            p_s = 1.0

                        # Effective neighbor transition probability
                        p_burn = min(max(p0 * p_w * p_s / dist, 0.0), 0.98)

                        if np.random.rand() < p_burn:
                            new_ignitions.add((ny, nx))

            # 3. Firebrand spotting (ember flight)
            if spotting_enabled and wind_speed_ms > 5.0 and len(burning_cells) > 5:
                # Stochastic ember generation from a sample of active flaming cells
                num_embers = min(int(len(burning_cells) * 0.02 * (wind_speed_ms / 10.0)), 15)
                for _ in range(num_embers):
                    sample_idx = np.random.randint(len(burning_cells))
                    sy, sx = burning_cells[sample_idx]
                    # Ember flight distance follows Weibull / log-normal distribution
                    jump_dist_cells = int(np.random.weibull(1.5) * (wind_speed_ms * 1.8))
                    if 3 <= jump_dist_cells <= 35:
                        # Dispersion angle centered on wind direction
                        dispersion = np.radians(np.random.normal(0, 15))
                        flight_rad = wind_rad + dispersion
                        ey = int(sy - jump_dist_cells * np.cos(flight_rad))
                        ex = int(sx + jump_dist_cells * np.sin(flight_rad))
                        if 0 <= ey < self.ny and 0 <= ex < self.nx and state[ey, ex] == 0:
                            # Ember ignition probability
                            p_spot_ignite = max(0.40 * (1.0 - 3.0 * fuel_moisture), 0.05)
                            if np.random.rand() < p_spot_ignite:
                                new_ignitions.add((ey, ex))

            # Decrement burning timers
            burn_timer[state == 1] -= 1
            burned_out = (state == 1) & (burn_timer <= 0)
            state[burned_out] = 2  # Consumed

            # Apply new ignitions
            for iy, ix in new_ignitions:
                state[iy, ix] = 1
                burn_timer[iy, ix] = self.burn_duration
                arrival_step[iy, ix] = step

            # Record step metrics
            if step % 5 == 0 or step == total_steps:
                consumed = int(np.sum(state >= 1))
                consumed_ha = (consumed * self.cell_size_m * self.cell_size_m) / 10000.0
                time_history.append(step)
                area_history.append(round(consumed_ha, 3))

        total_burned_cells = int(np.sum(state >= 1))
        total_burned_ha = round((total_burned_cells * self.cell_size_m * self.cell_size_m) / 10000.0, 3)

        return {
            "model": "Alexandridis Stochastic Cellular Automata",
            "grid_dimensions": {"nx": self.nx, "ny": self.ny, "resolution_m": self.cell_size_m},
            "burned_cells": total_burned_cells,
            "burned_area_ha": total_burned_ha,
            "total_steps": total_steps,
            "time_history": time_history,
            "area_history": area_history,
            "arrival_step_grid": arrival_step.tolist(),
            "final_burned_mask": (state >= 1).tolist(),
        }
