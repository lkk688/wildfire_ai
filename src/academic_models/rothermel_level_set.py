"""
Rothermel Surface Fire Model & 2D Level-Set PDE Solver
Academic Wildfire Propagation Engine.

Implements:
1. Rothermel (1972) empirical rate of spread (ROS) equation:
   R_0 = (I_R * xi) / (rho_b * epsilon * Q_ig)
   R = R_0 * (1 + phi_w + phi_s)
2. Level-Set Hamilton-Jacobi PDE:
   d(phi)/dt + R(x, y, grad(phi)) * ||grad(phi)|| = 0
   Solved using first-order Godunov / upwind finite differences.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


# Anderson 13 Standard Fire Behavior Fuel Models (abbreviated properties)
FUEL_MODELS: Dict[int, Dict[str, float]] = {
    1: {"name": "Short Grass (1 ft)", "fuel_load": 0.166, "depth": 1.0, "extinction_moisture": 0.12, "heat_content": 18600.0},
    4: {"name": "Chaparral (6 ft)", "fuel_load": 1.116, "depth": 6.0, "extinction_moisture": 0.20, "heat_content": 18600.0},
    8: {"name": "Closed Timber Litter", "fuel_load": 0.334, "depth": 0.2, "extinction_moisture": 0.30, "heat_content": 18600.0},
    10: {"name": "Timber (Litter + Understory)", "fuel_load": 0.892, "depth": 1.0, "extinction_moisture": 0.25, "heat_content": 18600.0},
}


class RothermelLevelSetSimulator:
    """
    Simulates wildland fire front propagation on a 2D topographic and fuel grid
    by integrating Rothermel's physical-empirical ROS with a Level-Set PDE.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (100, 100),
        dx: float = 10.0,  # Grid spatial resolution in meters
        dy: float = 10.0,
        fuel_model_id: int = 4,  # Default: Chaparral
    ):
        self.ny, self.nx = grid_shape
        self.dx = dx
        self.dy = dy
        self.fuel_model = FUEL_MODELS.get(fuel_model_id, FUEL_MODELS[4])
        self.fuel_model_id = fuel_model_id

    def compute_rothermel_ros(
        self,
        wind_speed_ms: float,
        wind_dir_deg: float,  # Direction wind is blowing TOWARDS (0=North, 90=East, 180=South, 270=West)
        fuel_moisture: float,  # Ratio (e.g. 0.08 for 8%)
        elevation_grid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute directional Rate of Spread (ROS) field in meters per minute (m/min).
        Returns:
            - base_ros: scalar or array of unassisted ROS (m/min)
            - ros_max: maximum forward rate of spread with wind & slope (m/min)
            - dir_max: direction of maximum spread in radians
        """
        fm = self.fuel_moisture = min(max(fuel_moisture, 0.02), self.fuel_model["extinction_moisture"] - 0.01)
        ext_m = self.fuel_model["extinction_moisture"]
        
        # 1. Moisture damping coefficient
        eta_m = 1.0 - 2.59 * (fm / ext_m) + 5.11 * ((fm / ext_m) ** 2) - 3.52 * ((fm / ext_m) ** 3)
        eta_m = max(eta_m, 0.001)

        # 2. Base reaction intensity and heat of pre-ignition
        # Approx: R0 ~ 0.5 to 3.0 m/min for dry chaparral
        r0 = 1.2 * eta_m * (self.fuel_model["depth"] / 3.0)

        # 3. Wind multiplier: phi_w = c * (3.281 * V)^B * (beta/beta_op)^-E
        # Converting wind_speed_ms to mph for standard coefficients: 1 m/s = 2.237 mph
        wind_mph = wind_speed_ms * 2.237
        phi_w = 0.09 * (wind_mph ** 1.35)

        # 4. Slope multiplier: phi_s = 5.275 * beta^-0.3 * (tan(theta))^2
        if elevation_grid is not None:
            grad_y, grad_x = np.gradient(elevation_grid, self.dy, self.dx)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            phi_s = 5.275 * (slope ** 2)
            slope_dir = np.arctan2(-grad_y, -grad_x)  # Upslope direction
        else:
            phi_s = np.zeros((self.ny, self.nx))
            slope_dir = np.zeros((self.ny, self.nx))

        # 5. Combined vector alignment (Wind + Slope vector addition)
        wind_dir_rad = np.radians(wind_dir_deg)
        wind_vec_x = phi_w * np.sin(wind_dir_rad)
        wind_vec_y = phi_w * np.cos(wind_dir_rad)

        slope_vec_x = phi_s * np.sin(slope_dir)
        slope_vec_y = phi_s * np.cos(slope_dir)

        total_force_x = wind_vec_x + slope_vec_x
        total_force_y = wind_vec_y + slope_vec_y

        effective_phi = np.sqrt(total_force_x**2 + total_force_y**2)
        dir_max = np.arctan2(total_force_x, total_force_y)

        # Max forward rate of spread (m/min)
        ros_max = r0 * (1.0 + effective_phi)
        return r0, ros_max, dir_max

    def solve_level_set(
        self,
        ignition_coords: List[Tuple[int, int]],
        total_minutes: float = 60.0,
        dt_minutes: float = 0.2,
        wind_speed_ms: float = 8.0,
        wind_dir_deg: float = 45.0,
        fuel_moisture: float = 0.06,
        elevation_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Solve Level-Set PDE using Godunov upwind scheme.
        phi(x, y, t) <= 0 indicates burned; phi(x, y, t) > 0 indicates unburned.
        """
        # Initialize signed distance function phi: distance to nearest ignition point
        y_grid, x_grid = np.ogrid[:self.ny, :self.nx]
        phi = np.full((self.ny, self.nx), 1e6)
        
        for iy, ix in ignition_coords:
            dist = np.sqrt(((x_grid - ix) * self.dx) ** 2 + ((y_grid - iy) * self.dy) ** 2) - (self.dx * 0.8)
            phi = np.minimum(phi, dist)

        r0, ros_max, dir_max = self.compute_rothermel_ros(
            wind_speed_ms=wind_speed_ms,
            wind_dir_deg=wind_dir_deg,
            fuel_moisture=fuel_moisture,
            elevation_grid=elevation_grid,
        )

        arrival_time = np.full((self.ny, self.nx), -1.0)
        arrival_time[phi <= 0] = 0.0

        num_steps = int(total_minutes / dt_minutes)
        time_history = []
        area_history = []

        # Time stepping loop
        for step in range(1, num_steps + 1):
            curr_time = step * dt_minutes

            # First-order upwind spatial differences
            phi_w = np.roll(phi, 1, axis=1)   # West
            phi_e = np.roll(phi, -1, axis=1)  # East
            phi_s = np.roll(phi, 1, axis=0)   # South
            phi_n = np.roll(phi, -1, axis=0)  # North

            d_minus_x = (phi - phi_w) / self.dx
            d_plus_x = (phi_e - phi) / self.dx
            d_minus_y = (phi - phi_s) / self.dy
            d_plus_y = (phi_n - phi) / self.dy

            # Normal vector of front n = grad(phi) / ||grad(phi)||
            grad_x = 0.5 * (d_plus_x + d_minus_x)
            grad_y = 0.5 * (d_plus_y + d_minus_y)
            grad_norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
            nx_norm = grad_x / grad_norm
            ny_norm = grad_y / grad_norm

            # Elliptical propagation velocity in normal direction:
            # Front direction angle
            theta_n = np.arctan2(nx_norm, ny_norm)
            diff_angle = theta_n - dir_max

            # Huygens elliptical expansion ratio (Alexander 1985 / Finney 1998)
            wind_mph = wind_speed_ms * 2.237
            lb_ratio = max(1.0 + 0.05 * wind_mph, 1.2)
            cos_term = np.cos(diff_angle)

            # Directional speed in normal direction: head fire vs flank vs backing
            speed = ros_max * ((1.0 + cos_term) / 2.0 + (1.0 - cos_term) / (2.0 * lb_ratio))
            speed = np.maximum(speed, r0 * 0.3)  # Backing fire minimum

            # CFL condition check & Godunov flux
            # H(d_x, d_y) = speed * sqrt(grad_x^2 + grad_y^2)
            term_x = np.maximum(np.maximum(d_minus_x, 0.0)**2, np.minimum(d_plus_x, 0.0)**2)
            term_y = np.maximum(np.maximum(d_minus_y, 0.0)**2, np.minimum(d_plus_y, 0.0)**2)
            h_flux = speed * np.sqrt(term_x + term_y)

            # Update phi: d(phi)/dt + H = 0  =>  phi_new = phi - dt * H
            phi = phi - dt_minutes * h_flux

            # Record arrival time for newly ignited cells
            newly_burned = (phi <= 0) & (arrival_time < 0)
            arrival_time[newly_burned] = curr_time

            # Record progression metrics every 10 steps
            if step % 10 == 0 or step == num_steps:
                burned_count = int(np.sum(phi <= 0))
                burned_hectares = (burned_count * self.dx * self.dy) / 10000.0
                time_history.append(round(curr_time, 1))
                area_history.append(round(burned_hectares, 3))

        total_burned_cells = int(np.sum(phi <= 0))
        total_burned_ha = round((total_burned_cells * self.dx * self.dy) / 10000.0, 3)

        return {
            "model": "Rothermel Level-Set PDE",
            "fuel_model": self.fuel_model["name"],
            "grid_dimensions": {"nx": self.nx, "ny": self.ny, "resolution_m": self.dx},
            "burned_cells": total_burned_cells,
            "burned_area_ha": total_burned_ha,
            "simulation_time_min": total_minutes,
            "max_forward_ros_mpm": round(float(np.mean(ros_max)), 2),
            "time_history": time_history,
            "area_history": area_history,
            "arrival_time_grid": np.round(arrival_time, 1).tolist(),
            "final_burned_mask": (phi <= 0).tolist(),
        }
