"""
VLM-to-Physics Data Assimilation Bridge (Physical Simulation Parameter Gateway)
Translates high-dimensional multimodal VLM structural vulnerability attributions
into quantitative 2D parameter fields for:
1. Continuous Rothermel Level-Set PDE Solvers (fuel loading, pre-ignition heat, WRF)
2. Discrete Alexandridis Stochastic Cellular Automata (ember spotting capture, structural wicks)
3. Dynamic Network Contagion Graphs (anisotropic radiant view-factor corridors)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class VLMPhysicsBridge:
    """
    Data assimilation gateway that ingests structured VLM physical assessments
    and County Assessor records to synthesize 2D spatial parameter grids
    for coupled wildfire spread simulation engines.
    """

    @staticmethod
    def extract_parcel_hardening_coefficients(assessment: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract normalized physical coefficients from VLM structured assessment:
        - alpha_fuel: Fuel load / bulk density modifier (0.15 to 2.0)
        - delta_q: Added pre-ignition heat resistance in kJ/kg (0 to 1200)
        - p_spotting_capture: Firebrand capture efficiency (0.01 to 0.95)
        - has_structural_wick: Whether attached wooden fences act as conductive conduits
        - wrf: Local wind reduction factor (0.2 to 1.0)
        """
        if not assessment:
            return {
                "alpha_fuel": 1.0,
                "delta_q": 0.0,
                "p_spotting_capture": 0.35,
                "has_structural_wick": False,
                "wrf": 0.70,
            }

        # 1. Parse Micro Facade Envelope
        attribution = assessment.get("visual_evidence_attribution", {})
        facade = attribution.get("micro_facade_envelope", {})
        macro = attribution.get("macro_ortho_context", {})
        matrix = assessment.get("physical_vulnerability_matrix", {})
        payloads = assessment.get("downstream_task_payloads", {})

        cladding = str(facade.get("exterior_cladding", "")).lower()
        roof = str(facade.get("roof_assembly", "")).lower()
        vents = str(facade.get("vent_mesh_rating", "")).lower()
        eaves = str(facade.get("eave_soffit_construction", "")).lower()
        zone0 = str(facade.get("zone0_perimeter_buffer", "")).lower()
        attachments = str(facade.get("attached_combustible_attachments", "")).lower()

        # 2. Compute Fuel Multiplier (alpha_fuel)
        # Baseline = 1.0. Hardened stucco with stone buffer = 0.20-0.35x.
        # Combustible wood siding with organic mulch = 1.35-1.80x.
        alpha_fuel = 1.0
        if "stucco" in cladding or "brick" in cladding or "cement" in cladding:
            alpha_fuel *= 0.45
        elif "wood" in cladding or "shingle" in cladding:
            alpha_fuel *= 1.40

        if "mulch" in zone0 or "pine needle" in zone0 or "dry" in zone0:
            alpha_fuel *= 1.35
        elif "concrete" in zone0 or "gravel" in zone0 or "hardened" in zone0:
            alpha_fuel *= 0.65

        alpha_fuel = float(np.clip(alpha_fuel, 0.15, 2.20))

        # 3. Compute Heat of Pre-Ignition Addition (delta_q)
        # Class A roof + compliant vents dramatically delay structural ignition
        delta_q = 0.0
        if "class a" in roof or "tile" in roof or "metal" in roof:
            delta_q += 500.0
        elif "wood shake" in roof:
            delta_q -= 150.0

        if "<= 1/8" in vents or "<=1/8" in vents or "chapter 7a" in vents:
            delta_q += 350.0
        elif "1/4" in vents or "unmeshed" in vents or ">1/8" in vents or "> 1/8" in vents or "non-compliant" in vents:
            delta_q -= 150.0

        delta_q = float(np.clip(delta_q, 0.0, 1200.0))

        # 4. Compute Ember Spotting Capture Probability (p_spotting_capture)
        # Governed by open rafter eddies and vent screen aperture
        p_base = 0.35
        if "<= 1/8" in vents or "<=1/8" in vents or "corrosion-resistant" in vents:
            k_vent = 0.08  # 92% of embers blocked (IBHS standard)
        elif "1/4" in vents or ">1/8" in vents or "> 1/8" in vents or "non-compliant" in vents or "unmeshed" in vents:
            k_vent = 0.85  # Easily penetrates into attic insulation
        else:
            k_vent = 0.60

        if "enclosed" in eaves or "boxed" in eaves:
            k_eaves = 0.25  # Smooth aerodynamic flow shedding embers
        elif "open" in eaves or "rafter" in eaves:
            k_eaves = 0.90  # Turbulent stagnation vortex trapping embers
        else:
            k_eaves = 0.55

        p_spotting = float(np.clip(p_base * (k_vent * 1.5 + k_eaves * 1.0), 0.01, 0.95))

        # 5. Structural Wick Conduction Flag
        att_norm = attachments.lower().replace("-", " ")
        is_safe_fence = any(k in att_norm for k in ["non combustible", "metal gate", "no fence", "detached", "eliminating", "zero fence"])
        has_wick = any(k in att_norm for k in ["fence", "wood", "deck", "bolted", "privacy fence"]) and not is_safe_fence

        # 6. Wind Reduction Factor based on Structure Separation Distance (SSD)
        ssd = macro.get("structure_separation_distance_m", 15.0)
        wrf = float(np.clip(0.35 + 0.02 * ssd, 0.30, 0.85))

        return {
            "alpha_fuel": round(alpha_fuel, 3),
            "delta_q": round(delta_q, 1),
            "p_spotting_capture": round(p_spotting, 3),
            "has_structural_wick": bool(has_wick),
            "wrf": round(wrf, 2),
        }

    @classmethod
    def generate_rothermel_spatial_grids(
        cls,
        grid_shape: Tuple[int, int],
        parcel_center: Tuple[int, int],
        assessment: Dict[str, Any],
        parcel_radius_cells: int = 4,
        buffer_radius_cells: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D spatial parameter grids for Rothermel Level-Set PDE:
        - fuel_modifier_grid: (ny, nx) array modulating local fuel loading
        - delta_q_grid: (ny, nx) array adding pre-ignition heat capacity
        - wrf_grid: (ny, nx) array with aerodynamic sheltering
        """
        ny, nx = grid_shape
        coeffs = cls.extract_parcel_hardening_coefficients(assessment)

        # Default wildland background
        fuel_mod = np.ones((ny, nx), dtype=float)
        delta_q = np.zeros((ny, nx), dtype=float)
        wrf = np.ones((ny, nx), dtype=float)

        cy, cx = parcel_center
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist_sq = (y_grid - cy) ** 2 + (x_grid - cx) ** 2

        # 1. Parcel Core (Building Envelope)
        core_mask = dist_sq <= (parcel_radius_cells ** 2)
        fuel_mod[core_mask] = coeffs["alpha_fuel"]
        delta_q[core_mask] = coeffs["delta_q"]
        wrf[core_mask] = coeffs["wrf"]

        # 2. Defensible Space Transition Zone (Zone 1: 5-30ft / cells buffer_radius)
        buffer_mask = (dist_sq > (parcel_radius_cells ** 2)) & (dist_sq <= (buffer_radius_cells ** 2))
        dist_from_core = np.sqrt(dist_sq[buffer_mask]) - parcel_radius_cells
        blend = dist_from_core / float(max(buffer_radius_cells - parcel_radius_cells, 1))

        # Smooth linear blend from parcel core property to wildland baseline (1.0)
        fuel_mod[buffer_mask] = coeffs["alpha_fuel"] * (1.0 - blend) + 1.0 * blend
        delta_q[buffer_mask] = coeffs["delta_q"] * (1.0 - blend)
        wrf[buffer_mask] = coeffs["wrf"] * (1.0 - blend) + 1.0 * blend

        # 3. Structural Wick Conduction Channel (Linear attached wooden fence)
        if coeffs["has_structural_wick"]:
            # Fence extending 6 cells windward or southward from structure wall
            fence_len = 7
            for offset in range(1, fence_len + 1):
                fy = cy + offset
                fx = cx
                if 0 <= fy < ny and 0 <= fx < nx:
                    fuel_mod[fy, fx] = max(fuel_mod[fy, fx], 1.65)  # Accelerated conductive line
                    delta_q[fy, fx] = 0.0

        return fuel_mod, delta_q, wrf

    @classmethod
    def generate_cellular_automata_grids(
        cls,
        grid_shape: Tuple[int, int],
        parcel_center: Tuple[int, int],
        assessment: Dict[str, Any],
        parcel_radius_cells: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D spatial parameter grids for Alexandridis Stochastic CA:
        - fuel_modifier_grid: (ny, nx) float array
        - spotting_prob_grid: (ny, nx) float array of firebrand capture efficiency
        - wick_mask: (ny, nx) boolean mask indicating structural conductive wicks
        """
        ny, nx = grid_shape
        coeffs = cls.extract_parcel_hardening_coefficients(assessment)

        fuel_mod = np.ones((ny, nx), dtype=float)
        spotting_grid = np.full((ny, nx), 0.35, dtype=float)  # Default wildland spotting
        wick_mask = np.zeros((ny, nx), dtype=bool)

        cy, cx = parcel_center
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist_sq = (y_grid - cy) ** 2 + (x_grid - cx) ** 2

        # Core Parcel
        core_mask = dist_sq <= (parcel_radius_cells ** 2)
        fuel_mod[core_mask] = coeffs["alpha_fuel"]
        spotting_grid[core_mask] = coeffs["p_spotting_capture"]

        # Attached Wooden Fence Conduit
        if coeffs["has_structural_wick"]:
            fence_len = 7
            for offset in range(1, fence_len + 1):
                fy = cy + offset
                fx = cx
                if 0 <= fy < ny and 0 <= fx < nx:
                    wick_mask[fy, fx] = True
                    fuel_mod[fy, fx] = 1.70

        return fuel_mod, spotting_grid, wick_mask

    @classmethod
    def compute_anisotropic_contagion_weight(
        cls,
        assessment: Dict[str, Any],
        distance_meters: float,
        wind_speed_ms: float,
        wind_dir_deg: float,
        bearing_to_target_deg: float,
    ) -> Dict[str, Any]:
        """
        Compute directional radiant and ember contagion weight between structures
        based on Eric Saylors Network Contagion model and Stefan-Boltzmann radiant flux:
        q'' = epsilon * sigma * F12 * T^4
        """
        coeffs = cls.extract_parcel_hardening_coefficients(assessment)

        # 1. Radiant Heat Flux Decay (Stefan-Boltzmann view factor approximation)
        # Critical ignition threshold for unhardened wood siding = 12.5 kW/m^2
        # Hardened stucco can withstand up to 25.0 kW/m^2
        t_flame_k = 1200.0  # Blackbody flame temperature in Kelvin
        sigma = 5.67e-8
        epsilon = 0.90
        # Geometric view factor approx F12 ~ (H * W) / (pi * d^2)
        flame_area = 40.0  # 4m high x 10m wide facade on fire
        view_factor = min(flame_area / (np.pi * max(distance_meters, 1.0) ** 2), 1.0)
        incident_radiant_flux_kw = (epsilon * sigma * view_factor * (t_flame_k ** 4)) / 1000.0

        # Critical resistance threshold
        is_hardened = coeffs["delta_q"] > 300.0
        critical_flux_threshold = 25.0 if is_hardened else 12.5
        radiant_breach_probability = float(np.clip((incident_radiant_flux_kw - 8.0) / (critical_flux_threshold - 8.0), 0.0, 1.0))

        # 2. Wind Directional Alignment (Flame Tilt Factor)
        angle_diff_rad = np.radians(wind_dir_deg - bearing_to_target_deg)
        flame_tilt_factor = max(np.cos(angle_diff_rad), 0.0)

        # 3. Spotting Contagion Weight
        ember_flight_factor = float(coeffs["p_spotting_capture"] * (1.0 + 0.15 * wind_speed_ms * flame_tilt_factor))
        if distance_meters > 120.0:
            ember_flight_factor *= np.exp(-0.01 * (distance_meters - 120.0))

        # Total combined dynamic edge contagion weight (0.0 to 1.0)
        total_contagion_weight = float(np.clip(radiant_breach_probability * (0.6 + 0.4 * flame_tilt_factor) + ember_flight_factor * 0.4, 0.0, 0.99))

        return {
            "incident_radiant_flux_kw_m2": round(incident_radiant_flux_kw, 2),
            "critical_threshold_kw_m2": critical_flux_threshold,
            "flame_tilt_factor": round(flame_tilt_factor, 3),
            "radiant_breach_probability": round(radiant_breach_probability, 3),
            "ember_spotting_weight": round(ember_flight_factor, 3),
            "total_dynamic_edge_weight": round(total_contagion_weight, 3),
            "is_within_critical_130ft_corridor": distance_meters <= 39.6,  # 130ft = 39.6m
        }
