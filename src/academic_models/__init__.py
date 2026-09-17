"""
Academic Wildfire Models Package
Export simulators and benchmark suite.
"""

from src.academic_models.rothermel_level_set import RothermelLevelSetSimulator
from src.academic_models.cellular_automata import CellularAutomataSimulator
from src.academic_models.benchmark_suite import (
    AcademicBenchmarkRunner,
    calculate_spatial_metrics,
)

__all__ = [
    "RothermelLevelSetSimulator",
    "CellularAutomataSimulator",
    "AcademicBenchmarkRunner",
    "calculate_spatial_metrics",
]
