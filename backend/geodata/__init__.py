"""
California data layers kept in a local cache, for the map and for the AI agent.

Rasters live in a tile pyramid in California Albers (EPSG:3310): each tile is fetched from its public source the first
time anything asks for it and is read from disk afterwards. Vectors are statewide GeoParquet files. The same functions
serve the web map (PNG tiles, see render.py) and the agent (query.py: values at a point, summaries over an area,
features nearby).

    from backend import geodata
    geodata.catalog()                                   # every layer with source, units, dates, licence
    geodata.point(-122.43, 37.62)                       # all layer values at a point
    geodata.area(-122.43, 37.62, radius_m=500)          # summaries within 500 m
    geodata.features("ignitions", bbox=(...))           # vector features

Data root: GEODATA_ROOT (default /data/wildfire_benchmark_data/ca). Build and prefetch: python -m backend.geodata.build
"""
from backend.geodata.catalog import catalog, layer  # noqa: F401
try:  # query.py imports the store; keep the package importable while building
    from backend.geodata.query import area, features, point, raster_window  # noqa: F401
except ImportError:  # pragma: no cover
    pass
