"""
Data fetchers. The live map sources (perimeters, FIRMS, CDEC RAWS, HRRR, 3DEP, Sentinel-2, FHSZ, WorldCover,
building footprints) live in backend/live_sources.py; this package keeps the Bellwether GeoTIFF cropper.
"""

from .gcs_bellwether_downloader import crop_bellwether_by_bbox
