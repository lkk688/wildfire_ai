// California data cache (backend/geodata): layer catalog shared by the layer panel, the map and the legend.
import { useEffect, useState } from 'react';
import { getJSON } from './api';

export type GeoLegend = {
  swatches?: Array<[string, string]>;
  shape?: 'dot' | 'box';
  ramp?: string[];
  min?: number;
  max?: number;
  units?: string;
} | null;

export type GeoLayer = {
  id: string;
  type: 'raster' | 'vector';
  title: string;
  group: string;
  kind?: 'continuous' | 'categorical' | 'rgb';
  geom?: 'polygon' | 'point';
  units?: string;
  res?: number;
  source: string;
  url?: string;
  date?: string;
  licence?: string;
  notes?: string;
  min_zoom?: number;
  prefetch?: boolean;
  available?: boolean;
  cached_tiles_L0?: number;
  legend: GeoLegend;
};

// Order of the groups in the layer panel
export const GEO_GROUPS = [
  'Fuels and terrain (LANDFIRE, 30 m)',
  'Modelled wildfire hazard (USFS WRC, 30 m)',
  'Fire weather 2000-2025 (gridMET, 4 km)',
  'Imagery and vegetation',
  'Official hazard and history',
  'Fuel treatments',
  'Buildings and places',
];

let catalogPromise: Promise<GeoLayer[]> | null = null;

export function geoCatalog(): Promise<GeoLayer[]> {
  if (!catalogPromise) {
    catalogPromise = getJSON('/api/geodata/catalog')
      .then((d) => d.layers as GeoLayer[])
      .catch((e) => {
        catalogPromise = null; // try again next time
        throw e;
      });
  }
  return catalogPromise;
}

export function useGeoCatalog(): { layers: GeoLayer[]; error: string | null } {
  const [layers, setLayers] = useState<GeoLayer[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    geoCatalog()
      .then((l) => alive && setLayers(l))
      .catch((e) => alive && setError(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, []);
  return { layers, error };
}

export const compass = (deg: number) => ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round((((deg % 360) + 360) % 360) / 45) % 8];
