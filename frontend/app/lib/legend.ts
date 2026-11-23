// Legend entries for the map layers. Colours and cut-offs mirror what the backend / MapView actually draw:
//   Bellwether: src/data_fetchers/gcs_bellwether_downloader.py (crop_bellwether_by_bbox)
//   USDA burn probability, slope: backend/live_sources.py (wrc_bp_overlay, slope_overlay)
//   FHSZ: backend/live_sources.py FHSZ_COLORS; points and lines: components/MapView.tsx

export type Swatch = { label: string; color: string; shape?: 'box' | 'dot' | 'line' | 'dash' | 'arrow' | 'ring' };
export type LegendItem = { id: string; title: string; note?: string; swatches: Swatch[] };

// Wind speed classes (mph), shared by the animated wind layer (windParticles.ts) and the legend
export const WIND_CLASSES: Array<{ max: number; label: string; color: string }> = [
  { max: 5, label: '< 5 mph', color: '#cbd5e1' },
  { max: 10, label: '5–10', color: '#7dd3fc' },
  { max: 20, label: '10–20', color: '#fde047' },
  { max: 30, label: '20–30', color: '#fb923c' },
  { max: 40, label: '30–40', color: '#ef4444' },
  { max: Infinity, label: '≥ 40 mph', color: '#e879f9' },
];

const BW_COLORS = ['#228B22', '#90EE90', '#FFFF66', '#FFCC00', '#FF8000', '#EE2222', '#9400D3'];
const BW_NAMES = ['Very Low', 'Low', 'Moderate', 'Significant', 'High', 'Very High', 'Extreme']; // same keys as the backend stats
const BW_CUTS: Record<'1yr' | '5yr', string[]> = {
  '1yr': ['0.004%', '0.02%', '0.1%', '0.4%', '0.67%', '1.33%'],
  '5yr': ['0.02%', '0.1%', '0.5%', '2%', '3.3%', '6.7%'],
};

export function bellwetherCategories(which: '1yr' | '5yr') {
  const c = BW_CUTS[which];
  return BW_NAMES.map((name, i) => ({
    name,
    color: BW_COLORS[i],
    range: i === 0 ? `< ${c[0]}` : i === BW_NAMES.length - 1 ? `> ${c[i - 1]}` : `${c[i - 1]} – ${c[i]}`,
  }));
}

export function bellwetherLegend(which: '1yr' | '5yr'): LegendItem {
  return {
    id: 'bellwether',
    title: `Bellwether: chance a 100 m cell burns in the next ${which === '1yr' ? 'year' : '5 years'}`,
    note: 'Partner model; pilot area only',
    swatches: bellwetherCategories(which).map((c) => ({ label: c.range, color: c.color })),
  };
}

export const LAYER_LEGENDS: Record<string, LegendItem> = {
  wrc: {
    id: 'wrc',
    title: 'USDA: chance of burning in a given year',
    note: 'Wildfire Risk to Communities, 30 m; clear = below 0.05%',
    swatches: [
      { label: '0.05–0.2%', color: '#BFDBFE' },
      { label: '0.2–0.5%', color: '#60A5FA' },
      { label: '0.5–1%', color: '#2563EB' },
      { label: '≥ 1%', color: '#1E3A8A' },
    ],
  },
  fhsz: {
    id: 'fhsz',
    title: 'CAL FIRE hazard severity zone',
    swatches: [
      { label: 'Moderate', color: '#EAB308' },
      { label: 'High', color: '#F97316' },
      { label: 'Very high', color: '#DC2626' },
    ],
  },
  perimeters: {
    id: 'perimeters',
    title: 'Fires (CAL FIRE, NIFC)',
    swatches: [
      { label: 'Active fire', color: '#ef4444', shape: 'line' },
      { label: 'Incident', color: '#dc2626', shape: 'ring' },
      { label: 'Past fire (1950+)', color: '#f59e0b', shape: 'dash' },
    ],
  },
  hotspots: {
    id: 'hotspots',
    title: 'Satellite heat detections, last 24 h',
    swatches: [
      { label: 'Detection', color: '#fbbf24', shape: 'dot' },
      { label: 'Intense (> 20 MW)', color: '#f97316', shape: 'dot' },
    ],
  },
  fuel: {
    id: 'fuel',
    title: 'Dead-fuel moisture at weather stations',
    note: 'Drier fuel ignites and spreads more easily',
    swatches: [
      { label: '≤ 6% very dry', color: '#dc2626', shape: 'dot' },
      { label: '6–10%', color: '#f97316', shape: 'dot' },
      { label: '10–15%', color: '#eab308', shape: 'dot' },
      { label: '> 15% moist', color: '#14b8a6', shape: 'dot' },
    ],
  },
  wind: {
    id: 'wind',
    title: 'Wind at the slider hour (streaks flow downwind)',
    note: 'NOAA HRRR / NBM models, 10 m above ground',
    swatches: WIND_CLASSES.map((c) => ({ label: c.label, color: c.color, shape: 'line' as const })),
  },
  alerts: {
    id: 'alerts',
    title: 'NWS watches, warnings, advisories',
    note: 'In effect at the slider hour',
    swatches: [],
  },
  slope: {
    id: 'slope',
    title: 'Terrain slope (fire runs faster uphill)',
    swatches: [
      { label: '15–30%', color: '#FACC15' },
      { label: '30–50%', color: '#F97316' },
      { label: '≥ 50%', color: '#DC2626' },
    ],
  },
  ndvi: {
    id: 'ndvi',
    title: 'Vegetation greenness (NDVI)',
    swatches: [
      { label: 'Bare / dry', color: '#a16207' },
      { label: 'Sparse', color: '#d9f99d' },
      { label: 'Dense green', color: '#15803d' },
    ],
  },
  buildings: {
    id: 'buildings',
    title: 'Buildings (zoom in to see)',
    swatches: [{ label: 'Footprint', color: '#059669' }],
  },
};

export const LEGEND_ORDER = ['alerts', 'wrc', 'fhsz', 'perimeters', 'hotspots', 'fuel', 'wind', 'slope', 'ndvi', 'buildings'];
