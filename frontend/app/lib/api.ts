// Backend base URL. Set NEXT_PUBLIC_API_BASE at build time to point the UI at another backend.
export const API_BASE: string = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

// Region boxes shared with the backend (backend/services.py REGION_BOUNDS): [min_lon, min_lat, max_lon, max_lat]
export const REGIONS: Record<string, { name: string; center: [number, number]; bbox: [number, number, number, number] }> = {
  san_bruno: { name: 'San Bruno WUI (Peninsula)', center: [37.618, -122.425], bbox: [-122.46, 37.58, -122.39, 37.65] },
  san_jose: { name: 'San Jose Foothills (Santa Clara)', center: [37.338, -121.886], bbox: [-121.98, 37.25, -121.78, 37.42] },
  santa_cruz: { name: 'Santa Cruz Mountains (CZU WUI)', center: [37.125, -122.05], bbox: [-122.18, 37.02, -121.92, 37.22] },
};

export type LayerState = {
  state: 'loading' | 'ok' | 'error' | 'off' | 'info';
  detail?: string;
  swatches?: Array<{ label: string; color: string }>; // legend entries that depend on live data (e.g. which NWS alerts are active)
};

export async function getJSON(path: string, init?: RequestInit): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, init);
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    throw new Error(body?.detail || `HTTP ${res.status}`);
  }
  return body;
}
