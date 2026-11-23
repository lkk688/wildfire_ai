'use client';

import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { API_BASE, REGIONS, LayerState, getJSON } from '../lib/api';

export type LayerToggles = {
  perimeters: boolean;
  hotspots: boolean;
  fuel: boolean;
  slope: boolean;
  buildings: boolean;
  ndvi: boolean;
  fhsz: boolean;
  wrc: boolean;
  wind: boolean;
};

interface MapViewProps {
  initialView?: { lat: number; lng: number; zoom: number } | null;
  activeLayer: '1yr' | '5yr' | 'none';
  selectedRegion: string;
  layers: LayerToggles;
  opacity: number;
  onSelectLocation: (data: any) => void;
  onLayerStatsUpdate: (stats: any) => void;
  onLayerStatus: (id: string, s: LayerState) => void;
}

const BUILDING_MIN_ZOOM = 15;

// Text from external feeds goes into tooltips: escape it.
const esc = (v: any) =>
  String(v ?? '—').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string));
const fmt = (v: any, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
const bboxParam = (b: [number, number, number, number]) => b.map((v) => v.toFixed(5)).join(',');

export default function MapView({
  initialView,
  activeLayer,
  selectedRegion,
  layers,
  opacity,
  onSelectLocation,
  onLayerStatsUpdate,
  onLayerStatus,
}: MapViewProps) {
  const [map, setMap] = useState<L.Map | null>(null);
  const refs = useRef<Record<string, L.Layer | null>>({});
  const clickRefs = useRef<{ marker: L.Marker | null; circle: L.Circle | null }>({ marker: null, circle: null });
  const statusRef = useRef(onLayerStatus);
  statusRef.current = onLayerStatus;
  const region = REGIONS[selectedRegion] || REGIONS.san_bruno;

  const setLayer = (id: string, layer: L.Layer | null) => {
    if (!map) return;
    const old = refs.current[id];
    if (old) map.removeLayer(old);
    refs.current[id] = layer;
    if (layer) layer.addTo(map);
  };
  const status = (id: string, s: LayerState) => statusRef.current(id, s);

  // Map init
  useEffect(() => {
    const m = L.map('leaflet-map', {
      center: initialView ? [initialView.lat, initialView.lng] : region.center,
      zoom: initialView ? initialView.zoom : 13,
      zoomControl: false,
    });
    // Keyless Esri basemaps (the CARTO tiles used before now require an API key)
    const esri = (svc: string) => `https://server.arcgisonline.com/ArcGIS/rest/services/${svc}/MapServer/tile/{z}/{y}/{x}`;
    const data =
      'Bellwether (X), NIFC, CAL FIRE, NASA FIRMS, CDEC, NOAA HRRR/Open-Meteo, NWS, USGS 3DEP, USDA Forest Service WRC, Copernicus Sentinel-2/Esri, ESA WorldCover, Microsoft Building Footprints';
    const dark = L.layerGroup([
      L.tileLayer(esri('Canvas/World_Dark_Gray_Base'), { maxZoom: 19, maxNativeZoom: 16, attribution: `Esri, HERE, Garmin, OSM | ${data}` }),
      L.tileLayer(esri('Canvas/World_Dark_Gray_Reference'), { maxZoom: 19, maxNativeZoom: 16 }),
    ]);
    const imagery = L.tileLayer(esri('World_Imagery'), { maxZoom: 19, attribution: `Esri, Maxar, Earthstar Geographics | ${data}` });
    const topo = L.tileLayer(esri('World_Topo_Map'), { maxZoom: 19, attribution: `Esri, USGS, NOAA | ${data}` });
    dark.addTo(m);
    if (!initialView) {
      const [w, so, e, n] = region.bbox;
      m.fitBounds([[so, w], [n, e]], { padding: [20, 20] }); // whole region box in view
    }
    L.control.layers({ 'Dark gray': dark, 'Satellite imagery': imagery, Topographic: topo }, {}, { position: 'topleft' }).addTo(m);
    L.control.zoom({ position: 'bottomleft' }).addTo(m);
    L.control.scale({ position: 'bottomleft', imperial: true }).addTo(m);
    setMap(m);
    return () => {
      m.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const firstRegion = useRef(true);
  useEffect(() => {
    if (!map) return;
    if (firstRegion.current && initialView) {
      firstRegion.current = false; // keep the URL's view on first load
      return;
    }
    firstRegion.current = false;
    const [w, so, e, n] = region.bbox;
    map.flyToBounds([[so, w], [n, e]], { padding: [20, 20], duration: 1.2 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, selectedRegion]);

  // ---------------------------------------------------------------- Bellwether (partner GeoTIFF)
  useEffect(() => {
    if (!map) return;
    setLayer('bellwether', null);
    if (activeLayer === 'none') {
      status('bellwether', { state: 'off' });
      return;
    }
    status('bellwether', { state: 'loading' });
    getJSON(`/api/layers/bellwether?is_5_year=${activeLayer === '5yr'}&region=${selectedRegion}`)
      .then((d) => {
        const lyr = d.layer;
        if (!lyr?.covered) {
          onLayerStatsUpdate(null);
          status('bellwether', { state: 'info', detail: lyr?.reason || 'No coverage here' });
          return;
        }
        setLayer('bellwether', L.imageOverlay(`${API_BASE}${lyr.image_url}`, lyr.bounds, { opacity, interactive: false }));
        onLayerStatsUpdate(lyr.stats);
        status('bellwether', {
          state: 'ok',
          detail: `${lyr.window}; values for ${fmt(lyr.valid_share_pct, 1)}% of this region`,
        });
      })
      .catch((e) => status('bellwether', { state: 'error', detail: String(e.message || e) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, activeLayer, selectedRegion]);

  useEffect(() => {
    const o = refs.current.bellwether as L.ImageOverlay | null;
    if (o) o.setOpacity(opacity);
  }, [opacity]);

  // ---------------------------------------------------------------- Fire perimeters and incidents
  useEffect(() => {
    if (!map) return;
    setLayer('perimeters', null);
    if (!layers.perimeters) return status('perimeters', { state: 'off' });
    status('perimeters', { state: 'loading' });
    getJSON(`/api/layers/perimeters?bbox=${bboxParam(region.bbox)}`)
      .then((d) => {
        const group = L.layerGroup();
        L.geoJSON(d.historic, {
          style: () => ({ color: '#f59e0b', weight: 1.2, dashArray: '4 4', fillColor: '#f59e0b', fillOpacity: 0.08 }),
          onEachFeature: (f, l) => {
            const p = f.properties;
            l.bindTooltip(
              `<div class="text-xs"><b>${esc(p.name)} (${esc(p.year)})</b><br/>${fmt(p.acres)} acres<br/><span class="opacity-70">${esc(p.source)}</span></div>`,
              { sticky: true },
            );
          },
        }).addTo(group);
        L.geoJSON(d.current, {
          style: () => ({ color: '#ef4444', weight: 2.5, fillColor: '#ef4444', fillOpacity: 0.3 }),
          onEachFeature: (f, l) => {
            const p = f.properties;
            l.bindTooltip(
              `<div class="text-xs"><b>${esc(p.name)}</b><br/>${fmt(p.acres)} acres · contained ${p.percent_contained ?? '—'}%<br/>Discovered ${esc(p.discovered)}<br/>Perimeter updated ${esc(p.perimeter_updated)}<br/><span class="opacity-70">${esc(p.source)}</span></div>`,
              { sticky: true },
            );
          },
        }).addTo(group);
        L.geoJSON(d.incidents, {
          pointToLayer: (_f, ll) =>
            L.circleMarker(ll, { radius: 7, color: '#fff', weight: 1.5, fillColor: '#dc2626', fillOpacity: 0.95 }),
          onEachFeature: (f, l) => {
            const p = f.properties;
            l.bindTooltip(
              `<div class="text-xs"><b>${esc(p.name)}</b> (${esc(p.county)})<br/>${fmt(p.acres)} acres · ${p.percent_contained ?? '—'}% contained<br/>Updated ${esc(p.updated)}<br/><span class="opacity-70">${esc(p.source)}</span></div>`,
            );
          },
        }).addTo(group);
        setLayer('perimeters', group);
        const err = Object.keys(d.errors || {}).length ? ` (failed: ${Object.keys(d.errors).join(', ')})` : '';
        status('perimeters', {
          state: err ? 'error' : 'ok',
          detail: `${d.current.features.length} current perimeters and ${d.incidents.features.length} CAL FIRE incidents statewide; ${d.historic.features.length} past fires (1950+) in this region${err}`,
        });
      })
      .catch((e) => status('perimeters', { state: 'error', detail: String(e.message || e) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.perimeters, selectedRegion]);

  // ---------------------------------------------------------------- NASA FIRMS hotspots (California, 24 h)
  useEffect(() => {
    if (!map) return;
    setLayer('hotspots', null);
    if (!layers.hotspots) return status('hotspots', { state: 'off' });
    status('hotspots', { state: 'loading' });
    getJSON('/api/layers/firms-hotspots')
      .then((d) => {
        const fc = d.hotspots;
        setLayer(
          'hotspots',
          L.geoJSON(fc, {
            pointToLayer: (f, ll) =>
              L.circleMarker(ll, {
                radius: Math.min(4 + Math.sqrt(f.properties.frp_mw || 0), 12),
                color: '#7c2d12',
                weight: 1,
                fillColor: (f.properties.frp_mw || 0) > 20 ? '#f97316' : '#fbbf24',
                fillOpacity: 0.85,
              }),
            onEachFeature: (f, l) => {
              const p = f.properties;
              l.bindTooltip(
                `<div class="text-xs"><b>${esc(p.sensor)} detection</b><br/>${esc(p.acquired_utc)} (${p.day_night === 'D' ? 'day' : 'night'})<br/>FRP ${fmt(p.frp_mw, 1)} MW · I4 ${fmt(p.brightness_ti4_k, 1)} K · confidence ${esc(p.confidence)}<br/><span class="opacity-70">${esc(p.source)}</span></div>`,
              );
            },
          }),
        );
        status('hotspots', {
          state: 'ok',
          detail: `${fc.features.length} VIIRS detections in California, last 24 h${fc.errors?.length ? ` (missing: ${fc.errors.length} sensor file)` : ''}`,
        });
      })
      .catch((e) => status('hotspots', { state: 'error', detail: String(e.message || e) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.hotspots]);

  // ---------------------------------------------------------------- CDEC RAWS fuel moisture
  useEffect(() => {
    if (!map) return;
    setLayer('fuel', null);
    if (!layers.fuel) return status('fuel', { state: 'off' });
    status('fuel', { state: 'loading' });
    getJSON(`/api/layers/fuel-moisture?bbox=${bboxParam(region.bbox)}`)
      .then((d) => {
        const fc = d.stations;
        const color = (fm: number | null) =>
          fm === null || fm === undefined ? '#64748b' : fm <= 6 ? '#dc2626' : fm <= 10 ? '#f97316' : fm <= 15 ? '#eab308' : '#14b8a6';
        setLayer(
          'fuel',
          L.geoJSON(fc, {
            pointToLayer: (f, ll) =>
              L.circleMarker(ll, { radius: 9, color: '#0f172a', weight: 2, fillColor: color(f.properties.fuel_moisture_10h_pct), fillOpacity: 0.95 }),
            onEachFeature: (f, l) => {
              const p = f.properties;
              l.bindTooltip(
                `<div class="text-xs"><b>RAWS ${esc(p.station_name)} (${esc(p.station_id)})</b><br/>10-h fuel moisture: <b>${p.fuel_moisture_10h_pct ?? '—'}%</b> (${esc(p.dryness_class)})<br/>${p.air_temp_f ?? '—'}°F · RH ${p.rel_humidity_pct ?? '—'}% · wind ${p.wind_speed_mph ?? '—'} mph${p.wind_gust_mph != null ? ` (gust ${p.wind_gust_mph})` : ''}<br/>Observed ${esc(p.fuel_moisture_10h_pct_time_pst)} PST · ${esc(p.operator)}<br/><span class="opacity-70">${esc(p.source)}</span></div>`,
              );
            },
          }),
        );
        const withData = fc.features.filter((f: any) => f.properties.fuel_moisture_10h_pct != null).length;
        status('fuel', { state: 'ok', detail: `${withData}/${fc.features.length} stations reporting (hourly, PST)` });
      })
      .catch((e) => status('fuel', { state: 'error', detail: String(e.message || e) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.fuel, selectedRegion]);

  // ---------------------------------------------------------------- Raster overlays: slope, USDA burn probability, NDVI
  const rasterLayer = (id: string, on: boolean, path: string, op: number, detail: (l: any) => string) => {
    if (!map) return;
    setLayer(id, null);
    if (!on) return status(id, { state: 'off' });
    status(id, { state: 'loading' });
    getJSON(path)
      .then((d) => {
        setLayer(id, L.imageOverlay(`${API_BASE}${d.layer.image_url}`, d.layer.bounds, { opacity: op, interactive: false }));
        status(id, { state: 'ok', detail: detail(d.layer) });
      })
      .catch((e) => status(id, { state: 'error', detail: String(e.message || e) }));
  };
  useEffect(() => {
    rasterLayer('slope', layers.slope, `/api/layers/terrain-slope?bbox=${bboxParam(region.bbox)}`, 0.75, (l) =>
      `~${l.pixel_m} m pixels; ≥30% slope on ${fmt((l.area_share_pct['30_50'] || 0) + (l.area_share_pct.ge50 || 0), 1)}% of the region`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.slope, selectedRegion]);
  useEffect(() => {
    rasterLayer('wrc', layers.wrc, `/api/layers/wrc-burn-probability?bbox=${bboxParam(region.bbox)}`, 0.8, (l) =>
      `30 m; max ${fmt(l.max_annual_bp * 100, 2)}%/yr, median ${fmt((l.median_annual_bp_nonzero ?? 0) * 100, 2)}%/yr (landscape as of 2020)`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.wrc, selectedRegion]);
  useEffect(() => {
    rasterLayer('ndvi', layers.ndvi, `/api/layers/ndvi?bbox=${bboxParam(region.bbox)}`, 0.7, (l) =>
      `Scene ${l.acquired_utc?.slice(0, 10)} (${l.cloud_cover_pct}% cloud)`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.ndvi, selectedRegion]);

  // ---------------------------------------------------------------- CAL FIRE FHSZ
  useEffect(() => {
    if (!map) return;
    setLayer('fhsz', null);
    if (!layers.fhsz) return status('fhsz', { state: 'off' });
    status('fhsz', { state: 'loading' });
    getJSON(`/api/layers/fhsz?bbox=${bboxParam(region.bbox)}`)
      .then((d) => {
        setLayer(
          'fhsz',
          L.geoJSON(d.zones, {
            style: (f) => ({ color: f?.properties?.color, weight: 1, fillColor: f?.properties?.color, fillOpacity: 0.18 }),
            onEachFeature: (f, l) =>
              l.bindTooltip(`<div class="text-xs"><b>FHSZ ${esc(f.properties.zone)}</b><br/>${esc(f.properties.source)}</div>`, { sticky: true }),
          }),
        );
        status('fhsz', { state: 'ok', detail: `${d.zones.features.length} zone polygons (SRA 2023, LRA 2025)` });
      })
      .catch((e) => status('fhsz', { state: 'error', detail: String(e.message || e) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.fhsz, selectedRegion]);

  // ---------------------------------------------------------------- HRRR wind grid
  useEffect(() => {
    if (!map) return;
    setLayer('wind', null);
    if (!layers.wind) return status('wind', { state: 'off' });
    status('wind', { state: 'loading' });
    getJSON(`/api/weather/wind-grid?bbox=${bboxParam(region.bbox)}&n=6`)
      .then((d) => {
        const g = L.layerGroup();
        let t = '';
        for (const f of d.wind.features) {
          const p = f.properties;
          t = p.valid_time_utc;
          const to = (p.wind_from_deg + 180) % 360;
          const icon = L.divIcon({
            className: '',
            iconSize: [44, 44],
            iconAnchor: [22, 22],
            html: `<div style="transform:rotate(${to}deg);width:44px;height:44px;display:flex;align-items:center;justify-content:center"><svg width="22" height="34" viewBox="0 0 22 34"><path d="M11 0 L20 12 L13 12 L13 34 L9 34 L9 12 L2 12 Z" fill="#60a5fa" stroke="#0f172a" stroke-width="1.5"/></svg></div>`,
          });
          L.marker([f.geometry.coordinates[1], f.geometry.coordinates[0]], { icon, interactive: true })
            .bindTooltip(
              `<div class="text-xs"><b>HRRR 10 m wind</b><br/>from ${fmt(p.wind_from_deg)}° at ${fmt(p.wind_speed_mph, 1)} mph (gust ${fmt(p.wind_gust_mph, 1)})<br/>valid ${esc(p.valid_time_utc)}</div>`,
            )
            .addTo(g);
        }
        setLayer('wind', g);
        status('wind', { state: 'ok', detail: `HRRR via Open-Meteo, valid ${t}` });
      })
      .catch((e) => status('wind', { state: 'error', detail: String(e.message || e) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.wind, selectedRegion]);

  // ---------------------------------------------------------------- Building footprints (viewport, zoom >= 15)
  useEffect(() => {
    if (!map) return;
    let cancelled = false;
    const load = () => {
      if (!layers.buildings) {
        setLayer('buildings', null);
        return status('buildings', { state: 'off' });
      }
      if (map.getZoom() < BUILDING_MIN_ZOOM) {
        setLayer('buildings', null);
        return status('buildings', { state: 'info', detail: `Zoom in to level ${BUILDING_MIN_ZOOM}+ to load footprints` });
      }
      const b = map.getBounds();
      status('buildings', { state: 'loading' });
      getJSON(`/api/layers/buildings?bbox=${bboxParam([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()])}`)
        .then((d) => {
          if (cancelled) return;
          const fc = d.buildings;
          setLayer(
            'buildings',
            L.geoJSON(fc, {
              style: () => ({ color: '#10b981', weight: 1, fillColor: '#059669', fillOpacity: 0.4 }),
              onEachFeature: (f, l) =>
                l.bindTooltip(
                  `<div class="text-xs"><b>Building footprint</b><br/>${fmt(f.properties.area_m2)} m² (${fmt(f.properties.area_sqft)} sq ft)<br/><span class="opacity-70">Microsoft US Building Footprints</span></div>`,
                  { sticky: true },
                ),
            }),
          );
          status('buildings', {
            state: 'ok',
            detail: `${fc.features.length.toLocaleString()} footprints${fc.truncated ? ` (sample of ${fc.n_in_view.toLocaleString()} in view)` : ''}`,
          });
        })
        .catch((e) => status('buildings', { state: 'error', detail: String(e.message || e) }));
    };
    load();
    map.on('moveend', load);
    return () => {
      cancelled = true;
      map.off('moveend', load);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.buildings]);

  // ---------------------------------------------------------------- Click: point query
  useEffect(() => {
    if (!map) return;
    const onClick = (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      const c = clickRefs.current;
      if (c.marker) map.removeLayer(c.marker);
      if (c.circle) map.removeLayer(c.circle);
      c.marker = L.marker([lat, lng], {
        icon: L.divIcon({
          className: '',
          iconSize: [18, 18],
          iconAnchor: [9, 9],
          html: '<div style="width:18px;height:18px;border-radius:9px;background:#f97316;border:3px solid #fde68a"></div>',
        }),
      }).addTo(map);
      onSelectLocation({ loading: true, center: { lat, lng } });
      getJSON('/api/query-corridor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng, radius_feet: 130 }),
      })
        .then((d) => {
          onSelectLocation(d.data);
          c.circle = L.circle([lat, lng], {
            radius: d.data.wind_adjusted_major_radius_meters || d.data.radius_meters,
            color: '#f97316',
            weight: 2,
            fillColor: '#ea580c',
            fillOpacity: 0.25,
            dashArray: '4 6',
          }).addTo(map);
        })
        .catch((err) => onSelectLocation({ error: String(err.message || err), center: { lat, lng } }));
    };
    map.on('click', onClick);
    return () => {
      map.off('click', onClick);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map]);

  return (
    <div className="w-full h-full relative">
      <div id="leaflet-map" className="w-full h-full z-10" />
    </div>
  );
}
