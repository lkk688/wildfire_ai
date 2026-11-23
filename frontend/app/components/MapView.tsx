'use client';

import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { API_BASE, REGIONS, LayerState, getJSON } from '../lib/api';
import { WindParticles, ScalarField } from '../lib/windParticles';
import { swrJSON, CacheMeta } from '../lib/cache';
import { WeatherStore } from '../lib/weatherStore';
import type { WxVar } from '../lib/weatherVars';

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
  alerts: boolean;
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
  wxStore: WeatherStore;
  wxHour: number | null; // epoch hour picked on the time slider; null = now
  wxField: WxVar | 'none';
}

const BUILDING_MIN_ZOOM = 15;
const MIN = 60 * 1000;
const DAY = 24 * 60 * MIN;

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
  wxStore,
  wxHour,
  wxField,
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
  // Cached copies are shown at once and refreshed in the background (lib/cache.ts); the status line says so.
  const meta = useRef<Record<string, CacheMeta>>({});
  const status = (id: string, s: LayerState) => {
    const m = meta.current[id];
    if (m?.cached && s.state === 'ok' && s.detail)
      s = { ...s, detail: `${s.detail} · cached copy from ${new Date(m.fetchedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}` };
    statusRef.current(id, s);
  };
  const fail = (id: string) => (e: Error, hadCopy: boolean, at: number | null) =>
    status(id, hadCopy
      ? { state: 'info', detail: `Source not answering (${String(e.message || e).slice(0, 120)}); showing the copy from ${at ? new Date(at).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : 'earlier'}` }
      : { state: 'error', detail: String(e.message || e) });

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
      'Bellwether (X), NIFC, CAL FIRE, NASA FIRMS, CDEC, NOAA HRRR/Open-Meteo, NOAA NWS, USGS 3DEP, USDA Forest Service WRC, Copernicus Sentinel-2/Esri, ESA WorldCover, Microsoft Building Footprints';
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
    return swrJSON(`/api/layers/bellwether?is_5_year=${activeLayer === '5yr'}&region=${selectedRegion}`, DAY, (d, m) => {
      meta.current['bellwether'] = m;
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
    }, fail('bellwether'));
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
    return swrJSON(`/api/layers/perimeters?bbox=${bboxParam(region.bbox)}`, 5 * MIN, (d, m) => {
      meta.current['perimeters'] = m;
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
    }, fail('perimeters'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.perimeters, selectedRegion]);

  // ---------------------------------------------------------------- NASA FIRMS hotspots (California, 24 h)
  useEffect(() => {
    if (!map) return;
    setLayer('hotspots', null);
    if (!layers.hotspots) return status('hotspots', { state: 'off' });
    status('hotspots', { state: 'loading' });
    return swrJSON('/api/layers/firms-hotspots', 10 * MIN, (d, m) => {
      meta.current['hotspots'] = m;
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
    }, fail('hotspots'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.hotspots]);

  // ---------------------------------------------------------------- CDEC RAWS fuel moisture
  useEffect(() => {
    if (!map) return;
    setLayer('fuel', null);
    if (!layers.fuel) return status('fuel', { state: 'off' });
    status('fuel', { state: 'loading' });
    return swrJSON(`/api/layers/fuel-moisture?bbox=${bboxParam(region.bbox)}`, 15 * MIN, (d, m) => {
      meta.current['fuel'] = m;
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
    }, fail('fuel'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.fuel, selectedRegion]);

  // ---------------------------------------------------------------- Raster overlays: slope, USDA burn probability, NDVI
  const rasterLayer = (id: string, on: boolean, path: string, op: number, detail: (l: any) => string, ttl = 7 * DAY) => {
    if (!map) return;
    setLayer(id, null);
    if (!on) return status(id, { state: 'off' });
    status(id, { state: 'loading' });
    return swrJSON(path, ttl, (d, m) => {
      meta.current[id] = m;
        setLayer(id, L.imageOverlay(`${API_BASE}${d.layer.image_url}`, d.layer.bounds, { opacity: op, interactive: false }));
        status(id, { state: 'ok', detail: detail(d.layer) });
    }, fail(id));
  };
  useEffect(() => {
    return rasterLayer('slope', layers.slope, `/api/layers/terrain-slope?bbox=${bboxParam(region.bbox)}`, 0.75, (l) =>
      `~${l.pixel_m} m pixels; ≥30% slope on ${fmt((l.area_share_pct['30_50'] || 0) + (l.area_share_pct.ge50 || 0), 1)}% of the region`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.slope, selectedRegion]);
  useEffect(() => {
    return rasterLayer('wrc', layers.wrc, `/api/layers/wrc-burn-probability?bbox=${bboxParam(region.bbox)}`, 0.8, (l) =>
      `30 m; max ${fmt(l.max_annual_bp * 100, 2)}%/yr, median ${fmt((l.median_annual_bp_nonzero ?? 0) * 100, 2)}%/yr (landscape as of 2020)`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.wrc, selectedRegion]);
  useEffect(() => {
    return rasterLayer('ndvi', layers.ndvi, `/api/layers/ndvi?bbox=${bboxParam(region.bbox)}`, 0.7, (l) =>
      `Scene ${l.acquired_utc?.slice(0, 10)} (${l.cloud_cover_pct}% cloud)`, DAY,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.ndvi, selectedRegion]);

  // ---------------------------------------------------------------- CAL FIRE FHSZ
  useEffect(() => {
    if (!map) return;
    setLayer('fhsz', null);
    if (!layers.fhsz) return status('fhsz', { state: 'off' });
    status('fhsz', { state: 'loading' });
    return swrJSON(`/api/layers/fhsz?bbox=${bboxParam(region.bbox)}`, DAY, (d, m) => {
      meta.current['fhsz'] = m;
        setLayer(
          'fhsz',
          L.geoJSON(d.zones, {
            style: (f) => ({ color: f?.properties?.color, weight: 1, fillColor: f?.properties?.color, fillOpacity: 0.18 }),
            onEachFeature: (f, l) =>
              l.bindTooltip(`<div class="text-xs"><b>FHSZ ${esc(f.properties.zone)}</b><br/>${esc(f.properties.source)}</div>`, { sticky: true }),
          }),
        );
        status('fhsz', { state: 'ok', detail: `${d.zones.features.length} zone polygons (SRA 2023, LRA 2025)` });
    }, fail('fhsz'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.fhsz, selectedRegion]);

  // ---------------------------------------------------------------- Hourly HRRR weather (time slider)
  const wxOn = layers.wind || wxField !== 'none';
  const hourNow = () => Math.floor(Date.now() / 3600000);
  const hourRef = useRef<number>(wxHour ?? hourNow());
  hourRef.current = wxHour ?? hourNow();

  // load the tiles that cover the view, now and after every move
  useEffect(() => {
    if (!map || !wxOn) return;
    const load = () => {
      const b = map.getBounds().pad(0.1);
      wxStore.setView(b.getWest(), b.getSouth(), b.getEast(), b.getNorth());
    };
    load();
    map.on('moveend', load);
    const refresh = setInterval(load, 10 * MIN);
    return () => {
      map.off('moveend', load);
      clearInterval(refresh);
    };
  }, [map, wxOn, wxStore]);

  // status lines for the weather layers, from the tile store
  useEffect(() => {
    const upd = () => {
      const st = wxStore.status();
      const base = `NOAA HRRR, hourly; ${st.tiles} tile${st.tiles === 1 ? '' : 's'} at ${st.step}° spacing${st.fromCache ? `, ${st.fromCache} from cache` : ''}`;
      const s: LayerState = st.loading && !st.tiles
        ? { state: 'loading' }
        : st.failed && !st.tiles
          ? { state: 'error', detail: `Weather source not answering (${st.error}); retrying every minute` }
          : st.stale || st.failed
            ? { state: 'info', detail: `${base}. Open-Meteo is not answering for part of the view; showing the last good values${st.fetchedAt ? ` (from ${new Date(st.fetchedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })})` : ''}` }
            : { state: 'ok', detail: `${base}. Pick the hour on the time slider; hover the map for values.` };
      if (layers.wind) status('wind', s);
      if (wxField !== 'none') status('wxfield', s);
    };
    if (!layers.wind) status('wind', { state: 'off' });
    if (wxField === 'none') status('wxfield', { state: 'off' });
    if (!wxOn) return;
    upd();
    return wxStore.subscribe(upd);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wxStore, layers.wind, wxField, wxOn]);

  // animated wind
  const windRef = useRef<WindParticles | null>(null);
  useEffect(() => {
    if (!map || !layers.wind) return;
    const layer = new WindParticles(wxStore, hourRef.current);
    windRef.current = layer;
    layer.addTo(map);
    return () => {
      windRef.current = null;
      map.removeLayer(layer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.wind, wxStore]);

  // colour field (temperature, humidity, gusts, rain, cloud, fire weather)
  const fieldRef = useRef<ScalarField | null>(null);
  useEffect(() => {
    if (!map || wxField === 'none') return;
    const layer = new ScalarField(wxStore, wxField, hourRef.current);
    fieldRef.current = layer;
    layer.addTo(map);
    return () => {
      fieldRef.current = null;
      map.removeLayer(layer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, wxField === 'none', wxStore]);

  useEffect(() => {
    const h = wxHour ?? hourNow();
    windRef.current?.setHour(h);
    if (wxField !== 'none') fieldRef.current?.set(wxField, h);
  }, [wxHour, wxField]);

  // hover position feeds the readout on the time slider
  useEffect(() => {
    if (!map) return;
    const mv = (e: L.LeafletMouseEvent) => wxStore.setHover({ lat: e.latlng.lat, lng: e.latlng.lng });
    const out = () => wxStore.setHover(null);
    map.on('mousemove', mv);
    map.on('mouseout', out);
    return () => {
      map.off('mousemove', mv);
      map.off('mouseout', out);
    };
  }, [map, wxStore]);

  // ---------------------------------------------------------------- NWS watches / warnings / advisories (all of CA)
  // The feed lists alerts that are active or upcoming; the map shows those in effect at the slider's hour.
  const alertsRef = useRef<any>(null);
  const drawAlerts = () => {
    const fc = alertsRef.current;
    if (!map || !fc || !layers.alerts) return;
    const t = (wxHour ?? hourNow()) * 3600000;
    const when = (v: string | null) => (v ? new Date(v).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '—');
    const rank = (p: any) =>
      (/Warning/.test(p.event) ? 30 : /Watch/.test(p.event) ? 20 : /Advisory/.test(p.event) ? 10 : 0) +
      (({ Extreme: 4, Severe: 3, Moderate: 2, Minor: 1 } as Record<string, number>)[p.severity] || 0);
    const live = fc.features
      .filter((f: any) => {
        const on = Date.parse(f.properties.onset || '') || 0, end = Date.parse(f.properties.ends || '') || Infinity;
        return on <= t + 3599999 && t < end;
      })
      .sort((x: any, y: any) => rank(x.properties) - rank(y.properties)); // warnings drawn on top
    setLayer(
      'alerts',
      L.geoJSON({ type: 'FeatureCollection', features: live } as any, {
        style: (f) => ({ color: f?.properties?.color, weight: 2, fillColor: f?.properties?.color, fillOpacity: 0.12 }),
        onEachFeature: (f, l) =>
          l.bindTooltip(
            `<div class="text-xs" style="width:300px;white-space:normal"><b>${esc(f.properties.event)}</b> <span style="opacity:.7">(${esc(f.properties.severity)})</span><br/>` +
              `${esc(when(f.properties.onset))} – ${esc(when(f.properties.ends))}<br/>` +
              `<span style="opacity:.8">${esc(f.properties.area)}</span><br/><span style="opacity:.7">${esc(f.properties.sender)} · National Weather Service</span></div>`,
            { sticky: true },
          ),
      }),
    );
    const events = new Map<string, string>();
    for (const f of live) events.set(f.properties.event, f.properties.color);
    const n = new Set(live.map((f: any) => f.properties.headline)).size;
    const at = wxHour == null ? 'now' : `at ${new Date(t).toLocaleString([], { weekday: 'short', hour: 'numeric' })}`;
    status('alerts', {
      state: fc.stale ? 'info' : 'ok',
      detail:
        (fc.stale ? `api.weather.gov is not answering; alerts as of ${new Date(fc.fetched_utc).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}. ` : '') +
        `${n} of ${fc.alert_count} active or upcoming California alerts in effect ${at}. Past alerts that have expired are not in the NWS feed.`,
      swatches: [...events].map(([label, color]) => ({ label, color })),
    });
  };
  useEffect(() => {
    if (!map) return;
    setLayer('alerts', null);
    if (!layers.alerts) return status('alerts', { state: 'off' });
    status('alerts', { state: 'loading' });
    let cancel = () => {};
    const load = () => {
      cancel();
      cancel = swrJSON('/api/weather/alerts?area=CA', 5 * MIN, (d) => {
        alertsRef.current = d.alerts;
        drawAlerts();
      }, fail('alerts'));
    };
    load();
    const refresh = setInterval(load, 5 * MIN);
    return () => {
      cancel();
      clearInterval(refresh);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, layers.alerts]);
  useEffect(() => {
    drawAlerts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wxHour]);

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
