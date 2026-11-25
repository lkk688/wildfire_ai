'use client';

import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Map, Eye, Sparkles, HelpCircle, MousePointerClick, X } from 'lucide-react';
import Sidebar from './components/Sidebar';
import CorridorMetricsCard from './components/CorridorMetricsCard';
import RiskDriversChart from './components/RiskDriversChart';
import DataCatalogModal from './components/DataCatalogModal';
import ModelEvaluationModal from './components/ModelEvaluationModal';
import AICopilotDrawer from './components/AICopilotDrawer';
import VisionBenchmarkView from './components/VisionBenchmarkView';
import GuideModal from './components/GuideModal';
import MapLegend from './components/MapLegend';
import TimeSlider from './components/TimeSlider';
import { WeatherStore } from './lib/weatherStore';
import type { WxVar } from './lib/weatherVars';
import { REGIONS } from './lib/api';
import type { LayerToggles } from './components/MapView';
import type { LayerState } from './lib/api';

const MapView = dynamic(() => import('./components/MapView'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-slate-950 flex flex-col items-center justify-center text-slate-400 gap-3">
      <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-sm font-medium">Loading map...</span>
    </div>
  ),
});


export default function Home() {
  const [appMode, setAppMode] = useState<'map' | 'vision'>('map');


  const [activeLayer, setActiveLayer] = useState<'1yr' | '5yr' | 'none'>('1yr');
  const [selectedRegion, setSelectedRegion] = useState<string>('san_bruno');
  const [layers, setLayers] = useState<LayerToggles>({
    perimeters: true,
    hotspots: true,
    fuel: true,
    slope: false,
    buildings: true,
    ndvi: false,
    fhsz: false,
    wrc: false,
    wind: true,
    alerts: true,
  });
  const [layerStatus, setLayerStatus] = useState<Record<string, LayerState>>({});
  // Hourly weather: one tile store for the map layers and the time slider; hour null = now
  const [wxStore] = useState(() => new WeatherStore());
  const [wxHour, setWxHour] = useState<number | null>(null);
  const [wxField, setWxField] = useState<WxVar | 'none'>('none');
  const [opacity, setOpacity] = useState<number>(0.75);
  // California data cache layers (backend/geodata), drawn as map tiles; ?geo=fuel_model,ignitions
  const [geoLayers, setGeoLayers] = useState<string[]>([]);
  const [geoOpacity, setGeoOpacity] = useState<number>(0.8);
  const [mapZoom, setMapZoom] = useState<number>(13);
  const [selectedCorridorData, setSelectedCorridorData] = useState<any>(null);
  const [layerStats, setLayerStats] = useState<any>(null);
  const [isDataCatalogOpen, setIsDataCatalogOpen] = useState<boolean>(false);
  const [isModelEvalOpen, setIsModelEvalOpen] = useState<boolean>(false);
  const [initialView, setInitialView] = useState<{ lat: number; lng: number; zoom: number } | null>(null);
  const [ready, setReady] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [hintDismissed, setHintDismissed] = useState(false);
  const GUIDE_KEY = 'wildfire_ai_guide_seen_v1';
  const closeGuide = useCallback(() => {
    setGuideOpen(false);
    try {
      localStorage.setItem(GUIDE_KEY, '1');
    } catch {
      /* storage unavailable: the guide simply shows again next time */
    }
  }, []);

  // Shareable views: ?mode=vision | ?region=santa_cruz&layers=fhsz,slope&bw=5yr&lat=37.1&lng=-122.1&z=15
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('mode') === 'vision' || params.get('case')) setAppMode('vision');
    const r = params.get('region');
    if (r) setSelectedRegion(r);
    const bw = params.get('bw');
    if (bw === '1yr' || bw === '5yr' || bw === 'none') setActiveLayer(bw);
    const ls = params.get('layers');
    if (ls !== null) {
      const on = new Set(ls.split(',').filter(Boolean));
      setLayers((prev) => Object.fromEntries(Object.keys(prev).map((k) => [k, on.has(k)])) as LayerToggles);
    }
    const geo = params.get('geo');
    if (geo) setGeoLayers(geo.split(',').filter(Boolean));
    const lat = parseFloat(params.get('lat') || ''), lng = parseFloat(params.get('lng') || ''), z = parseInt(params.get('z') || '', 10);
    if (params.get('panel') === 'eval') setIsModelEvalOpen(true);
    if (params.get('panel') === 'sources') setIsDataCatalogOpen(true);
    if (Number.isFinite(lat) && Number.isFinite(lng)) setInitialView({ lat, lng, zoom: Number.isFinite(z) ? z : 14 });
    // First visit: show the guide, unless the link points somewhere specific (?guide=1 forces it, ?guide=0 hides it).
    let seen = false;
    try {
      seen = localStorage.getItem(GUIDE_KEY) === '1';
    } catch {
      seen = false;
    }
    const deepLink = ['case', 'lat', 'panel', 'layers', 'mode', 'geo'].some((k) => params.has(k));
    const g = params.get('guide');
    if (g === '1' || (g !== '0' && !seen && !deepLink)) setGuideOpen(true);
    setReady(true);
  }, []);

  const updateStatus = useCallback(
    (id: string, s: LayerState) => setLayerStatus((prev) => ({ ...prev, [id]: s })),
    [],
  );

  return (
    <main className="flex w-screen h-screen overflow-hidden bg-slate-950 relative">
      {/* Floating Application Mode Switcher (Global Top Navigation) */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 bg-slate-900/90 backdrop-blur-xl border border-slate-700/80 p-1 rounded-2xl shadow-2xl flex items-center gap-1">
        <button
          id="nav-btn-map"
          onClick={() => setAppMode('map')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
            appMode === 'map'
              ? 'bg-gradient-to-r from-orange-500 to-rose-600 text-white shadow-lg shadow-orange-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Map className="w-3.5 h-3.5" />
          <span>Risk Map (live data)</span>
        </button>

        <button
          id="nav-btn-vision"
          onClick={() => setAppMode('vision')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer relative ${
            appMode === 'vision'
              ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg shadow-purple-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Eye className="w-3.5 h-3.5 text-purple-300" />
          <span>VLM Evaluation (held-out fires)</span>
        </button>

        <button
          id="nav-btn-guide"
          onClick={() => setGuideOpen(true)}
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold text-slate-300 hover:text-white hover:bg-slate-800/60 cursor-pointer"
          title="How to use this site"
        >
          <HelpCircle className="w-3.5 h-3.5 text-sky-300" />
          <span>Guide</span>
        </button>
      </div>

      {/* Conditionally Render Map Application vs Vision Application */}
      {appMode === 'map' ? (
        <>
          {/* Sidebar Controls */}
          <Sidebar
            activeLayer={activeLayer}
            setActiveLayer={setActiveLayer}
            selectedRegion={selectedRegion}
            setSelectedRegion={setSelectedRegion}
            layers={layers}
            setLayers={setLayers}
            layerStatus={layerStatus}
            opacity={opacity}
            setOpacity={setOpacity}
            layerStats={layerStats}
            onOpenDataCatalog={() => setIsDataCatalogOpen(true)}
            onOpenModelEvaluation={() => setIsModelEvalOpen(true)}
            onSwitchToVision={() => setAppMode('vision')}
            wxField={wxField}
            setWxField={setWxField}
            geoLayers={geoLayers}
            setGeoLayers={setGeoLayers}
            geoOpacity={geoOpacity}
            setGeoOpacity={setGeoOpacity}
            mapZoom={mapZoom}
          />

          {/* Main Interactive Map View */}
          <div className="flex-1 h-full relative">
            {ready && <MapView
              initialView={initialView}
              activeLayer={activeLayer}
              selectedRegion={selectedRegion}
              layers={layers}
              opacity={opacity}
              onSelectLocation={(data) => {
                setSelectedCorridorData(data);
                setHintDismissed(true);
              }}
              onLayerStatsUpdate={(stats) => setLayerStats(stats)}
              onLayerStatus={updateStatus}
              wxStore={wxStore}
              wxHour={wxHour}
              wxField={wxField}
              geoLayers={geoLayers}
              geoOpacity={geoOpacity}
              onZoomChange={setMapZoom}
            />}

            {(layers.wind || layers.alerts || wxField !== 'none') && (
              <TimeSlider
                store={wxStore}
                hour={wxHour}
                setHour={setWxHour}
                field={wxField}
                center={{
                  lat: (REGIONS[selectedRegion] || REGIONS.san_bruno).center[0],
                  lng: (REGIONS[selectedRegion] || REGIONS.san_bruno).center[1],
                  name: (REGIONS[selectedRegion] || REGIONS.san_bruno).name.split(' (')[0],
                }}
              />
            )}

            <MapLegend activeLayer={activeLayer} layers={layers} layerStatus={layerStatus} wxField={wxField} geoLayers={geoLayers} />

            {/* First-time hint: the point query is the most useful thing on the map */}
            {!selectedCorridorData && !hintDismissed && (
              <div id="map-hint" className="absolute top-20 left-1/2 -translate-x-1/2 z-[500] flex items-center gap-2 pl-3 pr-1.5 py-1.5 rounded-full bg-orange-500/95 text-white text-xs font-semibold shadow-xl pointer-events-auto">
                <MousePointerClick className="w-4 h-4" />
                <span>Click anywhere on the map to see everything known about that spot</span>
                <button onClick={() => setHintDismissed(true)} aria-label="Dismiss hint" className="p-0.5 rounded-full hover:bg-white/20 cursor-pointer">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Point query panel (real data at the clicked location) */}
            <CorridorMetricsCard
              data={selectedCorridorData}
              onClose={() => setSelectedCorridorData(null)}
            />

            {/* Aggregated Top Risk Factors Bar Chart */}
            <RiskDriversChart region={selectedRegion} raised={layers.wind || layers.alerts || wxField !== 'none'} />
          </div>

          {/* Multi-Source Data Documentation Modal */}
          <DataCatalogModal
            isOpen={isDataCatalogOpen}
            onClose={() => setIsDataCatalogOpen(false)}
            region={selectedRegion}
          />

          {/* Off-the-Shelf Model Evaluation & Gap Analysis Modal */}
          <ModelEvaluationModal
            isOpen={isModelEvalOpen}
            onClose={() => setIsModelEvalOpen(false)}
            selectedRegion={selectedRegion}
          />
        </>
      ) : (
        /* Dedicated VLM Vision & Physical Fire Reasoning Application */
        <VisionBenchmarkView onSwitchToMap={() => setAppMode('map')} />
      )}

      <GuideModal
        open={guideOpen}
        onClose={closeGuide}
        onGo={(m) => {
          setAppMode(m);
          closeGuide();
        }}
      />

      {/* Floating AI Wildfire Copilot (Accessible in both Map and Vision modes) */}
      <AICopilotDrawer selectedRegion={selectedRegion} />
    </main>
  );
}



