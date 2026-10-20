'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Map, Eye, Sparkles } from 'lucide-react';
import Sidebar from './components/Sidebar';
import CorridorMetricsCard from './components/CorridorMetricsCard';
import RiskDriversChart from './components/RiskDriversChart';
import DataCatalogModal from './components/DataCatalogModal';
import ModelEvaluationModal from './components/ModelEvaluationModal';
import AICopilotDrawer from './components/AICopilotDrawer';
import VisionBenchmarkView from './components/VisionBenchmarkView';

const MapView = dynamic(() => import('./components/MapView'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-slate-950 flex flex-col items-center justify-center text-slate-400 gap-3">
      <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-sm font-medium">Loading Wildfire Risk Intelligence GIS Map & Multi-Source Overlays...</span>
    </div>
  ),
});


export default function Home() {
  const [appMode, setAppMode] = useState<'map' | 'vision'>('map');

  // Automatically activate vision mode if URL specifies ?mode=vision or ?case=...
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('mode') === 'vision' || params.get('case')) {
        setAppMode('vision');
      }
    }
  }, []);
  const [activeLayer, setActiveLayer] = useState<'1yr' | '5yr' | 'none'>('1yr');
  const [selectedRegion, setSelectedRegion] = useState<string>('san_bruno');
  const [showSentinel, setShowSentinel] = useState<boolean>(true);
  const [showCalFire, setShowCalFire] = useState<boolean>(false);
  const [showWeather, setShowWeather] = useState<boolean>(true);
  const [showBuildings, setShowBuildings] = useState<boolean>(true);
  const [showCalfirePerimeters, setShowCalfirePerimeters] = useState<boolean>(true);
  const [showFirmsHotspots, setShowFirmsHotspots] = useState<boolean>(true);
  const [showFuelMoisture, setShowFuelMoisture] = useState<boolean>(true);
  const [showTerrainSlope, setShowTerrainSlope] = useState<boolean>(false);
  const [opacity, setOpacity] = useState<number>(0.75);
  const [selectedCorridorData, setSelectedCorridorData] = useState<any>(null);
  const [layerStats, setLayerStats] = useState<any>(null);
  const [isDataCatalogOpen, setIsDataCatalogOpen] = useState<boolean>(false);
  const [isModelEvalOpen, setIsModelEvalOpen] = useState<boolean>(false);


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
          <span>3D Risk GIS Map</span>
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
          <span>VLM Fire Vision & Reasoning</span>
          <span className="px-1.5 py-0.5 rounded-full text-[9px] font-extrabold bg-orange-500 text-white border border-orange-400/40 animate-pulse">
            NEW
          </span>
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
            showSentinel={showSentinel}
            setShowSentinel={setShowSentinel}
            showCalFire={showCalFire}
            setShowCalFire={setShowCalFire}
            showWeather={showWeather}
            setShowWeather={setShowWeather}
            showBuildings={showBuildings}
            setShowBuildings={setShowBuildings}
            showCalfirePerimeters={showCalfirePerimeters}
            setShowCalfirePerimeters={setShowCalfirePerimeters}
            showFirmsHotspots={showFirmsHotspots}
            setShowFirmsHotspots={setShowFirmsHotspots}
            showFuelMoisture={showFuelMoisture}
            setShowFuelMoisture={setShowFuelMoisture}
            showTerrainSlope={showTerrainSlope}
            setShowTerrainSlope={setShowTerrainSlope}
            opacity={opacity}
            setOpacity={setOpacity}
            layerStats={layerStats}
            onOpenDataCatalog={() => setIsDataCatalogOpen(true)}
            onOpenModelEvaluation={() => setIsModelEvalOpen(true)}
            onSwitchToVision={() => setAppMode('vision')}
          />

          {/* Main Interactive Map View */}
          <div className="flex-1 h-full relative">
            <MapView
              activeLayer={activeLayer}
              selectedRegion={selectedRegion}
              showSentinel={showSentinel}
              showCalFire={showCalFire}
              showWeather={showWeather}
              showBuildings={showBuildings}
              showCalfirePerimeters={showCalfirePerimeters}
              showFirmsHotspots={showFirmsHotspots}
              showFuelMoisture={showFuelMoisture}
              showTerrainSlope={showTerrainSlope}
              opacity={opacity}
              onSelectLocation={(data) => setSelectedCorridorData(data)}
              onLayerStatsUpdate={(stats) => setLayerStats(stats)}
            />

            {/* Real-Time Wind-Adjusted Corridor ROI Panel */}
            <CorridorMetricsCard
              data={selectedCorridorData}
              onClose={() => setSelectedCorridorData(null)}
            />

            {/* Aggregated Top Risk Factors Bar Chart */}
            <RiskDriversChart />
          </div>

          {/* Multi-Source Data Documentation Modal */}
          <DataCatalogModal
            isOpen={isDataCatalogOpen}
            onClose={() => setIsDataCatalogOpen(false)}
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

      {/* Floating AI Wildfire Copilot (Accessible in both Map and Vision modes) */}
      <AICopilotDrawer selectedRegion={selectedRegion} />
    </main>
  );
}



