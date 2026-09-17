'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from './components/Sidebar';
import CorridorMetricsCard from './components/CorridorMetricsCard';
import RiskDriversChart from './components/RiskDriversChart';
import DataCatalogModal from './components/DataCatalogModal';
import ModelEvaluationModal from './components/ModelEvaluationModal';
import AICopilotDrawer from './components/AICopilotDrawer';

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

      {/* Floating AI Wildfire Copilot (OpenAI-Compatible LLM with live GIS tools) */}
      <AICopilotDrawer selectedRegion={selectedRegion} />
    </main>
  );
}


