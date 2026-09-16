'use client';

import React from 'react';
import { Flame, Layers, Eye, ShieldAlert, Sparkles, MapPin, Shield, Wind, BookOpen, Home, Mountain, Droplet, Radio, Scale } from 'lucide-react';

interface SidebarProps {
  activeLayer: '1yr' | '5yr' | 'none';
  setActiveLayer: (layer: '1yr' | '5yr' | 'none') => void;
  selectedRegion: string;
  setSelectedRegion: (region: string) => void;
  showSentinel: boolean;
  setShowSentinel: (show: boolean) => void;
  showCalFire: boolean;
  setShowCalFire: (show: boolean) => void;
  showWeather: boolean;
  setShowWeather: (show: boolean) => void;
  showBuildings: boolean;
  setShowBuildings: (show: boolean) => void;
  showCalfirePerimeters: boolean;
  setShowCalfirePerimeters: (show: boolean) => void;
  showFirmsHotspots: boolean;
  setShowFirmsHotspots: (show: boolean) => void;
  showFuelMoisture: boolean;
  setShowFuelMoisture: (show: boolean) => void;
  showTerrainSlope: boolean;
  setShowTerrainSlope: (show: boolean) => void;
  opacity: number;
  setOpacity: (val: number) => void;
  layerStats: any;
  onOpenDataCatalog: () => void;
  onOpenModelEvaluation: () => void;
}

export default function Sidebar({
  activeLayer,
  setActiveLayer,
  selectedRegion,
  setSelectedRegion,
  showSentinel,
  setShowSentinel,
  showCalFire,
  setShowCalFire,
  showWeather,
  setShowWeather,
  showBuildings,
  setShowBuildings,
  showCalfirePerimeters,
  setShowCalfirePerimeters,
  showFirmsHotspots,
  setShowFirmsHotspots,
  showFuelMoisture,
  setShowFuelMoisture,
  showTerrainSlope,
  setShowTerrainSlope,
  opacity,
  setOpacity,
  layerStats,
  onOpenDataCatalog,
  onOpenModelEvaluation,
}: SidebarProps) {

  const riskCategories = [
    { name: 'Very Low', range: '< 0.004%', color: '#228B22' },
    { name: 'Low', range: '0.004% - 0.02%', color: '#90EE90' },
    { name: 'Moderate', range: '0.02% - 0.10%', color: '#FFFF66' },
    { name: 'Significant', range: '0.10% - 0.40%', color: '#FFCC00' },
    { name: 'High', range: '0.40% - 0.67%', color: '#FF8000' },
    { name: 'Very High', range: '0.67% - 1.33%', color: '#EE2222' },
    { name: 'Extreme', range: '> 1.33%', color: '#9400D3' },
  ];

  return (
    <aside className="w-96 glass-panel h-screen flex flex-col z-20 shadow-2xl overflow-y-auto border-r border-slate-800">
      {/* Header */}
      <div className="p-5 border-b border-slate-800/80 bg-slate-900/40">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-orange-500/20 text-orange-400 border border-orange-500/30 glow-orange">
              <Flame className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="font-bold text-lg text-slate-100 tracking-wide">Wildfire AI Pilot</h1>
              <p className="text-xs text-orange-400 font-medium">San Bruno FD × SJSU × Google X</p>
            </div>
          </div>
        </div>

        {/* Region Selector Dropdown */}
        <div className="mt-3">
          <label className="text-[11px] font-semibold uppercase text-slate-400 tracking-wider mb-1 flex items-center gap-1.5">
            <MapPin className="w-3.5 h-3.5 text-orange-400" /> Target Region / City BBox
          </label>
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 text-slate-100 text-xs font-semibold rounded-xl p-2.5 outline-none focus:border-orange-500 transition-all cursor-pointer"
          >
            <option value="san_bruno">📍 San Bruno WUI (Peninsula)</option>
            <option value="san_jose">📍 San Jose Foothills (Santa Clara)</option>
            <option value="santa_cruz">📍 Santa Cruz Mountains (CZU WUI)</option>
          </select>
        </div>

        {/* Model Evaluation & Gap Analysis Button */}
        <button
          onClick={onOpenModelEvaluation}
          className="w-full mt-2.5 py-2.5 px-3 rounded-xl bg-gradient-to-r from-orange-500/25 via-slate-800 to-purple-500/25 hover:from-orange-500/35 hover:to-purple-500/35 border border-orange-500/40 text-xs font-bold text-orange-200 flex items-center justify-center gap-2 transition-all shadow-lg glow-orange group"
        >
          <Scale className="w-4 h-4 text-orange-400 group-hover:scale-110 transition-transform" />
          <span>📊 Model Evaluation & Gap Analysis</span>
        </button>

        {/* Data Catalog Documentation Button */}
        <button
          onClick={onOpenDataCatalog}
          className="w-full mt-2 py-2 px-3 rounded-xl bg-slate-800/70 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-300 flex items-center justify-center gap-2 transition-all hover:border-slate-600 shadow-sm"
        >
          <BookOpen className="w-4 h-4 text-amber-400" />
          <span>📖 Multi-Source Data Catalog</span>
        </button>
      </div>


      {/* Layer Controls */}
      <div className="p-5 space-y-4 flex-1">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5">
            <Layers className="w-4 h-4 text-orange-400" />
            Bellwether Hazard Models (100m Grid)
          </div>
          <div className="grid grid-cols-2 gap-2.5">
            <button
              onClick={() => setActiveLayer('1yr')}
              className={`p-3 rounded-xl border text-left transition-all ${
                activeLayer === '1yr'
                  ? 'bg-orange-500/20 border-orange-500/60 text-orange-300 shadow-lg glow-orange'
                  : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              <div className="text-xs font-medium text-slate-400">Model 1</div>
              <div className="text-sm font-bold mt-0.5">1-Year Horizon</div>
            </button>
            <button
              onClick={() => setActiveLayer('5yr')}
              className={`p-3 rounded-xl border text-left transition-all ${
                activeLayer === '5yr'
                  ? 'bg-orange-500/20 border-orange-500/60 text-orange-300 shadow-lg glow-orange'
                  : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
              }`}
            >
              <div className="text-xs font-medium text-slate-400">Model 2</div>
              <div className="text-sm font-bold mt-0.5">5-Year Horizon</div>
            </button>
          </div>
        </div>

        {/* Multi-Source Overlays Toggle Section */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Multi-Source Overlays & Infra
          </div>

          {/* CAL FIRE Active Perimeters */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20">
                <Flame className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">CAL FIRE Fire Perimeters</div>
                <div className="text-[10px] text-slate-400">NIFC Active/Historical Boundaries</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showCalfirePerimeters}
              onChange={(e) => setShowCalfirePerimeters(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* NASA FIRMS Satellite Hotspots */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <Radio className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">NASA FIRMS Thermal Hotspots</div>
                <div className="text-[10px] text-slate-400">VIIRS 375m Satellite Anomaly</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showFirmsHotspots}
              onChange={(e) => setShowFirmsHotspots(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* CDEC RAWS Fuel Moisture */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20">
                <Droplet className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">CDEC Fuel Moisture (DFM)</div>
                <div className="text-[10px] text-slate-400">10-hr/100-hr RAWS Stations</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showFuelMoisture}
              onChange={(e) => setShowFuelMoisture(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* USGS Terrain Slope */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <Mountain className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">USGS 3DEP Terrain Slope</div>
                <div className="text-[10px] text-slate-400">10m LiDAR Steepness (%)</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showTerrainSlope}
              onChange={(e) => setShowTerrainSlope(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* Microsoft / OSM / FEMA Building Footprints */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Home className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">Building Footprints (MS/OSM/FEMA)</div>
                <div className="text-[10px] text-slate-400">3D Polygon Geometries & Hydrants</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showBuildings}
              onChange={(e) => setShowBuildings(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* Sentinel-2 */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">Sentinel-2 Satellite</div>
                <div className="text-[10px] text-slate-400">10m NDVI Vegetation Index</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showSentinel}
              onChange={(e) => setShowSentinel(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* CAL FIRE Insurance Map */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20">
                <Shield className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">CAL FIRE Insurance FHSZ</div>
                <div className="text-[10px] text-slate-400">SRA/LRA Fire Hazard Zones</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showCalFire}
              onChange={(e) => setShowCalFire(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>

          {/* NOAA HRRR Weather Stream */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
                <Wind className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-200">NOAA Live Wind Plume</div>
                <div className="text-[10px] text-slate-400">HRRR Hourly Wind Vector</div>
              </div>
            </div>
            <input
              type="checkbox"
              checked={showWeather}
              onChange={(e) => setShowWeather(e.target.checked)}
              className="w-4 h-4 accent-orange-500 cursor-pointer rounded"
            />
          </div>
        </div>

        {/* Opacity Slider */}
        <div className="space-y-1.5 bg-slate-900/40 p-2.5 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <span className="flex items-center gap-1.5 font-medium">
              <Eye className="w-3.5 h-3.5 text-slate-400" /> Layer Opacity
            </span>
            <span className="font-mono text-orange-400 font-bold">{Math.round(opacity * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-500"
          />
        </div>

        {/* Risk Classification Legend */}
        <div className="space-y-1.5">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-orange-400" />
            7-Level Wildfire Risk Scale
          </div>
          <div className="space-y-0.5 bg-slate-900/60 p-2 rounded-xl border border-slate-800/80 text-xs">
            {riskCategories.map((cat) => {
              const stat = layerStats?.[cat.name];
              return (
                <div key={cat.name} className="flex items-center justify-between py-0.5">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-sm shadow-sm"
                      style={{ backgroundColor: cat.color }}
                    />
                    <span className="font-medium text-slate-200">{cat.name}</span>
                  </div>
                  <div className="flex items-center gap-3 font-mono text-slate-400">
                    <span>{cat.range}</span>
                    {stat && (
                      <span className="text-orange-400 font-bold w-10 text-right">
                        {stat.percentage}%
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </aside>
  );
}
