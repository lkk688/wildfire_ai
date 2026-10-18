'use client';

import React, { useState, useEffect } from 'react';
import {
  Eye,
  Flame,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Play,
  RotateCcw,
  Sparkles,
  Info,
  Layers,
  MapPin,
  Wind,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ExternalLink,
  BookOpen,
  Maximize2,
  Minimize2,
  Calendar,
  Clock,
  Compass,
  Columns,
  Activity,
} from 'lucide-react';

interface MultiEpochEntry {
  url: string;
  label: string;
  date: string;
  role: string;
  badge: string;
  exists: boolean;
}

interface CaseItem {
  id: string | number;
  incident: string;
  county: string;
  city?: string;
  street_address?: string;
  latitude: number;
  longitude: number;
  year_built: string | number;
  structure_type: string;
  roof_construction: string;
  vent_screen: string;
  exterior_siding: string;
  deck_on_grade: string;
  deck_elevated: string;
  fence_attached: string;
  eaves: string;
  survival_label: 'Destroyed' | 'Survived';
  raw_damage: string;
  ignition_point: string;
  ignition_source: string;
  defensive_actions: string;
  image_url: string;
  has_image: boolean;
  multi_epoch_images?: {
    prefire: MultiEpochEntry;
    postfire: MultiEpochEntry;
    current: MultiEpochEntry;
  };
  incident_forensics?: {
    fire_name: string;
    ignition_date: string;
    location_text: string;
    weather_forcing: string;
    fuel_model: string;
    flame_spread_rate: string;
    observed_physical_outcome: string;
    forensic_breach_mechanism: string;
    firefighter_action_log: string;
    flame_azimuth_deg: number;
    wind_speed_mph: number;
  };
  expert_diagnosis: {
    title: string;
    diagnosis: string;
    recommendation: string;
  };
}

interface ModelOutput {
  predicted_survival: 'Destroyed' | 'Survived' | 'Unknown';
  predicted_vulnerability: string;
  predicted_mechanism: string;
  is_survival_correct: boolean;
  locus_match: boolean;
  latency_sec: number;
  thinking_trace_snippet?: string;
  full_reasoning_trace?: string;
  analysis_text?: string;
}

interface ModelScoreboardEntry {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  false_alarm_rate: number;
  locus_attribution_accuracy: number;
  confusion_matrix: {
    TP: number;
    FP: number;
    TN: number;
    FN: number;
  };
}

interface VisionBenchmarkViewProps {
  onSwitchToMap: () => void;
}

const MODEL_CONFIGS: Record<string, { label: string; badge: string; color: string; border: string }> = {
  'MiniMax-M3': {
    label: 'MiniMax-M3',
    badge: 'CoT Deductive (2026)',
    color: 'from-amber-500/20 to-orange-500/10 text-orange-300',
    border: 'border-orange-500/30',
  },
  'claude-haiku-5-5': {
    label: 'Claude Haiku 5.5',
    badge: 'Anthropic Fast Native Reasoning',
    color: 'from-cyan-500/20 to-blue-500/10 text-cyan-300',
    border: 'border-cyan-500/30',
  },
  'claude-sonnet-5-5': {
    label: 'Claude Sonnet 5.5',
    badge: 'Anthropic Frontier Reasoning',
    color: 'from-purple-500/20 to-indigo-500/10 text-purple-300',
    border: 'border-purple-500/30',
  },
  'Qwen3.8-27B': {
    label: 'Qwen3.8-27B',
    badge: 'Remote vLLM Open Weights',
    color: 'from-emerald-500/20 to-teal-500/10 text-emerald-300',
    border: 'border-emerald-500/30',
  },
};

export default function VisionBenchmarkView({ onSwitchToMap }: VisionBenchmarkViewProps) {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [modelResults, setModelResults] = useState<Record<string, Record<string, ModelOutput>>>({});
  const [scoreboard, setScoreboard] = useState<ModelScoreboardEntry[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [filterType, setFilterType] = useState<'all' | 'survived' | 'destroyed' | 'saves'>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [showDataSourceModal, setShowDataSourceModal] = useState<boolean>(false);
  const [liveModel, setLiveModel] = useState<string>('claude-sonnet-5-5');
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [liveResult, setLiveResult] = useState<any>(null);
  const [selectedEpoch, setSelectedEpoch] = useState<'prefire' | 'postfire' | 'current'>('prefire');
  const [splitComparison, setSplitComparison] = useState<boolean>(false);

  // Fetch benchmark data from backend
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [casesRes, resultsRes, metricsRes] = await Promise.all([
          fetch('http://localhost:8000/api/vlm/benchmark/cases').catch(() => null),
          fetch('http://localhost:8000/api/vlm/benchmark/results').catch(() => null),
          fetch('http://localhost:8000/api/vlm/benchmark/metrics').catch(() => null),
        ]);

        if (casesRes && casesRes.ok) {
          const casesData = await casesRes.json();
          setCases(casesData.items || []);
        }

        if (resultsRes && resultsRes.ok) {
          const resData = await resultsRes.json();
          setModelResults(resData.cases || {});
        }

        if (metricsRes && metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setScoreboard(metricsData.scoreboard || []);
        }
      } catch (err) {
        console.error('Error fetching VLM benchmark data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Filter cases based on user selection
  const filteredCases = cases.filter((item) => {
    if (filterType === 'survived') return item.survival_label === 'Survived';
    if (filterType === 'destroyed') return item.survival_label === 'Destroyed';
    if (filterType === 'saves') {
      return (
        item.survival_label === 'Survived' &&
        (item.defensive_actions.toLowerCase().includes('hand crew') ||
          item.defensive_actions.toLowerCase().includes('engine') ||
          item.vent_screen.includes('<= 1/8'))
      );
    }
    return true;
  });

  const currentCase = filteredCases[currentIndex] || cases[0];
  const currentCaseResults = currentCase ? modelResults[String(currentCase.id)] || {} : {};

  const toggleThinking = (modelName: string) => {
    setExpandedThinking((prev) => ({ ...prev, [modelName]: !prev[modelName] }));
  };

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : filteredCases.length - 1));
    setLiveResult(null);
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev < filteredCases.length - 1 ? prev + 1 : 0));
    setLiveResult(null);
  };

  const handleRunLivePrediction = async () => {
    if (!currentCase) return;
    setLiveLoading(true);
    setLiveResult(null);
    try {
      const res = await fetch('http://localhost:8000/api/vlm/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: liveModel,
          case_id: String(currentCase.id),
        }),
      });
      const data = await res.json();
      setLiveResult(data);
    } catch (err) {
      setLiveResult({ success: false, error: String(err) });
    } finally {
      setLiveLoading(false);
    }
  };

  return (
    <div className="flex-1 h-full flex flex-col bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Top Header & Application Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-rose-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-wide">
                WildfireReason-VLM
              </h1>
              <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-orange-500/20 text-orange-400 border border-orange-500/30">
                CVPR 2026 Frontier Track
              </span>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                CAL FIRE DINS Ground-Truth
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Zero-Shot Forensic Ignition Attribution & Defect Discovery Engine across 4 Frontier Reasoning Models
            </p>
          </div>
        </div>

        {/* Global Action Bar */}
        <div className="flex items-center gap-3">
          {/* Switch to GIS Map View */}
          <button
            onClick={onSwitchToMap}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all shadow-sm"
          >
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            <span>Switch to 3D GIS Map</span>
          </button>

          {/* Data Source & Methodology Button */}
          <button
            onClick={() => setShowDataSourceModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/80 transition-all"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>Data Sources & Benchmark Protocol</span>
          </button>
        </div>
      </header>

      {/* Cross-Model Benchmark Scoreboard (Top Metric Ribbon) */}
      <section className="bg-slate-900/40 border-b border-slate-800/60 px-6 py-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Cross-Model Empirical Scoreboard (50 Ground-Truth Cases: 25 Destroyed vs 25 Survived)
            </span>
          </div>
          <span className="text-[11px] text-rose-400 font-medium">
            ⚠️ Key Finding: All frontier models perform at ~50% (coin-toss) with 88%–100% False Alarm Rates
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {scoreboard.map((item) => {
            const isQwen = item.model_name.includes('Qwen');
            const isSonnet = item.model_name.includes('sonnet');
            const isHaiku = item.model_name.includes('haiku');
            return (
              <div
                key={item.model_name}
                className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 hover:border-slate-700 transition-all shadow-sm relative overflow-hidden"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-white tracking-wide">
                    {item.model_name}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                      isSonnet
                        ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                        : isHaiku
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : isQwen
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                    }`}
                  >
                    {isSonnet ? 'SOTA Frontier' : isHaiku ? 'Fast CoT' : isQwen ? 'Open vLLM' : 'Commercial CoT'}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center pt-1 border-t border-slate-800/60">
                  <div>
                    <div className="text-[10px] text-slate-400">Accuracy</div>
                    <div className="text-sm font-bold text-slate-100">{item.accuracy}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-rose-400 font-medium">False Alarm</div>
                    <div className="text-sm font-bold text-rose-400">{item.false_alarm_rate}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">Locus Acc</div>
                    <div className="text-sm font-bold text-amber-300">
                      {item.locus_attribution_accuracy}%
                    </div>
                  </div>
                </div>

                <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/40">
                  <span>CM: TP={item.confusion_matrix.TP} | FP={item.confusion_matrix.FP}</span>
                  <span>TN={item.confusion_matrix.TN} | FN={item.confusion_matrix.FN}</span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Case Navigation & Filtering Sub-Header */}
      <section className="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Filter Pills */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-slate-400 mr-1 font-medium">Filters:</span>
          {(
            [
              { key: 'all', label: `All (${cases.length})` },
              { key: 'survived', label: 'Survived (Defect Zero)' },
              { key: 'destroyed', label: 'Destroyed' },
              { key: 'saves', label: 'Tactical Firefighter Saves' },
            ] as const
          ).map((t) => (
            <button
              key={t.key}
              onClick={() => {
                setFilterType(t.key);
                setCurrentIndex(0);
                setLiveResult(null);
              }}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                filterType === t.key
                  ? 'bg-orange-500 text-white shadow-md shadow-orange-500/20'
                  : 'bg-slate-800/60 text-slate-300 hover:bg-slate-800 border border-slate-700/60'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Case Selector Dropdown & Stepper */}
        <div className="flex items-center gap-2">
          <button
            onClick={handlePrev}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
            title="Previous Case"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <select
            value={currentCase ? currentCase.id : ''}
            onChange={(e) => {
              const idx = filteredCases.findIndex((c) => String(c.id) === e.target.value);
              if (idx !== -1) {
                setCurrentIndex(idx);
                setLiveResult(null);
              }
            }}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-orange-500"
          >
            {filteredCases.map((c, i) => (
              <option key={c.id} value={c.id}>
                Case #{c.id} - {c.survival_label} ({c.incident} Fire) {c.defensive_actions !== 'None' ? '★ Saved by Hand Crew' : ''}
              </option>
            ))}
          </select>

          <button
            onClick={handleNext}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
            title="Next Case"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <span className="text-xs text-slate-400 font-mono ml-1">
            {filteredCases.length > 0 ? currentIndex + 1 : 0} of {filteredCases.length}
          </span>
        </div>
      </section>

      {/* Main Case Inspector Workstation */}
      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-slate-400">
          <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading Ground-Truth DINS Cases & Model Reasoning Matrix...</span>
        </div>
      ) : currentCase ? (
        <main className="flex-1 p-6 space-y-6">
          {/* Top Split: Left Parcel Imagery & GT Details vs Right Multi-Model Predictions */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Aerial Imagery & Forensic GT Card (5 Cols) */}
            <div className="lg:col-span-5 space-y-4">
              {/* High-Resolution Multi-Temporal Aerial Imagery Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                {/* Multi-Epoch Timeline Switcher Bar */}
                <div className="bg-slate-950 p-2 border-b border-slate-800 flex items-center justify-between gap-1 text-xs">
                  <div className="flex items-center gap-1 overflow-x-auto">
                    {/* Epoch 1: Pre-Fire */}
                    <button
                      onClick={() => {
                        setSelectedEpoch('prefire');
                        setSplitComparison(false);
                      }}
                      className={`px-2.5 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
                        !splitComparison && selectedEpoch === 'prefire'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-sm shadow-emerald-500/20'
                          : 'bg-slate-850 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-transparent'
                      }`}
                    >
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      <span>2020.02 灾前完整态</span>
                    </button>

                    {/* Epoch 2: Post-Fire GT */}
                    <button
                      onClick={() => {
                        setSelectedEpoch('postfire');
                        setSplitComparison(false);
                      }}
                      className={`px-2.5 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
                        !splitComparison && selectedEpoch === 'postfire'
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/50 shadow-sm shadow-rose-500/20'
                          : 'bg-slate-850 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-transparent'
                      }`}
                    >
                      <span className="w-2 h-2 rounded-full bg-rose-400" />
                      <span>2021.10 灾后现场真值</span>
                      <span className="text-[10px] px-1 py-0.2 rounded bg-rose-500/30 text-rose-200 font-mono">★ 真值</span>
                    </button>

                    {/* Epoch 3: Current */}
                    <button
                      onClick={() => {
                        setSelectedEpoch('current');
                        setSplitComparison(false);
                      }}
                      className={`px-2.5 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
                        !splitComparison && selectedEpoch === 'current'
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm shadow-cyan-500/20'
                          : 'bg-slate-850 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-transparent'
                      }`}
                    >
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      <span>2024+ 当前现貌</span>
                    </button>
                  </div>

                  {/* Split Comparison Toggle Button */}
                  <button
                    onClick={() => setSplitComparison(!splitComparison)}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap shrink-0 ${
                      splitComparison
                        ? 'bg-gradient-to-r from-orange-500 to-rose-600 text-white shadow-md shadow-orange-500/20'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                    title="并排对比灾前航拍与灾后真值"
                  >
                    <Columns className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">并排对比</span>
                  </button>
                </div>

                {/* Imagery Viewport */}
                {splitComparison ? (
                  /* 2-Column Split View: Pre-Fire (2020) vs Post-Fire GT (2021) */
                  <div className="grid grid-cols-2 divide-x divide-slate-800 bg-slate-950">
                    {/* Left: Pre-Fire */}
                    <div className="relative aspect-square overflow-hidden group">
                      <img
                        src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=prefire`}
                        alt={`Pre-fire 2020 Case ${currentCase.id}`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                      />
                      <div className="absolute top-2 left-2 bg-emerald-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-emerald-300 border border-emerald-500/40">
                        2020.02 灾前完整态 (VLM输入)
                      </div>
                      <div className="absolute bottom-2 left-2 right-2 bg-slate-950/80 backdrop-blur-md px-2 py-1 rounded text-[10px] text-slate-300 truncate">
                        完好屋面 / 绿化树冠 / 30m 防御圈
                      </div>
                    </div>

                    {/* Right: Post-Fire GT */}
                    <div className="relative aspect-square overflow-hidden group">
                      <img
                        src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=postfire`}
                        alt={`Post-fire 2021 Case ${currentCase.id}`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                      />
                      <div className="absolute top-2 left-2 bg-rose-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-rose-300 border border-rose-500/40">
                        2021.10 灾后现场物证 (真值)
                      </div>
                      <div className="absolute bottom-2 left-2 right-2 bg-slate-950/80 backdrop-blur-md px-2 py-1 rounded text-[10px] font-semibold truncate text-rose-300">
                        {currentCase.survival_label === 'Destroyed'
                          ? '🔥 建筑烧毁 / 废墟残骸推平'
                          : '🛡️ 建筑完好存活 / 成功防守'}
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Single Epoch View */
                  <div className="relative aspect-video w-full bg-slate-950 overflow-hidden group">
                    <img
                      src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=${selectedEpoch}`}
                      alt={`Aerial view ${selectedEpoch} for Case ${currentCase.id}`}
                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    />

                    {/* Temporal Epoch Floating Badge */}
                    <div className="absolute top-3 left-3 flex flex-wrap gap-1.5">
                      <span className={`px-2 py-0.5 rounded-md text-[11px] font-bold backdrop-blur-md border shadow-md ${
                        selectedEpoch === 'prefire'
                          ? 'bg-emerald-950/90 text-emerald-300 border-emerald-500/40'
                          : selectedEpoch === 'postfire'
                          ? 'bg-rose-950/90 text-rose-300 border-rose-500/40'
                          : 'bg-cyan-950/90 text-cyan-300 border-cyan-500/40'
                      }`}>
                        {selectedEpoch === 'prefire' && '2020.02 灾前完整态 (Pre-Fire Baseline)'}
                        {selectedEpoch === 'postfire' && '2021.10 灾后实证真值 (Post-Fire Ground Truth)'}
                        {selectedEpoch === 'current' && '2024-2026 地块当前现貌 (Current Recovery)'}
                      </span>
                      <span className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-black/70 text-slate-300 backdrop-blur-md">
                        0.3m / Pixel Orthorectified
                      </span>
                    </div>

                    <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[11px] bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10">
                      <div className="flex items-center gap-1.5 text-slate-300">
                        <MapPin className="w-3.5 h-3.5 text-rose-400" />
                        <span>{currentCase.latitude.toFixed(4)}°N, {currentCase.longitude.toFixed(4)}°W</span>
                      </div>
                      <span className="text-slate-400 font-medium">
                        {currentCase.incident} Fire, {currentCase.county} Co.
                      </span>
                    </div>
                  </div>
                )}

                {/* Epoch Functional Description Bar */}
                <div className="bg-slate-950/90 px-3.5 py-2 border-t border-slate-800 text-[11px] flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-slate-300">
                    <Clock className="w-3.5 h-3.5 text-orange-400 shrink-0" />
                    <span>
                      {selectedEpoch === 'prefire' && '科学用途：提供给 VLM 评测物理结构与周边植被的真实防御基准'}
                      {selectedEpoch === 'postfire' && '法医事实：火灾熄灭后的正射航拍，直观验证建筑是否化为焦炭'}
                      {selectedEpoch === 'current' && '恢复现状：展示火灾数年后地块重建或自然演替现貌'}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-400 shrink-0">
                    {selectedEpoch === 'prefire' ? 'ESRI Wayback 23001' : selectedEpoch === 'postfire' ? 'ESRI Wayback 26120' : 'ESRI World Imagery'}
                  </span>
                </div>

                {/* Weather & Meteorological Forcing Banner */}
                <div className="bg-rose-950/20 border-t border-rose-900/30 p-2.5 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 text-rose-300">
                    <Wind className="w-4 h-4 text-rose-400 animate-pulse" />
                    <span className="font-semibold">Red Flag Warning Forcing:</span>
                    <span>Gusts 25-35 mph | DFM &lt; 6%</span>
                  </div>
                  <span className="text-rose-400/80 font-mono text-[11px]">Chaparral/Timber Fuel</span>
                </div>
              </div>

              {/* CAL FIRE DINS Forensic Ground-Truth Panel */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white tracking-wide">
                      CAL FIRE DINS Forensic Inspection
                    </h3>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider ${
                      currentCase.survival_label === 'Survived'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/20'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-500/20'
                    }`}
                  >
                    Actual GT: {currentCase.survival_label}
                  </span>
                </div>

                {/* Key Ground Truth Facts Grid */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">Inspection Damage Level</span>
                    <span className="font-semibold text-slate-200">{currentCase.raw_damage}</span>
                  </div>

                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">Actual Forensic Ignition Point</span>
                    <span className="font-semibold text-amber-300">{currentCase.ignition_point || 'None'}</span>
                  </div>

                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80 col-span-2">
                    <span className="text-slate-400 block text-[11px]">Emergency Responder Tactical Actions</span>
                    <span className={`font-semibold ${currentCase.defensive_actions !== 'None' ? 'text-cyan-300' : 'text-slate-400'}`}>
                      {currentCase.defensive_actions}
                      {currentCase.defensive_actions !== 'None' && ' (Active Fire Suppression Save)'}
                    </span>
                  </div>
                </div>

                {/* Building Hardening Specs */}
                <div className="pt-2 border-t border-slate-800/60 space-y-2">
                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                    Physical Structural Hardening Envelope
                  </span>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Year Built:</span>
                      <span className="text-slate-200 font-medium">{currentCase.year_built}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Structure Type:</span>
                      <span className="text-slate-200 font-medium">{currentCase.structure_type.replace('Single Family Residence', 'SFR')}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Roof Material:</span>
                      <span className="text-slate-200 font-medium">{currentCase.roof_construction}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Vent Mesh:</span>
                      <span className={`font-semibold ${currentCase.vent_screen.includes('<= 1/8') ? 'text-emerald-400' : 'text-slate-200'}`}>
                        {currentCase.vent_screen}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Exterior Siding:</span>
                      <span className="text-slate-200 font-medium">{currentCase.exterior_siding}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Ground Deck:</span>
                      <span className="text-slate-200 font-medium">{currentCase.deck_on_grade}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Eaves Design:</span>
                      <span className="text-slate-200 font-medium">{currentCase.eaves}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Attached Fence:</span>
                      <span className="text-slate-200 font-medium">{currentCase.fence_attached}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Forensic Wildfire Incident Fact Sheet & Graphic (真实大火案卷与火场事实) */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4 relative overflow-hidden">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Flame className="w-5 h-5 text-orange-400" />
                    <div>
                      <h3 className="text-sm font-bold text-white tracking-wide">
                        真实大火事实与火场案卷 (Ground-Truth Incident Dossier)
                      </h3>
                      <span className="text-[11px] text-slate-400">
                        {currentCase.incident_forensics?.fire_name || `${currentCase.incident} Fire`} 现场环境与蔓延动力学
                      </span>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-orange-500/20 text-orange-300 border border-orange-500/30">
                    {currentCase.incident_forensics?.ignition_date || '2020-06-06'}
                  </span>
                </div>

                {/* SVG Visual: Flame Spread & Defensible Space Vector Graphic */}
                <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 flex flex-col sm:flex-row items-center gap-4">
                  {/* SVG Diagram Canvas */}
                  <div className="w-full sm:w-44 h-36 relative shrink-0 flex items-center justify-center">
                    <svg viewBox="0 0 200 160" className="w-full h-full">
                      {/* Defensible Space Concentric Zones */}
                      <circle cx="100" cy="80" r="70" fill="none" stroke="#10b981" strokeWidth="1" strokeDasharray="4 3" opacity="0.4" />
                      <circle cx="100" cy="80" r="45" fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
                      <circle cx="100" cy="80" r="22" fill="none" stroke="#ef4444" strokeWidth="1.2" opacity="0.7" />

                      {/* Zone Labels */}
                      <text x="175" y="80" fill="#10b981" fontSize="7" opacity="0.8">100ft</text>
                      <text x="150" y="80" fill="#f59e0b" fontSize="7" opacity="0.8">30ft</text>
                      <text x="125" y="80" fill="#ef4444" fontSize="7" opacity="0.8">5ft</text>

                      {/* Wind / Flame Spread Vector Arrow (SW 230° -> NE) */}
                      <defs>
                        <linearGradient id="flameGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#ef4444" stopOpacity="0.9" />
                          <stop offset="60%" stopColor="#f97316" stopOpacity="0.8" />
                          <stop offset="100%" stopColor="#fbbf24" stopOpacity="0.2" />
                        </linearGradient>
                      </defs>
                      <path d="M 25 140 Q 60 110 85 92" fill="none" stroke="url(#flameGrad)" strokeWidth="4" strokeLinecap="round" />
                      <polygon points="85,92 73,95 80,105" fill="#f97316" />

                      {/* Wind Vector Label */}
                      <text x="15" y="152" fill="#f87171" fontSize="8" fontWeight="bold">SW 230° 32mph</text>

                      {/* Embers Drifting (Particles) */}
                      <circle cx="65" cy="100" r="1.5" fill="#fde047" className="animate-ping" />
                      <circle cx="75" cy="88" r="1.5" fill="#fb923c" />
                      <circle cx="82" cy="78" r="1.5" fill="#ef4444" />

                      {/* Center Building Envelope */}
                      <rect x="91" y="71" width="18" height="18" rx="2" fill={currentCase.survival_label === 'Survived' ? '#059669' : '#dc2626'} stroke="#ffffff" strokeWidth="1.5" />
                      <path d="M 88 71 L 100 62 L 112 71 Z" fill={currentCase.survival_label === 'Survived' ? '#10b981' : '#f87171'} stroke="#ffffff" strokeWidth="1" />

                      {/* Outcome Badge in SVG */}
                      <g transform="translate(100, 20)">
                        <rect x="-42" y="-10" width="84" height="20" rx="10" fill={currentCase.survival_label === 'Survived' ? '#064e3b' : '#7f1d1d'} stroke={currentCase.survival_label === 'Survived' ? '#34d399' : '#f87171'} strokeWidth="1" />
                        <text x="0" y="3" fill="#ffffff" fontSize="8" fontWeight="bold" textAnchor="middle">
                          {currentCase.survival_label === 'Survived' ? '🛡️ 建筑存活 (GT)' : '🔥 建筑烧毁 (GT)'}
                        </text>
                      </g>
                    </svg>
                  </div>

                  {/* Factual Spread & Environment Breakdown */}
                  <div className="flex-1 space-y-2 text-xs">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">强风与烈焰推进方向</span>
                        <span className="font-semibold text-rose-300">西南 (230° SW) 32 mph 阵风推进</span>
                      </div>
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">周边植被与可燃物</span>
                        <span className="font-semibold text-amber-300">Chaparral 灌木 + 枯草干燥层</span>
                      </div>
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">真实物理击穿 / 防护点</span>
                        <span className="font-semibold text-cyan-300">{currentCase.ignition_point || '无击穿 (全防御幸存)'}</span>
                      </div>
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">消防员现场战术干预</span>
                        <span className="font-semibold text-emerald-300">{currentCase.defensive_actions}</span>
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-relaxed bg-slate-900/40 p-2 rounded border border-slate-800/60">
                      {currentCase.incident_forensics?.observed_physical_outcome ||
                        `加州消防局现场鉴定：建筑在强风火场中经历烈焰与飞火考验，最终实际结果为 ${currentCase.raw_damage}。`}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Multi-Model Deductive Reasoning Matrix (7 Cols) */}
            <div className="lg:col-span-7 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Flame className="w-5 h-5 text-orange-400" />
                  <h3 className="text-sm font-bold text-white tracking-wide">
                    Multi-Model Deductive Reasoning & Prediction Matrix
                  </h3>
                </div>
                <span className="text-xs text-slate-400">
                  Pre-computed Zero-Shot Chain-of-Thought (CoT)
                </span>
              </div>

              {/* 4 Models Display Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.keys(MODEL_CONFIGS).map((modelKey) => {
                  const cfg = MODEL_CONFIGS[modelKey];
                  const res = currentCaseResults[modelKey];
                  const isCorrect = res ? res.is_survival_correct : false;
                  const isExpanded = expandedThinking[modelKey] || false;

                  return (
                    <div
                      key={modelKey}
                      className={`bg-slate-900 border ${cfg.border} rounded-2xl p-4 shadow-lg flex flex-col justify-between space-y-3 transition-all hover:border-slate-600`}
                    >
                      {/* Model Card Header */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-bold text-white tracking-wide">
                            {cfg.label}
                          </span>
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${cfg.color}`}>
                            {cfg.badge}
                          </span>
                        </div>

                        {/* Prediction vs Ground Truth Outcome */}
                        {res ? (
                          <div className="space-y-2">
                            <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
                              <span className="text-xs text-slate-400">Predicted Survival:</span>
                              <div className="flex items-center gap-1.5">
                                <span
                                  className={`text-xs font-bold px-2 py-0.5 rounded ${
                                    res.predicted_survival === 'Destroyed'
                                      ? 'bg-rose-500/20 text-rose-300'
                                      : 'bg-emerald-500/20 text-emerald-300'
                                  }`}
                                >
                                  {res.predicted_survival}
                                </span>
                                {isCorrect ? (
                                  <span className="flex items-center text-[11px] text-emerald-400 font-semibold gap-0.5">
                                    <CheckCircle2 className="w-3.5 h-3.5" /> Correct
                                  </span>
                                ) : (
                                  <span className="flex items-center text-[11px] text-rose-400 font-semibold gap-0.5">
                                    <XCircle className="w-3.5 h-3.5" />
                                    {currentCase.survival_label === 'Survived' ? 'False Alarm' : 'Miss'}
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Attributions: Vulnerability & Failure Mechanism */}
                            <div className="grid grid-cols-2 gap-2 text-[11px]">
                              <div className="bg-slate-950/50 p-2 rounded-lg border border-slate-800/60">
                                <span className="text-slate-400 block text-[10px]">Primary Vulnerability</span>
                                <span className="font-semibold text-amber-300">
                                  {res.predicted_vulnerability || 'Unknown'}
                                </span>
                              </div>
                              <div className="bg-slate-950/50 p-2 rounded-lg border border-slate-800/60">
                                <span className="text-slate-400 block text-[10px]">Failure Mechanism</span>
                                <span className="font-semibold text-orange-300">
                                  {res.predicted_mechanism || 'Unknown'}
                                </span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="p-4 bg-slate-950/40 rounded-xl text-center text-xs text-slate-500">
                            No evaluation record for this case
                          </div>
                        )}
                      </div>

                      {/* Collapsible Reasoning Trace (Thinking tokens) */}
                      {res && (
                        <div className="pt-2 border-t border-slate-800/80">
                          <button
                            onClick={() => toggleThinking(modelKey)}
                            className="w-full flex items-center justify-between text-left text-xs font-medium text-slate-300 hover:text-white py-1 transition-colors"
                          >
                            <span className="flex items-center gap-1.5 text-purple-300">
                              <Sparkles className="w-3.5 h-3.5" />
                              <span>{isExpanded ? 'Hide' : 'View'} CoT Thinking Trace</span>
                            </span>
                            {isExpanded ? (
                              <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                            )}
                          </button>

                          {isExpanded && (
                            <div className="mt-2 p-2.5 rounded-xl bg-slate-950 text-[11px] font-mono text-slate-300 max-h-48 overflow-y-auto border border-purple-500/20 leading-relaxed">
                              <div className="text-[10px] text-purple-400 font-bold mb-1 uppercase tracking-wider">
                                Deductive Reasoning Trace (&lt;think&gt; tokens):
                              </div>
                              {res.thinking_trace_snippet || res.full_reasoning_trace || 'No raw trace captured'}
                            </div>
                          )}

                          {/* Analysis Excerpt */}
                          {res.analysis_text && !isExpanded && (
                            <p className="mt-2 text-[11px] text-slate-400 line-clamp-3 leading-relaxed italic bg-slate-950/30 p-2 rounded-lg">
                              "{res.analysis_text}"
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Live Inference Testing Bar */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col md:flex-row items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Play className="w-4 h-4 text-emerald-400" />
                  <div>
                    <span className="text-xs font-bold text-white block">Interactive Live Model Query</span>
                    <span className="text-[11px] text-slate-400">Trigger on-demand inference against current case</span>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 w-full md:w-auto">
                  <select
                    value={liveModel}
                    onChange={(e) => setLiveModel(e.target.value)}
                    className="bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-orange-500"
                  >
                    <option value="MiniMax-M3">MiniMax-M3 (CoT)</option>
                    <option value="claude-haiku-5-5">Claude Haiku 5.5</option>
                    <option value="claude-sonnet-5-5">Claude Sonnet 5.5</option>
                    <option value="Qwen3.8-27B">Qwen3.8-27B (vLLM)</option>
                  </select>

                  <button
                    onClick={handleRunLivePrediction}
                    disabled={liveLoading}
                    className="px-4 py-1.5 rounded-lg text-xs font-bold bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-600 hover:to-rose-700 text-white shadow-md shadow-orange-500/20 disabled:opacity-50 transition-all flex items-center gap-1.5 whitespace-nowrap"
                  >
                    {liveLoading ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Inferring...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Run Live Call</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Live Inference Output Display */}
              {liveResult && (
                <div className="bg-slate-900/90 border border-orange-500/40 rounded-2xl p-4 shadow-xl space-y-2 animate-fadeIn">
                  <div className="flex items-center justify-between text-xs border-b border-slate-800 pb-2">
                    <span className="font-bold text-orange-400">Live Result ({liveResult.model_name})</span>
                    <span className="text-slate-400 font-mono">Latency: {liveResult.latency_sec}s</span>
                  </div>
                  {liveResult.success ? (
                    <div className="space-y-2 text-xs">
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400">Predicted:</span>
                        <span className="font-bold text-white px-2 py-0.5 rounded bg-slate-800">
                          {liveResult.predicted_survival}
                        </span>
                        <span className="text-slate-400">Vulnerability:</span>
                        <span className="font-bold text-amber-300">{liveResult.predicted_vulnerability}</span>
                        <span className="text-slate-400">Mechanism:</span>
                        <span className="font-bold text-orange-300">{liveResult.predicted_mechanism}</span>
                      </div>
                      {liveResult.thinking_trace && (
                        <div className="p-2.5 rounded-xl bg-slate-950 font-mono text-[11px] text-slate-300 max-h-36 overflow-y-auto border border-slate-800">
                          {liveResult.thinking_trace}
                        </div>
                      )}
                      {liveResult.analysis_text && (
                        <p className="text-slate-300 text-xs italic bg-slate-950/40 p-2 rounded-lg">
                          "{liveResult.analysis_text}"
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-rose-400">Error: {liveResult.error}</div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Bottom Full-Width Section: Expert Fire Forensic Defect Analysis (错在哪) */}
          <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-96 h-96 bg-orange-500/5 rounded-full blur-3xl pointer-events-none" />

            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4 mb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-orange-500/20 text-orange-400 flex items-center justify-center border border-orange-500/30">
                  <AlertTriangle className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white tracking-wide">
                    Expert Fire Forensic Defect Analysis: Why the Models Failed (错在哪)
                  </h3>
                  <span className="text-xs text-slate-400">
                    Physical Causality Audit & Discrepancy Breakdown for Case #{currentCase.id}
                  </span>
                </div>
              </div>

              <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                {currentCase.expert_diagnosis.title}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 text-xs leading-relaxed">
              <div className="md:col-span-8 space-y-3">
                <div className="text-slate-300 text-sm leading-relaxed bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  {currentCase.expert_diagnosis.diagnosis}
                </div>
              </div>

              <div className="md:col-span-4 bg-slate-950/80 p-4 rounded-xl border border-slate-800/80 flex flex-col justify-between space-y-3">
                <div>
                  <span className="text-[11px] font-bold text-purple-400 uppercase tracking-wider block mb-1">
                    CVPR 2026 Paper Implication:
                  </span>
                  <p className="text-slate-300 text-xs leading-relaxed">
                    {currentCase.expert_diagnosis.recommendation}
                  </p>
                </div>
                <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                  <span>Targeted RLVR Reward</span>
                  <span className="font-mono text-emerald-400 font-semibold">R_physics + R_locus</span>
                </div>
              </div>
            </div>
          </section>
        </main>
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          No benchmark cases found matching current filter.
        </div>
      )}

      {/* Data Source & Methodology Modal */}
      {showDataSourceModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-white font-bold text-base">
                <BookOpen className="w-5 h-5 text-cyan-400" />
                <span>WildfireReason-DINS Benchmark Methodology</span>
              </div>
              <button
                onClick={() => setShowDataSourceModal(false)}
                className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
              <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
                <span className="font-bold text-cyan-300 block mb-1 text-sm">
                  1. Ground-Truth Data Source: CAL FIRE DINS
                </span>
                <p>
                  All structural parcel labels originate from the official State of California Damage Inspection Specialist (DINS)
                  forensic database. Trained fire marshals physically survey post-incident structures to document verified damage severity,
                  roof class, vent mesh sizes, decking combustibility, and the precise initial ignition locus (<code>WHEREFIRESTARTEDONSTRUCTURE</code>).
                </p>
              </div>

              <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
                <span className="font-bold text-orange-300 block mb-1 text-sm">
                  2. Sub-Meter Aerial Orthoimagery Pairing
                </span>
                <p>
                  Each structural parcel coordinate is paired with Zoom 18 (0.3m ground resolution) aerial imagery tiles from ESRI World Imagery,
                  capturing the actual building geometry, 0-5ft non-combustible perimeter, tree canopy overhangs, and adjacent driveway fuel breaks.
                </p>
              </div>

              <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
                <span className="font-bold text-purple-300 block mb-1 text-sm">
                  3. Standardized Multi-Model Evaluation Protocol
                </span>
                <p>
                  Models are evaluated on an identical deductive prompt instructing Chain-of-Thought (CoT) reasoning over physical wildfire dynamics,
                  weather forcing (Red Flag Warning, 25-35 mph gusts, DFM &lt; 6%), and structural hardening specs. Outputs are parsed into
                  binary survival predictions, primary vulnerability, and failure mechanism.
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setShowDataSourceModal(false)}
                className="px-4 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-white transition-all"
              >
                Close Documentation
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
