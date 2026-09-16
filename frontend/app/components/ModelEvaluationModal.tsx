'use client';

import React, { useState, useEffect } from 'react';
import {
  X,
  Layers,
  Shield,
  ShieldAlert,
  AlertTriangle,
  Scale,
  Sliders,
  CheckCircle2,
  Info,
  ExternalLink,
  Flame,
  Home,
  Building,
  HelpCircle,
  FileText,
  TrendingDown,
  TrendingUp,
  Cpu,
  Database,
  BrainCircuit,
  Sparkles,
  Compass,
  Wind,
  Mountain,
} from 'lucide-react';

interface ModelEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedRegion: string;
}

export default function ModelEvaluationModal({
  isOpen,
  onClose,
  selectedRegion,
}: ModelEvaluationModalProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'matrix' | 'cases' | 'simulator' | 'guidelines' | 'academic_ai'>('overview');
  const [selectedScenarioIdx, setSelectedScenarioIdx] = useState<number>(0);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetch(`http://localhost:8000/api/model-evaluation?region=${selectedRegion}`)
        .then((res) => res.json())
        .then((resData) => {
          if (resData.success && resData.data) {
            setData(resData.data);
          }
          setLoading(false);
        })
        .catch((err) => {
          console.error('Failed to fetch model evaluation data:', err);
          setLoading(false);
        });
    }
  }, [isOpen, selectedRegion]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-slate-950/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="glass-panel w-full max-w-6xl max-h-[92vh] rounded-3xl p-6 shadow-2xl border border-slate-700/80 flex flex-col overflow-hidden bg-slate-900/95 text-slate-100">
        
        {/* Header Section */}
        <div className="flex items-start justify-between pb-4 border-b border-slate-800 shrink-0">
          <div className="flex items-center gap-3.5">
            <div className="p-3 rounded-2xl bg-orange-500/20 text-orange-400 border border-orange-500/40 shadow-lg glow-orange">
              <Scale className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-orange-500/20 text-orange-300 border border-orange-500/30">
                  Risk Intelligence & Benchmark
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                  Target: {data?.region_name || selectedRegion}
                </span>
              </div>
              <h2 className="text-xl font-extrabold text-slate-100 tracking-wide mt-1">
                Off-the-Shelf Model Evaluation & Gap Analysis
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Demystifying Wildfire Predictions for Residents & Firefighters: Bellwether ML, CAL FIRE FHSZ, Insurance & Academic SOTA
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2.5 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800 border border-transparent hover:border-slate-700 transition-colors"
            title="Close Evaluation Modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 py-3 border-b border-slate-800/80 shrink-0 overflow-x-auto text-xs font-semibold">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3.5 py-2 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'overview'
                ? 'bg-orange-500 text-slate-950 font-bold shadow-md shadow-orange-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            1. Overview & Core Models
          </button>

          <button
            onClick={() => setActiveTab('matrix')}
            className={`px-3.5 py-2 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'matrix'
                ? 'bg-orange-500 text-slate-950 font-bold shadow-md shadow-orange-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Scale className="w-3.5 h-3.5" />
            2. Comparison Matrix
          </button>

          <button
            onClick={() => setActiveTab('cases')}
            className={`px-3.5 py-2 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'cases'
                ? 'bg-orange-500 text-slate-950 font-bold shadow-md shadow-orange-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            3. Discrepancy Case Studies
          </button>

          <button
            onClick={() => setActiveTab('simulator')}
            className={`px-3.5 py-2 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'simulator'
                ? 'bg-orange-500 text-slate-950 font-bold shadow-md shadow-orange-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            4. Interactive Gap Simulator
          </button>

          <button
            onClick={() => setActiveTab('academic_ai')}
            className={`px-3.5 py-2 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'academic_ai'
                ? 'bg-orange-500 text-slate-950 font-bold shadow-md shadow-orange-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <BrainCircuit className="w-3.5 h-3.5" />
            5. Academic SOTA & Frontier AI Models
          </button>

          <button
            onClick={() => setActiveTab('guidelines')}
            className={`px-3.5 py-2 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === 'guidelines'
                ? 'bg-orange-500 text-slate-950 font-bold shadow-md shadow-orange-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            6. Actionable Guidelines
          </button>
        </div>

        {/* Tab Body Content */}
        <div className="flex-1 overflow-y-auto py-4 pr-1 space-y-4">
          {loading ? (
            <div className="py-20 flex flex-col items-center justify-center gap-3 text-slate-400">
              <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-medium">Loading Cross-Model Evaluation & Actuarial Gap Metrics...</span>
            </div>
          ) : data ? (
            <>
              {/* TAB 1: OVERVIEW & CORE MODELS */}
              {activeTab === 'overview' && (
                <div className="space-y-4">
                  {/* Platform User Value Proposition (Dual Persona Emphasis) */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                    {/* For Residents & Homeowners */}
                    <div className="p-4 rounded-2xl bg-gradient-to-br from-emerald-500/10 via-slate-900 to-slate-950 border border-emerald-500/30 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-400">
                        <Home className="w-4 h-4" /> Built For Residents & Property Owners
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {data.user_value_proposition?.for_residents}
                      </p>
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                          ✓ Insurance Transparency
                        </span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                          ✓ Zone 0 (0-5ft) Audit
                        </span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                          ✓ CDI Mandatory Discounts
                        </span>
                      </div>
                    </div>

                    {/* For Firefighters & Incident Commanders */}
                    <div className="p-4 rounded-2xl bg-gradient-to-br from-orange-500/10 via-slate-900 to-slate-950 border border-orange-500/30 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-orange-400">
                        <Shield className="w-4 h-4" /> Built For Firefighters & Incident Commanders
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {data.user_value_proposition?.for_firefighters}
                      </p>
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-orange-500/10 text-orange-300 border border-orange-500/20">
                          ✓ 100m Dynamic Probabilities
                        </span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-orange-500/10 text-orange-300 border border-orange-500/20">
                          ✓ 130ft Radiant Contagion ROI
                        </span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-orange-500/10 text-orange-300 border border-orange-500/20">
                          ✓ Live HRRR Wind Plumes
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Regional Metric Highlights */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                      <div className="text-[11px] text-slate-400 font-medium">Model Agreement Overlap</div>
                      <div className="text-xl font-black text-emerald-400 mt-0.5">
                        {data.regional_metrics?.agreement_area_percentage}%
                      </div>
                      <div className="text-[10px] text-slate-500">Concordant risk classification area</div>
                    </div>
                    <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                      <div className="text-[11px] text-slate-400 font-medium">Regulatory Under-Warning Gap</div>
                      <div className="text-xl font-black text-rose-400 mt-0.5">
                        {data.regional_metrics?.under_warning_percentage}%
                      </div>
                      <div className="text-[10px] text-slate-500">Bellwether AI High vs CAL FIRE Moderate</div>
                    </div>
                    <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
                      <div className="text-[11px] text-slate-400 font-medium">FAIR Plan Surge (Insurance Crisis)</div>
                      <div className="text-xl font-black text-purple-400 mt-0.5">
                        {data.regional_metrics?.fair_plan_enrollment_surge}
                      </div>
                      <div className="text-[10px] text-slate-500">Driven by commercial black-box models</div>
                    </div>
                  </div>

                  {/* 3 Model Cards Deep Dive */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    {data.models?.map((m: any) => (
                      <div
                        key={m.id}
                        className="p-4 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col justify-between hover:border-slate-700 transition-all shadow-md"
                      >
                        <div className="space-y-2.5">
                          <div className="flex items-center justify-between">
                            <span
                              className="text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wider"
                              style={{
                                backgroundColor: `${m.color}20`,
                                color: m.color,
                                border: `1px solid ${m.color}40`,
                              }}
                            >
                              {m.badge}
                            </span>
                            <span className="text-[11px] text-slate-400 font-mono">{m.spatial_resolution}</span>
                          </div>

                          <div>
                            <h3 className="text-base font-bold text-slate-100">{m.name}</h3>
                            <div className="text-[11px] text-orange-400 font-medium mt-0.5">{m.type}</div>
                            <div className="text-[10px] text-slate-400">By {m.developer}</div>
                          </div>

                          <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/50 p-2.5 rounded-xl border border-slate-800/60">
                            {m.core_methodology}
                          </p>

                          {/* Key Specs */}
                          <div className="space-y-1 text-[11px] bg-slate-950/30 p-2.5 rounded-xl border border-slate-800/40">
                            <div className="flex justify-between">
                              <span className="text-slate-400">Temporal Cadence:</span>
                              <span className="text-slate-200 font-semibold">{m.update_cadence}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-400">Output Metric:</span>
                              <span className="text-slate-200 font-semibold text-right">{m.output_metric}</span>
                            </div>
                          </div>

                          {/* Strengths */}
                          <div>
                            <div className="text-[11px] font-bold text-emerald-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                              <CheckCircle2 className="w-3 h-3" /> Core Strengths
                            </div>
                            <ul className="space-y-1">
                              {m.strengths?.map((s: string, idx: number) => (
                                <li key={idx} className="text-[11px] text-slate-300 flex items-start gap-1.5 leading-snug">
                                  <span className="text-emerald-400 mt-0.5">•</span>
                                  <span>{s}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          {/* Limitations */}
                          <div>
                            <div className="text-[11px] font-bold text-amber-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" /> Limitations & Gaps
                            </div>
                            <ul className="space-y-1">
                              {m.limitations?.map((l: string, idx: number) => (
                                <li key={idx} className="text-[11px] text-slate-400 flex items-start gap-1.5 leading-snug">
                                  <span className="text-amber-400 mt-0.5">•</span>
                                  <span>{l}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 2: SIDE-BY-SIDE MATRIX */}
              {activeTab === 'matrix' && (
                <div className="space-y-3">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 flex items-center justify-between">
                    <span>
                      Comparing 8 critical operational and technical dimensions across the three off-the-shelf paradigms.
                    </span>
                    <span className="text-orange-400 font-semibold text-[11px]">
                      Interactive GIS Architecture
                    </span>
                  </div>

                  <div className="overflow-x-auto rounded-2xl border border-slate-800">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 font-semibold uppercase text-[10px] tracking-wider">
                          <th className="p-3.5 w-1/5">Comparison Dimension</th>
                          <th className="p-3.5 w-1/4 text-orange-400 border-l border-slate-800/80">
                            🤖 Google X Bellwether ML
                          </th>
                          <th className="p-3.5 w-1/4 text-rose-400 border-l border-slate-800/80">
                            🏛️ CAL FIRE Statutory FHSZ
                          </th>
                          <th className="p-3.5 w-1/4 text-purple-400 border-l border-slate-800/80">
                            💼 Commercial Insurance (Verisk/Zesty)
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 bg-slate-900/30">
                        {data.comparison_matrix?.map((row: any, idx: number) => (
                          <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                            <td className="p-3.5 font-bold text-slate-200 bg-slate-950/20">
                              {row.dimension}
                            </td>
                            <td className="p-3.5 text-slate-300 leading-relaxed border-l border-slate-800/60">
                              {row.bellwether}
                            </td>
                            <td className="p-3.5 text-slate-300 leading-relaxed border-l border-slate-800/60">
                              {row.calfire}
                            </td>
                            <td className="p-3.5 text-slate-300 leading-relaxed border-l border-slate-800/60">
                              {row.insurance}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* TAB 3: DISCREPANCY & GAP CASE STUDIES */}
              {activeTab === 'cases' && (
                <div className="space-y-4">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
                    <p>
                      Real-world geographic case studies in the Wildland-Urban Interface (WUI) and San Francisco Watershed,
                      demonstrating the root causes of divergence between machine learning models and statutory regulations.
                    </p>
                  </div>

                  <div className="space-y-3.5">
                    {data.gap_analysis_cases?.map((c: any) => (
                      <div
                        key={c.id}
                        className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all space-y-3"
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                          <div>
                            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-rose-500/20 text-rose-300 border border-rose-500/30">
                              {c.divergence_type}
                            </span>
                            <h4 className="text-base font-bold text-slate-100 mt-1">{c.title}</h4>
                            <div className="text-xs text-orange-400 font-medium">📍 {c.location}</div>
                          </div>

                          {/* Scores Badge Cluster */}
                          <div className="flex items-center gap-2 shrink-0">
                            <div className="p-2 rounded-xl bg-orange-500/10 border border-orange-500/20 text-right">
                              <div className="text-[9px] uppercase text-orange-400 font-semibold">Bellwether AI</div>
                              <div className="text-xs font-bold text-slate-100">{c.bellwether_score}</div>
                            </div>
                            <div className="p-2 rounded-xl bg-rose-500/10 border border-rose-500/20 text-right">
                              <div className="text-[9px] uppercase text-rose-400 font-semibold">CAL FIRE FHSZ</div>
                              <div className="text-xs font-bold text-slate-100">{c.calfire_rating}</div>
                            </div>
                            <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-right">
                              <div className="text-[9px] uppercase text-purple-400 font-semibold">Insurance Actuarial</div>
                              <div className="text-xs font-bold text-slate-100">{c.insurance_rating}</div>
                            </div>
                          </div>
                        </div>

                        {/* Root Cause Diagnosis */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 text-xs">
                          <div className="bg-slate-950/40 p-3 rounded-xl border border-slate-800/60 space-y-1">
                            <div className="font-bold text-amber-400 flex items-center gap-1.5">
                              <HelpCircle className="w-3.5 h-3.5" /> Root Cause of Discrepancy
                            </div>
                            <p className="text-slate-300 leading-relaxed text-[11px]">{c.root_cause}</p>
                          </div>

                          <div className="bg-slate-950/40 p-3 rounded-xl border border-slate-800/60 space-y-1">
                            <div className="font-bold text-emerald-400 flex items-center gap-1.5">
                              <ShieldAlert className="w-3.5 h-3.5" /> Operational Implication
                            </div>
                            <p className="text-slate-300 leading-relaxed text-[11px]">{c.operational_implication}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 4: INTERACTIVE GAP SIMULATOR */}
              {activeTab === 'simulator' && (
                <div className="space-y-4">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
                    Select a simulated WUI condition to inspect real-time discrepancies across Bellwether ML, CAL FIRE FHSZ,
                    and Insurance Risk Scores with automated root-cause diagnostics.
                  </div>

                  {/* Scenario Buttons */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {data.simulator_scenarios?.map((sc: any, idx: number) => (
                      <button
                        key={sc.id}
                        onClick={() => setSelectedScenarioIdx(idx)}
                        className={`p-3 rounded-2xl border text-left transition-all ${
                          selectedScenarioIdx === idx
                            ? 'bg-orange-500/20 border-orange-500/80 text-orange-300 shadow-lg glow-orange'
                            : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                        }`}
                      >
                        <div className="text-[10px] uppercase font-bold text-orange-400">Scenario {idx + 1}</div>
                        <div className="text-xs font-bold text-slate-100 mt-0.5 leading-snug">{sc.name}</div>
                      </button>
                    ))}
                  </div>

                  {/* Active Scenario Diagnostic Dashboard */}
                  {data.simulator_scenarios?.[selectedScenarioIdx] && (
                    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
                      {(() => {
                        const sc = data.simulator_scenarios[selectedScenarioIdx];
                        return (
                          <>
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                              <div>
                                <h3 className="text-base font-bold text-slate-100">{sc.name}</h3>
                                <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400 mt-1">
                                  <span>⛰️ Slope: <strong className="text-slate-200">{sc.terrain}</strong></span>
                                  <span>•</span>
                                  <span>🌿 Fuel State: <strong className="text-slate-200">{sc.fuel_state}</strong></span>
                                  <span>•</span>
                                  <span>🏠 Hardening: <strong className="text-slate-200">{sc.hardening}</strong></span>
                                </div>
                              </div>
                            </div>

                            {/* 3 Model Output Gauges */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
                              {/* Gauge 1: Bellwether */}
                              <div className="p-4 rounded-xl bg-slate-950/60 border border-orange-500/30 space-y-2">
                                <div className="text-[11px] font-bold text-orange-400 uppercase tracking-wider">
                                  Google X Bellwether AI
                                </div>
                                <div className="text-2xl font-black text-slate-100">{sc.bellwether_pred}</div>
                                <div className="text-[11px] text-slate-400">
                                  Dynamic forward burn probability based on satellite fuel moisture.
                                </div>
                                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                  <div
                                    className="bg-orange-500 h-full rounded-full transition-all duration-500"
                                    style={{
                                      width: sc.bellwether_pred.includes('Low')
                                        ? '15%'
                                        : sc.bellwether_pred.includes('Significant')
                                        ? '45%'
                                        : '90%',
                                    }}
                                  />
                                </div>
                              </div>

                              {/* Gauge 2: CAL FIRE */}
                              <div className="p-4 rounded-xl bg-slate-950/60 border border-rose-500/30 space-y-2">
                                <div className="text-[11px] font-bold text-rose-400 uppercase tracking-wider">
                                  CAL FIRE Statutory FHSZ
                                </div>
                                <div className="text-2xl font-black text-slate-100">{sc.calfire_fhsz}</div>
                                <div className="text-[11px] text-slate-400">
                                  Statutory zoning mandate based on historical 30-year flame length.
                                </div>
                                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                  <div
                                    className="bg-rose-500 h-full rounded-full transition-all duration-500"
                                    style={{
                                      width: sc.calfire_fhsz.includes('Moderate')
                                        ? '35%'
                                        : sc.calfire_fhsz.includes('High')
                                        ? '65%'
                                        : '95%',
                                    }}
                                  />
                                </div>
                              </div>

                              {/* Gauge 3: Insurance Cat Model */}
                              <div className="p-4 rounded-xl bg-slate-950/60 border border-purple-500/30 space-y-2">
                                <div className="text-[11px] font-bold text-purple-400 uppercase tracking-wider">
                                  Insurance Cat-Model Rating
                                </div>
                                <div className="text-2xl font-black text-slate-100">{sc.insurance_score}</div>
                                <div className="text-[11px] text-slate-400">
                                  Commercial underwriting score determining policy retention & pricing.
                                </div>
                                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                  <div
                                    className="bg-purple-500 h-full rounded-full transition-all duration-500"
                                    style={{
                                      width: sc.insurance_score.includes('38')
                                        ? '38%'
                                        : sc.insurance_score.includes('68')
                                        ? '68%'
                                        : '85%',
                                    }}
                                  />
                                </div>
                              </div>
                            </div>

                            {/* Automated AI Diagnostic Box */}
                            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-1.5">
                              <div className="flex items-center gap-2 text-xs font-bold text-amber-400 uppercase tracking-wider">
                                <AlertTriangle className="w-4 h-4" /> AI Root-Cause Diagnostic & Gap Explanation
                              </div>
                              <p className="text-xs text-slate-200 leading-relaxed font-medium">
                                {sc.diagnosis}
                              </p>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}

              {/* TAB 5: ACADEMIC SOTA & FRONTIER AI MODELS */}
              {activeTab === 'academic_ai' && (
                <div className="space-y-5">
                  <div className="p-4 rounded-2xl bg-gradient-to-r from-cyan-500/15 via-slate-900 to-indigo-500/15 border border-slate-700/80 space-y-2">
                    <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-cyan-400">
                      <BrainCircuit className="w-4 h-4" /> Frontier AI & Open-Source Research Landscape
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      Beyond off-the-shelf commercial models, computational wildfire science has witnessed groundbreaking advances in
                      <strong> Physics-Informed Neural Networks (PINNs)</strong>, <strong>Fourier Neural Operators (FNO)</strong>, and
                      <strong> Multimodal Foundation Models (LLMs)</strong>. Below is the state of the art in open-source research and
                      how foundation models reason over fire environments.
                    </p>
                  </div>

                  {/* Open-Source SOTA Models Grid */}
                  <div>
                    <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2 mb-3">
                      <Cpu className="w-4 h-4 text-orange-400" />
                      Academic Open-Source Wildfire SOTA Models & Benchmarks
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                      {data.academic_and_frontier_models?.open_source_models?.map((model: any, idx: number) => (
                        <div
                          key={idx}
                          className="p-4 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-2 hover:border-slate-700 transition-all shadow-md"
                        >
                          <div className="flex items-center justify-between">
                            <h4 className="text-xs font-bold text-slate-100">{model.name}</h4>
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                              {model.open_access}
                            </span>
                          </div>
                          <div className="text-[11px] font-semibold text-orange-400">{model.paradigm}</div>
                          <p className="text-[11px] text-slate-300 leading-relaxed bg-slate-950/40 p-2.5 rounded-xl border border-slate-800/60">
                            {model.highlights}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* What Data is Needed for LLM-based Wildfire Prediction */}
                  <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
                    <div className="border-b border-slate-800 pb-3">
                      <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                        <Database className="w-4 h-4 text-emerald-400" />
                        If Foundation Models (LLMs) Predict Wildfires: What Data is Needed?
                      </h3>
                      <p className="text-xs text-slate-300 leading-relaxed mt-1">
                        {data.academic_and_frontier_models?.llm_prediction_requirements?.overview}
                      </p>
                    </div>

                    {/* 6 Essential Data Categories */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {data.academic_and_frontier_models?.llm_prediction_requirements?.required_data_inputs?.map(
                        (inp: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1.5"
                          >
                            <div className="text-xs font-bold text-emerald-400">{inp.category}</div>
                            <p className="text-[11px] text-slate-300 leading-relaxed">{inp.variables}</p>
                            <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/60 flex items-center justify-between">
                              <span>Update Cadence:</span>
                              <span className="font-semibold text-slate-400">{inp.frequency}</span>
                            </div>
                          </div>
                        )
                      )}
                    </div>

                    {/* Agentic ReAct Architecture */}
                    <div className="p-4 rounded-xl bg-gradient-to-r from-orange-500/10 via-slate-950 to-indigo-500/10 border border-orange-500/30 space-y-1.5">
                      <div className="text-xs font-bold text-orange-400 uppercase tracking-wider flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5" /> Agentic ReAct Tool-Use Architecture
                      </div>
                      <p className="text-xs text-slate-200 leading-relaxed">
                        {data.academic_and_frontier_models?.llm_prediction_requirements?.agentic_reasoning_architecture}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 6: GUIDELINES & ACTIONABLE RECOMMENDATIONS */}
              {activeTab === 'guidelines' && (
                <div className="space-y-4">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
                    Bridging scientific gap analysis with on-the-ground operational decisions for Fire Marshals and legal rights for property owners.
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* For Fire Chiefs & Marshals */}
                    <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
                      <div className="flex items-center gap-2.5 text-sm font-bold text-orange-400 border-b border-slate-800 pb-2.5">
                        <Shield className="w-5 h-5" />
                        For Fire Marshals & Operational Commanders
                      </div>
                      <ul className="space-y-2.5">
                        {data.actionable_recommendations?.for_fire_marshals?.map((item: string, idx: number) => (
                          <li key={idx} className="text-xs text-slate-300 flex items-start gap-2 leading-relaxed">
                            <span className="p-1 rounded bg-orange-500/20 text-orange-400 font-mono text-[10px] mt-0.5">
                              0{idx + 1}
                            </span>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* For Homeowners & Policyholders */}
                    <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
                      <div className="flex items-center gap-2.5 text-sm font-bold text-emerald-400 border-b border-slate-800 pb-2.5">
                        <Home className="w-5 h-5" />
                        For Homeowners & Insurance Policyholders
                      </div>
                      <ul className="space-y-2.5">
                        {data.actionable_recommendations?.for_homeowners?.map((item: string, idx: number) => (
                          <li key={idx} className="text-xs text-slate-300 flex items-start gap-2 leading-relaxed">
                            <span className="p-1 rounded bg-emerald-500/20 text-emerald-400 font-mono text-[10px] mt-0.5">
                              0{idx + 1}
                            </span>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Regulatory Reference Box */}
                  <div className="p-4 rounded-2xl bg-slate-950/50 border border-slate-800 text-xs text-slate-400 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <span className="font-bold text-slate-200">Applicable California Standards: </span>
                      CDI "Safer from Wildfires" (10 CCR § 2644.9) • California Building Code Chapter 7A • IBHS Wildfire Prepared Home™
                    </div>
                    <a
                      href="https://www.insurance.ca.gov/01-consumers/140-catastrophes/SaferfromWildfires.cfm"
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 text-orange-400 hover:text-orange-300 font-semibold shrink-0"
                    >
                      <span>CDI Regulation Portal</span>
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="py-20 text-center text-slate-400 text-sm">
              Failed to load evaluation dataset. Please verify backend service on port 8000.
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400 shrink-0">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Wildfire Risk Intelligence Platform • Serving Residents, Firefighters & Municipal Planners</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
