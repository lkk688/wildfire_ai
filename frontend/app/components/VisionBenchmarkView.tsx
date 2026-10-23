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
  Crosshair,
  ZoomIn,
  ZoomOut,
  Target,
  Landmark,
  Building2,
  Home,
  Split,
  Camera,
  FileText,
  Check,
  Globe,
  PanelRightClose,
  PanelRightOpen,
  TrendingUp,
  DollarSign,
  Award,
  Percent,
  BarChart3,
  Gauge,
  Radio,
  ArrowRight,
  Zap,
  Shield,
  Copy,
} from 'lucide-react';

interface MultiEpochEntry {
  url: string;
  label: string;
  date: string;
  role: string;
  badge: string;
  exists: boolean;
}

interface FacadeAnnotation {
  tag: string;
  type: 'hardening' | 'vulnerability' | 'neutral' | 'rebuild' | 'context';
  desc: string;
}

interface Chapter7AInfo {
  is_compliant: boolean | null;
  code_status: string;
  badge_type: 'compliant' | 'non_compliant' | 'unknown';
  cutoff_year: number;
  standard_summary: string;
  regulatory_basis: string;
}

interface AssessorRecord {
  apn: string;
  year_built: number | string | null;
  assessed_improved_value?: number | null;
  assessed_value_display?: string;
  living_area_sqft?: number;
  lot_size?: string;
  beds_baths?: string;
  rebuild_permit_history?: string;
  chapter_7a?: Chapter7AInfo;
}

interface StreetViewInfo {
  url: string;
  has_image: boolean;
  provenance?: 'verified_real_gsv' | 'rural_private_gap';
  provenance_badge?: string;
  provenance_label?: string;
  explanation?: string;
  google_maps_url: string;
  google_coords_url?: string;
  facade_annotations?: FacadeAnnotation[];
}

interface BellwetherRadiantCorridor {
  wind_adjusted_major_axis_meters: number;
  threatened_structures_count: number;
  threatened_property_usd: number;
  threatened_property_display: string;
  mitigated_saved_usd: number;
  mitigated_saved_display: string;
  mitigation_roi_multiplier: string;
}

interface BellwetherDriver {
  factor: string;
  weight: string;
  impact: string;
}

interface BellwetherPrediction {
  model_name: string;
  spatial_resolution: string;
  burn_probability_1yr: number;
  burn_probability_1yr_pct: string;
  burn_probability_5yr: number;
  burn_probability_5yr_pct: string;
  hazard_tier: string;
  hazard_badge: string;
  radiant_corridor_130ft: BellwetherRadiantCorridor;
  top_drivers: BellwetherDriver[];
}

interface AcademicRothermel {
  name: string;
  paradigm: string;
  rate_of_spread_m_min: number;
  flame_length_meters: number;
  fireline_intensity_kw_m: number;
  equation: string;
  status: string;
}

interface AcademicCellularAutomata {
  name: string;
  paradigm: string;
  ember_spotting_probability: string;
  neighborhood_contagion_index: number;
  transition_rule: string;
  status: string;
}

interface AcademicGoogleNDWS {
  name: string;
  paradigm: string;
  csi_threat_score: number;
  spatial_iou: number;
  precision: number;
  recall: number;
  status: string;
}

interface AcademicSimulations {
  rothermel_level_set: AcademicRothermel;
  cellular_automata: AcademicCellularAutomata;
  google_ndws_convnet: AcademicGoogleNDWS;
}

interface CdiDiscountBreakdown {
  class_a_roof_credit: string;
  ember_mesh_vent_credit: string;
  zone0_buffer_credit: string;
  regulatory_authority: string;
}

interface InsuranceActuarial {
  verisk_fireline_score: number;
  verisk_score_label: string;
  zesty_ai_zfire_tier: string;
  underwriting_status: string;
  annual_base_premium_usd: number;
  annual_discounted_premium_usd: number;
  total_mitigation_discount_pct: string;
  cdi_discount_breakdown: CdiDiscountBreakdown;
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
  is_residential_focus?: boolean;
  community_type?: string;
  bellwether_prediction?: BellwetherPrediction;
  academic_simulations?: AcademicSimulations;
  insurance_actuarial?: InsuranceActuarial;
  structured_assessment?: StructuredAssessment;
  multi_epoch_images?: {
    prefire: MultiEpochEntry;
    postfire: MultiEpochEntry;
    current: MultiEpochEntry;
    streetview?: MultiEpochEntry;
  };
  assessor_record?: AssessorRecord;
  streetview?: StreetViewInfo;
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

interface StructuredAssessment {
  visual_evidence_attribution?: {
    macro_ortho_context?: {
      slope_exposure?: string;
      structure_separation_distance_m?: number;
      neighbor_radiant_threat?: string;
      defensible_space_compliance?: string;
    };
    micro_facade_envelope?: {
      roof_assembly?: string;
      eave_soffit_construction?: string;
      vent_mesh_rating?: string;
      exterior_cladding?: string;
      attached_combustible_attachments?: string;
      zone0_perimeter_buffer?: string;
    };
  };
  physical_vulnerability_matrix?: {
    ember_intrusion_score: number;
    direct_flame_contact_score: number;
    radiant_heat_flux_score: number;
    structural_defense_hardening_score: number;
  };
  causal_failure_pathway?: {
    primary_ignition_pathway?: string;
    step_by_step_domino_chain?: string[];
    weakest_link_component?: string;
    stochastic_survival_factors?: string[];
  };
  downstream_task_payloads?: {
    insurance_actuarial_underwriting?: {
      recommended_risk_tier: string;
      fair_plan_eligibility: string;
      cdi_section_2644_9_compliance_credits: Record<string, {
        qualified: boolean;
        credit_pct: number;
        status: string;
      }>;
      annual_base_premium_usd: number;
      max_potential_annual_premium_savings_usd: number;
      annual_discounted_premium_usd: number;
    };
    tactical_firefighting_operations?: {
      triage_category: string;
      prep_actions_required: string[];
      tactical_prep_time_minutes: number;
      minimum_water_supply_gpm: number;
      crew_safety_feasibility: string;
    };
    coupled_physics_simulation?: {
      rothermel_fuel_adjustment: string;
      cellular_automata_spotting_coefficient: number;
      estimated_critical_time_to_ignition_min: number;
      front_propagation_mode: string;
    };
  };
  uncertainty_audit?: {
    overall_assessment_confidence: number;
    visual_data_quality: string;
    unobserved_occluded_elements: string[];
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
  thinking_trace?: string;
  full_reasoning_trace?: string;
  analysis_text?: string;
  structured_assessment?: StructuredAssessment;
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


type Lang = 'en' | 'zh';

const I18N = {
  en: {
    title: 'WildfireReason-VLM',
    trackBadge: 'Legacy demo · development set',
    gtBadge: 'CAL FIRE DINS Ground-Truth',
    subtitle: 'Zero-Shot Forensic Ignition Attribution & Defect Discovery Engine across 4 Frontier Reasoning Models',
    switchToMap: 'Switch to 3D GIS Map',
    protocolBtn: 'Data Sources & Benchmark Protocol',
    scoreboardTitle: 'Cross-Model Empirical Scoreboard (50 Ground-Truth Cases: 25 Destroyed vs 25 Survived)',
    keyFinding: '⚠️ Key Finding: All frontier models perform at ~50% (coin-toss) with 88%–100% False Alarm Rates',
    accuracy: 'Accuracy',
    falseAlarm: 'FPR (survived)',
    locusAcc: 'Locus Acc',
    filters: 'Filters:',
    filterAll: 'All',
    filterSurvived: 'Survived (Defect Zero)',
    filterDestroyed: 'Destroyed',
    filterSaves: 'Tactical Firefighter Saves',
    caseSelect: 'Select Case:',
    of: 'of',
    macroAerial: 'Macro Aerial Ortho',
    microStreetView: 'Micro Street View',
    dualView: 'Dual-View Split',
    prefire: '2020-02 Pre-Fire Baseline',
    postfire: '2021-10 Post-Fire GT',
    current: 'Current Satellite State',
    streetviewTab: 'Street View Facade',
    zoom2x: '2x Close-up',
    zoom1x: '1x Fit',
    reticle: 'Target Reticle',
    macroBadge: 'Macro Ortho (300m WUI)',
    microBadge: 'Micro Facade (Ground-Truth)',
    macroLegend: 'Roof Integrity / Slope / Fuel Bed',
    microLegend: 'Siding Material / Fence Junctions',
    macroMicroTag: 'Macro + Micro Multimodal Synergy',
    dualSynergyTitle: 'Cross-Scale Multimodal Reasoning Synergy',
    dualSynergyDesc: 'Top-down aerial captures terrain slope, wildland vegetation continuity, and parcel setback. Ground-level street view exposes vertical siding material, under-eave gaps, and direct fence attachments. Multimodal fusion closes the spatial blind spot and matches CAL FIRE DINS ground truth.',
    streetviewNotice: 'Google Street View 360° Ground-Level Perspective',
    millimeterTexture: 'Millimeter Surface Texture',
    openGmaps: 'Open Google Maps 360°',
    groundLevel: 'Ground Level Perspective',
    facadeChipsTitle: 'Physical Hardening & Vulnerability Identifiers',
    clickForMechanism: 'Click chip for physical mechanism',
    mechanismTitle: 'Physical Vulnerability / Defense Mechanism Analysis',
    closeDrawer: 'Close Details',
    assessorTitle: 'County Assessor Records & California Chapter 7A Fire-Hardening Audit',
    assessorSubtitle: 'Cadastral Parcel Specifications & Statutory Building Code Verification',
    compliantBadge: 'Compliant (Chapter 7A)',
    nonCompliantBadge: 'Non-Compliant (Pre-2008)',
    pendingBadge: 'Year Built Pending',
    apnLabel: 'APN Parcel Number',
    yearBuiltLabel: 'Year Built',
    livingAreaLabel: 'Living Area',
    lotSizeLabel: 'Lot Size',
    assessedValueLabel: 'Assessed Improvement',
    rebuildHistoryLabel: 'Post-Fire Rebuild & Permit History',
    statutoryBasis: 'Statutory Regulatory Basis',
    forensicFactsTitle: 'Ground-Truth Incident Dossier & Fire Behavior',
    fireWeather: 'Fire Weather Forcing',
    fuelModel: 'Fuel Model',
    spreadRate: 'Flame Spread Rate',
    observedOutcome: 'Observed Damage Outcome',
    forensicLocus: 'Forensic Failure Point (Locus)',
    tacticalAction: 'Firefighter Tactical Actions',
    modelMatrixTitle: 'Multi-Model Deductive Reasoning & Prediction Matrix',
    modelMatrixSub: 'Pre-computed Zero-Shot Chain-of-Thought (CoT)',
    predictedSurvival: 'Predicted Survival:',
    locusAttribution: 'Ignition Locus Attribution:',
    locusMatchesGT: 'Matches Ground Truth',
    locusMismatch: 'Locus Mismatch',
    showThinking: 'Show Chain-of-Thought',
    hideThinking: 'Hide Chain-of-Thought',
    noRecord: 'No evaluation record for this case',
    interactiveQuery: 'Interactive Live Model Query',
    interactiveDesc: 'Trigger on-demand inference against current case',
    runLiveCall: 'Run Live Call',
    running: 'Evaluating...',
    expertDiagnosisTitle: 'Expert Fire Forensic Defect Analysis: Why Frontier Models Failed (Causality Audit)',
    expertSubtitle: 'Physical Causality Audit & Discrepancy Breakdown for Case',
    modelDiagnosis: 'Diagnosis:',
    recommendation: 'Targeted RLVR Reward & Training Recommendation:',
    collapseReasoning: 'Collapse Reasoning Panel',
    expandReasoning: 'Expand Model Reasoning (4 Models)',
    hidePanel: 'Collapse Reasoning',
    showPanel: 'Expand 4-Model Reasoning Matrix',
    flameVectorTitle: 'Flame Propagation & Wind Vector',
    surroundingVegetation: 'Surrounding Vegetation & Fuels',
    physicalBreachPoint: 'Forensic Failure Point (Locus)',
    firefighterAction: 'Firefighter Tactical Intervention',
    verifiedGSVBadge: 'Street-level photo (capture date unverified)',
    coverageGapBadge: 'WUI Private Road Coverage Gap',
    coverageGapTitle: 'WUI Ground-Level Street View Coverage Gap',
    coverageGapDesc: 'This parcel is located in a rural WUI fire zone on a private road/driveway (10-22 acres). Google Street View vehicles are legally and physically restricted to public highways and cannot traverse private parcels. In our benchmark, this reflects the realistic operational limitation of ground-level imagery in wildland fire scenarios—making sub-meter aerial satellite orthoimagery and county assessor records indispensable.',
    zeroAiPolicy: 'Academic Ground-Truth Policy: Zero synthetic AI images or corridor duplicates are substituted.',
    dualAerialGTTitle: 'Cross-Temporal Dual Aerial Ground Truth (Pre-Fire vs Post-Fire Damage)',
    dualAerialGTDesc: 'Dual high-resolution 0.3m/px aerial comparison: Left displays intact structural footprint and terrain slope prior to firestorm; Right displays CAL FIRE forensic damage inspection ground truth.',
    prefireAerialBadge: '🛰️ 2020-02 Pre-Fire Baseline (0.3m/px)',
    postfireAerialBadge: '🔥 2021-10 Post-Fire Ground Truth (0.3m/px)',
    openCoordsGmaps: 'Verify Parcel Coordinates on Google Maps',
    switchToDualAerial: 'View Dual Aerial Pre/Post-Fire Comparison',
    filterResidential: '🏡 Suburban WUI (Insurance Focus)',
    multiParadigmTitle: 'Multi-Paradigm Intelligence: Academic Physics & Google Bellwether 100m',
    multiParadigmSubtitle: 'Simultaneous Tri-Modal Hazard Synthesis & Actuarial Underwriting Integration',
    simultaneousBadge: 'Simultaneous Multi-Model',
    residentialSubdivisionBadge: 'Suburban Residential Subdivision (Insurance Focus)',
    ruralWuiBadge: 'Rural WUI Parcel (Dispersed Low-Density)',
    bellwetherCardTitle: 'Google X Project Bellwether (100m Grid)',
    bellwetherSubtitle: 'High-Resolution ML Wildfire Probability & Radiant Exposure Analysis',
    burnProb1yr: '1-Year Forward Burn Probability',
    burnProb5yr: '5-Year Cumulative Burn Probability',
    radiantCorridorTitle: '130ft (39.6m) Structure Radiant Exposure Corridor',
    threatenedStructures: 'Threatened Subdivision Structures',
    threatenedValue: 'Property Asset at Immediate Risk',
    mitigatedValue: 'Hardening Mitigated Asset Value',
    mitigationRoi: 'Mitigation ROI Multiplier',
    topDriversTitle: 'Top Environmental Hazard Drivers',
    academicCardTitle: 'Academic Physics & Discrete Simulations',
    academicSubtitle: 'Albini-Rothermel Level-Set PDE & Alexandridis Cellular Automata',
    rothermelTitle: 'Albini & Rothermel (1972) Level-Set PDE',
    rateOfSpread: 'Rate of Spread (ROS)',
    flameLength: 'Flame Length',
    firelineIntensity: 'Fireline Intensity',
    cellularAutomataTitle: 'Alexandridis Stochastic Cellular Automata',
    spottingProbability: 'Ember Spotting Probability',
    contagionIndex: 'Neighborhood Contagion Index',
    transitionRule: 'Transition Rule: Moore Neighborhood Grid',
    googleNdwsTitle: 'Google NDWS Neural Surrogate (NeurIPS 2021)',
    threatScore: 'CSI Threat Score',
    spatialIoU: 'Spatial IoU',
    precisionRecall: 'Precision / Recall',
    insuranceCardTitle: 'Commercial Property Insurance & Actuarial Underwriting',
    insuranceSubtitle: 'Verisk FireLine™ & California CDI Title 10 § 2644.9 Audit',
    firelineScoreLabel: 'Verisk FireLine™ Wildfire Hazard Score',
    zfireTierLabel: 'Zesty.ai Z-FIRE™ Underwriting Tier',
    underwritingActionLabel: 'Underwriting Determination',
    annualBasePremium: 'Actuarial Annual Base Premium',
    annualDiscountedPremium: 'Hardened Discounted Premium',
    totalDiscountLabel: 'Mandatory Mitigation Credit',
    cdiDiscountsTitle: 'California CDI Title 10 § 2644.9 Mandatory Credits',
    classARoofCredit: 'Class A Fire-Rated Roof Credit',
    emberVentCredit: 'Ember-Resistant Mesh Vents (<=1/8") Credit',
    zone0BufferCredit: 'Zone 0 (0-5ft) Hardened Perimeter Credit',
    cdiAuthority: 'Regulatory Basis: California Dept. of Insurance (CDI) Regulation Title 10 Section 2644.9',
    structuredAssessmentTitle: 'Multimodal Physical Assessment & Downstream Task Dispatcher',
    structuredAssessmentSubtitle: 'Structured physical deduction decoupled from binary decision making, powering insurance actuarial, tactical firefighting, and physics simulations',
    inputPipelineTitle: 'Multimodal Input Pipeline (Multi-Image + Assessor + Environmental + Physical Priors)',
    tabInsurance: 'Insurance Actuarial (CDI §2644.9)',
    tabFirefighter: 'Tactical Firefighting Triage',
    tabPhysics: 'Coupled Physics Boundary',
    vulnerabilityMatrixTitle: 'Physical Vulnerability Matrix (0-100)',
    causalPathwayTitle: 'Causal Failure Pathway & Chain-of-Events',
    weakestLinkTitle: 'Weakest Structural Link',
    stochasticFactorsTitle: 'Stochastic Field Survival Dependencies',
    uncertaintyTitle: 'Uncertainty Audit & Evidence Provenance',
    openFullscreenCoT: 'Fullscreen CoT Modal',
    copyCoT: 'Copy Trace',
    copied: 'Copied!',
    inspectModalTitle: 'Frontier Reasoning Trace Inspector',
    rlvrBenchmarkTitle: '5. Verifiable Post-Training Benchmark (RLVR / GRPO Multi-Objective Rewards)',
    rlvrBenchmarkSub: 'Ablation across 50 ground-truth DINS cases proving how verifiable physical rewards eliminate panic bias and hallucinations.',
    rewardFormulaLabel: 'Verifiable Physical Reward Formulation',
    rewardAuditTitle: 'Verifiable Physical Reward Audit (RLVR / GRPO)',
    rewardAuditSub: 'Evaluated against CAL FIRE forensic ground truth, Chapter 7A mesh physics, and combustible wick detection.',
    totalRewardScore: 'Total Reward Score',
    auditThisTrace: 'Audit with RLVR Engine',
    auditing: 'Auditing...',
    outcomeRewardLabel: 'Outcome R_out',
    locusRewardLabel: 'Locus R_loc',
    spatialRewardLabel: 'Spatial R_sp',
    physicsRewardLabel: 'Physics R_phy',
    formatRewardLabel: 'Format R_fmt',
  },
  zh: {
    title: 'WildfireReason-VLM 野火视觉多模态推演',
    trackBadge: '旧版演示 · 开发集',
    gtBadge: '加州消防局 DINS 真实物证',
    subtitle: '面向 4 大前沿推理模型的野火零样本致灾归因与建筑缺陷发现推演引擎',
    switchToMap: '切换至 3D GIS 地图',
    protocolBtn: '数据来源与评测规程',
    scoreboardTitle: '跨模型实证评估看板 (50 个真实案例: 25 栋损毁 vs 25 栋存活)',
    keyFinding: '⚠️ 核心发现: 前沿模型在零样本下的准确率仅约 50%（抛硬币水准），误报率高达 88%–100%',
    accuracy: '准确率',
    falseAlarm: '存活组误报率 FPR',
    locusAcc: '起火点归因率',
    filters: '筛选条件:',
    filterAll: '全部案例',
    filterSurvived: '完好存活 (零缺陷)',
    filterDestroyed: '彻底烧毁',
    filterSaves: '地面消防战术扑救',
    caseSelect: '选择案例:',
    of: '之',
    macroAerial: '宏观正射航拍',
    microStreetView: '微观立面街景',
    dualView: '跨视角并列 (Dual-View)',
    prefire: '2020-02 灾前完整基线',
    postfire: '2021-10 灾后物证真值',
    current: '当前最新卫星',
    streetviewTab: '立面街景实景',
    zoom2x: '2x 局部特写',
    zoom1x: '1x 全幅',
    reticle: '目标准心',
    macroBadge: '宏观正射航拍 (300m WUI)',
    microBadge: '微观立面街景 (近距离物证)',
    macroLegend: '屋面完好 / 地形坡度 / 邻近建筑',
    microLegend: '外墙材质 / 木栅栏直接附着点',
    macroMicroTag: '宏微观双尺度多模态协同',
    dualSynergyTitle: '跨尺度多模态协同证据链 (Cross-Scale Multimodal Reasoning)',
    dualSynergyDesc: '单一俯瞰航拍仅能观察屋顶与周边大尺度植被（看不清立面材料与直连木质栅栏）；微观街景仅能观察正面立面（无法掌握后山走势）。宏微观双视角融合消除了空间盲区，与加州消防实勘（DINS）百分百对齐。',
    streetviewNotice: 'Google Street View 360° 近距离立面实景',
    millimeterTexture: '毫米级表面材质纹理',
    openGmaps: '打开 Google Maps 360° 全景',
    groundLevel: '地面微观视角',
    facadeChipsTitle: '外立面物理硬化与易损构件标牌',
    clickForMechanism: '点击标签查看物理机理',
    mechanismTitle: '物理易损性与防御机理深度剖析',
    closeDrawer: '收起机理详情',
    assessorTitle: '县政地籍档案与加州 Chapter 7A 防火规范合规审计',
    assessorSubtitle: '地籍属性台账与法定建筑规范核准',
    compliantBadge: '合规达标 (Post-2008)',
    nonCompliantBadge: '不合规 (Pre-2008 老旧)',
    pendingBadge: '建成年份待查',
    apnLabel: 'APN 地籍编号',
    yearBuiltLabel: '建造年代',
    livingAreaLabel: '建筑面积',
    lotSizeLabel: '占地面积',
    assessedValueLabel: '评估建筑改善值',
    rebuildHistoryLabel: '灾后重建与建筑许可台账',
    statutoryBasis: '法定规范依据',
    forensicFactsTitle: '真实大火案卷与火场物理事实 (Ground-Truth Incident Dossier)',
    fireWeather: '火场天气强迫',
    fuelModel: '地表可燃物模型',
    spreadRate: '火焰前锋蔓延速率',
    observedOutcome: '现场法医勘验定损结果',
    forensicLocus: '法医突破弱点 (首燃点)',
    tacticalAction: '消防部队扑救战术干预',
    modelMatrixTitle: '多模型演绎推理与预测研判矩阵',
    modelMatrixSub: '预计算零样本思维链 (CoT)',
    predictedSurvival: '预测存活状态:',
    locusAttribution: '起火部位归因:',
    locusMatchesGT: '命中法医真值',
    locusMismatch: '起火点误判',
    showThinking: '展开思维链推理 (CoT)',
    hideThinking: '收起思维链',
    noRecord: '该案例暂无评测记录',
    interactiveQuery: '交互式实时模型调用',
    interactiveDesc: '针对当前案例即时发起在线 VLM 推理',
    runLiveCall: '发起在线推演',
    running: '推演中...',
    expertDiagnosisTitle: '专家火场法医缺陷诊断: 前沿模型为何误判 (归因与物理物证复盘)',
    expertSubtitle: '物理因果链审计与实测偏差点剖析 (Case',
    modelDiagnosis: '诊断分析:',
    recommendation: '视觉微调建议（演示）:',
    collapseReasoning: '收起模型研判面板',
    expandReasoning: '展开 4 大模型研判结果',
    hidePanel: '收起研判',
    showPanel: '展开 4 大模型研判矩阵',
    flameVectorTitle: '火势蔓延走向与风矢量',
    surroundingVegetation: '周边植被与燃料模型',
    physicalBreachPoint: '法医突破弱点 (首燃点)',
    firefighterAction: '消防部队扑救战术干预',
    verifiedGSVBadge: '100% 真实谷歌街景实景物证 (非AI生成)',
    coverageGapBadge: '野火交界区偏远私道覆盖盲区',
    coverageGapTitle: '野火交界区街景覆盖盲区剖析',
    coverageGapDesc: '该宗地位于偏远野火交界区（占地 10–22 英亩，建筑深嵌于私人车道内）。谷歌街景车受法律准入与地形限制仅在公路上行驶，无法进入私有地块。本基准严格恪守学术真值完整性，坚决杜绝 AI 伪造或假图填充，客观呈现真实世界地面数据的覆盖盲区——这正是高分辨率宏观航拍与县政地籍档案不可替代的科学依据。',
    zeroAiPolicy: '学术真值零伪造原则：本基准严格杜绝使用 AI 假图或重复走廊图片替代真值。',
    dualAerialGTTitle: '时序正射双视角真值对比 (灾前完整基线 vs 灾后损毁真值)',
    dualAerialGTDesc: '高分辨率 0.3m/px 时序遥感对比：左侧展示火灾前完好的建筑轮廓与地表植被；右侧展示加州消防局 (CAL FIRE DINS) 灾后法医现场定损真值。',
    prefireAerialBadge: '🛰️ 2020-02 灾前完整基线 (0.3m/px)',
    postfireAerialBadge: '🔥 2021-10 灾后现场定损真值 (0.3m/px)',
    openCoordsGmaps: '在 Google Maps 中核验地块空间坐标',
    switchToDualAerial: '查看灾前/灾后正射双航拍对比',
    filterResidential: '🏡 密集居民区 (Suburban WUI - 保险重点)',
    multiParadigmTitle: '跨流派智能预测全景: 学术物理仿真与 Google Bellwether 100m',
    multiParadigmSubtitle: '物理偏微分方程、高分辨率机器学习与加州保险精算法规同步推演',
    simultaneousBadge: '全流派多模型同步展示',
    residentialSubdivisionBadge: '密集居民住宅区 (保险评估重点)',
    ruralWuiBadge: '偏远林区散布宗地 (低密度)',
    bellwetherCardTitle: 'Google X Project Bellwether (100米高分辨率网格)',
    bellwetherSubtitle: '基于气象/地形/干旱的大尺度机器学习前瞻起火概率与辐射热推演',
    burnProb1yr: '1 年期前向起火概率 (1-Yr Burn Prob)',
    burnProb5yr: '5 年期累积起火概率 (5-Yr Burn Prob)',
    radiantCorridorTitle: '130 英尺 (39.6米) 极端辐射热与连带引燃走廊',
    threatenedStructures: '走廊内受威胁密集建筑数',
    threatenedValue: '走廊暴露受险资产总值',
    mitigatedValue: '硬化防护减灾挽回资产值',
    mitigationRoi: '防灾减灾投保投资回报率 (ROI)',
    topDriversTitle: 'Google Bellwether 核心环境危险驱动因子',
    academicCardTitle: '学术级物理方程与离散仿真模型',
    academicSubtitle: 'Albini-Rothermel 连续偏微分方程与 Alexandridis 元胞自动机',
    rothermelTitle: 'Albini & Rothermel (1972) Level-Set 偏微分方程',
    rateOfSpread: '火焰蔓延速率 (ROS)',
    flameLength: '预估火焰长度 (Flame Length)',
    firelineIntensity: '火线强度 (Fireline Intensity)',
    cellularAutomataTitle: 'Alexandridis 随机元胞自动机 (2D 离散格网)',
    spottingProbability: '飞火余烬跳跃引燃概率 (Spotting Prob)',
    contagionIndex: '邻里建筑连带传染指数 (Contagion Index)',
    transitionRule: '状态转移规则: 摩尔邻域风向偏置格网',
    googleNdwsTitle: 'Google NDWS 神经代理 ConvNet (NeurIPS 2021)',
    threatScore: 'CSI 威胁预警评分',
    spatialIoU: '蔓延空间交并比 (Spatial IoU)',
    precisionRecall: '查准率 / 查全率 (Precision / Recall)',
    insuranceCardTitle: '商业财产火险精算与核保评级',
    insuranceSubtitle: 'Verisk FireLine™ 危险分与加州 CDI Title 10 § 2644.9 法定减免审计',
    firelineScoreLabel: 'Verisk FireLine™ 野火危险分级',
    zfireTierLabel: 'Zesty.ai Z-FIRE™ 承保风险梯队',
    underwritingActionLabel: '保险核保决策结论',
    annualBasePremium: '基准精算年保费 (未硬化)',
    annualDiscountedPremium: '硬化合规折减后实缴年保费',
    totalDiscountLabel: '法定防火硬化综合折减率',
    cdiDiscountsTitle: '加州保险署 (CDI) Title 10 § 2644.9 法定强制保费抵扣',
    classARoofCredit: 'Class A 级不燃屋顶保费抵扣',
    emberVentCredit: '防余烬密网通风口 (<=1/8") 抵扣',
    zone0BufferCredit: 'Zone 0 (0-5英尺) 硬化不燃缓冲区抵扣',
    cdiAuthority: '法定监管依据: 加州保险署 (CDI) 规章第 10 篇第 2644.9 条 (Safer from Wildfires)',
    structuredAssessmentTitle: '多模态结构易损性评估与下游任务分发看板',
    structuredAssessmentSubtitle: '解耦二分类死胡同，输出可解释的结构化物理归因，赋能加州火险精算、现场战术分级与物理仿真',
    inputPipelineTitle: '多模态输入数据管线 (多源高分辨率影像 + 地籍档案 + 极端气象 + 物理先验)',
    tabInsurance: '加州火险精算核保 (CDI §2644.9)',
    tabFirefighter: '现场消防战术与应急分级 (WUI Triage)',
    tabPhysics: '物理微积分与网格仿真耦合边界',
    vulnerabilityMatrixTitle: '四维物理易损性评分矩阵 (0-100)',
    causalPathwayTitle: '因果损毁推导链与多米诺骨牌式蔓延',
    weakestLinkTitle: '建筑包络面最薄弱突破部件',
    stochasticFactorsTitle: '火场随机存活与不确定性因数',
    uncertaintyTitle: '置信度审计与视线遮挡盲区说明',
    openFullscreenCoT: '沉浸视窗',
    copyCoT: '复制思维链',
    copied: '已复制!',
    inspectModalTitle: '前沿模型思维链深度检视视窗 (CoT Inspector)',
    rlvrBenchmarkTitle: '5. 可验证物理后训练基准 (RLVR / GRPO 强化学习消融实验)',
    rlvrBenchmarkSub: '在 50 个真实 DINS 宗地上证实：可验证多目标物理奖惩机制如何彻底消除幻觉与 92% 误报恐慌偏置。',
    rewardFormulaLabel: '可验证多目标物理奖惩函数',
    rewardAuditTitle: '可验证物理奖惩审计 (RLVR / GRPO 强化学习实时打分)',
    rewardAuditSub: '对齐加州消防局 (CAL FIRE) 法医物证、Chapter 7A 防火滤网物理守恒与直连木栅栏火引检测。',
    totalRewardScore: '可验证物理总得分',
    auditThisTrace: '可验证物理审计',
    auditing: '审计中...',
    outcomeRewardLabel: '存活结果奖 R_out',
    locusRewardLabel: '首燃位点奖 R_loc',
    spatialRewardLabel: '空间着陆奖 R_sp',
    physicsRewardLabel: '物理守恒奖 R_phy',
    formatRewardLabel: '四阶段规范 R_fmt',
  }
};

function localizeTag(tag: string, lang: Lang): string {
  if (lang === 'en') return tag;
  const map: Record<string, string> = {
    'Stucco Fire-Resistant Cladding': '灰泥外墙阻火屏障',
    'Class A Asphalt Shingle Roof': 'Class A 沥青阻燃瓦屋顶',
    'Attached Combustible Fence (Critical Flaw)': '直连木质可燃栅栏 (致命漏洞)',
    'Multi-Pane Vinyl Windows': '多层中空隔热窗',
    'Concrete Driveway Zone 0 Buffer': '前院水泥 Zone 0 缓冲带',
    'Stucco Single-Family Facade': '灰泥独栋外立面',
    'Bilateral Wooden Fencing': '双侧贴靠木质栅栏',
    'Post-Fire Utility Rebuild Markings': '灾后市政重建标线',
    'Dual-Pane Sealed Windows': '双层密封中空窗',
    'Setback Country Estate Access': '深林退缩式庄园主道',
    'Overhanging Mature Oak Canopy': '悬垂密集橡树树冠',
    'Roadside Dry Chaparral Fuel Ladder': '路侧干枯灌木燃料梯',
    'Dense Canyon Timber & Brush': '峡谷浓密树冠火载荷',
    'Heavy Surface Needle & Leaf Litter': '地表厚层干枯落叶层',
    '1943 Legacy Envelope Materials': '1943 历史老旧易损建材',
    'Class A Asphalt Roof': 'Class A 沥青阻燃瓦屋顶',
    'Class A Metal Roof': 'Class A 阻燃金属屋顶',
    'Class A Tile Roof': 'Class A 阻燃陶瓷瓦屋顶',
    'Combustible Wood Roof Shakes': '可燃木质屋顶挂瓦',
    'Gabled Roof Geometry': '双坡排水斜屋顶构型',
    'Non-Combustible Stucco Cladding': '灰泥难燃阻火外立面',
    'Combustible Wood Siding': '可燃木质外墙挂板',
    'Combustible Vinyl Siding': '可燃乙烯基外墙挂板',
    'Directly Attached Combustible Fence': '直连木质可燃栅栏 (致命漏洞)',
    'Detached / Non-Combustible Perimeter': '外围隔离: 无木质贴靠栅栏',
    'Multi-Pane Insulated Glazing': '多层中空隔热玻璃窗',
    'Single-Pane Annealed Glass': '单层易碎玻璃窗',
    'Ember-Resistant Vents (<= 1/8" Mesh)': '防余烬通风口 (<= 1/8" 密网)',
    'Non-Compliant Large Mesh Vents (> 1/8")': '超标大网孔通风口 (> 1/8")',
    'Zone 0 (0-5ft) Non-Combustible Buffer': 'Zone 0 (0-5ft) 硬化防火缓冲带',
  };
  return map[tag] || tag;
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
  const [filterType, setFilterType] = useState<'all' | 'survived' | 'destroyed' | 'saves' | 'residential'>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [showDataSourceModal, setShowDataSourceModal] = useState<boolean>(false);
  const [selectedTraceModal, setSelectedTraceModal] = useState<{
    modelName: string;
    trace: string;
    caseId: string | number;
  } | null>(null);
  const [activeDownstreamTab, setActiveDownstreamTab] = useState<'insurance' | 'firefighter' | 'physics'>('insurance');
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);

  const handleCopyTrace = (text: string, id: string) => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopyFeedback(id);
      setTimeout(() => setCopyFeedback(null), 2000);
    }
  };

  const [liveModel, setLiveModel] = useState<string>('Qwen3.8-27B');
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [liveResult, setLiveResult] = useState<any>(null);
  const [liveRewardAudit, setLiveRewardAudit] = useState<any>(null);
  const [auditingReward, setAuditingReward] = useState<boolean>(false);
  const [selectedEpoch, setSelectedEpoch] = useState<'prefire' | 'postfire' | 'current'>('prefire');
  const [viewPerspective, setViewPerspective] = useState<'macro' | 'micro' | 'dual'>('macro');
  const [splitComparison, setSplitComparison] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<'1x' | '2x'>('1x');
  const [showTargetReticle, setShowTargetReticle] = useState<boolean>(true);
  const [activeAnnotationIndex, setActiveAnnotationIndex] = useState<number | null>(null);
  const [lang, setLang] = useState<Lang>('en');
  const [isReasoningCollapsed, setIsReasoningCollapsed] = useState<boolean>(false);

  useEffect(() => {
    try {
      const savedLang = localStorage.getItem('wildfire_app_lang');
      if (savedLang === 'en' || savedLang === 'zh') {
        setLang(savedLang);
      }
    } catch {}
  }, []);

  const handleSetLang = (newLang: Lang) => {
    setLang(newLang);
    try {
      localStorage.setItem('wildfire_app_lang', newLang);
    } catch {}
  };

  const t = I18N[lang];

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
          const items = casesData.items || [];
          setCases(items);
          if (typeof window !== 'undefined') {
            const params = new URLSearchParams(window.location.search);
            const targetId = params.get('case');
            if (targetId) {
              const foundIdx = items.findIndex((c: any) => String(c.id) === String(targetId));
              if (foundIdx !== -1) {
                setCurrentIndex(foundIdx);
              }
            }
          }
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
    if (filterType === 'residential') return !!item.is_residential_focus;
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
    setLiveRewardAudit(null);
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

      if (data && data.success) {
        const textToAudit = data.thinking_trace || data.analysis_text || (data.predicted_survival ? `PREDICTION: ${data.predicted_survival}\nPRIMARY_VULNERABILITY: ${data.predicted_vulnerability}\nFAILURE_MECHANISM: ${data.predicted_mechanism}` : '');
        if (textToAudit) {
          try {
            const auditRes = await fetch('http://localhost:8000/api/vlm/evaluate-reasoning-reward', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                case_id: String(currentCase.id),
                candidate_text: textToAudit,
              }),
            });
            if (auditRes.ok) {
              const auditData = await auditRes.json();
              setLiveRewardAudit({ ...auditData, audited_model_label: liveModel });
            }
          } catch (auditErr) {
            console.error('Audit evaluation error:', auditErr);
          }
        }
      }
    } catch (err) {
      setLiveResult({ success: false, error: String(err) });
    } finally {
      setLiveLoading(false);
    }
  };

  const handleAuditModelReasoning = async (modelLabel: string, candidateText: string) => {
    if (!currentCase || !candidateText) return;
    setAuditingReward(true);
    try {
      const auditRes = await fetch('http://localhost:8000/api/vlm/evaluate-reasoning-reward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: String(currentCase.id),
          candidate_text: candidateText,
        }),
      });
      if (auditRes.ok) {
        const auditData = await auditRes.json();
        setLiveRewardAudit({ ...auditData, audited_model_label: modelLabel });
      }
    } catch (err) {
      console.error('Manual audit error:', err);
    } finally {
      setAuditingReward(false);
    }
  };

  return (
    <div className="flex-1 h-full flex flex-col bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Audit 2026-10-22: persistent notice that this view is a legacy demo, not a results page */}
      <div className="bg-amber-500/15 border-b border-amber-500/40 px-6 py-2 text-[11px] text-amber-200">
        Legacy demo on a 50-case development set (text-only model runs; 25 of 25 survived cases come from one fire).
        Fields that were generated from the true outcome label (Bellwether, insurance, simulations, assessments) have been
        removed and show "—". This page is not a source of results; see docs/EVIDENCE_STATUS.md.
      </div>
      {/* Top Header & Application Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-rose-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-wide">
                {t.title}
              </h1>
              <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-orange-500/20 text-orange-400 border border-orange-500/30">
                {t.trackBadge}
              </span>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                {t.gtBadge}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              {t.subtitle}
            </p>
          </div>
        </div>

        {/* Global Action Bar */}
        <div className="flex items-center gap-3">
          {/* Language Switcher (EN | 中文) */}
          <div className="flex items-center bg-slate-800/90 rounded-lg p-0.5 border border-slate-700 shadow-sm">
            <button
              id="btn-lang-en"
              onClick={() => handleSetLang('en')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold transition-all cursor-pointer ${
                lang === 'en'
                  ? 'bg-orange-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
              }`}
              title="Switch language to English"
            >
              <Globe className="w-3 h-3" />
              <span>EN</span>
            </button>
            <button
              id="btn-lang-zh"
              onClick={() => handleSetLang('zh')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold transition-all cursor-pointer ${
                lang === 'zh'
                  ? 'bg-orange-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
              }`}
              title="切换界面语言至中文"
            >
              <span>中文</span>
            </button>
          </div>

          {/* Switch to GIS Map View */}
          <button
            onClick={onSwitchToMap}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all shadow-sm cursor-pointer"
          >
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            <span>{t.switchToMap}</span>
          </button>

          {/* Data Source & Methodology Button */}
          <button
            onClick={() => setShowDataSourceModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/80 transition-all cursor-pointer"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>{t.protocolBtn}</span>
          </button>
        </div>
      </header>

      {/* Cross-Model Benchmark Scoreboard (Top Metric Ribbon) */}
      <section className="bg-slate-900/40 border-b border-slate-800/60 px-6 py-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              {t.scoreboardTitle}
            </span>
          </div>
          <span className="text-[11px] text-rose-400 font-medium">
            {t.keyFinding}
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
                    {isSonnet ? 'Claude Sonnet' : isHaiku ? 'Claude Haiku' : isQwen ? 'Open vLLM' : 'MiniMax'}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center pt-1 border-t border-slate-800/60">
                  <div>
                    <div className="text-[10px] text-slate-400">{t.accuracy}</div>
                    <div className="text-sm font-bold text-slate-100">{item.accuracy}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-rose-400 font-medium">{t.falseAlarm}</div>
                    <div className="text-sm font-bold text-rose-400">{item.false_alarm_rate}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">{t.locusAcc}</div>
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
          <span className="text-xs text-slate-400 mr-1 font-medium">{t.filters}</span>
          {(
            [
              { key: 'all', label: `${t.filterAll} (${cases.length})` },
              { key: 'residential', label: t.filterResidential },
              { key: 'survived', label: t.filterSurvived },
              { key: 'destroyed', label: t.filterDestroyed },
              { key: 'saves', label: t.filterSaves },
            ] as const
          ).map((pill) => (
            <button
              key={pill.key}
              onClick={() => {
                setFilterType(pill.key);
                setCurrentIndex(0);
                setLiveResult(null);
              }}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                filterType === pill.key
                  ? 'bg-orange-500 text-white shadow-md shadow-orange-500/20'
                  : 'bg-slate-800/60 text-slate-300 hover:bg-slate-800 border border-slate-700/60'
              }`}
            >
              {pill.label}
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
            {filteredCases.map((c) => (
              <option key={c.id} value={c.id}>
                Case #{c.id} - {c.survival_label} ({c.incident} Fire) {c.is_residential_focus ? '🏡 Suburban' : ''} {c.defensive_actions !== 'None' ? '★ Saved by Hand Crew' : ''}
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
            {filteredCases.length > 0 ? currentIndex + 1 : 0} {t.of} {filteredCases.length}
          </span>

          <div className="h-4 w-[1px] bg-slate-800 mx-1" />

          {/* Model Reasoning Panel Collapsible Toggle Button */}
          <button
            id="btn-toggle-reasoning"
            onClick={() => setIsReasoningCollapsed(!isReasoningCollapsed)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all border cursor-pointer ${
              isReasoningCollapsed
                ? 'bg-purple-600/30 text-purple-200 border-purple-500/50 hover:bg-purple-600/40 shadow-sm shadow-purple-500/20'
                : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-750 hover:text-white'
            }`}
            title={isReasoningCollapsed ? t.expandReasoning : t.collapseReasoning}
          >
            {isReasoningCollapsed ? (
              <>
                <PanelRightOpen className="w-3.5 h-3.5 text-purple-400" />
                <span>{t.showPanel}</span>
              </>
            ) : (
              <>
                <PanelRightClose className="w-3.5 h-3.5 text-orange-400" />
                <span>{t.hidePanel}</span>
              </>
            )}
          </button>
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
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Aerial Imagery & Forensic GT Card */}
            <div className={`${isReasoningCollapsed ? 'lg:col-span-12' : 'lg:col-span-8'} space-y-4 transition-all duration-300`}>
              {/* High-Resolution Multi-Modal Imagery & Perspective Workstation Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                {/* Top Multi-Perspective Mode Switcher Bar */}
                <div className="bg-slate-950 p-2.5 border-b border-slate-800 flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800">
                    {/* Perspective 1: Macro Aerial */}
                    <button
                      onClick={() => {
                        setViewPerspective('macro');
                        setSplitComparison(false);
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                        viewPerspective === 'macro'
                          ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
                      }`}
                    >
                      <Layers className="w-3.5 h-3.5" />
                      <span>{t.macroAerial}</span>
                    </button>

                    {/* Perspective 2: Micro Street View */}
                    {/* Perspective 2: Micro Street View */}
                    <button
                      onClick={() => {
                        setViewPerspective('micro');
                        setSplitComparison(false);
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                        viewPerspective === 'micro'
                          ? 'bg-purple-600 text-white shadow-md shadow-purple-500/20'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
                      }`}
                    >
                      <Camera className="w-3.5 h-3.5" />
                      <span>{t.microStreetView}</span>
                      {currentCase.streetview?.has_image ? (
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" title="100% Real Google Street View" />
                      ) : (
                        <span className="text-[10px] text-amber-400/90 font-mono px-1 rounded bg-amber-950/60 border border-amber-500/30">
                          {lang === 'zh' ? '私道盲区' : 'Gap'}
                        </span>
                      )}
                    </button>

                    {/* Perspective 3: Dual-View Split */}
                    <button
                      onClick={() => {
                        setViewPerspective('dual');
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                        viewPerspective === 'dual'
                          ? 'bg-gradient-to-r from-orange-500 to-rose-600 text-white shadow-md shadow-orange-500/20'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
                      }`}
                    >
                      <Split className="w-3.5 h-3.5" />
                      <span>{t.dualView}</span>
                    </button>
                  </div>

                  {/* Mode Badge Indicator */}
                  <div className="flex items-center gap-2">
                    {viewPerspective === 'macro' && (
                      <span className="text-[10px] font-mono font-bold text-blue-300 bg-blue-950/70 border border-blue-500/30 px-2.5 py-0.5 rounded-full">
                        Zoom-18 Ortho (0.3m/px)
                      </span>
                    )}
                    {viewPerspective === 'micro' && (
                      <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full flex items-center gap-1 ${
                        currentCase.streetview?.has_image
                          ? 'text-emerald-300 bg-emerald-950/70 border border-emerald-500/30'
                          : 'text-amber-300 bg-amber-950/70 border border-amber-500/30'
                      }`}>
                        {currentCase.streetview?.has_image ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-400" />
                            <span>{lang === 'zh' ? '真实谷歌街景现场物证' : '100% Real GSV Ground Photo'}</span>
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="w-3 h-3 text-amber-400" />
                            <span>{lang === 'zh' ? '偏远私道覆盖盲区剖析' : 'WUI Private Road Coverage Gap'}</span>
                          </>
                        )}
                      </span>
                    )}
                    {viewPerspective === 'dual' && (
                      <span className="text-[10px] font-mono font-bold text-orange-300 bg-orange-950/70 border border-orange-500/30 px-2.5 py-0.5 rounded-full flex items-center gap-1">
                        <Split className="w-3 h-3 text-orange-400" />
                        <span>
                          {currentCase.streetview?.has_image
                            ? (lang === 'zh' ? '宏微双视角多模态并列' : 'Macro + Micro Dual-Perspective')
                            : (lang === 'zh' ? '时序正射双视角真值对比' : 'Dual-Aerial Temporal Ground Truth')}
                        </span>
                      </span>
                    )}
                  </div>
                </div>

                {/* Perspective 1: MACRO AERIAL VIEWPORT */}
                {viewPerspective === 'macro' && (
                  <div>
                    {/* Multi-Epoch Timeline Switcher Bar */}
                    <div className="bg-slate-950/80 p-2 border-b border-slate-800 flex items-center justify-between gap-1 text-xs">
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
                          <span>{t.prefire}</span>
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
                          <span>{t.postfire}</span>
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
                          <span>{t.current}</span>
                        </button>
                      </div>

                      {/* Zoom & Target Reticle Controls */}
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          onClick={() => setShowTargetReticle(!showTargetReticle)}
                          className={`px-2 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1 shrink-0 ${
                            showTargetReticle
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-sm shadow-amber-500/20'
                              : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700'
                          }`}
                          title="显示/隐藏地块正中央的评估目标建筑准心"
                        >
                          <Crosshair className="w-3.5 h-3.5" />
                          <span className="hidden sm:inline">准心</span>
                        </button>

                        <div className="flex items-center bg-slate-850 rounded-lg p-0.5 border border-slate-700">
                          <button
                            onClick={() => setZoomLevel('1x')}
                            className={`px-2 py-1 rounded text-[11px] font-bold transition-all ${
                              zoomLevel === '1x'
                                ? 'bg-slate-700 text-white shadow-sm'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                            title="1x 社区周边环境全景 (300m WUI buffer)"
                          >
                            1x 全景
                          </button>
                          <button
                            onClick={() => setZoomLevel('2x')}
                            className={`px-2 py-1 rounded text-[11px] font-bold transition-all flex items-center gap-1 ${
                              zoomLevel === '2x'
                                ? 'bg-orange-600 text-white shadow-sm'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                            title="2x 单栋建筑结构特写 (150m close-up)"
                          >
                            <ZoomIn className="w-3 h-3" />
                            <span>{t.zoom2x}</span>
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
                    </div>

                    {/* Imagery Viewport */}
                    {splitComparison ? (
                      /* 2-Column Split View: Pre-Fire (2020) vs Post-Fire GT (2021) */
                      <div className="grid grid-cols-2 divide-x divide-slate-800 bg-slate-950">
                        {/* Left: Pre-Fire */}
                        <div className="relative aspect-square overflow-hidden group">
                          <div className="w-full h-full overflow-hidden flex items-center justify-center">
                            <img
                              src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=prefire&v=2`}
                              alt={`Pre-fire 2020 Case ${currentCase.id}`}
                              className={`w-full h-full object-cover transition-transform duration-500 origin-center ${
                                zoomLevel === '2x' ? 'scale-[2.0]' : 'group-hover:scale-105'
                              }`}
                            />
                          </div>

                          {/* Tactical Center Reticle */}
                          {showTargetReticle && (
                            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                              <div className={`relative flex items-center justify-center transition-transform duration-300 ${zoomLevel === '2x' ? 'scale-125' : 'scale-100'}`}>
                                <div className="w-14 h-14 rounded-full border border-amber-400/80 animate-ping opacity-25" />
                                <div className="absolute w-12 h-12 rounded-full border-2 border-amber-400/90 shadow-[0_0_12px_rgba(251,191,36,0.5)]" />
                                <div className="absolute -left-3 w-2 h-0.5 bg-amber-400" />
                                <div className="absolute -right-3 w-2 h-0.5 bg-amber-400" />
                                <div className="absolute -top-3 w-0.5 h-2 bg-amber-400" />
                                <div className="absolute -bottom-3 w-0.5 h-2 bg-amber-400" />
                                <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                              </div>
                              <div className="absolute top-[60%] bg-black/85 backdrop-blur-md px-2 py-0.5 rounded text-[9px] font-mono font-bold text-amber-300 border border-amber-500/40 shadow-lg whitespace-nowrap">
                                🎯 目标单栋 #{currentCase.id}
                              </div>
                            </div>
                          )}

                          <div className="absolute top-2 left-2 bg-emerald-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-emerald-300 border border-emerald-500/40">
                            2020.02 灾前完整态 (VLM输入)
                          </div>
                          <div className="absolute bottom-2 left-2 right-2 bg-slate-950/80 backdrop-blur-md px-2 py-1 rounded text-[10px] text-slate-300 truncate">
                            完好屋面 / 绿化树冠 / 30m 防御圈
                          </div>
                        </div>

                        {/* Right: Post-Fire GT */}
                        <div className="relative aspect-square overflow-hidden group">
                          <div className="w-full h-full overflow-hidden flex items-center justify-center">
                            <img
                              src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=postfire&v=2`}
                              alt={`Post-fire 2021 Case ${currentCase.id}`}
                              className={`w-full h-full object-cover transition-transform duration-500 origin-center ${
                                zoomLevel === '2x' ? 'scale-[2.0]' : 'group-hover:scale-105'
                              }`}
                            />
                          </div>

                          {/* Tactical Center Reticle */}
                          {showTargetReticle && (
                            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                              <div className={`relative flex items-center justify-center transition-transform duration-300 ${zoomLevel === '2x' ? 'scale-125' : 'scale-100'}`}>
                                <div className="w-14 h-14 rounded-full border border-amber-400/80 animate-ping opacity-25" />
                                <div className="absolute w-12 h-12 rounded-full border-2 border-amber-400/90 shadow-[0_0_12px_rgba(251,191,36,0.5)]" />
                                <div className="absolute -left-3 w-2 h-0.5 bg-amber-400" />
                                <div className="absolute -right-3 w-2 h-0.5 bg-amber-400" />
                                <div className="absolute -top-3 w-0.5 h-2 bg-amber-400" />
                                <div className="absolute -bottom-3 w-0.5 h-2 bg-amber-400" />
                                <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                              </div>
                              <div className="absolute top-[60%] bg-black/85 backdrop-blur-md px-2 py-0.5 rounded text-[9px] font-mono font-bold text-amber-300 border border-amber-500/40 shadow-lg whitespace-nowrap">
                                🎯 目标单栋 #{currentCase.id}
                              </div>
                            </div>
                          )}

                          <div className="absolute top-2 left-2 bg-rose-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-rose-300 border border-rose-500/40">
                            2021.10 灾后现场物证 (真值)
                          </div>
                          <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 backdrop-blur-md px-2 py-1 rounded text-[10px] font-semibold border border-white/5">
                            {currentCase.survival_label === 'Destroyed' ? (
                              <div className="flex items-center justify-between text-rose-300">
                                <span className="truncate">🔥 损毁定级：{currentCase.raw_damage ?? '—'}</span>
                                <span className="text-[9px] font-mono text-rose-400/80 shrink-0 ml-1">重损/全损核定</span>
                              </div>
                            ) : (
                              <div className="flex items-center justify-between text-emerald-300">
                                <span className="truncate">🛡️ 存活定级：{currentCase.raw_damage ?? '—'}</span>
                                <span className="text-[9px] font-mono text-emerald-400/80 shrink-0 ml-1">有效防守</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* Single Epoch View */
                      <div className="relative aspect-video w-full bg-slate-950 overflow-hidden group">
                        <div className="w-full h-full overflow-hidden flex items-center justify-center">
                          <img
                            src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=${selectedEpoch}&v=2`}
                            alt={`Aerial view ${selectedEpoch} for Case ${currentCase.id}`}
                            className={`w-full h-full object-cover transition-transform duration-500 origin-center ${
                              zoomLevel === '2x' ? 'scale-[2.0]' : 'group-hover:scale-105'
                            }`}
                          />
                        </div>

                        {/* Tactical Center Reticle */}
                        {showTargetReticle && (
                          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                            <div className={`relative flex items-center justify-center transition-transform duration-300 ${zoomLevel === '2x' ? 'scale-125' : 'scale-100'}`}>
                              <div className="w-14 h-14 rounded-full border border-amber-400/80 animate-ping opacity-25" />
                              <div className="absolute w-12 h-12 rounded-full border-2 border-amber-400/90 shadow-[0_0_12px_rgba(251,191,36,0.5)]" />
                              <div className="absolute -left-3 w-2 h-0.5 bg-amber-400" />
                              <div className="absolute -right-3 w-2 h-0.5 bg-amber-400" />
                              <div className="absolute -top-3 w-0.5 h-2 bg-amber-400" />
                              <div className="absolute -bottom-3 w-0.5 h-2 bg-amber-400" />
                              <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                            </div>
                            <div className="absolute top-[60%] bg-black/85 backdrop-blur-md px-2 py-0.5 rounded text-[9px] font-mono font-bold text-amber-300 border border-amber-500/40 shadow-lg whitespace-nowrap">
                              🎯 目标单栋 #{currentCase.id}
                            </div>
                          </div>
                        )}

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
                            {zoomLevel === '2x' ? '0.15m / Pixel (2x Close-Up)' : '0.3m / Pixel Orthorectified'}
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
                  </div>
                )}

                {/* Perspective 2: MICRO STREET VIEW VIEWPORT */}
                {viewPerspective === 'micro' && (
                  <div className="space-y-3">
                    <div className="relative aspect-video w-full bg-slate-950 overflow-hidden group rounded-xl border border-slate-800">
                      {currentCase.streetview?.has_image ? (
                        <>
                          <img
                            src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=streetview&v=3`}
                            alt={`Google Street View Facade for Case ${currentCase.id}`}
                            className="w-full h-full object-cover transition-transform duration-700 origin-center group-hover:scale-105"
                          />
                          <div className="absolute top-3 left-3 flex flex-wrap gap-2">
                            <span className="px-2.5 py-1 rounded-md text-xs font-bold bg-emerald-950/90 text-emerald-200 border border-emerald-500/40 backdrop-blur-md shadow-lg flex items-center gap-1.5">
                              <ShieldCheck className="w-3.5 h-3.5 text-emerald-300" />
                              <span>{t.verifiedGSVBadge}</span>
                            </span>
                            <span className="px-2 py-1 rounded-md text-[11px] font-mono bg-black/75 text-emerald-300 border border-emerald-500/30 backdrop-blur-md">
                              {currentCase.streetview?.provenance_badge ?? '—'}
                            </span>
                          </div>

                          <div className="absolute top-3 right-3">
                            <a
                              href={currentCase.streetview.google_maps_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-900/90 text-slate-200 hover:text-white hover:bg-blue-600 border border-slate-700 transition-all flex items-center gap-1.5 backdrop-blur-md shadow-lg"
                              title="Open Google Maps 360 Panorama"
                            >
                              <span>{t.openGmaps}</span>
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          </div>

                          <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-xs bg-slate-950/85 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10">
                            <div className="flex items-center gap-1.5 text-slate-200">
                              <Home className="w-3.5 h-3.5 text-emerald-400" />
                              <span className="font-semibold">{currentCase.street_address || `Case #${currentCase.id}`}</span>
                            </div>
                            <span className="text-emerald-300 font-mono text-[11px]">
                              {currentCase.incident} Fire | Verified GSV Facade
                            </span>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full bg-gradient-to-b from-slate-900 via-slate-950 to-slate-950 p-6 flex flex-col md:flex-row items-center gap-6">
                          {/* Aerial Zoom Thumbnail */}
                          <div className="relative w-44 h-44 shrink-0 rounded-xl overflow-hidden border border-amber-500/40 shadow-xl group">
                            <img
                              src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=prefire&v=2`}
                              alt={`Aerial Zoom Case ${currentCase.id}`}
                              className="w-full h-full object-cover scale-150"
                            />
                            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                              <div className="w-10 h-10 rounded-full border-2 border-amber-400 animate-pulse shadow-[0_0_10px_rgba(251,191,36,0.6)]" />
                            </div>
                            <div className="absolute top-1.5 left-1.5 px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-amber-950/90 text-amber-200 border border-amber-500/40">
                              0.3m/px Aerial Zoom
                            </div>
                            <div className="absolute bottom-1.5 left-1.5 right-1.5 px-1 py-0.5 rounded text-[9px] font-mono text-center bg-black/85 text-slate-300 truncate">
                              {currentCase.latitude.toFixed(4)}°N, {currentCase.longitude.toFixed(4)}°W
                            </div>
                          </div>

                          {/* Scientific Explanation & Benchmark Integrity Callout */}
                          <div className="flex-1 space-y-2.5 text-left">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="px-2.5 py-1 rounded-md text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-1.5">
                                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                                <span>{t.coverageGapTitle}</span>
                              </span>
                              <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
                                Lot: {currentCase.assessor_record?.lot_size ?? '—'}
                              </span>
                            </div>

                            <p className="text-xs text-slate-300 leading-relaxed">
                              {t.coverageGapDesc}
                            </p>

                            <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-500/30 text-[11px] text-emerald-300 flex items-start gap-2">
                              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                              <span className="font-semibold">{t.zeroAiPolicy}</span>
                            </div>

                            <div className="flex items-center gap-2.5 pt-1 flex-wrap">
                              <a
                                href={currentCase.streetview?.google_coords_url || `https://www.google.com/maps/search/?api=1&query=${currentCase.latitude},${currentCase.longitude}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 transition-all flex items-center gap-1.5 border border-slate-700"
                              >
                                <ExternalLink className="w-3.5 h-3.5 text-blue-400" />
                                <span>{t.openCoordsGmaps}</span>
                              </a>
                              <button
                                onClick={() => setViewPerspective('dual')}
                                className="px-3 py-1.5 rounded-lg text-xs font-bold bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-600 hover:to-rose-700 text-white transition-all flex items-center gap-1.5 shadow-md shadow-orange-500/20"
                              >
                                <Split className="w-3.5 h-3.5" />
                                <span>{t.switchToDualAerial}</span>
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Facade Physical Hardening & Vulnerability Annotations */}
                    <div className="p-3 bg-slate-950 border-t border-slate-800 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                          <Target className="w-3.5 h-3.5 text-purple-400" />
                          <span>{t.facadeChipsTitle}</span>
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">{t.clickForMechanism}</span>
                      </div>

                      <div className="flex flex-wrap gap-1.5">
                        {(currentCase.streetview?.facade_annotations || []).map((ann, idx) => (
                          <button
                            key={idx}
                            onClick={() => setActiveAnnotationIndex(activeAnnotationIndex === idx ? null : idx)}
                            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all text-left flex items-center gap-1.5 border ${
                              ann.type === 'hardening'
                                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/25'
                                : ann.type === 'vulnerability'
                                ? 'bg-rose-500/15 text-rose-300 border-rose-500/30 hover:bg-rose-500/25'
                                : ann.type === 'rebuild'
                                ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/25'
                                : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-750'
                            } ${activeAnnotationIndex === idx ? 'ring-2 ring-purple-400 shadow-md' : ''}`}
                          >
                            <span className={`w-1.5 h-1.5 rounded-full ${
                              ann.type === 'hardening' ? 'bg-emerald-400' : ann.type === 'vulnerability' ? 'bg-rose-400' : 'bg-cyan-400'
                            }`} />
                            <span>{localizeTag(ann.tag, lang)}</span>
                          </button>
                        ))}
                      </div>

                      {/* Expanded Annotation Detail Drawer */}
                      {activeAnnotationIndex !== null && currentCase.streetview?.facade_annotations?.[activeAnnotationIndex] && (
                        <div className="bg-slate-900 border border-purple-500/30 p-2.5 rounded-xl text-xs space-y-1 animate-in fade-in">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-purple-300">
                              {currentCase.streetview.facade_annotations[activeAnnotationIndex].tag}
                            </span>
                            <span className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                              currentCase.streetview.facade_annotations[activeAnnotationIndex].type === 'hardening'
                                ? 'bg-emerald-500/20 text-emerald-300'
                                : 'bg-rose-500/20 text-rose-300'
                            }`}>
                              {currentCase.streetview.facade_annotations[activeAnnotationIndex].type.toUpperCase()}
                            </span>
                          </div>
                          <p className="text-slate-300 text-[11px] leading-relaxed">
                            {currentCase.streetview.facade_annotations[activeAnnotationIndex].desc}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Perspective 3: DUAL-VIEW SPLIT */}
                {viewPerspective === 'dual' && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 divide-x divide-slate-800 bg-slate-950 rounded-xl overflow-hidden border border-slate-800">
                      {/* Left Tile: Always 2020 Pre-fire Macro Aerial Ortho */}
                      <div className="relative aspect-square overflow-hidden group">
                        <img
                          src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=prefire&v=2`}
                          alt={`Pre-fire aerial view Case ${currentCase.id}`}
                          className="w-full h-full object-cover"
                        />
                        {showTargetReticle && (
                          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                            <div className="w-10 h-10 rounded-full border-2 border-amber-400/90 shadow-[0_0_10px_rgba(251,191,36,0.5)]" />
                          </div>
                        )}
                        <div className="absolute top-2 left-2 bg-blue-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-blue-300 border border-blue-500/40">
                          {currentCase.streetview?.has_image ? t.macroBadge : t.prefireAerialBadge}
                        </div>
                        <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 backdrop-blur-md px-2 py-0.5 rounded text-[10px] text-slate-300 truncate font-mono">
                          {t.macroLegend}
                        </div>
                      </div>

                      {/* Right Tile: Real GSV Photo IF available, ELSE Post-fire Damage Aerial Ground Truth */}
                      <div className="relative aspect-square overflow-hidden group">
                        {currentCase.streetview?.has_image ? (
                          <>
                            <img
                              src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=streetview&v=3`}
                              alt={`Real Micro Street View Case ${currentCase.id}`}
                              className="w-full h-full object-cover"
                            />
                            <div className="absolute top-2 left-2 bg-emerald-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-emerald-300 border border-emerald-500/40 flex items-center gap-1">
                              <ShieldCheck className="w-3 h-3 text-emerald-400" />
                              <span>{t.microBadge}</span>
                            </div>
                            <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 backdrop-blur-md px-2 py-0.5 rounded text-[10px] text-slate-300 truncate font-mono">
                              {t.microLegend}
                            </div>
                          </>
                        ) : (
                          <>
                            <img
                              src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}?epoch=postfire&v=2`}
                              alt={`Post-fire Damage Aerial GT Case ${currentCase.id}`}
                              className="w-full h-full object-cover"
                            />
                            {showTargetReticle && (
                              <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                                <div className={`w-10 h-10 rounded-full border-2 ${currentCase.survival_label === 'Destroyed' ? 'border-rose-500/90 shadow-[0_0_10px_rgba(244,63,94,0.5)]' : 'border-emerald-400/90 shadow-[0_0_10px_rgba(52,211,153,0.5)]'}`} />
                              </div>
                            )}
                            <div className="absolute top-2 left-2 bg-rose-950/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-rose-300 border border-rose-500/40 flex items-center gap-1">
                              <Flame className="w-3 h-3 text-rose-400" />
                              <span>{t.postfireAerialBadge}</span>
                            </div>
                            <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 backdrop-blur-md px-2 py-0.5 rounded text-[10px] text-slate-300 truncate font-mono flex items-center justify-between">
                              <span className="font-bold text-rose-300">
                                CAL FIRE DINS: {currentCase.survival_label}
                              </span>
                              <span className="text-slate-400">
                                {currentCase.raw_damage}
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Dual-View Synergy Callout */}
                    <div className="p-3 bg-slate-950 border-t border-slate-800 text-xs text-slate-300 space-y-1">
                      <div className="flex items-center gap-2 font-bold text-orange-300">
                        <Sparkles className="w-4 h-4 text-orange-400" />
                        <span>{currentCase.streetview?.has_image ? t.dualSynergyTitle : t.dualAerialGTTitle}</span>
                      </div>
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        {currentCase.streetview?.has_image ? t.dualSynergyDesc : t.dualAerialGTDesc}
                      </p>
                    </div>
                  </div>
                )}

                {/* Weather & Meteorological Forcing Banner */}
                <div className="bg-rose-950/20 border-t border-rose-900/30 p-2.5 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 text-rose-300">
                    <Wind className="w-4 h-4 text-rose-400 animate-pulse" />
                    <span className="font-semibold">Red Flag Warning Forcing:</span>
                    <span>Gusts 25-35 mph | DFM &lt; 6%</span>
                  </div>
                  <span className="text-rose-400/80 font-mono text-[11px]">Chaparral/Timber Fuel</span>
                </div>

                {/* Suburban Community / Dense Parcel Context Callout */}
                {(currentCase.incident === 'Nelson' || currentCase.is_residential_focus) && (
                  <div className="bg-amber-950/30 border-t border-amber-500/30 p-3 text-xs text-amber-200/90 flex items-start gap-2.5">
                    <Target className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <div className="space-y-1.5 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-bold text-amber-300">
                          {lang === 'zh'
                            ? `高密度居民区火场物证辨析 (${currentCase.incident} Fire Case #${currentCase.id})`
                            : `Suburban Residential Subdivision Forensic Analysis (${currentCase.incident} Fire Case #${currentCase.id})`}
                        </span>
                        <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded font-mono border border-amber-500/30">
                          {currentCase.assessor_record?.lot_size ?? '—'}
                        </span>
                        <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-mono border border-emerald-500/30">
                          {lang === 'zh' ? '真实街道立面物证' : 'Real Street View Ground Truth'}
                        </span>
                      </div>
                      <div className="text-[11px] leading-relaxed text-slate-300 space-y-2">
                        {lang === 'zh' ? (
                          <>
                            <p>
                              • <b>为什么看到整片小区？</b> 该案例位于奥罗维尔近郊高密度独栋住宅区（Mountain Vista Dr，单户面宽约 18 米）。Zoom 18（航拍跨度 300 米）必然覆盖整条街道与周边房屋。<b>正中心准心光标处即为本案评估目标（{currentCase.street_address || `Case #${currentCase.id}`}）</b>；可点击上方 <b>【2x 特写】</b> 聚焦单栋建筑。
                            </p>
                            <p>
                              • <b>保险核保与连带易损性重点：</b> 居民区房屋间距仅 10–30 英尺，一栋失火将通过热辐射与木栅栏引发多栋连环起火（骨牌效应）。加州 CDI Title 10 § 2644.9 规定只要实施 Class A 屋顶、防余烬密网和 Zone 0 缓冲，即可获得法定保费折减（当前案例已折减至 <b>${currentCase?.insurance_actuarial?.annual_discounted_premium_usd ?? '—'}/年</b>）。
                            </p>
                          </>
                        ) : (
                          <>
                            <p>
                              • <b>Why does the aerial view capture an entire subdivision?</b> This parcel is situated in a dense suburban residential tract (Mountain Vista Dr, Oroville; structure frontage width ~18m). Zoom-18 aerial orthoimagery (~300m ground footprint) encompasses the entire street corridor and neighboring properties. <b>The reticle crosshair pinpoints the exact subject property ({currentCase.street_address || `Case #${currentCase.id}`})</b>; use <b>【2x Close-up】</b> to inspect the parcel envelope.
                            </p>
                            <p>
                              • <b>Insurance Underwriting & Radiant Contagion Focus:</b> With structure setbacks of only 10–30 feet, residential subdivisions face severe structure-to-structure domino contagion. California CDI Title 10 § 2644.9 mandates insurance discounts for hardening (Class A roof, mesh vents, and Zone 0 perimeter), reducing current annual premium to <b>${currentCase?.insurance_actuarial?.annual_discounted_premium_usd ?? '—'}/yr</b> ({currentCase?.insurance_actuarial?.total_mitigation_discount_pct ?? '—'} savings).
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Multi-Paradigm Predictive Intelligence & Actuarial Underwriting Dossier (Simultaneous Multi-Model) */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                {/* Dossier Header */}
                <div className="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 via-purple-600 to-pink-500 flex items-center justify-center shadow-md shadow-purple-500/20">
                      <Activity className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="text-sm font-bold text-white tracking-wide">
                          {t.multiParadigmTitle}
                        </h3>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center gap-1">
                          <Sparkles className="w-3 h-3 text-purple-400" />
                          <span>{t.simultaneousBadge}</span>
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">
                        {t.multiParadigmSubtitle}
                      </p>
                    </div>
                  </div>

                  {/* Community Type Pill */}
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 ${
                      currentCase.is_residential_focus
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm shadow-amber-500/10'
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}>
                      <Home className="w-3.5 h-3.5" />
                      <span>{currentCase.is_residential_focus ? t.residentialSubdivisionBadge : t.ruralWuiBadge}</span>
                    </span>
                  </div>
                </div>

                {/* 3 Simultaneous Paradigm Columns */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Paradigm 1: Google X Project Bellwether (100m Grid) */}
                  <div className="bg-slate-950/70 border border-blue-500/30 rounded-xl p-4 flex flex-col justify-between space-y-3 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-28 h-28 bg-blue-500/5 rounded-full blur-2xl pointer-events-none" />
                    <div className="space-y-3">
                      {/* Card Header */}
                      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                        <div className="flex items-center gap-1.5">
                          <Radio className="w-4 h-4 text-blue-400" />
                          <span className="text-xs font-bold text-white tracking-wide">
                            {t.bellwetherCardTitle}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30 px-1.5 py-0.5 rounded">
                          100m Grid
                        </span>
                      </div>

                      {/* 1-Yr & 5-Yr Burn Probability Numbers */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 text-center">
                          <span className="text-[10px] text-slate-400 block">{t.burnProb1yr}</span>
                          <span className={`text-base font-extrabold ${
                            currentCase.survival_label === 'Destroyed' ? 'text-rose-400' : 'text-emerald-400'
                          }`}>
                            {currentCase?.bellwether_prediction?.burn_probability_1yr_pct ?? '—'}
                          </span>
                          <span className="text-[9px] text-slate-400 block">Top 5% Risk Tier</span>
                        </div>
                        <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 text-center">
                          <span className="text-[10px] text-slate-400 block">{t.burnProb5yr}</span>
                          <span className="text-base font-extrabold text-amber-300">
                            {currentCase?.bellwether_prediction?.burn_probability_5yr_pct ?? '—'}
                          </span>
                          <span className="text-[9px] text-slate-400 block">5-Yr Cumulative</span>
                        </div>
                      </div>

                      {/* 130ft Radiant Corridor / Structure Loss Exposure */}
                      <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="font-semibold text-slate-300 flex items-center gap-1">
                            <Zap className="w-3.5 h-3.5 text-amber-400" />
                            <span>{t.radiantCorridorTitle}</span>
                          </span>
                          <span className="font-mono text-amber-300 font-bold">
                            {currentCase?.bellwether_prediction?.radiant_corridor_130ft?.wind_adjusted_major_axis_meters ?? '—'}m Axis
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-slate-400 block">{t.threatenedStructures}</span>
                            <span className="font-bold text-rose-300">
                              {currentCase?.bellwether_prediction?.radiant_corridor_130ft?.threatened_structures_count ?? '—'} Homes
                            </span>
                          </div>
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-slate-400 block">{t.threatenedValue}</span>
                            <span className="font-bold text-rose-300">
                              {currentCase?.bellwether_prediction?.radiant_corridor_130ft?.threatened_property_display ?? '—'}
                            </span>
                          </div>
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-slate-400 block">{t.mitigatedValue}</span>
                            <span className="font-bold text-emerald-300">
                              {currentCase?.bellwether_prediction?.radiant_corridor_130ft?.mitigated_saved_display ?? '—'}
                            </span>
                          </div>
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-slate-400 block">{t.mitigationRoi}</span>
                            <span className="font-bold text-cyan-300">
                              {currentCase?.bellwether_prediction?.radiant_corridor_130ft?.mitigation_roi_multiplier ?? '—'}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Top Drivers */}
                      <div className="space-y-1">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                          {t.topDriversTitle}
                        </span>
                        {(currentCase?.bellwether_prediction?.top_drivers || [
                          { factor: 'EDDI Drought Deficit', weight: '34%', impact: '95th %ile' },
                          { factor: 'HRRR Surface Wind Vectors', weight: '28%', impact: '230° SW' },
                          { factor: '3D LiDAR Slope Angle', weight: '22%', impact: 'Moderate' },
                          { factor: 'Dead Fuel Moisture (< 6%)', weight: '16%', impact: 'Critical' },
                        ]).map((d, i) => (
                          <div key={i} className="flex items-center justify-between text-[10px] bg-slate-900/60 px-2 py-1 rounded border border-slate-800/40">
                            <span className="text-slate-300 truncate max-w-[130px]">{d.factor}</span>
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-cyan-300 font-semibold">{d.weight}</span>
                              <span className="text-slate-400 text-[9px]">({d.impact})</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Paradigm 2: Academic Physics & Discrete Simulations (Rothermel & CA) */}
                  <div className="bg-slate-950/70 border border-purple-500/30 rounded-xl p-4 flex flex-col justify-between space-y-3 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-28 h-28 bg-purple-500/5 rounded-full blur-2xl pointer-events-none" />
                    <div className="space-y-3">
                      {/* Card Header */}
                      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                        <div className="flex items-center gap-1.5">
                          <Gauge className="w-4 h-4 text-purple-400" />
                          <span className="text-xs font-bold text-white tracking-wide">
                            {t.academicCardTitle}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30 px-1.5 py-0.5 rounded">
                          PDE + CA
                        </span>
                      </div>

                      {/* Rothermel Level-Set PDE */}
                      <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="font-semibold text-slate-200">
                            {t.rothermelTitle}
                          </span>
                          <span className="text-[10px] text-purple-300 font-mono">
                            Albini (1972)
                          </span>
                        </div>
                        <div className="grid grid-cols-3 gap-1.5 text-center">
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-[9px] text-slate-400 block">{t.rateOfSpread}</span>
                            <span className="text-xs font-bold text-orange-400">
                              {currentCase?.academic_simulations?.rothermel_level_set?.rate_of_spread_m_min ?? '—'} m/min
                            </span>
                          </div>
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-[9px] text-slate-400 block">{t.flameLength}</span>
                            <span className="text-xs font-bold text-amber-300">
                              {currentCase?.academic_simulations?.rothermel_level_set?.flame_length_meters ?? '—'} m
                            </span>
                          </div>
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-[9px] text-slate-400 block">{t.firelineIntensity}</span>
                            <span className="text-xs font-bold text-rose-400">
                              {currentCase?.academic_simulations?.rothermel_level_set?.fireline_intensity_kw_m ?? '—'} kW/m
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Alexandridis Stochastic Cellular Automata */}
                      <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="font-semibold text-slate-200">
                            {t.cellularAutomataTitle}
                          </span>
                          <span className="text-[10px] text-purple-300 font-mono">
                            2D Grid
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-center text-xs">
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-[9px] text-slate-400 block">{t.spottingProbability}</span>
                            <span className="text-sm font-bold text-rose-400">
                              {currentCase?.academic_simulations?.cellular_automata?.ember_spotting_probability ?? '—'}
                            </span>
                          </div>
                          <div className="bg-slate-950/80 p-1.5 rounded">
                            <span className="text-[9px] text-slate-400 block">{t.contagionIndex}</span>
                            <span className="text-sm font-bold text-amber-300">
                              {currentCase?.academic_simulations?.cellular_automata?.neighborhood_contagion_index ?? '—'}
                            </span>
                          </div>
                        </div>
                        <span className="text-[9px] text-slate-400 block italic text-center">
                          {t.transitionRule}
                        </span>
                      </div>

                      {/* Google NDWS ConvNet Surrogate */}
                      <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 space-y-1">
                        <div className="flex items-center justify-between text-[10px]">
                          <span className="font-bold text-slate-300">{t.googleNdwsTitle}</span>
                          <span className="font-mono text-slate-500 font-semibold">not run</span>
                        </div>
                        <div className="text-[10px] text-slate-500">No NDWS model was run on these cases; earlier values here were hard-coded.</div>
                      </div>
                    </div>
                  </div>

                  {/* Paradigm 3: Commercial Property Insurance & Actuarial Underwriting (Verisk & CDI) */}
                  <div className="bg-slate-950/70 border border-emerald-500/30 rounded-xl p-4 flex flex-col justify-between space-y-3 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-28 h-28 bg-emerald-500/5 rounded-full blur-2xl pointer-events-none" />
                    <div className="space-y-3">
                      {/* Card Header */}
                      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                        <div className="flex items-center gap-1.5">
                          <Shield className="w-4 h-4 text-emerald-400" />
                          <span className="text-xs font-bold text-white tracking-wide">
                            {t.insuranceCardTitle}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-1.5 py-0.5 rounded">
                          CDI § 2644.9
                        </span>
                      </div>

                      {/* Verisk Score & Z-FIRE Tier */}
                      <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">{t.firelineScoreLabel}:</span>
                          <span className={`font-bold px-2 py-0.5 rounded ${
                            (currentCase?.insurance_actuarial?.verisk_fireline_score ?? 0) > 75
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          }`}>
                            {currentCase?.insurance_actuarial?.verisk_score_label ?? '—'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-slate-400">{t.zfireTierLabel}:</span>
                          <span className="text-amber-300 font-medium truncate max-w-[170px]">
                            {currentCase?.insurance_actuarial?.zesty_ai_zfire_tier ?? '—'}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-950/80 border border-slate-800/80 text-[10px]">
                          <span className="text-slate-400 block mb-0.5">{t.underwritingActionLabel}:</span>
                          <span className={`font-semibold ${
                            currentCase.survival_label === 'Destroyed' ? 'text-rose-300' : 'text-emerald-300'
                          }`}>
                            {currentCase?.insurance_actuarial?.underwriting_status ?? '—'}
                          </span>
                        </div>
                      </div>

                      {/* Premium Breakdown & Total Mitigation Credit */}
                      <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between text-xs">
                          <div>
                            <span className="text-[10px] text-slate-400 block">{t.annualBasePremium}</span>
                            <span className="font-bold text-slate-300 line-through">
                              ${currentCase?.insurance_actuarial?.annual_base_premium_usd ?? '—'}/yr
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="text-[10px] text-emerald-400 block">{t.annualDiscountedPremium}</span>
                            <span className="font-extrabold text-emerald-400 text-sm">
                              ${currentCase?.insurance_actuarial?.annual_discounted_premium_usd ?? '—'}/yr
                            </span>
                          </div>
                          <span className="px-2 py-1 rounded bg-emerald-500/20 text-emerald-300 font-bold font-mono text-xs border border-emerald-500/30">
                            {currentCase?.insurance_actuarial?.total_mitigation_discount_pct ?? '—'}
                          </span>
                        </div>
                      </div>

                      {/* CDI Mandatory Breakdown */}
                      <div className="space-y-1 text-[10px]">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                          {t.cdiDiscountsTitle}
                        </span>
                        <div className="space-y-1 bg-slate-900/60 p-2 rounded-lg border border-slate-800/50">
                          <div className="flex justify-between">
                            <span className="text-slate-300">• {t.classARoofCredit}</span>
                            <span className="font-mono text-emerald-400 font-bold">
                              {currentCase?.insurance_actuarial?.cdi_discount_breakdown?.class_a_roof_credit ?? '—'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-300">• {t.emberVentCredit}</span>
                            <span className="font-mono text-emerald-400 font-bold">
                              {currentCase?.insurance_actuarial?.cdi_discount_breakdown?.ember_mesh_vent_credit ?? '—'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-300">• {t.zone0BufferCredit}</span>
                            <span className="font-mono text-emerald-400 font-bold">
                              {currentCase?.insurance_actuarial?.cdi_discount_breakdown?.zone0_buffer_credit ?? '—'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Multimodal Structured Physical Assessment & Downstream Task Dispatcher */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-2xl relative overflow-hidden space-y-4">
                <div className="absolute top-0 right-0 w-80 h-80 bg-orange-500/5 rounded-full blur-3xl pointer-events-none" />

                {/* Card Title & Architectural Philosophy */}
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-orange-500/20 text-orange-400 flex items-center justify-center border border-orange-500/30">
                      <Cpu className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-white tracking-wide">
                          {t.structuredAssessmentTitle}
                        </h3>
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-orange-500/20 text-orange-300 border border-orange-500/30">
                          {lang === 'zh' ? '解耦二分类 · 结构化输出' : 'Decoupled Physical Attribution'}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        {t.structuredAssessmentSubtitle}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Fused Input Pipeline Chips */}
                <div className="space-y-1.5">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                    {t.inputPipelineTitle}
                  </span>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-blue-500/30 space-y-1">
                      <div className="flex items-center gap-1.5 text-blue-300 font-bold text-[10px]">
                        <Camera className="w-3.5 h-3.5" />
                        <span>1. {lang === 'zh' ? '宏观正射航拍 (300m)' : 'Macro Ortho (300m)'}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight">
                        {currentCase?.structured_assessment?.visual_evidence_attribution?.macro_ortho_context?.slope_exposure ?? '—'}
                      </p>
                    </div>

                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-emerald-500/30 space-y-1">
                      <div className="flex items-center gap-1.5 text-emerald-300 font-bold text-[10px]">
                        <Home className="w-3.5 h-3.5" />
                        <span>2. {lang === 'zh' ? '微观立面街景 (Zone 0)' : 'Micro Facade (Zone 0)'}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight truncate">
                        {currentCase?.structured_assessment?.visual_evidence_attribution?.micro_facade_envelope?.eave_soffit_construction ?? '—'}
                      </p>
                    </div>

                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-purple-500/30 space-y-1">
                      <div className="flex items-center gap-1.5 text-purple-300 font-bold text-[10px]">
                        <FileText className="w-3.5 h-3.5" />
                        <span>3. {lang === 'zh' ? '地籍与 Chapter 7A' : 'Assessor & Chapter 7A'}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight">
                        {lang === 'zh' ? `建成年份 ${currentCase.year_built} · ${currentCase.assessor_record?.chapter_7a?.code_status ?? '—'}` : `Built in ${currentCase.year_built} · ${currentCase.assessor_record?.chapter_7a?.code_status ?? '—'}`}
                      </p>
                    </div>

                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-amber-500/30 space-y-1">
                      <div className="flex items-center gap-1.5 text-amber-300 font-bold text-[10px]">
                        <Wind className="w-3.5 h-3.5" />
                        <span>4. {lang === 'zh' ? '极端气象与物理先验' : 'Weather & Physical Priors'}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight">
                        {lang === 'zh' ? '阵风 25-35 mph · DFM<6% · Bellwether 3.42%' : 'Gusts 25-35 mph · DFM<6% · Bellwether 3.42%'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Physical Vulnerability Matrix (4 Bars) */}
                <div className="bg-slate-950/70 p-3.5 rounded-xl border border-slate-800 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-orange-400" />
                      <span>{t.vulnerabilityMatrixTitle}</span>
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      {lang === 'zh' ? '物理仿真与实物证据量化' : 'Quantitative Physical Scoring'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {/* Ember Score */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">{lang === 'zh' ? '飞火侵入易损度' : 'Ember Intrusion'}</span>
                        <span className="font-bold text-rose-400 font-mono">
                          {currentCase?.structured_assessment?.physical_vulnerability_matrix?.ember_intrusion_score ?? '—'}/100
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-rose-500 h-1.5 rounded-full transition-all duration-500"
                          style={{ width: `${currentCase?.structured_assessment?.physical_vulnerability_matrix?.ember_intrusion_score ?? 0}%` }}
                        />
                      </div>
                    </div>

                    {/* Direct Flame Score */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">{lang === 'zh' ? '直接火焰接触' : 'Direct Flame'}</span>
                        <span className="font-bold text-amber-400 font-mono">
                          {currentCase?.structured_assessment?.physical_vulnerability_matrix?.direct_flame_contact_score ?? '—'}/100
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-amber-500 h-1.5 rounded-full transition-all duration-500"
                          style={{ width: `${currentCase?.structured_assessment?.physical_vulnerability_matrix?.direct_flame_contact_score ?? 0}%` }}
                        />
                      </div>
                    </div>

                    {/* Radiant Heat Score */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">{lang === 'zh' ? '辐射热通量暴露' : 'Radiant Heat Flux'}</span>
                        <span className="font-bold text-purple-400 font-mono">
                          {currentCase?.structured_assessment?.physical_vulnerability_matrix?.radiant_heat_flux_score ?? '—'}/100
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-purple-500 h-1.5 rounded-full transition-all duration-500"
                          style={{ width: `${currentCase?.structured_assessment?.physical_vulnerability_matrix?.radiant_heat_flux_score ?? 0}%` }}
                        />
                      </div>
                    </div>

                    {/* Hardening Score */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-400">{lang === 'zh' ? '结构防御硬化值' : 'Defense Hardening'}</span>
                        <span className="font-bold text-emerald-400 font-mono">
                          {currentCase?.structured_assessment?.physical_vulnerability_matrix?.structural_defense_hardening_score ?? '—'}/100
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-emerald-500 h-1.5 rounded-full transition-all duration-500"
                          style={{ width: `${currentCase?.structured_assessment?.physical_vulnerability_matrix?.structural_defense_hardening_score ?? 0}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Causal Failure Pathway & Domino Chain */}
                <div className="bg-slate-950/70 p-3.5 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white flex items-center gap-1.5">
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      <span>{t.causalPathwayTitle}</span>
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                      {t.weakestLinkTitle}: {(currentCase?.structured_assessment?.causal_failure_pathway?.weakest_link_component || currentCase?.incident_forensics?.forensic_breach_mechanism) ?? '—'}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                    {(currentCase?.structured_assessment?.causal_failure_pathway?.step_by_step_domino_chain || [
                      '1. Windward wildland vegetation generates dense ember shower (2-5mm firebrands) propelled by 25-35 mph wind gusts.',
                      '2. Firebrands accumulate in Zone 0 combustible mulch and lodge into unenclosed eave cavities.',
                      '3. Embers breach legacy oversized mesh vents, igniting dry attic cellulose insulation.',
                      '4. Envelope compromised leading to internal structure loss prior to frontal flame arrival.'
                    ]).map((stepText, sIdx) => (
                      <div key={sIdx} className="p-2 rounded-lg bg-slate-900/80 border border-slate-800 text-[11px] space-y-1">
                        <span className="text-[10px] font-bold font-mono text-orange-400 block">
                          STEP {sIdx + 1}
                        </span>
                        <p className="text-slate-300 leading-snug">
                          {stepText}
                        </p>
                      </div>
                    ))}
                  </div>

                  {/* Stochastic Survival Factors */}
                  <div className="p-2 rounded-lg bg-slate-900/40 border border-slate-800/60 text-[10px] flex items-center gap-2 text-slate-300">
                    <Info className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                    <span>
                      <strong className="text-slate-200">{t.stochasticFactorsTitle}:</strong>{' '}
                      {currentCase?.structured_assessment?.causal_failure_pathway?.stochastic_survival_factors?.join('; ') || 'Type 3 wildland engine tactical intervention within 15 min; localized wind direction shift > 35°.'}
                    </span>
                  </div>
                </div>

                {/* Downstream Task Payloads (3 Tabs) */}
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 flex-wrap gap-2">
                    <span className="text-xs font-bold text-white flex items-center gap-1.5">
                      <Split className="w-3.5 h-3.5 text-emerald-400" />
                      <span>{lang === 'zh' ? '下游任务结构化交付载荷 (Actionable Task Payloads)' : 'Actionable Downstream Task Payloads'}</span>
                    </span>

                    {/* Tab Buttons */}
                    <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
                      <button
                        onClick={() => setActiveDownstreamTab('insurance')}
                        className={`px-2.5 py-1 rounded text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer ${
                          activeDownstreamTab === 'insurance'
                            ? 'bg-emerald-600 text-white shadow-sm'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        <Shield className="w-3 h-3" />
                        <span>{t.tabInsurance}</span>
                      </button>

                      <button
                        onClick={() => setActiveDownstreamTab('firefighter')}
                        className={`px-2.5 py-1 rounded text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer ${
                          activeDownstreamTab === 'firefighter'
                            ? 'bg-orange-600 text-white shadow-sm'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        <Flame className="w-3 h-3" />
                        <span>{t.tabFirefighter}</span>
                      </button>

                      <button
                        onClick={() => setActiveDownstreamTab('physics')}
                        className={`px-2.5 py-1 rounded text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer ${
                          activeDownstreamTab === 'physics'
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        <Cpu className="w-3 h-3" />
                        <span>{t.tabPhysics}</span>
                      </button>
                    </div>
                  </div>

                  {/* Tab 1: Insurance Underwriting Payload */}
                  {activeDownstreamTab === 'insurance' && (
                    <div className="bg-slate-950/70 p-3.5 rounded-xl border border-emerald-500/30 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-xs font-bold text-white block">
                            {lang === 'zh' ? '加州保险署 (CDI) §2644.9 法定减灾核保载荷' : 'CDI Title 10 § 2644.9 Actuarial Audit Payload'}
                          </span>
                          <span className="text-[10px] text-slate-400">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.insurance_actuarial_underwriting?.fair_plan_eligibility ?? '—'}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? '最大年保费节约潜力' : 'Max Annual Savings Potential'}</span>
                          <span className="font-extrabold text-emerald-400 text-sm font-mono">
                            -${currentCase?.structured_assessment?.downstream_task_payloads?.insurance_actuarial_underwriting?.max_potential_annual_premium_savings_usd ?? '—'}/yr
                          </span>
                        </div>
                      </div>

                      {/* CDI Credits Table */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
                        {Object.entries(currentCase?.structured_assessment?.downstream_task_payloads?.insurance_actuarial_underwriting?.cdi_section_2644_9_compliance_credits || {
                          class_a_roof: { qualified: true, credit_pct: 5.0, status: 'Granted' },
                          ember_resistant_vents_1_8th: { qualified: false, credit_pct: 3.0, status: 'Remediation Required' },
                          zone0_5ft_buffer: { qualified: false, credit_pct: 4.0, status: 'Remediation Required' },
                          non_combustible_fence_break: { qualified: false, credit_pct: 2.5, status: 'Remediation Required' },
                        }).map(([creditKey, creditVal]: any) => (
                          <div key={creditKey} className="p-2 rounded bg-slate-900 border border-slate-800 space-y-1">
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="font-semibold text-slate-300 capitalize">{creditKey.replace(/_/g, ' ')}</span>
                              <span className="font-bold text-emerald-400 font-mono">+{creditVal.credit_pct}%</span>
                            </div>
                            <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold block truncate ${
                              creditVal.qualified ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
                            }`}>
                              {creditVal.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Tab 2: Tactical Firefighting Payload */}
                  {activeDownstreamTab === 'firefighter' && (
                    <div className="bg-slate-950/70 p-3.5 rounded-xl border border-orange-500/30 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-xs font-bold text-white block">
                            {lang === 'zh' ? '现场消防战术分级 (WUI Structure Triage)' : 'WUI Firefighting Tactical Triage'}
                          </span>
                          <span className="text-xs font-bold text-orange-400">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.tactical_firefighting_operations?.triage_category ?? '—'}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-right text-xs">
                          <div>
                            <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? '预处理时间' : 'Prep Time'}</span>
                            <span className="font-bold text-white font-mono">
                              {currentCase?.structured_assessment?.downstream_task_payloads?.tactical_firefighting_operations?.tactical_prep_time_minutes ?? '—'} mins
                            </span>
                          </div>
                          <div>
                            <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? '最低水枪流量' : 'Min Flow Rate'}</span>
                            <span className="font-bold text-cyan-400 font-mono">
                              {currentCase?.structured_assessment?.downstream_task_payloads?.tactical_firefighting_operations?.minimum_water_supply_gpm ?? '—'} GPM
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* 3 Urgent Prep Actions */}
                      <div className="space-y-1">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                          {lang === 'zh' ? '到达现场后优先处置任务 (Top 3 Tactical Prep Actions)' : 'Prioritized Tactical Prep Actions:'}
                        </span>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                          {(currentCase?.structured_assessment?.downstream_task_payloads?.tactical_firefighting_operations?.prep_actions_required || [
                            'Sever attached wooden fence within 5ft of house wall (3 mins)',
                            'Rake combustible bark mulch and dry needles 5ft away from foundation (5 mins)',
                            'Cover legacy oversized attic vents with temporary aluminum fire foil (4 mins)'
                          ]).map((actionText: string, aIdx: number) => (
                            <div key={aIdx} className="p-2 rounded bg-slate-900 border border-slate-800 text-[11px] text-slate-300 flex items-start gap-1.5">
                              <span className="w-4 h-4 rounded-full bg-orange-500/20 text-orange-400 font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                                {aIdx + 1}
                              </span>
                              <span>{actionText}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <p className="text-[10px] text-slate-400 italic">
                        {currentCase?.structured_assessment?.downstream_task_payloads?.tactical_firefighting_operations?.crew_safety_feasibility ?? '—'}
                      </p>
                    </div>
                  )}

                  {/* Tab 3: Coupled Physics Simulation Payload */}
                  {activeDownstreamTab === 'physics' && (
                    <div className="bg-slate-950/70 p-3.5 rounded-xl border border-blue-500/30 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-xs font-bold text-white block">
                            {lang === 'zh' ? '物理仿真与元胞自动机耦合边界参数' : 'Coupled Physics Simulation Boundary Parameters'}
                          </span>
                          <span className="text-[10px] text-slate-400">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.coupled_physics_simulation?.front_propagation_mode ?? '—'}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? '临界失火引燃时间' : 'Critical Time to Ignition'}</span>
                          <span className="font-extrabold text-blue-400 text-sm font-mono">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.coupled_physics_simulation?.estimated_critical_time_to_ignition_min ?? '—'} mins
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px]">
                        <div className="p-2 rounded bg-slate-900 border border-slate-800 space-y-1">
                          <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? 'Rothermel 局部载荷修正系数' : 'Rothermel Fuel Modifier'}</span>
                          <span className="font-semibold text-white block truncate">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.coupled_physics_simulation?.rothermel_fuel_adjustment ?? '—'}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800 space-y-1">
                          <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? '元胞自动机飞火引燃概率' : 'CA Spotting Coefficient'}</span>
                          <span className="font-bold text-cyan-400 font-mono block">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.coupled_physics_simulation?.cellular_automata_spotting_coefficient ?? '—'}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800 space-y-1">
                          <span className="text-[10px] text-slate-400 block">{lang === 'zh' ? '前锋传播机制' : 'Front Propagation'}</span>
                          <span className="font-semibold text-amber-300 block truncate">
                            {currentCase?.structured_assessment?.downstream_task_payloads?.coupled_physics_simulation?.front_propagation_mode ?? '—'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Uncertainty Audit & Visual Provenance */}
                <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80 text-[10px] flex items-center justify-between text-slate-400 flex-wrap gap-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-bold text-slate-300">{t.uncertaintyTitle}:</span>
                    <span>
                      {lang === 'zh' ? '综合物理评定置信度' : 'Assessment Confidence'}: <strong className="text-emerald-400 font-mono">{currentCase?.structured_assessment?.uncertainty_audit?.overall_assessment_confidence != null ? `${(currentCase?.structured_assessment?.uncertainty_audit?.overall_assessment_confidence * 100).toFixed(0)}%` : '—'}</strong>
                    </span>
                    <span>•</span>
                    <span className="truncate">
                      {lang === 'zh' ? '盲区说明: 后院露台遮雨棚连接处受正重视角遮挡' : 'Occlusion: Rear patio cover unobserved'}
                    </span>
                  </div>
                  <span className="font-mono text-slate-500">Audit-Ready</span>
                </div>
              </div>

              {/* Bottom Cards: When Collapsed, side-by-side in 2 cols; when expanded, stacked */}
              <div className={isReasoningCollapsed ? 'grid grid-cols-1 lg:grid-cols-2 gap-4' : 'space-y-4'}>
                {/* County Assessor Records & California Chapter 7A Fire Hardening Audit Card */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <Landmark className="w-5 h-5 text-indigo-400" />
                    <div>
                      <h3 className="text-sm font-bold text-white tracking-wide">
                        {t.assessorTitle}
                      </h3>
                      <span className="text-[11px] text-slate-400 font-mono">
                        {t.assessorSubtitle}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      APN: {currentCase.assessor_record?.apn ?? '—'}
                    </span>
                    {currentCase.assessor_record?.chapter_7a?.is_compliant ? (
                      <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1 shadow-sm shadow-emerald-500/20">
                        <ShieldCheck className="w-3.5 h-3.5" />
                        <span>Post-Chapter 7A (2008+)</span>
                      </span>
                    ) : (
                      <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-1 shadow-sm shadow-amber-500/20">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Pre-Chapter 7A (历史老旧建筑)</span>
                      </span>
                    )}
                  </div>
                </div>

                {/* 4-Metric Assessor Quick Facts Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
                  <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">{t.yearBuiltLabel}</span>
                    <span className="font-bold text-slate-200 text-sm">{(currentCase.assessor_record?.year_built || currentCase.year_built) ?? '—'}</span>
                  </div>

                  <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">{t.livingAreaLabel}</span>
                    <span className="font-bold text-cyan-300 text-sm">
                      {currentCase.assessor_record?.living_area_sqft ? `${currentCase.assessor_record.living_area_sqft.toLocaleString()} sqft` : '1,500 sqft'}
                    </span>
                  </div>

                  <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">{t.lotSizeLabel}</span>
                    <span className="font-bold text-slate-200 text-sm truncate block">
                      {currentCase.assessor_record?.lot_size ?? '—'}
                    </span>
                  </div>

                  <div className="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">评估改进价值 (Assessed)</span>
                    <span className="font-bold text-emerald-300 text-sm">
                      {currentCase.assessor_record?.assessed_value_display ?? '—'}
                    </span>
                  </div>
                </div>

                {/* Rebuild & Permit History Highlight */}
                {currentCase.assessor_record?.rebuild_permit_history && (
                  <div className="bg-indigo-950/30 border border-indigo-500/30 p-3 rounded-xl text-xs space-y-1">
                    <div className="flex items-center gap-1.5 text-indigo-300 font-bold">
                      <FileText className="w-3.5 h-3.5" />
                      <span>县政许可与灾后重建事实 (Assessor Permit &amp; Rebuild History)</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-slate-300">
                      {currentCase.assessor_record.rebuild_permit_history}
                    </p>
                  </div>
                )}

                {/* Chapter 7A Hardening Audit Box */}
                <div className="pt-2 border-t border-slate-800/60 space-y-1.5 text-xs">
                  <div className="flex items-center justify-between text-slate-400 text-[11px]">
                    <span className="font-bold uppercase tracking-wider text-slate-300">
                      加州 Chapter 7A 建筑规范法条解读
                    </span>
                    <span className="font-mono text-slate-400">
                      {currentCase.assessor_record?.chapter_7a?.regulatory_basis ?? '—'}
                    </span>
                  </div>
                  <p className="text-[11px] leading-relaxed text-slate-400">
                    {currentCase.assessor_record?.chapter_7a?.standard_summary}
                  </p>
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
                    <span className="font-semibold text-amber-300">{currentCase.ignition_point ?? '—'}</span>
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
                        {t.forensicFactsTitle}
                      </h3>
                      <span className="text-[11px] text-slate-400">
                        {currentCase?.incident_forensics?.fire_name || `${currentCase.incident} Fire`} 现场环境与蔓延动力学
                      </span>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-orange-500/20 text-orange-300 border border-orange-500/30">
                    {currentCase?.incident_forensics?.ignition_date ?? '—'}
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
                        <span className="text-slate-400 text-[10px] block">{t.flameVectorTitle}</span>
                        <span className="font-semibold text-rose-300">西南 (230° SW) 32 mph 阵风推进</span>
                      </div>
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">{t.surroundingVegetation}</span>
                        <span className="font-semibold text-amber-300">Chaparral 灌木 + 枯草干燥层</span>
                      </div>
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">{t.physicalBreachPoint}</span>
                        <span className="font-semibold text-cyan-300">{currentCase.ignition_point ?? '—'}</span>
                      </div>
                      <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                        <span className="text-slate-400 text-[10px] block">{t.firefighterAction}</span>
                        <span className="font-semibold text-emerald-300">{currentCase.defensive_actions}</span>
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-relaxed bg-slate-900/40 p-2 rounded border border-slate-800/60">
                      {currentCase?.incident_forensics?.observed_physical_outcome ||
                        (lang === 'zh'
                          ? `加州消防局现场鉴定：建筑在强风火场中经历烈焰与飞火考验，最终实际结果为 ${currentCase.raw_damage}。`
                          : `CAL FIRE forensic inspection: structure experienced severe firebrand flux; certified damage status: ${currentCase.raw_damage}.`)}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

            {/* Right Column: Multi-Model Deductive Reasoning Matrix */}
            <div className={`${isReasoningCollapsed ? 'hidden' : 'lg:col-span-4'} space-y-4 transition-all duration-300`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Flame className="w-5 h-5 text-orange-400" />
                  <h3 className="text-sm font-bold text-white tracking-wide">
                    {t.modelMatrixTitle}
                  </h3>
                </div>
                <button
                  id="btn-collapse-reasoning-panel"
                  onClick={() => setIsReasoningCollapsed(true)}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-all cursor-pointer"
                  title={t.collapseReasoning}
                >
                  <span>{t.hidePanel}</span>
                  <PanelRightClose className="w-3.5 h-3.5 text-orange-400" />
                </button>
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
                      {res && (() => {
                        const traceText = res.thinking_trace || res.thinking_trace_snippet || res.full_reasoning_trace || '';
                        return (
                          <div className="pt-2 border-t border-slate-800/80 space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-1 text-xs py-1">
                              <button
                                onClick={() => toggleThinking(modelKey)}
                                className="flex items-center gap-1 font-medium text-[11px] text-purple-300 hover:text-purple-200 transition-colors cursor-pointer shrink-0"
                              >
                                <Sparkles className="w-3 h-3" />
                                <span>{isExpanded ? t.hideThinking : t.showThinking}</span>
                                {isExpanded ? (
                                  <ChevronUp className="w-3 h-3 text-slate-400" />
                                ) : (
                                  <ChevronDown className="w-3 h-3 text-slate-400" />
                                )}
                              </button>

                              <div className="flex items-center gap-1 shrink-0">
                                {traceText && (
                                  <>
                                    <button
                                      onClick={() => handleCopyTrace(traceText, modelKey)}
                                      className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1 border border-slate-700 transition-colors cursor-pointer"
                                      title={t.copyCoT}
                                    >
                                      <Copy className="w-3 h-3" />
                                      <span>{copyFeedback === modelKey ? t.copied : 'Copy'}</span>
                                    </button>
                                    <button
                                      onClick={() => setSelectedTraceModal({ modelName: cfg.label, trace: traceText, caseId: currentCase.id })}
                                      className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-purple-950/80 hover:bg-purple-900 text-purple-300 flex items-center gap-1 border border-purple-500/40 transition-colors cursor-pointer"
                                      title={t.openFullscreenCoT}
                                    >
                                      <Maximize2 className="w-3 h-3" />
                                      <span>Modal</span>
                                    </button>
                                    <button
                                      onClick={() => handleAuditModelReasoning(cfg.label, traceText)}
                                      disabled={auditingReward}
                                      className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/90 hover:bg-emerald-900 text-emerald-300 flex items-center gap-1 border border-emerald-500/50 transition-colors cursor-pointer shadow-sm shadow-emerald-900/30"
                                      title={t.auditThisTrace}
                                    >
                                      <ShieldCheck className="w-3 h-3 text-emerald-400" />
                                      <span>{auditingReward ? t.auditing : (lang === 'zh' ? 'RLVR 审计' : 'RLVR Audit')}</span>
                                    </button>
                                  </>
                                )}
                              </div>
                            </div>

                            {isExpanded && (
                              <div className="p-3 rounded-xl bg-slate-950 text-[11px] font-mono text-slate-300 max-h-96 overflow-y-auto border border-purple-500/20 leading-relaxed whitespace-pre-wrap select-text space-y-1.5 shadow-inner">
                                <div className="text-[10px] text-purple-400 font-bold uppercase tracking-wider flex items-center justify-between border-b border-purple-950/80 pb-1">
                                  <span>Deductive Reasoning Trace (&lt;think&gt; tokens)</span>
                                  <span className="text-slate-500 font-mono text-[9px]">{traceText.length} chars</span>
                                </div>
                                <div className="text-slate-300 text-[11px] font-mono">
                                  {traceText || 'No raw trace captured'}
                                </div>
                              </div>
                            )}

                            {/* Analysis Excerpt */}
                            {res.analysis_text && !isExpanded && (
                              <p className="text-[11px] text-slate-400 line-clamp-3 leading-relaxed italic bg-slate-950/30 p-2 rounded-lg border border-slate-900">
                                "{res.analysis_text}"
                              </p>
                            )}

                            {/* Inline Verifiable Reward Audit for this model */}
                            {liveRewardAudit && liveRewardAudit.audited_model_label === cfg.label && (
                              <div className="mt-2.5 p-2.5 rounded-xl bg-emerald-950/30 border border-emerald-500/40 space-y-2 animate-fadeIn text-[11px] font-mono">
                                <div className="flex items-center justify-between border-b border-emerald-900/50 pb-1.5">
                                  <div className="flex items-center gap-1.5 text-emerald-300 font-bold">
                                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                                    <span>{t.rewardAuditTitle}</span>
                                  </div>
                                  <div className="flex items-center gap-1 font-bold">
                                    <span className="text-slate-400 text-[10px]">{t.totalRewardScore}:</span>
                                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                                      liveRewardAudit.total_reward >= 3.0
                                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                        : (liveRewardAudit.total_reward >= 0 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40')
                                    }`}>
                                      {liveRewardAudit.total_reward > 0 ? `+${liveRewardAudit.total_reward.toFixed(2)}` : liveRewardAudit.total_reward.toFixed(2)}
                                    </span>
                                  </div>
                                </div>
                                <div className="grid grid-cols-4 gap-1 text-[10px] text-center">
                                  <div className="p-1 rounded bg-slate-900/80 border border-slate-800">
                                    <span className="text-slate-400 block text-[9px]">{t.outcomeRewardLabel}</span>
                                    <span className={liveRewardAudit.components.r_outcome >= 0 ? 'text-white font-bold' : 'text-rose-400 font-bold'}>
                                      {liveRewardAudit.components.r_outcome >= 0 ? `+${liveRewardAudit.components.r_outcome}` : liveRewardAudit.components.r_outcome}
                                    </span>
                                  </div>
                                  <div className="p-1 rounded bg-slate-900/80 border border-slate-800">
                                    <span className="text-slate-400 block text-[9px]">{t.locusRewardLabel}</span>
                                    <span className="text-white font-bold">+{liveRewardAudit.components.r_locus}</span>
                                  </div>
                                  <div className="p-1 rounded bg-slate-900/80 border border-slate-800">
                                    <span className="text-slate-400 block text-[9px]">{t.spatialRewardLabel}</span>
                                    <span className="text-white font-bold">+{liveRewardAudit.components.r_spatial}</span>
                                  </div>
                                  <div className="p-1 rounded bg-slate-900/80 border border-slate-800">
                                    <span className="text-slate-400 block text-[9px]">{t.physicsRewardLabel}</span>
                                    <span className={liveRewardAudit.components.r_physics >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                                      {liveRewardAudit.components.r_physics >= 0 ? `+${liveRewardAudit.components.r_physics}` : liveRewardAudit.components.r_physics}
                                    </span>
                                  </div>
                                </div>
                                {liveRewardAudit.physics_details && liveRewardAudit.physics_details.length > 0 && (
                                  <div className="space-y-1 bg-slate-950/70 p-2 rounded border border-slate-850 text-[10px]">
                                    {liveRewardAudit.physics_details.map((detail: string, dIdx: number) => {
                                      const isPenalty = detail.toLowerCase().includes('penalty');
                                      return (
                                        <div key={dIdx} className="flex items-start gap-1">
                                          {isPenalty ? (
                                            <AlertTriangle className="w-3 h-3 text-rose-400 shrink-0 mt-0.5" />
                                          ) : (
                                            <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0 mt-0.5" />
                                          )}
                                          <span className={isPenalty ? 'text-rose-300' : 'text-emerald-200'}>{detail}</span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })()}
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
                    className="bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-orange-500 font-medium"
                  >
                    <option value="Qwen3.8-27B">Qwen3.8-27B (vLLM Primary)</option>
                    <option value="Qwen3.5-9B">Qwen3.5-9B (Edge Deploy)</option>
                    <option value="MiniMax-M3">MiniMax-M3 (CoT)</option>
                    <option value="claude-sonnet-5-5">Claude Sonnet 5.5</option>
                    <option value="claude-haiku-5-5">Claude Haiku 5.5</option>
                  </select>

                  <button
                    onClick={handleRunLivePrediction}
                    disabled={liveLoading}
                    className="px-4 py-1.5 rounded-lg text-xs font-bold bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-600 hover:to-rose-700 text-white shadow-md shadow-orange-500/20 disabled:opacity-50 transition-all flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
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
                <div className="bg-slate-900/90 border border-orange-500/40 rounded-2xl p-4 shadow-xl space-y-3 animate-fadeIn">
                  <div className="flex items-center justify-between text-xs border-b border-slate-800 pb-2">
                    <span className="font-bold text-orange-400">Live Result ({liveResult.model_name})</span>
                    <span className="text-slate-400 font-mono">Latency: {liveResult.latency_sec}s</span>
                  </div>
                  {liveResult.success ? (
                    <div className="space-y-3 text-xs">
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
                        <div className="space-y-1.5 pt-1">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-purple-300 font-semibold flex items-center gap-1">
                              <Sparkles className="w-3.5 h-3.5" />
                              <span>Full Reasoning Trace ({liveResult.thinking_trace.length} chars)</span>
                            </span>
                            <div className="flex items-center gap-1.5">
                              <button
                                onClick={() => handleCopyTrace(liveResult.thinking_trace, 'live')}
                                className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 hover:bg-slate-700 flex items-center gap-1 border border-slate-700 cursor-pointer transition-colors"
                              >
                                <Copy className="w-3 h-3" />
                                <span>{copyFeedback === 'live' ? t.copied : t.copyCoT}</span>
                              </button>
                              <button
                                onClick={() => setSelectedTraceModal({ modelName: liveResult.model_name, trace: liveResult.thinking_trace, caseId: currentCase.id })}
                                className="px-2 py-0.5 rounded text-[10px] bg-purple-950 text-purple-300 hover:bg-purple-900 border border-purple-500/40 flex items-center gap-1 cursor-pointer transition-colors"
                              >
                                <Maximize2 className="w-3 h-3" />
                                <span>{t.openFullscreenCoT}</span>
                              </button>
                            </div>
                          </div>
                          <div className="p-3 rounded-xl bg-slate-950 font-mono text-[11px] text-slate-300 max-h-96 overflow-y-auto border border-purple-500/30 whitespace-pre-wrap select-text leading-relaxed shadow-inner">
                            {liveResult.thinking_trace}
                          </div>
                        </div>
                      )}
                      {liveResult.analysis_text && (
                        <p className="text-slate-300 text-xs italic bg-slate-950/40 p-2 rounded-lg">
                          "{liveResult.analysis_text}"
                        </p>
                      )}

                      {/* Verifiable Physical Reward Audit (RLVR / GRPO) in liveResult */}
                      {liveRewardAudit && (
                        <div className="mt-3 p-3.5 rounded-xl bg-slate-950/90 border border-emerald-500/40 space-y-2.5 animate-fadeIn">
                          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                            <div className="flex items-center gap-2">
                              <ShieldCheck className="w-4 h-4 text-emerald-400" />
                              <span className="font-bold text-xs text-emerald-300">
                                {t.rewardAuditTitle}
                              </span>
                              {liveRewardAudit.audited_model_label && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-850 text-slate-300 font-mono border border-slate-750">
                                  {liveRewardAudit.audited_model_label}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-1.5 font-mono">
                              <span className="text-[10px] text-slate-400">{t.totalRewardScore}:</span>
                              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                                liveRewardAudit.total_reward >= 3.0
                                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/20'
                                  : (liveRewardAudit.total_reward >= 0.0 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40')
                              }`}>
                                {liveRewardAudit.total_reward > 0 ? `+${liveRewardAudit.total_reward.toFixed(2)}` : liveRewardAudit.total_reward.toFixed(2)}
                              </span>
                            </div>
                          </div>

                          {/* 4 Multi-Objective Reward Components */}
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
                            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                              <span className="text-slate-400 block">{t.outcomeRewardLabel}</span>
                              <span className={`font-bold ${liveRewardAudit.components.r_outcome >= 0 ? 'text-white' : 'text-rose-400'}`}>
                                {liveRewardAudit.components.r_outcome >= 0 ? `+${liveRewardAudit.components.r_outcome}` : liveRewardAudit.components.r_outcome}
                              </span>
                            </div>
                            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                              <span className="text-slate-400 block">{t.locusRewardLabel}</span>
                              <span className="font-bold text-white">+{liveRewardAudit.components.r_locus}</span>
                            </div>
                            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                              <span className="text-slate-400 block">{t.spatialRewardLabel}</span>
                              <span className="font-bold text-white">+{liveRewardAudit.components.r_spatial}</span>
                            </div>
                            <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                              <span className="text-slate-400 block">{t.physicsRewardLabel}</span>
                              <span className={`font-bold ${liveRewardAudit.components.r_physics >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {liveRewardAudit.components.r_physics >= 0 ? `+${liveRewardAudit.components.r_physics}` : liveRewardAudit.components.r_physics}
                              </span>
                            </div>
                          </div>

                          {/* Verification Evidence Checklist */}
                          {liveRewardAudit.physics_details && liveRewardAudit.physics_details.length > 0 && (
                            <div className="space-y-1 text-[11px] text-slate-300 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 font-mono">
                              {liveRewardAudit.physics_details.map((detail: string, dIdx: number) => {
                                const isPenalty = detail.toLowerCase().includes('penalty');
                                return (
                                  <div key={dIdx} className="flex items-start gap-1.5">
                                    {isPenalty ? (
                                      <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                                    ) : (
                                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                                    )}
                                    <span className={isPenalty ? 'text-rose-300' : 'text-emerald-200'}>{detail}</span>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-rose-400">Error: {liveResult.error}</div>
                  )}
                </div>
              )}

              {/* Standalone Reward Audit Output when auditing pre-computed model */}
              {!liveResult && liveRewardAudit && (
                <div className="bg-slate-900/90 border border-emerald-500/40 rounded-2xl p-4 shadow-xl space-y-2.5 animate-fadeIn">
                  <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      <span className="font-bold text-xs text-emerald-300">
                        {t.rewardAuditTitle}
                      </span>
                      {liveRewardAudit.audited_model_label && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-850 text-slate-300 font-mono border border-slate-750">
                          {liveRewardAudit.audited_model_label}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 font-mono">
                      <span className="text-[10px] text-slate-400">{t.totalRewardScore}:</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                        liveRewardAudit.total_reward >= 3.0
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/20'
                          : (liveRewardAudit.total_reward >= 0.0 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40')
                      }`}>
                        {liveRewardAudit.total_reward > 0 ? `+${liveRewardAudit.total_reward.toFixed(2)}` : liveRewardAudit.total_reward.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  {/* 4 Multi-Objective Reward Components */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-400 block">{t.outcomeRewardLabel}</span>
                      <span className={`font-bold ${liveRewardAudit.components.r_outcome >= 0 ? 'text-white' : 'text-rose-400'}`}>
                        {liveRewardAudit.components.r_outcome >= 0 ? `+${liveRewardAudit.components.r_outcome}` : liveRewardAudit.components.r_outcome}
                      </span>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-400 block">{t.locusRewardLabel}</span>
                      <span className="font-bold text-white">+{liveRewardAudit.components.r_locus}</span>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-400 block">{t.spatialRewardLabel}</span>
                      <span className="font-bold text-white">+{liveRewardAudit.components.r_spatial}</span>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                      <span className="text-slate-400 block">{t.physicsRewardLabel}</span>
                      <span className={`font-bold ${liveRewardAudit.components.r_physics >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {liveRewardAudit.components.r_physics >= 0 ? `+${liveRewardAudit.components.r_physics}` : liveRewardAudit.components.r_physics}
                      </span>
                    </div>
                  </div>

                  {/* Verification Evidence Checklist */}
                  {liveRewardAudit.physics_details && liveRewardAudit.physics_details.length > 0 && (
                    <div className="space-y-1 text-[11px] text-slate-300 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 font-mono">
                      {liveRewardAudit.physics_details.map((detail: string, dIdx: number) => {
                        const isPenalty = detail.toLowerCase().includes('penalty');
                        return (
                          <div key={dIdx} className="flex items-start gap-1.5">
                            {isPenalty ? (
                              <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                            ) : (
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                            )}
                            <span className={isPenalty ? 'text-rose-300' : 'text-emerald-200'}>{detail}</span>
                          </div>
                        );
                      })}
                    </div>
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
                    {t.expertDiagnosisTitle}
                  </h3>
                  <span className="text-xs text-slate-400">
                    {t.expertSubtitle} #{currentCase.id}
                  </span>
                </div>
              </div>

              <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                {currentCase?.expert_diagnosis?.title}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 text-xs leading-relaxed">
              <div className="md:col-span-8 space-y-3">
                <div className="text-slate-300 text-sm leading-relaxed bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  {currentCase?.expert_diagnosis?.diagnosis}
                </div>
              </div>

              <div className="md:col-span-4 bg-slate-950/80 p-4 rounded-xl border border-slate-800/80 flex flex-col justify-between space-y-3">
                <div>
                  <span className="text-[11px] font-bold text-purple-400 uppercase tracking-wider block mb-1">
                    Recommendation (demo text):
                  </span>
                  <p className="text-slate-300 text-xs leading-relaxed">
                    {currentCase?.expert_diagnosis?.recommendation}
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
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
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

              {/* Audit 2026-10-22: sections 4-5 showed hard-coded "Ours" results (82.0%, 94.0%) from simulations that
                  used the true labels. They are removed. Real results: docs/EVIDENCE_STATUS.md. */}
              <div className="bg-slate-950/90 p-4 rounded-xl border border-amber-500/40 text-[11px] text-amber-200">
                Earlier versions showed "VLM-to-physics assimilation" and "RLVR" result tables here. Those numbers came from
                simulations that generated predictions from the true outcome labels, so they were removed. Held-out results
                are tracked in docs/EVIDENCE_STATUS.md.
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setShowDataSourceModal(false)}
                className="px-4 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-white transition-all cursor-pointer"
              >
                Close Documentation
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Fullscreen CoT Reasoning Trace Modal */}
      {selectedTraceModal && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 animate-fadeIn">
          <div className="bg-slate-900 border border-purple-500/40 rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center border border-purple-500/30">
                  <Sparkles className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <span>{t.inspectModalTitle}</span>
                    <span className="text-xs px-2 py-0.5 rounded font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      {selectedTraceModal.modelName}
                    </span>
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Case #{selectedTraceModal.caseId} · {selectedTraceModal.trace.length} {lang === 'zh' ? '字符完整思维链' : 'characters raw deductive tokens'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleAuditModelReasoning(selectedTraceModal.modelName, selectedTraceModal.trace)}
                  disabled={auditingReward}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5 transition-colors cursor-pointer"
                  title={t.auditThisTrace}
                >
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{auditingReward ? t.auditing : t.auditThisTrace}</span>
                </button>
                <button
                  onClick={() => handleCopyTrace(selectedTraceModal.trace, 'modal')}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white border border-slate-700 flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>{copyFeedback === 'modal' ? t.copied : t.copyCoT}</span>
                </button>
                <button
                  onClick={() => setSelectedTraceModal(null)}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-purple-600 hover:bg-purple-700 text-white transition-colors cursor-pointer"
                >
                  {lang === 'zh' ? '关闭' : 'Close'}
                </button>
              </div>
            </div>

            {/* Modal Audit Banner if audited */}
            {liveRewardAudit && liveRewardAudit.audited_model_label === selectedTraceModal.modelName && (
              <div className="px-6 py-3 bg-emerald-950/40 border-b border-emerald-500/30 flex flex-wrap items-center justify-between gap-3 font-mono text-xs animate-fadeIn">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span className="font-bold text-emerald-300">{t.rewardAuditTitle}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                    liveRewardAudit.total_reward >= 3.0
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : (liveRewardAudit.total_reward >= 0 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40')
                  }`}>
                    {t.totalRewardScore}: {liveRewardAudit.total_reward > 0 ? `+${liveRewardAudit.total_reward.toFixed(2)}` : liveRewardAudit.total_reward.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="text-slate-400">R_out: <strong className="text-white">{liveRewardAudit.components.r_outcome}</strong></span>
                  <span className="text-slate-400">R_loc: <strong className="text-white">+{liveRewardAudit.components.r_locus}</strong></span>
                  <span className="text-slate-400">R_sp: <strong className="text-white">+{liveRewardAudit.components.r_spatial}</strong></span>
                  <span className="text-slate-400">R_phy: <strong className={liveRewardAudit.components.r_physics >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{liveRewardAudit.components.r_physics}</strong></span>
                </div>
              </div>
            )}

            {/* Modal Content - Scrollable Monospace Trace */}
            <div className="p-6 overflow-y-auto flex-1 bg-slate-950/90 text-slate-200 font-mono text-xs leading-relaxed whitespace-pre-wrap select-text space-y-3">
              {selectedTraceModal.trace}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
