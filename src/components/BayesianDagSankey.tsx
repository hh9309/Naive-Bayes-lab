import React, { useState, useMemo } from 'react';
import {
  GitFork,
  ArrowRight,
  Sparkles,
  Info,
  Scale,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Zap,
} from 'lucide-react';
import { BayesCaseScenario } from '../data/bayesCases';
import { BayesInferenceResult, TokenProbabilityStep } from '../utils/bayesEngine';
import { KatexMath } from './KatexMath';

interface BayesianDagSankeyProps {
  scenario: BayesCaseScenario;
  inference: BayesInferenceResult;
  alpha: number;
  inputText: string;
  onOpenAiModal?: (prompt?: string) => void;
}

export const BayesianDagSankey: React.FC<BayesianDagSankeyProps> = ({
  scenario,
  inference,
  alpha,
  inputText,
  onOpenAiModal,
}) => {
  // Visual Mode: DAG vs Sankey Flow
  const [subMode, setSubMode] = useState<'dag' | 'sankey'>('sankey');
  // In DAG mode: Generative (Y -> X) vs Discriminative (X -> Y)
  const [modelType, setModelType] = useState<'generative' | 'discriminative'>('generative');
  // Selected token for drill-down inspection
  const [selectedToken, setSelectedToken] = useState<string | null>(null);

  const activeSteps = useMemo(() => {
    // Sort by absolute log-likelihood ratio * count descending so top influential words are on top
    return [...inference.steps].sort(
      (a, b) =>
        Math.abs(b.logLikelihoodRatio) * b.countInDoc -
        Math.abs(a.logLikelihoodRatio) * a.countInDoc
    );
  }, [inference.steps]);

  const selectedStepData: TokenProbabilityStep | undefined = useMemo(() => {
    if (selectedToken) {
      return activeSteps.find((s) => s.token === selectedToken);
    }
    return activeSteps[0];
  }, [selectedToken, activeSteps]);

  // Calculations for Sankey dimensions
  const totalPosPull = useMemo(() => {
    return activeSteps
      .filter((s) => s.logLikelihoodRatio > 0)
      .reduce((sum, s) => sum + s.logLikelihoodRatio * s.countInDoc, 0);
  }, [activeSteps]);

  const totalNegPull = useMemo(() => {
    return activeSteps
      .filter((s) => s.logLikelihoodRatio < 0)
      .reduce((sum, s) => sum + Math.abs(s.logLikelihoodRatio) * s.countInDoc, 0);
  }, [activeSteps]);

  const logOddsMargin = inference.totalLogPos - inference.totalLogNeg;
  // Tilt angle for scale balance (-25deg to +25deg)
  const tiltDeg = Math.max(-25, Math.min(25, logOddsMargin * 4));

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-5 space-y-5">
      {/* Sub-header & Mode switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-sky-50 text-sky-700">
              <GitFork className="w-4 h-4" />
            </span>
            <h3 className="text-base font-semibold text-slate-900 font-serif-title">
              贝叶斯网络 DAG 拓扑图与对数似然权重桑基流
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            动态解析「文档类别变量 Y」与高维「词元特征 Xᵢ」之间的因果生成拓扑与似然流向拉扯
          </p>
        </div>

        <div className="flex items-center gap-1.5 p-1 bg-slate-100 rounded-lg self-start">
          <button
            onClick={() => setSubMode('sankey')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap flex items-center gap-1.5 ${
              subMode === 'sankey'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Scale className="w-3.5 h-3.5 text-rose-600" />
            <span>权重桑基流 (Sankey)</span>
          </button>
          <button
            onClick={() => setSubMode('dag')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap flex items-center gap-1.5 ${
              subMode === 'dag'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <GitFork className="w-3.5 h-3.5 text-sky-600" />
            <span>贝叶斯网络 DAG</span>
          </button>
        </div>
      </div>

      {/* Mode 1: Sankey Diagram View */}
      {subMode === 'sankey' && (
        <div className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 p-3 bg-slate-50 rounded-xl border border-slate-200/80 text-xs">
            <div className="flex items-center gap-3">
              <span className="font-semibold text-slate-700 flex items-center gap-1">
                <Scale className="w-4 h-4 text-emerald-600" />
                概率天平倾斜度：
              </span>
              <span className="font-mono-tabular">
                Δ对数奇数比 ={' '}
                <strong
                  className={logOddsMargin >= 0 ? 'text-rose-700' : 'text-sky-700'}
                >
                  {logOddsMargin >= 0 ? '+' : ''}
                  {logOddsMargin.toFixed(3)} nats
                </strong>
              </span>
              <span className="text-slate-400">|</span>
              <span>
                最终判决：
                <strong
                  className={`ml-1 px-2 py-0.5 rounded text-white text-[11px] ${
                    inference.predictedClass === 1 ? 'bg-rose-600' : 'bg-sky-600'
                  }`}
                >
                  {inference.predictedLabel} ({(inference.posteriorPos * 100).toFixed(1)}% vs {(inference.posteriorNeg * 100).toFixed(1)}%)
                </strong>
              </span>
            </div>

            <div className="flex items-center gap-4 text-[11px]">
              <span className="flex items-center gap-1.5 text-rose-700 font-medium">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" />
                向正类拉扯 (+LLR)
              </span>
              <span className="flex items-center gap-1.5 text-sky-700 font-medium">
                <span className="w-2.5 h-2.5 rounded-full bg-sky-500 inline-block" />
                向负类拉扯 (-LLR)
              </span>
              <span className="flex items-center gap-1.5 text-slate-500">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-300 inline-block" />
                中性背景 (≈0)
              </span>
            </div>
          </div>

          {/* SVG Sankey Diagram */}
          <div className="overflow-x-auto border border-slate-200/90 rounded-xl bg-gradient-to-b from-[#F8FAFC] to-[#F1F5F9] p-4">
            <svg
              className="w-full min-w-[700px] h-[400px]"
              viewBox="0 0 760 400"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                {/* Gradient for Positive pull streams */}
                <linearGradient id="posStreamGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#FB7185" stopOpacity="0.75" />
                  <stop offset="100%" stopColor="#E11D48" stopOpacity="0.85" />
                </linearGradient>
                {/* Gradient for Negative pull streams */}
                <linearGradient id="negStreamGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.75" />
                  <stop offset="100%" stopColor="#0284C7" stopOpacity="0.85" />
                </linearGradient>
                {/* Neutral Stream */}
                <linearGradient id="neutralStreamGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#94A3B8" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#64748B" stopOpacity="0.3" />
                </linearGradient>
              </defs>

              {/* Central Dynamic Scale Balance representation */}
              <g transform="translate(480, 200)">
                <line x1="0" y1="50" x2="0" y2="120" stroke="#94A3B8" strokeWidth="3" />
                <polygon points="-25,120 25,120 0,90" fill="#CBD5E1" />
                {/* Pivot beam tilting */}
                <g transform={`rotate(${tiltDeg})`}>
                  <line x1="-90" y1="0" x2="90" y2="0" stroke="#0F172A" strokeWidth="4" strokeLinecap="round" />
                  {/* Left Pan (Pos) */}
                  <line x1="90" y1="0" x2="90" y2="25" stroke="#E11D48" strokeWidth="1.5" />
                  <ellipse cx="90" cy="27" rx="20" ry="6" fill="#FFE4E6" stroke="#E11D48" strokeWidth="1.5" />
                  {/* Right Pan (Neg) */}
                  <line x1="-90" y1="0" x2="-90" y2="25" stroke="#0284C7" strokeWidth="1.5" />
                  <ellipse cx="-90" cy="27" rx="20" ry="6" fill="#E0F2FE" stroke="#0284C7" strokeWidth="1.5" />
                </g>
                <text x="0" y="85" textAnchor="middle" fill="#64748B" fontSize="10" fontFamily="'IBM Plex Mono', monospace">
                  天平平衡轴 (Δ={logOddsMargin.toFixed(2)})
                </text>
              </g>

              {/* Right Target Blocks: Class 1 (Top) & Class 0 (Bottom) */}
              {/* Positive Target Block */}
              <g transform="translate(620, 40)">
                <rect
                  x="0"
                  y="0"
                  width="125"
                  height="110"
                  rx="10"
                  fill="#FFF1F2"
                  stroke="#E11D48"
                  strokeWidth="2"
                  className="transition-all"
                />
                <text x="62" y="24" textAnchor="middle" fill="#9F1239" fontWeight="bold" fontSize="12">
                  正类: {scenario.posLabel.split(' ')[0]}
                </text>
                <text x="62" y="42" textAnchor="middle" fill="#BE123C" fontSize="10" fontFamily="'IBM Plex Mono', monospace">
                  先验: ln P={inference.logPriorPos.toFixed(2)}
                </text>
                <text x="62" y="60" textAnchor="middle" fill="#BE123C" fontSize="10" fontFamily="'IBM Plex Mono', monospace">
                  似然和: {(inference.totalLogPos - inference.logPriorPos).toFixed(2)}
                </text>
                <line x1="12" y1="72" x2="113" y2="72" stroke="#FECDD3" strokeWidth="1" />
                <text x="62" y="90" textAnchor="middle" fill="#881337" fontWeight="bold" fontSize="13" fontFamily="'IBM Plex Mono', monospace">
                  ℓ₁ = {inference.totalLogPos.toFixed(2)}
                </text>
              </g>

              {/* Negative Target Block */}
              <g transform="translate(620, 240)">
                <rect
                  x="0"
                  y="0"
                  width="125"
                  height="110"
                  rx="10"
                  fill="#F0F9FF"
                  stroke="#0284C7"
                  strokeWidth="2"
                  className="transition-all"
                />
                <text x="62" y="24" textAnchor="middle" fill="#075985" fontWeight="bold" fontSize="12">
                  负类: {scenario.negLabel.split(' ')[0]}
                </text>
                <text x="62" y="42" textAnchor="middle" fill="#0369A1" fontSize="10" fontFamily="'IBM Plex Mono', monospace">
                  先验: ln P={inference.logPriorNeg.toFixed(2)}
                </text>
                <text x="62" y="60" textAnchor="middle" fill="#0369A1" fontSize="10" fontFamily="'IBM Plex Mono', monospace">
                  似然和: {(inference.totalLogNeg - inference.logPriorNeg).toFixed(2)}
                </text>
                <line x1="12" y1="72" x2="113" y2="72" stroke="#BAE6FD" strokeWidth="1" />
                <text x="62" y="90" textAnchor="middle" fill="#0C4A6E" fontWeight="bold" fontSize="13" fontFamily="'IBM Plex Mono', monospace">
                  ℓ₀ = {inference.totalLogNeg.toFixed(2)}
                </text>
              </g>

              {/* Left Column Tokens & Curved Sankey Ribbons */}
              {activeSteps.slice(0, 8).map((step, idx) => {
                const totalVisible = Math.min(activeSteps.length, 8);
                const startY = 35 + idx * (330 / Math.max(1, totalVisible - 1));
                const isSelected = selectedToken === step.token;
                const isPos = step.logLikelihoodRatio > 0.12;
                const isNeg = step.logLikelihoodRatio < -0.12;

                // Target connection point on right
                const targetY = isPos
                  ? 55 + (idx % 4) * 20
                  : isNeg
                  ? 255 + (idx % 4) * 20
                  : 195;

                // Thickness based on absolute LLR * count
                const ribbonThickness = Math.max(
                  3,
                  Math.min(22, Math.abs(step.logLikelihoodRatio) * step.countInDoc * 6)
                );

                const sourceX = 145;
                const targetX = 620;
                const cX1 = sourceX + (targetX - sourceX) * 0.42;
                const cX2 = sourceX + (targetX - sourceX) * 0.58;

                const pathData = `
                  M ${sourceX} ${startY}
                  C ${cX1} ${startY}, ${cX2} ${targetY}, ${targetX} ${targetY}
                  L ${targetX} ${targetY + ribbonThickness}
                  C ${cX2} ${targetY + ribbonThickness}, ${cX1} ${startY + ribbonThickness}, ${sourceX} ${startY + ribbonThickness}
                  Z
                `;

                const strokeColor = isPos
                  ? 'url(#posStreamGrad)'
                  : isNeg
                  ? 'url(#negStreamGrad)'
                  : 'url(#neutralStreamGrad)';

                return (
                  <g key={step.token} className="cursor-pointer group" onClick={() => setSelectedToken(step.token)}>
                    {/* Ribbon Stream */}
                    <path
                      d={pathData}
                      fill={strokeColor}
                      opacity={isSelected ? 0.95 : 0.65}
                      className="transition-all hover:opacity-100"
                    />

                    {/* Left Token Node Pill */}
                    <g transform={`translate(15, ${startY - 12})`}>
                      <rect
                        x="0"
                        y="0"
                        width="130"
                        height="26"
                        rx="6"
                        fill={isSelected ? '#0F172A' : '#FFFFFF'}
                        stroke={
                          isSelected
                            ? '#0F172A'
                            : isPos
                            ? '#FDA4AF'
                            : isNeg
                            ? '#7DD3FC'
                            : '#CBD5E1'
                        }
                        strokeWidth={isSelected ? 2 : 1.2}
                        className="transition-colors shadow-2xs"
                      />
                      <text
                        x="10"
                        y="17"
                        fill={isSelected ? '#FFFFFF' : '#1E293B'}
                        fontWeight={isSelected ? 'bold' : '600'}
                        fontSize="11"
                        fontFamily="'Noto Sans SC', sans-serif"
                      >
                        {step.token}
                        {step.countInDoc > 1 ? ` ×${step.countInDoc}` : ''}
                      </text>
                      <text
                        x="120"
                        y="17"
                        textAnchor="end"
                        fill={
                          isSelected
                            ? '#38BDF8'
                            : isPos
                            ? '#E11D48'
                            : isNeg
                            ? '#0284C7'
                            : '#64748B'
                        }
                        fontWeight="bold"
                        fontSize="10"
                        fontFamily="'IBM Plex Mono', monospace"
                      >
                        {step.logLikelihoodRatio >= 0 ? '+' : ''}
                        {step.logLikelihoodRatio.toFixed(2)}
                      </text>
                    </g>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      )}

      {/* Mode 2: Bayesian Network DAG View */}
      {subMode === 'dag' && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 bg-slate-50 rounded-xl border border-slate-200/80 text-xs">
            <div className="flex items-center gap-3">
              <span className="font-semibold text-slate-700">拓扑因果范式切换：</span>
              <div className="flex items-center gap-1.5 bg-white p-1 rounded-lg border border-slate-200">
                <button
                  onClick={() => setModelType('generative')}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    modelType === 'generative'
                      ? 'bg-slate-900 text-white font-medium shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  生成式拓扑 (Y → Xᵢ) · 朴素贝叶斯
                </button>
                <button
                  onClick={() => setModelType('discriminative')}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    modelType === 'discriminative'
                      ? 'bg-slate-900 text-white font-medium shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  判别式拓扑 (Xᵢ → Y) · 逻辑回归/神经网络
                </button>
              </div>
            </div>

            <div className="text-[11px] text-slate-500">
              {modelType === 'generative'
                ? '★ 假设在已知类别 Y 的条件下，所有特征词 Xᵢ 相互独立，各箭头代表边缘条件似然 P(Xᵢ | Y)'
                : '★ 直接拟合条件概率 P(Y | X)，箭头自底向上汇聚，不构建特征生成模型'}
            </div>
          </div>

          {/* SVG DAG Representation */}
          <div className="overflow-x-auto border border-slate-200/90 rounded-xl bg-gradient-to-b from-[#F8FAFC] to-[#F1F5F9] p-4">
            <svg
              className="w-full min-w-[700px] h-[380px]"
              viewBox="0 0 760 380"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                <marker
                  id="arrowGenerative"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0284C7" />
                </marker>
                <marker
                  id="arrowDiscriminative"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7C3AED" />
                </marker>
              </defs>

              {/* Root Class Node Y */}
              <g transform="translate(380, 50)">
                <ellipse
                  cx="0"
                  cy="0"
                  rx="95"
                  ry="32"
                  fill="#0F172A"
                  stroke="#38BDF8"
                  strokeWidth="2.5"
                  className="shadow-md"
                />
                <text x="0" y="-6" textAnchor="middle" fill="#FFFFFF" fontWeight="bold" fontSize="13">
                  类别变量 Y ∈ &#123;0, 1&#125;
                </text>
                <text x="0" y="14" textAnchor="middle" fill="#94A3B8" fontSize="10" fontFamily="'IBM Plex Mono', monospace">
                  P(Y=1)={(inference.priorPos * 100).toFixed(0)}% | P(Y=0)={(inference.priorNeg * 100).toFixed(0)}%
                </text>
              </g>

              {/* Feature Nodes X_i in a circular arc below */}
              {activeSteps.slice(0, 7).map((step, idx) => {
                const total = Math.min(activeSteps.length, 7);
                const angle = Math.PI * 0.15 + (idx / Math.max(1, total - 1)) * (Math.PI * 0.7);
                const nodeX = 380 + Math.cos(angle) * 280;
                const nodeY = 80 + Math.sin(angle) * 220;
                const isSelected = selectedToken === step.token;

                // Arrows
                const startX = modelType === 'generative' ? 380 : nodeX;
                const startY = modelType === 'generative' ? 82 : nodeY - 20;
                const endX = modelType === 'generative' ? nodeX : 380;
                const endY = modelType === 'generative' ? nodeY - 24 : 82;

                return (
                  <g key={step.token} className="cursor-pointer group" onClick={() => setSelectedToken(step.token)}>
                    {/* Directed Arrow Edge */}
                    <line
                      x1={startX}
                      y1={startY}
                      x2={endX}
                      y2={endY}
                      stroke={
                        modelType === 'generative'
                          ? isSelected
                            ? '#0284C7'
                            : '#94A3B8'
                          : isSelected
                          ? '#7C3AED'
                          : '#A78BFA'
                      }
                      strokeWidth={isSelected ? 2.5 : 1.5}
                      strokeDasharray={isSelected ? 'none' : '4, 3'}
                      markerEnd={modelType === 'generative' ? 'url(#arrowGenerative)' : 'url(#arrowDiscriminative)'}
                    />

                    {/* Edge weight / condition tag */}
                    {modelType === 'generative' && (
                      <rect
                        x={(startX + endX) / 2 - 28}
                        y={(startY + endY) / 2 - 9}
                        width="56"
                        height="18"
                        rx="4"
                        fill="#FFFFFF"
                        stroke="#CBD5E1"
                        strokeWidth="1"
                      />
                    )}
                    {modelType === 'generative' && (
                      <text
                        x={(startX + endX) / 2}
                        y={(startY + endY) / 2 + 3}
                        textAnchor="middle"
                        fill="#0369A1"
                        fontSize="9"
                        fontWeight="600"
                        fontFamily="'IBM Plex Mono', monospace"
                      >
                        P(X_{idx + 1}|Y)
                      </text>
                    )}

                    {/* Feature Circle Node */}
                    <circle
                      cx={nodeX}
                      cy={nodeY}
                      r="26"
                      fill={
                        isSelected
                          ? '#0284C7'
                          : step.logLikelihoodRatio > 0.15
                          ? '#FFE4E6'
                          : step.logLikelihoodRatio < -0.15
                          ? '#E0F2FE'
                          : '#FFFFFF'
                      }
                      stroke={
                        isSelected
                          ? '#0F172A'
                          : step.logLikelihoodRatio > 0.15
                          ? '#E11D48'
                          : step.logLikelihoodRatio < -0.15
                          ? '#0284C7'
                          : '#94A3B8'
                      }
                      strokeWidth={isSelected ? 3 : 1.5}
                      className="transition-transform group-hover:scale-110"
                    />
                    <text
                      x={nodeX}
                      y={nodeY + 4}
                      textAnchor="middle"
                      fill={isSelected ? '#FFFFFF' : '#0F172A'}
                      fontWeight="bold"
                      fontSize="11"
                      fontFamily="'Noto Sans SC', sans-serif"
                    >
                      {step.token}
                    </text>

                    {/* Subscript indicator */}
                    <text
                      x={nodeX}
                      y={nodeY + 39}
                      textAnchor="middle"
                      fill="#64748B"
                      fontSize="10"
                      fontFamily="'IBM Plex Mono', monospace"
                    >
                      X_{idx + 1}
                    </text>
                  </g>
                );
              })}

              {/* Show correlation violation edge if present in scenario */}
              {modelType === 'generative' && scenario.correlatedHighlight.r > 0.5 && (
                <g>
                  <path
                    d="M 230 260 C 320 350, 440 350, 530 260"
                    fill="none"
                    stroke="#D97706"
                    strokeWidth="2"
                    strokeDasharray="5, 4"
                  />
                  <rect x="330" y="325" width="100" height="20" rx="4" fill="#FEF3C7" stroke="#F59E0B" />
                  <text x="380" y="339" textAnchor="middle" fill="#B45309" fontSize="10" fontWeight="bold">
                    共现违背 r={scenario.correlatedHighlight.r}
                  </text>
                </g>
              )}
            </svg>
          </div>
        </div>
      )}

      {/* Selected Token Drill-down Card & Algebraic Breakdown */}
      {selectedStepData && (
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          <div className="md:col-span-4 space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-xs font-bold bg-slate-900 text-white rounded">
                特征钻取
              </span>
              <h4 className="text-base font-bold text-slate-900">
                「{selectedStepData.token}」
                {selectedStepData.countInDoc > 1 && (
                  <span className="text-xs text-slate-500 font-normal">
                    (文档内频次: {selectedStepData.countInDoc})
                  </span>
                )}
              </h4>
            </div>
            <p className="text-xs text-slate-500">
              点击上方桑基图词流或 DAG 节点，可查看该词元的条件似然概率分布与卡方信息度。
            </p>
          </div>

          <div className="md:col-span-8 grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono-tabular text-xs">
            <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500">P(X|正类)</div>
              <div className="text-sm font-bold text-rose-700 mt-0.5">
                {selectedStepData.pGivenPos.toFixed(5)}
              </div>
              <div className="text-[10px] text-slate-400">频次: {selectedStepData.posCount}</div>
            </div>

            <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500">P(X|负类)</div>
              <div className="text-sm font-bold text-sky-700 mt-0.5">
                {selectedStepData.pGivenNeg.toFixed(5)}
              </div>
              <div className="text-[10px] text-slate-400">频次: {selectedStepData.negCount}</div>
            </div>

            <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500">对数似然比 LLR</div>
              <div
                className={`text-sm font-bold mt-0.5 ${
                  selectedStepData.logLikelihoodRatio >= 0 ? 'text-rose-700' : 'text-sky-700'
                }`}
              >
                {selectedStepData.logLikelihoodRatio >= 0 ? '+' : ''}
                {selectedStepData.logLikelihoodRatio.toFixed(3)}
              </div>
              <div className="text-[10px] text-slate-400">
                贡献: {(selectedStepData.logLikelihoodRatio * selectedStepData.countInDoc).toFixed(3)}
              </div>
            </div>

            <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500">卡方显著性 χ²</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">
                {selectedStepData.chiSquare.toFixed(2)}
              </div>
              <div className="text-[10px] text-slate-400">
                MI: {selectedStepData.mutualInfo.toFixed(4)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
