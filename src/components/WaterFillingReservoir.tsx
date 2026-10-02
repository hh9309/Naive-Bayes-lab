import React, { useState, useMemo } from 'react';
import {
  Waves,
  Droplets,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  TrendingDown,
  TrendingUp,
  RotateCcw,
  Sparkles,
  Info,
} from 'lucide-react';
import { BayesCaseScenario } from '../data/bayesCases';
import { KatexMath } from './KatexMath';

interface WaterFillingReservoirProps {
  scenario: BayesCaseScenario;
  alpha: number;
  setAlpha: (val: number) => void;
}

interface ReservoirTank {
  id: string;
  token: string;
  role: 'high_freq' | 'mid_freq' | 'low_freq' | 'zero_freq';
  roleName: string;
  count: number;
  isZero: boolean;
}

export const WaterFillingReservoir: React.FC<WaterFillingReservoirProps> = ({
  scenario,
  alpha,
  setAlpha,
}) => {
  const [selectedClass, setSelectedClass] = useState<'pos' | 'neg'>('pos');
  const [activeTankId, setActiveTankId] = useState<string>('zero_1');

  // Total tokens and vocabulary size for the selected class
  const isPos = selectedClass === 'pos';
  const totalTokens = isPos
    ? scenario.samples
        .filter((s) => s.actualClass === 1)
        .reduce((sum, s) => sum + s.text.length, 0)
    : scenario.samples
        .filter((s) => s.actualClass === 0)
        .reduce((sum, s) => sum + s.text.length, 0);

  const effectiveTotalTokens = Math.max(120, totalTokens);
  const vocabSize = scenario.vocab.length;

  // Select 5 representative tanks from scenario vocabulary
  const tanks: ReservoirTank[] = useMemo(() => {
    const sorted = [...scenario.vocab].sort((a, b) =>
      isPos ? b.posCount - a.posCount : b.negCount - a.negCount
    );

    const high1 = sorted[0] || { token: '高频词A', posCount: 28, negCount: 5 };
    const high2 = sorted[1] || { token: '高频词B', posCount: 18, negCount: 3 };
    const mid = sorted[Math.floor(sorted.length / 2)] || { token: '常规词', posCount: 6, negCount: 6 };
    const low = sorted[sorted.length - 2] || { token: '生僻词', posCount: 1, negCount: 2 };

    return [
      {
        id: 'high_1',
        token: high1.token,
        role: 'high_freq',
        roleName: '顶峰高频特征',
        count: isPos ? high1.posCount : high1.negCount,
        isZero: false,
      },
      {
        id: 'high_2',
        token: high2.token,
        role: 'high_freq',
        roleName: '次高频特征',
        count: isPos ? high2.posCount : high2.negCount,
        isZero: false,
      },
      {
        id: 'mid_1',
        token: mid.token,
        role: 'mid_freq',
        roleName: '中频基准特征',
        count: Math.max(1, isPos ? mid.posCount : mid.negCount),
        isZero: false,
      },
      {
        id: 'low_1',
        token: low.token,
        role: 'low_freq',
        roleName: '低频稀疏特征',
        count: 1,
        isZero: false,
      },
      {
        id: 'zero_1',
        token: scenario.id === 'spam' ? '量子加密币' : scenario.id === 'phishing' ? '伪造网银盾' : '超稀有未登录词',
        role: 'zero_freq',
        roleName: '未登录零频词 (OOV)',
        count: 0,
        isZero: true,
      },
    ];
  }, [scenario, isPos]);

  // Denominator: N_Y + alpha * |V|
  const denominator = effectiveTotalTokens + alpha * vocabSize;
  const denominatorZeroAlpha = effectiveTotalTokens;

  // Calculate probabilities and water percentages for display
  const tankMetrics = useMemo(() => {
    return tanks.map((tank) => {
      const pRaw = tank.count / denominatorZeroAlpha;
      const pSmoothed = (tank.count + alpha) / denominator;
      const delta = pSmoothed - pRaw; // negative means clipped/skimmed, positive means filled
      const isPeak = delta < -0.0001;
      const isFilled = delta > 0.0001;

      return {
        ...tank,
        pRaw,
        pSmoothed,
        delta,
        isPeak,
        isFilled,
      };
    });
  }, [tanks, alpha, denominator, denominatorZeroAlpha]);

  // Find max probability to scale water height proportionally
  const maxProb = Math.max(...tankMetrics.map((t) => Math.max(t.pRaw, t.pSmoothed)), 0.05);

  // Active tank for drilldown
  const activeTank = tankMetrics.find((t) => t.id === activeTankId) || tankMetrics[4];

  // Total transferred mass from peaks
  const totalSkimmedProb = tankMetrics
    .filter((t) => t.delta < 0)
    .reduce((sum, t) => sum + Math.abs(t.delta), 0);

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-5 space-y-5">
      {/* Top Banner & Class Channel Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-sky-50 text-sky-600">
              <Waves className="w-4 h-4" />
            </span>
            <h3 className="text-base font-semibold text-slate-900 font-serif-title">
              水位削峰填谷“水蓄池”重分配流（Water-Filling Reservoir Animation）
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            将全概率归一约束 <KatexMath math="\sum P(X_i \mid Y) \equiv 1" /> 拟物化为连通柱状水箱，观察高频“削峰”如何通过虹吸注入零频“填谷”
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg text-xs">
            <button
              onClick={() => setSelectedClass('pos')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors whitespace-nowrap ${
                selectedClass === 'pos'
                  ? 'bg-white text-rose-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              正类通道 P(X|{scenario.posLabel.split(' ')[0]})
            </button>
            <button
              onClick={() => setSelectedClass('neg')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors whitespace-nowrap ${
                selectedClass === 'neg'
                  ? 'bg-white text-sky-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              负类通道 P(X|{scenario.negLabel.split(' ')[0]})
            </button>
          </div>
        </div>
      </div>

      {/* Main Interactive Metrology Bar */}
      <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Droplets className="w-4 h-4 text-sky-600" />
            <span className="font-semibold text-slate-700">平滑系数 α = </span>
            <span className="font-mono-tabular font-bold text-sky-700 text-sm">
              {alpha.toFixed(2)}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">虹吸削峰总转移量：</span>
            <span className="font-mono-tabular font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
              -{(totalSkimmedProb * 100).toFixed(2)}%
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">零频水箱注水收益：</span>
            <span className="font-mono-tabular font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
              +{((tankMetrics[4].pSmoothed) * 100).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-1.5 self-start md:self-auto">
          {[
            { label: 'α=0 (干涸危机)', val: 0 },
            { label: 'α=0.1 (微量注水)', val: 0.1 },
            { label: 'α=1.0 (标准平滑)', val: 1.0 },
            { label: 'α=3.0 (高平滑)', val: 3.0 },
          ].map((preset) => (
            <button
              key={preset.label}
              onClick={() => setAlpha(preset.val)}
              className={`px-2 py-1 rounded text-[11px] font-medium border transition-colors whitespace-nowrap ${
                Math.abs(alpha - preset.val) < 0.02
                  ? 'bg-slate-900 text-white border-slate-900'
                  : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* SVG Water Reservoirs Visualization */}
      <div className="overflow-x-auto border border-slate-200/90 rounded-xl bg-gradient-to-b from-[#F8FAFC] via-[#F1F5F9] to-[#E2E8F0]/60 p-4">
        <svg
          className="w-full min-w-[720px] h-[340px]"
          viewBox="0 0 760 340"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {/* Water gradient for safe smoothed water */}
            <linearGradient id="safeWaterGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#0284C7" stopOpacity="0.95" />
            </linearGradient>

            {/* Skimmed water gradient for peak loss */}
            <linearGradient id="skimmedWaterGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#FB7185" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#E11D48" stopOpacity="0.95" />
            </linearGradient>

            {/* Siphon tube flow gradient */}
            <linearGradient id="tubeFlowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#FB7185" stopOpacity="0.9" />
              <stop offset="50%" stopColor="#38BDF8" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#0284C7" stopOpacity="0.9" />
            </linearGradient>

            {/* Dry tank hazard pattern */}
            <pattern id="hazardStripe" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="4" height="8" fill="#FEE2E2" />
              <rect x="4" width="4" height="8" fill="#FEF2F2" />
            </pattern>
          </defs>

          {/* Overhead Horizontal Siphon Pipe connecting all reservoirs */}
          <g transform="translate(40, 30)">
            {/* Main glass manifold pipe */}
            <rect x="25" y="0" width="670" height="12" rx="6" fill="#FFFFFF" stroke="#94A3B8" strokeWidth="1.5" />
            
            {/* Animated fluid pulse inside siphon manifold if alpha > 0 */}
            {alpha > 0 && (
              <rect
                x="25"
                y="2"
                width="670"
                height="8"
                rx="4"
                fill="url(#tubeFlowGrad)"
                opacity="0.85"
                className="animate-pulse"
              />
            )}

            {/* Downward feeder nozzles into each tank */}
            {[0, 1, 2, 3, 4].map((i) => {
              const xPos = 45 + i * 135 + 45;
              return (
                <g key={`nozzle-${i}`}>
                  <rect x={xPos - 5} y="11" width="10" height="16" fill="#E2E8F0" stroke="#94A3B8" strokeWidth="1" />
                  {alpha > 0 && (
                    <line
                      x1={xPos}
                      y1="13"
                      x2={xPos}
                      y2="28"
                      stroke={i < 2 ? '#E11D48' : '#0284C7'}
                      strokeWidth="2.5"
                      strokeDasharray="4, 3"
                    />
                  )}
                </g>
              );
            })}

            <text x="360" y="-8" textAnchor="middle" fill="#64748B" fontSize="10" fontWeight="600" fontFamily="'IBM Plex Mono', monospace">
              全概率质量连通虹吸总管 (∑ P ≡ 1.0000 守恒流通道)
            </text>
          </g>

          {/* Render 5 Vertical Reservoirs */}
          {tankMetrics.map((tank, idx) => {
            const tankWidth = 100;
            const tankHeight = 180;
            const tankX = 50 + idx * 135;
            const tankY = 56;
            const isSelected = activeTankId === tank.id;

            // Height scaling: max prob maps to 150px
            const currentWaterHeight = Math.min(
              165,
              Math.max(0, (tank.pSmoothed / Math.max(0.01, maxProb)) * 145)
            );
            const rawWaterHeight = Math.min(
              165,
              Math.max(0, (tank.pRaw / Math.max(0.01, maxProb)) * 145)
            );

            const waterY = tankY + tankHeight - currentWaterHeight;
            const isDry = tank.count === 0 && alpha === 0;

            return (
              <g
                key={tank.id}
                className="cursor-pointer group"
                onClick={() => setActiveTankId(tank.id)}
              >
                {/* Tank Header Annotation: Skimmed / Filled tag */}
                <g transform={`translate(${tankX + tankWidth / 2}, ${tankY - 4})`}>
                  {tank.delta < -0.0001 ? (
                    <g transform="translate(0, 0)">
                      <rect x="-35" y="-18" width="70" height="16" rx="4" fill="#FFE4E6" stroke="#FDA4AF" strokeWidth="1" />
                      <text x="0" y="-6" textAnchor="middle" fill="#E11D48" fontSize="9" fontWeight="bold" fontFamily="'IBM Plex Mono', monospace">
                        削峰 -{(Math.abs(tank.delta) * 100).toFixed(1)}%
                      </text>
                    </g>
                  ) : tank.delta > 0.0001 ? (
                    <g transform="translate(0, 0)">
                      <rect x="-35" y="-18" width="70" height="16" rx="4" fill="#DCFCE7" stroke="#86EFAC" strokeWidth="1" />
                      <text x="0" y="-6" textAnchor="middle" fill="#15803D" fontSize="9" fontWeight="bold" fontFamily="'IBM Plex Mono', monospace">
                        填谷 +{(tank.delta * 100).toFixed(2)}%
                      </text>
                    </g>
                  ) : (
                    <text x="0" y="-6" textAnchor="middle" fill="#94A3B8" fontSize="9" fontFamily="'IBM Plex Mono', monospace">
                      基准
                    </text>
                  )}
                </g>

                {/* Glass Tank Container Body */}
                <rect
                  x={tankX}
                  y={tankY}
                  width={tankWidth}
                  height={tankHeight}
                  rx="8"
                  fill={isDry ? 'url(#hazardStripe)' : '#FFFFFF'}
                  stroke={
                    isSelected
                      ? '#0F172A'
                      : isDry
                      ? '#E11D48'
                      : '#CBD5E1'
                  }
                  strokeWidth={isSelected ? 2.5 : isDry ? 2 : 1.5}
                  className="transition-all"
                />

                {/* Glass Measurement Grids / Tick marks */}
                {[0.25, 0.5, 0.75].map((ratio) => (
                  <line
                    key={ratio}
                    x1={tankX + 4}
                    y1={tankY + tankHeight * ratio}
                    x2={tankX + 16}
                    y2={tankY + tankHeight * ratio}
                    stroke="#CBD5E1"
                    strokeWidth="1"
                  />
                ))}

                {/* Previous Level Ghost Dashed Line when alpha > 0 */}
                {alpha > 0 && tank.pRaw > 0 && (
                  <line
                    x1={tankX + 2}
                    y1={tankY + tankHeight - rawWaterHeight}
                    x2={tankX + tankWidth - 2}
                    y2={tankY + tankHeight - rawWaterHeight}
                    stroke="#94A3B8"
                    strokeWidth="1.2"
                    strokeDasharray="3, 2"
                  />
                )}

                {/* Water Volume Body */}
                {!isDry && currentWaterHeight > 0 && (
                  <g>
                    {/* Water Rect */}
                    <rect
                      x={tankX + 2}
                      y={waterY}
                      width={tankWidth - 4}
                      height={currentWaterHeight - 2}
                      rx="6"
                      fill={tank.role === 'zero_freq' ? '#38BDF8' : 'url(#safeWaterGrad)'}
                      className="transition-all duration-300"
                    />

                    {/* Water surface wave line */}
                    <path
                      d={`
                        M ${tankX + 2} ${waterY}
                        Q ${tankX + tankWidth * 0.25} ${waterY - 3}, ${tankX + tankWidth * 0.5} ${waterY}
                        T ${tankX + tankWidth - 2} ${waterY}
                      `}
                      fill="none"
                      stroke="#E0F2FE"
                      strokeWidth="2"
                    />
                  </g>
                )}

                {/* DRY TANK DANGER CALLOUT */}
                {isDry && (
                  <g transform={`translate(${tankX + tankWidth / 2}, ${tankY + tankHeight / 2 - 10})`}>
                    <circle cx="0" cy="0" r="16" fill="#FEE2E2" stroke="#EF4444" strokeWidth="1.5" />
                    <text x="0" y="5" textAnchor="middle" fill="#DC2626" fontSize="16" fontWeight="bold">
                      !
                    </text>
                    <text x="0" y="28" textAnchor="middle" fill="#DC2626" fontSize="10" fontWeight="bold">
                      水箱见底干涸
                    </text>
                    <text x="0" y="42" textAnchor="middle" fill="#991B1B" fontSize="9" fontFamily="'IBM Plex Mono', monospace">
                      P=0 一票否决
                    </text>
                  </g>
                )}

                {/* Probability Value inside or above tank */}
                {!isDry && (
                  <text
                    x={tankX + tankWidth / 2}
                    y={Math.min(tankY + tankHeight - 12, waterY + 22)}
                    textAnchor="middle"
                    fill="#FFFFFF"
                    fontWeight="bold"
                    fontSize="11"
                    fontFamily="'IBM Plex Mono', monospace"
                  >
                    {(tank.pSmoothed * 100).toFixed(2)}%
                  </text>
                )}

                {/* Tank Base Stand */}
                <rect
                  x={tankX - 4}
                  y={tankY + tankHeight}
                  width={tankWidth + 8}
                  height="8"
                  rx="3"
                  fill="#64748B"
                />

                {/* Token Label & Role Beneath Tank */}
                <g transform={`translate(${tankX + tankWidth / 2}, ${tankY + tankHeight + 22})`}>
                  <text
                    x="0"
                    y="0"
                    textAnchor="middle"
                    fill={isSelected ? '#0284C7' : '#0F172A'}
                    fontWeight={isSelected ? 'bold' : '600'}
                    fontSize="12"
                    fontFamily="'Noto Sans SC', sans-serif"
                  >
                    {tank.token}
                  </text>

                  <text
                    x="0"
                    y="15"
                    textAnchor="middle"
                    fill={tank.role === 'zero_freq' ? '#DC2626' : '#64748B'}
                    fontSize="10"
                    fontFamily="'IBM Plex Mono', monospace"
                  >
                    {tank.role === 'zero_freq' ? '未登录零频 (0词)' : `频次: ${tank.count}`}
                  </text>
                </g>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Selected Reservoir Mathematical Analysis Card */}
      {activeTank && (
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
          <div className="md:col-span-5 space-y-1.5">
            <div className="flex items-center gap-2">
              <span
                className={`px-2 py-0.5 text-xs font-bold rounded ${
                  activeTank.isZero
                    ? 'bg-rose-600 text-white'
                    : 'bg-slate-900 text-white'
                }`}
              >
                {activeTank.roleName}
              </span>
              <h4 className="text-base font-bold text-slate-900">
                特征水箱：「{activeTank.token}」
              </h4>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              {activeTank.isZero
                ? `该词未在${isPos ? scenario.posLabel : scenario.negLabel}训练集出现过。α=0 时水位为绝对 0，导致联合似然连乘彻底坍塌归零；α=${alpha.toFixed(2)} 时通过虹吸注入平滑水量，确保概率底线。`
                : `该词在训练集中出现了 ${activeTank.count} 次。平滑机制让高频水箱让渡出部分质量，分摊至全词表 ${vocabSize} 个维度中。`}
            </p>
          </div>

          <div className="md:col-span-7 grid grid-cols-3 gap-3 font-mono-tabular text-xs">
            <div className="p-3 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500 font-sans">
                无平滑原始水位 (α=0)
              </div>
              <div
                className={`text-sm font-bold mt-1 ${
                  activeTank.isZero ? 'text-rose-600' : 'text-slate-800'
                }`}
              >
                {(activeTank.pRaw * 100).toFixed(3)}%
              </div>
              <div className="text-[10px] text-slate-400 mt-1">
                {activeTank.count} / {denominatorZeroAlpha}
              </div>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500 font-sans">
                当前平滑水位 (α={alpha.toFixed(2)})
              </div>
              <div className="text-sm font-bold text-sky-700 mt-1">
                {(activeTank.pSmoothed * 100).toFixed(3)}%
              </div>
              <div className="text-[10px] text-slate-400 mt-1">
                ({activeTank.count} + {alpha.toFixed(1)}) / {denominator.toFixed(0)}
              </div>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-lg">
              <div className="text-[11px] text-slate-500 font-sans">
                概率质量净变动量
              </div>
              <div
                className={`text-sm font-bold mt-1 flex items-center gap-1 ${
                  activeTank.delta < 0 ? 'text-rose-600' : 'text-emerald-700'
                }`}
              >
                {activeTank.delta < 0 ? (
                  <TrendingDown className="w-3.5 h-3.5" />
                ) : (
                  <TrendingUp className="w-3.5 h-3.5" />
                )}
                {activeTank.delta >= 0 ? '+' : ''}
                {(activeTank.delta * 100).toFixed(3)}%
              </div>
              <div className="text-[10px] text-slate-400 mt-1">
                {activeTank.delta < 0 ? '向全表分流削峰' : '注入先验填谷'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
