import React, { useState, useEffect, useRef } from 'react';
import {
  RotateCcw,
  Play,
  Pause,
  Sliders,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { BayesCaseScenario } from '../data/bayesCases';
import { BayesInferenceResult } from '../utils/bayesEngine';
import { KatexMath } from './KatexMath';
import { BayesianDagSankey } from './BayesianDagSankey';
import { WaterFillingReservoir } from './WaterFillingReservoir';

interface Modules1To4Props {
  activeModule: 1 | 2 | 3 | 4;
  scenario: BayesCaseScenario;
  inference: BayesInferenceResult;
  alpha: number;
  setAlpha: (val: number) => void;
  inputText: string;
  setInputText: (val: string) => void;
  onOpenAiModal: (prompt?: string) => void;
}

export const Modules1To4: React.FC<Modules1To4Props> = ({
  activeModule,
  scenario,
  inference,
  alpha,
  setAlpha,
  inputText,
  setInputText,
  onOpenAiModal,
}) => {
  // Module 1 state
  const [algSlice, setAlgSlice] = useState<'posterior' | 'log_derivation' | 'underflow_sim'>('posterior');
  const [selectedTerm, setSelectedTerm] = useState<'posterior' | 'prior' | 'likelihood' | 'evidence'>('likelihood');
  const [simTokenCount, setSimTokenCount] = useState<number>(45);
  const [avgTokenProb, setAvgTokenProb] = useState<number>(0.004);

  // Module 2 3D Canvas state
  const canvas3dRef = useRef<HTMLCanvasElement | null>(null);
  const [yaw, setYaw] = useState<number>(-0.55);
  const [pitch, setPitch] = useState<number>(0.42);
  const [zoom3d, setZoom3d] = useState<number>(1.0);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [animStepCount, setAnimStepCount] = useState<number>(99);
  const [view3dSlice, setView3dSlice] = useState<'prob_bars' | 'log_odds' | 'cumulative_posterior'>('prob_bars');
  const isDraggingRef = useRef<boolean>(false);
  const lastMouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Module 3 state
  const [smoothSlice, setSmoothSlice] = useState<'reservoir' | 'zero_trap' | 'conservation' | 'alpha_curve'>('reservoir');

  // Module 4 state
  const canvas2dRef = useRef<HTMLCanvasElement | null>(null);
  const [featureX, setFeatureX] = useState<string>(scenario.defaultFeaturePair[0]);
  const [featureY, setFeatureY] = useState<string>(scenario.defaultFeaturePair[1]);
  const [correlationFactor, setCorrelationFactor] = useState<number>(0.0);
  const [indepSlice, setIndepSlice] = useState<'dag_sankey' | 'orthogonal_proj' | 'collinear_inflation'>('dag_sankey');
  const [animPhase, setAnimPhase] = useState<number>(0);

  useEffect(() => {
    setFeatureX(scenario.defaultFeaturePair[0]);
    setFeatureY(scenario.defaultFeaturePair[1]);
  }, [scenario]);

  useEffect(() => {
    if (activeModule !== 2 || !autoRotate) return;
    let rafId: number;
    const tick = () => {
      setYaw((prev) => prev + 0.0045);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [activeModule, autoRotate]);

  useEffect(() => {
    if (activeModule !== 2) return;
    const canvas = canvas3dRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 760;
    const height = canvas.clientHeight || 430;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, '#F8FAFC');
    grad.addColorStop(1, '#F1F5F9');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    const cx = width * 0.5;
    const cy = height * 0.62;
    const scale = 155 * zoom3d;

    const project3D = (x: number, y: number, z: number) => {
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;

      const cosP = Math.cos(pitch);
      const sinP = Math.sin(pitch);
      const y2 = y * cosP - z1 * sinP;
      const z2 = y * sinP + z1 * cosP;

      const fov = 4.2;
      const persp = fov / Math.max(1.5, fov + z2 * 0.35);
      return {
        sx: cx + x1 * scale * persp,
        sy: cy - y2 * scale * persp,
        depth: z2,
      };
    };

    ctx.strokeStyle = '#CBD5E1';
    ctx.lineWidth = 1;
    const gridXMin = -1.45;
    const gridXMax = 1.45;
    const gridZMin = -0.85;
    const gridZMax = 0.85;

    for (let gx = gridXMin; gx <= gridXMax + 0.01; gx += 0.29) {
      const p1 = project3D(gx, 0, gridZMin);
      const p2 = project3D(gx, 0, gridZMax);
      ctx.beginPath();
      ctx.moveTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.stroke();
    }
    for (let gz = gridZMin; gz <= gridZMax + 0.01; gz += 0.28) {
      const p1 = project3D(gridXMin, 0, gz);
      const p2 = project3D(gridXMax, 0, gz);
      ctx.beginPath();
      ctx.moveTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.stroke();
    }

    const origin = project3D(gridXMin, 0, gridZMin);
    const topY = project3D(gridXMin, 1.35, gridZMin);
    ctx.strokeStyle = '#64748B';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(origin.sx, origin.sy);
    ctx.lineTo(topY.sx, topY.sy);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '500 11px "IBM Plex Mono", sans-serif';
    ctx.fillText(
      view3dSlice === 'prob_bars'
        ? 'P(X_i | Y) 条件概率高度'
        : view3dSlice === 'log_odds'
        ? '|ln P(X_i|Y)| 对数似然模长'
        : '后验累加演化轨迹',
      topY.sx - 10,
      topY.sy - 10
    );

    const displaySteps = inference.steps.slice(0, Math.min(8, Math.max(1, animStepCount)));
    const count = displaySteps.length;

    const draw3DBar = (
      xCenter: number,
      zCenter: number,
      barWidth: number,
      barDepth: number,
      barHeight: number,
      topColor: string,
      frontColor: string,
      sideColor: string
    ) => {
      const h = Math.max(0.02, barHeight);
      const x0 = xCenter - barWidth / 2;
      const x1 = xCenter + barWidth / 2;
      const z0 = zCenter - barDepth / 2;
      const z1 = zCenter + barDepth / 2;

      const b00 = project3D(x0, 0, z0);
      const b10 = project3D(x1, 0, z0);
      const b11 = project3D(x1, 0, z1);
      const b01 = project3D(x0, 0, z1);

      const t00 = project3D(x0, h, z0);
      const t10 = project3D(x1, h, z0);
      const t11 = project3D(x1, h, z1);
      const t01 = project3D(x0, h, z1);

      ctx.fillStyle = frontColor;
      ctx.beginPath();
      ctx.moveTo(b01.sx, b01.sy);
      ctx.lineTo(b11.sx, b11.sy);
      ctx.lineTo(t11.sx, t11.sy);
      ctx.lineTo(t01.sx, t01.sy);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.stroke();

      ctx.fillStyle = sideColor;
      ctx.beginPath();
      ctx.moveTo(b11.sx, b11.sy);
      ctx.lineTo(b10.sx, b10.sy);
      ctx.lineTo(t10.sx, t10.sy);
      ctx.lineTo(t11.sx, t11.sy);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = topColor;
      ctx.beginPath();
      ctx.moveTo(t00.sx, t00.sy);
      ctx.lineTo(t10.sx, t10.sy);
      ctx.lineTo(t11.sx, t11.sy);
      ctx.lineTo(t01.sx, t01.sy);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      return t01;
    };

    const maxProb = Math.max(
      0.08,
      ...displaySteps.map((s) => Math.max(s.pGivenPos, s.pGivenNeg))
    );

    const posteriorRibbonPoints: Array<{ sx: number; sy: number; token: string; post: number }> = [];
    let runningLogPos = inference.logPriorPos;
    let runningLogNeg = inference.logPriorNeg;

    displaySteps.forEach((step, idx) => {
      const xPos =
        count === 1 ? 0 : -1.15 + (idx / (count - 1)) * 2.3;

      runningLogPos += step.countInDoc * step.logPPos;
      runningLogNeg += step.countInDoc * step.logPNeg;
      const maxL = Math.max(runningLogPos, runningLogNeg);
      const pPos = Math.exp(runningLogPos - maxL) / (Math.exp(runningLogPos - maxL) + Math.exp(runningLogNeg - maxL));

      let hPos = 0.1;
      let hNeg = 0.1;

      if (view3dSlice === 'prob_bars') {
        hPos = (step.pGivenPos / maxProb) * 1.15;
        hNeg = (step.pGivenNeg / maxProb) * 1.15;
      } else if (view3dSlice === 'log_odds') {
        hPos = Math.max(0.08, ((12 + Math.max(-11.5, step.logPPos)) / 12) * 1.15);
        hNeg = Math.max(0.08, ((12 + Math.max(-11.5, step.logPNeg)) / 12) * 1.15);
      } else {
        hPos = pPos * 1.18;
        hNeg = (1 - pPos) * 1.18;
      }

      const topNegPt = draw3DBar(
        xPos,
        -0.32,
        0.16,
        0.18,
        hNeg,
        '#38BDF8',
        '#0284C7',
        '#0369A1'
      );

      const topPosPt = draw3DBar(
        xPos,
        0.32,
        0.16,
        0.18,
        hPos,
        '#F87171',
        '#E11D48',
        '#BE123C'
      );

      ctx.font = '600 10px "IBM Plex Mono", monospace';
      ctx.fillStyle = '#0369A1';
      ctx.fillText(
        view3dSlice === 'prob_bars'
          ? `${(step.pGivenNeg * 100).toFixed(1)}%`
          : view3dSlice === 'log_odds'
          ? step.logPNeg.toFixed(1)
          : `${((1 - pPos) * 100).toFixed(0)}%`,
        topNegPt.sx - 10,
        topNegPt.sy - 6
      );

      ctx.fillStyle = '#BE123C';
      ctx.fillText(
        view3dSlice === 'prob_bars'
          ? `${(step.pGivenPos * 100).toFixed(1)}%`
          : view3dSlice === 'log_odds'
          ? step.logPPos.toFixed(1)
          : `${(pPos * 100).toFixed(0)}%`,
        topPosPt.sx - 10,
        topPosPt.sy - 6
      );

      const labelPt = project3D(xPos, -0.05, 0.72);
      ctx.font = '600 12px "Noto Sans SC", sans-serif';
      ctx.fillStyle = step.isZeroInPos || step.isZeroInNeg ? '#D97706' : '#0F172A';
      ctx.fillText(step.token, labelPt.sx - 14, labelPt.sy + 14);

      const ribbonPt = project3D(xPos, pPos * 1.22 + 0.08, 0);
      posteriorRibbonPoints.push({ sx: ribbonPt.sx, sy: ribbonPt.sy, token: step.token, post: pPos });
    });

    if (posteriorRibbonPoints.length > 1) {
      ctx.strokeStyle = '#059669';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      posteriorRibbonPoints.forEach((pt, i) => {
        if (i === 0) ctx.moveTo(pt.sx, pt.sy);
        else ctx.lineTo(pt.sx, pt.sy);
      });
      ctx.stroke();

      posteriorRibbonPoints.forEach((pt) => {
        ctx.fillStyle = '#10B981';
        ctx.beginPath();
        ctx.arc(pt.sx, pt.sy, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
  }, [activeModule, yaw, pitch, zoom3d, inference, animStepCount, view3dSlice]);

  useEffect(() => {
    if (activeModule !== 4) return;
    let rafId: number;
    const animate = () => {
      setAnimPhase((prev) => (prev + 0.025) % (Math.PI * 2));
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [activeModule]);

  useEffect(() => {
    if (activeModule !== 4) return;
    const canvas = canvas2dRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 720;
    const height = canvas.clientHeight || 410;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#F8FAFC';
    ctx.fillRect(0, 0, width, height);

    const padLeft = 64;
    const padBottom = 52;
    const padTop = 32;
    const padRight = 36;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const x = padLeft + (plotW * i) / 5;
      const y = padTop + (plotH * i) / 5;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }

    const vX = scenario.vocab.find((v) => v.token === featureX) || scenario.vocab[0];
    const vY = scenario.vocab.find((v) => v.token === featureY) || scenario.vocab[1];

    const posCenterX = padLeft + plotW * 0.72;
    const posCenterY = padTop + plotH * 0.28;
    const negCenterX = padLeft + plotW * 0.28;
    const negCenterY = padTop + plotH * 0.72;

    const drawContourEllipses = (cx: number, cy: number, colorRgb: string, angleRad: number) => {
      [1.0, 0.65, 0.35].forEach((scaleR, i) => {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angleRad);
        ctx.beginPath();
        ctx.ellipse(0, 0, 88 * scaleR, 54 * scaleR * (1 - correlationFactor * 0.45), 0, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${colorRgb}, ${0.06 + i * 0.04})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(${colorRgb}, ${0.28 + i * 0.15})`;
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.restore();
      });
    };

    const tiltAngle = -correlationFactor * (Math.PI / 4);
    drawContourEllipses(posCenterX, posCenterY, '225, 29, 72', tiltAngle);
    drawContourEllipses(negCenterX, negCenterY, '2, 132, 199', tiltAngle);

    const queryX = padLeft + plotW * (0.62 + 0.04 * Math.sin(animPhase));
    const queryY = padTop + plotH * (0.36 + 0.04 * Math.cos(animPhase));

    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#0F172A';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(queryX, queryY);
    ctx.lineTo(queryX, padTop + plotH);
    ctx.moveTo(queryX, queryY);
    ctx.lineTo(padLeft, queryY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(225, 29, 72, 0.18)';
    ctx.beginPath();
    ctx.arc(queryX, padTop + plotH, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#BE123C';
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    ctx.fillText(`投影 P(${vX.token}|Y)`, queryX - 38, padTop + plotH + 20);

    ctx.fillStyle = 'rgba(2, 132, 199, 0.18)';
    ctx.beginPath();
    ctx.arc(padLeft, queryY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0369A1';
    ctx.fillText(`P(${vY.token}|Y)`, 6, queryY + 4);

    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(padLeft + plotW * 0.15, padTop + plotH * 0.12);
    ctx.lineTo(padLeft + plotW * 0.88, padTop + plotH * 0.88);
    ctx.stroke();

    ctx.fillStyle = '#065F46';
    ctx.font = '600 11px "Noto Sans SC", sans-serif';
    ctx.fillText(
      '朴素正交假设决策超平面: ln P(Y=1) + ∑ ln P(X_i|Y=1) = ln P(Y=0) + ∑ ln P(X_i|Y=0)',
      padLeft + 18,
      padTop + 22
    );

    if (correlationFactor > 0.05) {
      ctx.strokeStyle = '#D97706';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      const shift = correlationFactor * 95;
      ctx.moveTo(padLeft + plotW * 0.15 + shift, padTop + plotH * 0.12);
      ctx.lineTo(padLeft + plotW * 0.88 - shift, padTop + plotH * 0.88);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#B45309';
      ctx.fillText(
        `特征共现偏移边界 (相关系数 r=${correlationFactor.toFixed(2)} 导致概率膨胀区)`,
        padLeft + 18,
        padTop + 42
      );
    }

    ctx.fillStyle = '#0F172A';
    ctx.beginPath();
    ctx.arc(queryX, queryY, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#0F172A';
    ctx.font = '600 12px "Noto Sans SC", sans-serif';
    ctx.fillText(`待测文本向量 X(${vX.token}, ${vY.token})`, queryX + 10, queryY - 8);

    ctx.fillStyle = '#334155';
    ctx.font = '600 12px "Noto Sans SC", sans-serif';
    ctx.fillText(`特征轴 X₁：「${vX.token}」TF-IDF 响应强度 →`, padLeft + plotW * 0.55, height - 10);
  }, [activeModule, featureX, featureY, correlationFactor, animPhase, scenario]);

  const handleMouseDown3D = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    setAutoRotate(false);
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };
  const handleMouseMove3D = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - lastMouseRef.current.x;
    const dy = e.clientY - lastMouseRef.current.y;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
    setYaw((prev) => prev + dx * 0.01);
    setPitch((prev) => Math.max(-0.2, Math.min(1.15, prev + dy * 0.01)));
  };
  const handleMouseUp3D = () => {
    isDraggingRef.current = false;
  };

  if (activeModule === 1) {
    const rawUnderflowProduct = Math.pow(avgTokenProb, simTokenCount);
    const isUnderflowZero = rawUnderflowProduct === 0 || rawUnderflowProduct < 1e-300;
    const logSumValue = simTokenCount * Math.log(avgTokenProb);

    return (
      <div className="space-y-6">
        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 01</span>
              <span aria-hidden="true">·</span>
              <span>形式化贝叶斯后验推导</span>
              <span aria-hidden="true">·</span>
              <span>对数似然防下溢机理</span>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
              代数建模与贝叶斯定理：从连乘坍缩到对数线性空间
            </h2>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg self-start">
            <button
              onClick={() => setAlgSlice('posterior')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                algSlice === 'posterior'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片一：后验概率解剖
            </button>
            <button
              onClick={() => setAlgSlice('log_derivation')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                algSlice === 'log_derivation'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片二：对数似然推导
            </button>
            <button
              onClick={() => setAlgSlice('underflow_sim')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                algSlice === 'underflow_sim'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片三：浮点下溢实验台
            </button>
          </div>
        </div>

        {algSlice === 'posterior' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 bg-white border border-slate-200/90 rounded-xl p-6 space-y-6">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <h3 className="text-base font-semibold text-slate-900">
                  点击公式任意代数因子，查看当前文本实时数值映射
                </h3>
                <span className="text-xs text-slate-500">点击因子高亮解构</span>
              </div>

              <div className="bg-[#F8FAFC] border border-slate-200/80 rounded-xl p-6 flex flex-col items-center justify-center min-h-[210px]">
                <div className="flex flex-wrap items-center justify-center gap-3 text-lg md:text-xl font-mono-tabular">
                  <button
                    onClick={() => setSelectedTerm('posterior')}
                    className={`px-3 py-2 rounded-lg border transition-all ${
                      selectedTerm === 'posterior'
                        ? 'bg-emerald-50 border-emerald-500 text-emerald-900 font-semibold'
                        : 'bg-white border-slate-200 text-slate-800 hover:border-slate-400'
                    }`}
                  >
                    <KatexMath math="P(Y \mid X)" />
                  </button>

                  <span className="text-slate-400 font-bold">=</span>

                  <div className="flex flex-col items-center">
                    <div className="flex items-center gap-2 pb-2 border-b-2 border-slate-700 px-3">
                      <button
                        onClick={() => setSelectedTerm('prior')}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          selectedTerm === 'prior'
                            ? 'bg-sky-50 border-sky-500 text-sky-900 font-semibold'
                            : 'bg-white border-slate-200 text-slate-800 hover:border-slate-400'
                        }`}
                      >
                        <KatexMath math="P(Y)" />
                      </button>
                      <span className="text-slate-500">·</span>
                      <button
                        onClick={() => setSelectedTerm('likelihood')}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          selectedTerm === 'likelihood'
                            ? 'bg-rose-50 border-rose-500 text-rose-900 font-semibold'
                            : 'bg-white border-slate-200 text-slate-800 hover:border-slate-400'
                        }`}
                      >
                        <KatexMath math="\prod_{i=1}^n P(X_i \mid Y)" />
                      </button>
                    </div>

                    <div className="pt-2">
                      <button
                        onClick={() => setSelectedTerm('evidence')}
                        className={`px-4 py-1 rounded-lg border text-base transition-all ${
                          selectedTerm === 'evidence'
                            ? 'bg-amber-50 border-amber-500 text-amber-900 font-semibold'
                            : 'bg-white border-slate-200 text-slate-700 hover:border-slate-400'
                        }`}
                      >
                        <KatexMath math="P(X) = \sum_y P(Y=y) \prod_{i=1}^n P(X_i \mid Y=y)" />
                      </button>
                    </div>
                  </div>
                </div>

                <div className="mt-6 w-full pt-4 border-t border-slate-200/80 flex flex-wrap items-center justify-between gap-2 text-xs font-mono-tabular text-slate-600">
                  <div>
                    正类后验 P(Y=1|X) ={' '}
                    <strong className="text-rose-700">
                      {(inference.posteriorPos * 100).toFixed(2)}%
                    </strong>
                  </div>
                  <div>
                    负类后验 P(Y=0|X) ={' '}
                    <strong className="text-sky-700">
                      {(inference.posteriorNeg * 100).toFixed(2)}%
                    </strong>
                  </div>
                  <div>
                    有效词元数 n = <strong>{inference.steps.length}</strong>
                  </div>
                </div>
              </div>

              <div>
                <div className="text-xs font-medium text-slate-500 mb-2.5">
                  当前文本各分词条件似然比 P(Xᵢ | Y=1) vs P(Xᵢ | Y=0) 切片明细：
                </div>
                <div className="overflow-x-auto border border-slate-200 rounded-lg">
                  <table className="w-full text-left text-xs font-mono-tabular">
                    <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                      <tr>
                        <th className="py-2 px-3 font-sans">特征词 Xᵢ</th>
                        <th className="py-2 px-3">P(Xᵢ | 正类)</th>
                        <th className="py-2 px-3">P(Xᵢ | 负类)</th>
                        <th className="py-2 px-3">对数似然比 LLR</th>
                        <th className="py-2 px-3 font-sans">倾向贡献</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {inference.steps.map((s) => (
                        <tr key={s.token} className="hover:bg-slate-50/80">
                          <td className="py-2 px-3 font-sans font-medium text-slate-900">
                            {s.token}
                            {s.countInDoc > 1 ? ` ×${s.countInDoc}` : ''}
                          </td>
                          <td className="py-2 px-3 text-rose-700">
                            {s.pGivenPos.toFixed(5)}
                          </td>
                          <td className="py-2 px-3 text-sky-700">
                            {s.pGivenNeg.toFixed(5)}
                          </td>
                          <td className="py-2 px-3 font-semibold">
                            {s.logLikelihoodRatio >= 0 ? '+' : ''}
                            {s.logLikelihoodRatio.toFixed(3)}
                          </td>
                          <td className="py-2 px-3 font-sans">
                            {s.logLikelihoodRatio > 0.15 ? (
                              <span className="text-rose-700 font-medium">
                                ▲ 指向{scenario.posLabel.split(' ')[0]}
                              </span>
                            ) : s.logLikelihoodRatio < -0.15 ? (
                              <span className="text-sky-700 font-medium">
                                ▼ 指向{scenario.negLabel.split(' ')[0]}
                              </span>
                            ) : (
                              <span className="text-slate-500">● 中性背景</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="lg:col-span-5 bg-white border border-slate-200/90 rounded-xl p-6 flex flex-col justify-between space-y-5">
              {selectedTerm === 'posterior' && (
                <div className="space-y-4">
                  <div className="text-xs font-medium text-emerald-700">
                    当前选中因子 · 后验概率 P(Y | X)
                  </div>
                  <h4 className="text-lg font-semibold text-slate-900">
                    给定观测文本向量 X 后，类别 Y 的条件归属概率
                  </h4>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    在分类决策阶段，我们采用<strong>最大后验概率准则（MAP, Maximum A Posteriori）</strong>：比较正负类别的后验概率大小，将文本指派给后验概率最大的类别：
                  </p>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 font-mono-tabular text-xs space-y-2">
                    <div className="py-1">
                      <KatexMath math="\hat{Y} = \arg\max_{Y} P(Y \mid X)" />
                    </div>
                    <div className="text-rose-700">
                      P({scenario.posLabel} | X) = {(inference.posteriorPos * 100).toFixed(4)}%
                    </div>
                    <div className="text-sky-700">
                      P({scenario.negLabel} | X) = {(inference.posteriorNeg * 100).toFixed(4)}%
                    </div>
                  </div>
                </div>
              )}

              {selectedTerm === 'prior' && (
                <div className="space-y-4">
                  <div className="text-xs font-medium text-sky-700">
                    当前选中因子 · 类先验概率 P(Y)
                  </div>
                  <h4 className="text-lg font-semibold text-slate-900">
                    未观测任何特征词前的历史基准分布
                  </h4>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    先验概率 <KatexMath math="P(Y=c) = \frac{N_c}{N}" /> 反映了训练语料库中各类别的自然占比。当文本特征极少或模糊时，先验概率主导决策结果；若训练集严重类别不平衡，则会扭曲后验判别边界。
                  </p>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 font-mono-tabular text-xs space-y-2">
                    <div>
                      P(Y=1) = {scenario.totalPosDocs} / {scenario.totalPosDocs + scenario.totalNegDocs} ={' '}
                      <strong>{inference.priorPos.toFixed(2)}</strong> (ln P = {inference.logPriorPos.toFixed(3)})
                    </div>
                    <div>
                      P(Y=0) = {scenario.totalNegDocs} / {scenario.totalPosDocs + scenario.totalNegDocs} ={' '}
                      <strong>{inference.priorNeg.toFixed(2)}</strong> (ln P = {inference.logPriorNeg.toFixed(3)})
                    </div>
                  </div>
                </div>
              )}

              {selectedTerm === 'likelihood' && (
                <div className="space-y-4">
                  <div className="text-xs font-medium text-rose-700">
                    当前选中因子 · 类条件似然连乘 ∏ P(Xᵢ | Y)
                  </div>
                  <h4 className="text-lg font-semibold text-slate-900">
                    朴素条件独立假设的核心降维引擎
                  </h4>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    精确估计联合分布 P(X₁, X₂, …, Xₙ | Y) 需要指数级参数空间 O(|V|ⁿ)。朴素贝叶斯引入<strong>特征条件独立假设</strong>，将其分解为各分词边缘条件概率的连乘积：
                  </p>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 font-mono-tabular text-xs space-y-2">
                    <div className="py-1">
                      <KatexMath math="P(X_1, X_2, \dots, X_n \mid Y) \approx \prod_{i=1}^n P(X_i \mid Y)" />
                    </div>
                    <div className="text-rose-700">
                      正类连乘积 ∏ P(Xᵢ|Y=1) = {inference.rawProductPos.toExponential(4)}
                    </div>
                    <div className="text-sky-700">
                      负类连乘积 ∏ P(Xᵢ|Y=0) = {inference.rawProductNeg.toExponential(4)}
                    </div>
                  </div>
                </div>
              )}

              {selectedTerm === 'evidence' && (
                <div className="space-y-4">
                  <div className="text-xs font-medium text-amber-700">
                    当前选中因子 · 全概率证据因子 P(X)
                  </div>
                  <h4 className="text-lg font-semibold text-slate-900">
                    与类别 Y 无关的归一化常数
                  </h4>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    分母 <KatexMath math="P(X) = \sum_{y} P(Y=y)\prod_{i=1}^n P(X_i \mid Y=y)" /> 对所有候选类别完全相同。因此在做分类排序时可直接约去分母，仅需在输出校准概率百分比时通过 Log-Sum-Exp 还原。
                  </p>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 font-mono-tabular text-xs space-y-2">
                    <div className="py-1">
                      <KatexMath math="P(X) = P(Y=1)\prod_{i=1}^n P(X_i \mid 1) + P(Y=0)\prod_{i=1}^n P(X_i \mid 0)" />
                    </div>
                    <div>
                      当前样本 P(X) = {(inference.rawProductPos + inference.rawProductNeg).toExponential(4)}
                    </div>
                  </div>
                </div>
              )}

              <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
                <span className="text-xs text-slate-500">
                  对当前样本独立性假设有疑问？
                </span>
                <button
                  onClick={() =>
                    onOpenAiModal(
                      `请结合当前文本「${inputText}」推导其后验对数似然得分，并解释为何分母 P(X) 可以约去。`
                    )
                  }
                  className="px-3.5 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap"
                >
                  唤起 AI 随诊推导 →
                </button>
              </div>
            </div>
          </div>
        )}

        {algSlice === 'log_derivation' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
            <div className="border-b border-slate-100 pb-3">
              <h3 className="text-base font-semibold text-slate-900">
                四步代数变换切片：从高维概率连乘到线性对数加权求和
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                利用自然对数函数 ln(x) 在 (0, +∞) 上的严格单调递增性，将乘法运算同构映射为加法运算
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="text-xs font-semibold text-slate-500">
                  步骤 01 · 贝叶斯最优决策原始形式
                </div>
                <div className="py-2">
                  <KatexMath math="\hat{Y} = \arg\max_{Y} \left[ \frac{P(Y) \prod_{i=1}^n P(X_i \mid Y)}{P(X)} \right] = \arg\max_{Y} \left[ P(Y) \prod_{i=1}^n P(X_i \mid Y) \right]" block />
                </div>
                <p className="text-xs text-slate-600">
                  由于分母 P(X) 与类别变量 Y 无关，消去分母不改变 argmax 的极值点位置。
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="text-xs font-semibold text-slate-500">
                  步骤 02 · 施加严格单调递增自然对数 ln(·)
                </div>
                <div className="py-2">
                  <KatexMath math="\hat{Y} = \arg\max_{Y} \ln \left[ P(Y) \prod_{i=1}^n P(X_i \mid Y) \right]" block />
                </div>
                <p className="text-xs text-slate-600">
                  因为 a &gt; b &gt; 0 ⇔ ln(a) &gt; ln(b)，取对数后最优分类边界保持 100% 代数等价。
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="text-xs font-semibold text-slate-500">
                  步骤 03 · 积的对数等于对数的和
                </div>
                <div className="py-2">
                  <KatexMath math="\ell(Y) = \ln P(Y) + \sum_{i=1}^n \text{count}(X_i) \cdot \ln P(X_i \mid Y)" block />
                </div>
                <p className="text-xs text-slate-600">
                  连乘积 ∏ 彻底转化为线性累加 ∑，形式上等价于带截距 ln P(Y) 的广义线性分类器！
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="text-xs font-semibold text-slate-500">
                  步骤 04 · Log-Sum-Exp 数值稳定概率还原
                </div>
                <div className="py-2">
                  <KatexMath math="P(Y=1 \mid X) = \frac{\exp(\ell_1 - M)}{\exp(\ell_1 - M) + \exp(\ell_0 - M)}, \quad M = \max(\ell_1, \ell_0)" block />
                </div>
                <p className="text-xs text-slate-600">
                  令 M = max(ℓ₁, ℓ₀)，先平移再求指数，彻底杜绝 exp(-900) 下溢为 0。
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#F8FAFC] border border-slate-200 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="text-xs font-medium text-slate-500">
                  当前待测文本实时对数似然累加验证：
                </div>
                <div className="text-xs font-mono-tabular text-slate-800">
                  ℓ(正类) = {inference.logPriorPos.toFixed(3)} + ∑ ln P(Xᵢ|1) ={' '}
                  <strong className="text-rose-700">{inference.totalLogPos.toFixed(3)}</strong>
                  <span className="mx-3">|</span>
                  ℓ(负类) = {inference.logPriorNeg.toFixed(3)} + ∑ ln P(Xᵢ|0) ={' '}
                  <strong className="text-sky-700">{inference.totalLogNeg.toFixed(3)}</strong>
                </div>
              </div>
              <div className="text-xs font-mono-tabular px-3 py-2 bg-white border border-slate-200 rounded-lg">
                Δ 对数奇数比 (Log-Odds) ={' '}
                <strong className="text-emerald-700">
                  {(inference.totalLogPos - inference.totalLogNeg).toFixed(3)}
                </strong>
              </div>
            </div>
          </div>
        )}

        {algSlice === 'underflow_sim' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div>
                <h3 className="text-base font-semibold text-slate-900">
                  IEEE-754 双精度浮点数下溢（Underflow）动态压测切片
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  拖动文档词元长度 n 与单字平均条件概率 P(Xᵢ | Y)，观察为何长文本必须采用对数似然变换
                </p>
              </div>
              <div className="text-xs font-mono-tabular text-slate-500">
                IEEE-754 最小正数极限 ≈ 2.22 × 10⁻³⁰⁸
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs font-medium text-slate-700 mb-1.5">
                    <span>文本有效特征词元数量 (n)</span>
                    <span className="font-mono-tabular font-semibold">{simTokenCount} 个词</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={150}
                    step={1}
                    value={simTokenCount}
                    onChange={(e) => setSimTokenCount(Number(e.target.value))}
                    className="w-full accent-slate-900 cursor-pointer"
                  />
                  <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                    <span>短短信 (5词)</span>
                    <span>中长邮件 (75词)</span>
                    <span>长篇新闻 (150词)</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-medium text-slate-700 mb-1.5">
                    <span>单特征词平均条件概率 P(Xᵢ | Y)</span>
                    <span className="font-mono-tabular font-semibold">{avgTokenProb.toFixed(4)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0005}
                    max={0.01}
                    step={0.0005}
                    value={avgTokenProb}
                    onChange={(e) => setAvgTokenProb(Number(e.target.value))}
                    className="w-full accent-slate-900 cursor-pointer"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div
                  className={`p-4 rounded-xl border ${
                    isUnderflowZero
                      ? 'bg-rose-50/70 border-rose-300'
                      : 'bg-slate-50 border-slate-200'
                  } flex flex-col justify-between`}
                >
                  <div className="text-xs font-medium text-slate-600">
                    直接连乘 ∏ᵢ₌₁ⁿ P(Xᵢ | Y)
                  </div>
                  <div className="my-3 font-mono-tabular text-lg font-semibold text-slate-900 break-all">
                    {isUnderflowZero ? '0.000000e+0 (下溢归零!)' : rawUnderflowProduct.toExponential(4)}
                  </div>
                  <div className="text-xs flex items-center gap-1.5">
                    {isUnderflowZero ? (
                      <span className="text-rose-700 font-medium flex items-center gap-1">
                        <AlertTriangle className="w-3.5 h-3.5" /> 触发浮点下溢，产生 0/0 NaN 崩溃
                      </span>
                    ) : (
                      <span className="text-slate-500">尚未突破 10⁻³⁰⁸ 极限，但指数衰减极快</span>
                    )}
                  </div>
                </div>

                <div className="p-4 rounded-xl border bg-emerald-50/50 border-emerald-200 flex flex-col justify-between">
                  <div className="text-xs font-medium text-emerald-900">
                    对数似然累加 ∑ᵢ₌₁ⁿ ln P(Xᵢ | Y)
                  </div>
                  <div className="my-3 font-mono-tabular text-lg font-semibold text-emerald-800">
                    {logSumValue.toFixed(4)}
                  </div>
                  <div className="text-xs text-emerald-700 font-medium flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> 线性增长，百万词长文档依然数值稳定
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (activeModule === 2) {
    return (
      <div className="space-y-6">
        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 02</span>
              <span aria-hidden="true">·</span>
              <span>3D 手势旋转舞台</span>
              <span aria-hidden="true">·</span>
              <span>特征词分布升降与后验累加动画</span>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
              词频条件概率 3D 演播：双类别特征似然柱群与后验轨迹
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              <button
                onClick={() => setView3dSlice('prob_bars')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  view3dSlice === 'prob_bars'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                切片一：条件概率 P(Xᵢ|Y)
              </button>
              <button
                onClick={() => setView3dSlice('log_odds')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  view3dSlice === 'log_odds'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                切片二：对数似然模长
              </button>
              <button
                onClick={() => setView3dSlice('cumulative_posterior')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  view3dSlice === 'cumulative_posterior'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                切片三：后验概率逐词演进
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8 bg-white border border-slate-200/90 rounded-xl p-4 flex flex-col">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
              <div className="flex items-center gap-4 text-xs">
                <span className="flex items-center gap-1.5 font-medium text-rose-700">
                  <span className="w-2.5 h-2.5 rounded-xs bg-rose-600 inline-block" />
                  前排柱：{scenario.posLabel}
                </span>
                <span className="flex items-center gap-1.5 font-medium text-sky-700">
                  <span className="w-2.5 h-2.5 rounded-xs bg-sky-600 inline-block" />
                  后排柱：{scenario.negLabel}
                </span>
                <span className="flex items-center gap-1.5 font-medium text-emerald-700">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 inline-block" />
                  绿线：正类后验累加轨迹
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => setAutoRotate(!autoRotate)}
                  className="px-2.5 py-1 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-md flex items-center gap-1 whitespace-nowrap"
                >
                  {autoRotate ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {autoRotate ? '暂停环绕' : '自动环绕'}
                </button>
                <button
                  onClick={() => {
                    setYaw(-0.55);
                    setPitch(0.42);
                    setZoom3d(1.0);
                  }}
                  className="px-2.5 py-1 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-md flex items-center gap-1 whitespace-nowrap"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  复位视角
                </button>
              </div>
            </div>

            <div className="relative flex-1 min-h-[410px] rounded-lg overflow-hidden border border-slate-200/70">
              <canvas
                ref={canvas3dRef}
                onMouseDown={handleMouseDown3D}
                onMouseMove={handleMouseMove3D}
                onMouseUp={handleMouseUp3D}
                onMouseLeave={handleMouseUp3D}
                className="w-full h-[410px] cursor-grab active:cursor-grabbing block"
              />
              <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-xs border border-slate-200 rounded-lg px-3 py-1.5 text-[11px] text-slate-600 pointer-events-none">
                提示：按住鼠标左键拖拽可自由 3D 旋转视角 · 实时映射平滑系数 α = {alpha.toFixed(2)}
              </div>
            </div>
          </div>

          <div className="lg:col-span-4 bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col justify-between space-y-5">
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-900 border-b border-slate-100 pb-2.5">
                3D 几何相机与逐词累加演播控制
              </h3>

              <div>
                <div className="flex justify-between text-xs text-slate-600 mb-1">
                  <span>逐词演播进度 (前 k 个特征词)</span>
                  <span className="font-mono-tabular font-semibold">
                    {Math.min(inference.steps.length, animStepCount)} / {inference.steps.length}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={Math.max(1, inference.steps.length)}
                  value={Math.min(inference.steps.length, animStepCount)}
                  onChange={(e) => setAnimStepCount(Number(e.target.value))}
                  className="w-full accent-slate-900 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-600 mb-1">
                  <span>水平旋转角 Yaw (弧度)</span>
                  <span className="font-mono-tabular">{yaw.toFixed(2)} rad</span>
                </div>
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.05}
                  value={yaw}
                  onChange={(e) => {
                    setAutoRotate(false);
                    setYaw(Number(e.target.value));
                  }}
                  className="w-full accent-slate-900 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-600 mb-1">
                  <span>俯仰倾角 Pitch (弧度)</span>
                  <span className="font-mono-tabular">{pitch.toFixed(2)} rad</span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={1.05}
                  step={0.02}
                  value={pitch}
                  onChange={(e) => {
                    setAutoRotate(false);
                    setPitch(Number(e.target.value));
                  }}
                  className="w-full accent-slate-900 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-600 mb-1">
                  <span>拉普拉斯平滑系数 (α)</span>
                  <span className="font-mono-tabular font-semibold">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={0.1}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full accent-slate-900 cursor-pointer"
                />
              </div>

              <div className="pt-2">
                <div className="text-xs text-slate-500 mb-2">快速相机切片视角：</div>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => {
                      setAutoRotate(false);
                      setYaw(-0.55);
                      setPitch(0.42);
                    }}
                    className="px-2.5 py-1.5 text-xs bg-slate-100 hover:bg-slate-200 rounded-md text-slate-700 whitespace-nowrap"
                  >
                    等轴测 3D
                  </button>
                  <button
                    onClick={() => {
                      setAutoRotate(false);
                      setYaw(0);
                      setPitch(0.12);
                    }}
                    className="px-2.5 py-1.5 text-xs bg-slate-100 hover:bg-slate-200 rounded-md text-slate-700 whitespace-nowrap"
                  >
                    正视高度比
                  </button>
                  <button
                    onClick={() => {
                      setAutoRotate(false);
                      setYaw(0.2);
                      setPitch(0.95);
                    }}
                    className="px-2.5 py-1.5 text-xs bg-slate-100 hover:bg-slate-200 rounded-md text-slate-700 whitespace-nowrap"
                  >
                    俯视矩阵
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-1.5">
              <div className="text-xs font-medium text-slate-700">
                实时后验概率收敛状态
              </div>
              <div className="flex items-center justify-between text-xs font-mono-tabular">
                <span className="text-rose-700">
                  P(正类|X) = {(inference.posteriorPos * 100).toFixed(2)}%
                </span>
                <span className="text-sky-700">
                  P(负类|X) = {(inference.posteriorNeg * 100).toFixed(2)}%
                </span>
              </div>
              <div className="w-full h-2 bg-sky-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-rose-600 transition-all duration-200"
                  style={{ width: `${Math.min(100, Math.max(0, inference.posteriorPos * 100))}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (activeModule === 3) {
    const alphaSamples = [0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];

    return (
      <div className="space-y-6">
        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 03</span>
              <span aria-hidden="true">·</span>
              <span>零概率陷阱修复</span>
              <span aria-hidden="true">·</span>
              <span>Lidstone / Laplace 平滑切片</span>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
              拉普拉斯平滑 (α) 动态切片：消除未登录词“一票否决”坍缩
            </h2>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg self-start">
            <button
              onClick={() => setSmoothSlice('reservoir')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                smoothSlice === 'reservoir'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片一：水蓄池削峰填谷
            </button>
            <button
              onClick={() => setSmoothSlice('zero_trap')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                smoothSlice === 'zero_trap'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片二：零概率陷阱对比
            </button>
            <button
              onClick={() => setSmoothSlice('conservation')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                smoothSlice === 'conservation'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片三：概率质量守恒机理
            </button>
            <button
              onClick={() => setSmoothSlice('alpha_curve')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                smoothSlice === 'alpha_curve'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片四：α 连续演化曲线
            </button>
          </div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-5 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <div className="lg:col-span-7 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-900">
                滑动调节平滑系数 α（当前 α = <span className="font-mono-tabular text-sky-700">{alpha.toFixed(2)}</span>）
              </span>
              <div className="flex items-center gap-1.5">
                {[
                  { label: 'α=0 (无平滑陷阱)', val: 0 },
                  { label: 'α=0.1 (Lidstone)', val: 0.1 },
                  { label: 'α=1.0 (标准拉普拉斯)', val: 1.0 },
                  { label: 'α=3.0 (强平滑)', val: 3.0 },
                ].map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => setAlpha(preset.val)}
                    className={`px-2.5 py-1 text-xs rounded-md border transition-colors whitespace-nowrap ${
                      Math.abs(alpha - preset.val) < 0.01
                        ? 'bg-slate-900 text-white border-slate-900'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
            <input
              type="range"
              min={0}
              max={5}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              className="w-full accent-slate-900 cursor-pointer"
            />
            <div className="flex justify-between text-xs text-slate-500">
              <span>α = 0.00（极大似然估计 MLE，遇零即崩）</span>
              <span>α = 1.00（Add-1 Laplace 均匀Dirichlet先验）</span>
              <span>α = 5.00（过度平滑，趋向均匀分布）</span>
            </div>
          </div>

          <div className="lg:col-span-5 bg-slate-50 border border-slate-200 rounded-xl p-4 font-mono-tabular text-xs space-y-2">
            <div className="text-slate-500 font-sans font-medium">
              拉普拉斯修正代数公式：
            </div>
            <div className="py-1">
              <KatexMath math="P(X_i \mid Y) = \frac{N_{Y, i} + \alpha}{N_Y + \alpha \cdot |V|}" block />
            </div>
            <div className="text-slate-600 flex flex-wrap gap-x-4 gap-y-1 pt-1">
              <span>正类总词频 N₁ = {inference.totalPosTokens}</span>
              <span>负类总词频 N₀ = {inference.totalNegTokens}</span>
              <span>词表维度 |V| = {inference.vocabSize}</span>
            </div>
          </div>
        </div>

        {smoothSlice === 'reservoir' && (
          <WaterFillingReservoir
            scenario={scenario}
            alpha={alpha}
            setAlpha={setAlpha}
          />
        )}

        {smoothSlice === 'zero_trap' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-100 pb-4">
              <div>
                <h3 className="text-base font-semibold text-slate-900">
                  未登录词 / 零频词“一票否决”现象与平滑修复对比表
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  当待测文本出现某个类别训练集中未曾见过的词（N_&#123;Y,i&#125; = 0）时，α=0 会令该项概率为 0，进而抹杀全部其它特征词的贡献
                </p>
              </div>
              <button
                onClick={() => {
                  const oovSample = scenario.samples.find((s) =>
                    s.note?.includes('未登录词')
                  );
                  if (oovSample) setInputText(oovSample.text);
                }}
                className="px-3.5 py-2 text-xs font-medium bg-amber-50 text-amber-900 border border-amber-300 rounded-lg hover:bg-amber-100 transition-colors whitespace-nowrap"
              >
                一键载入含未登录词测试样本
              </button>
            </div>

            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="w-full text-left text-xs font-mono-tabular">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3 font-sans">分词 Xᵢ</th>
                    <th className="py-2.5 px-3">正类频次 N₁,ᵢ</th>
                    <th className="py-2.5 px-3">负类频次 N₀,ᵢ</th>
                    <th className="py-2.5 px-3">无平滑 P(Xᵢ|1) (α=0)</th>
                    <th className="py-2.5 px-3">平滑后 P(Xᵢ|1) (α={alpha.toFixed(2)})</th>
                    <th className="py-2.5 px-3">无平滑 P(Xᵢ|0) (α=0)</th>
                    <th className="py-2.5 px-3">平滑后 P(Xᵢ|0) (α={alpha.toFixed(2)})</th>
                    <th className="py-2.5 px-3 font-sans">诊断状态</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {inference.steps.map((s) => {
                    const hasZero = s.isZeroInPos || s.isZeroInNeg;
                    return (
                      <tr
                        key={s.token}
                        className={hasZero ? 'bg-amber-50/60' : 'hover:bg-slate-50'}
                      >
                        <td className="py-2.5 px-3 font-sans font-semibold text-slate-900">
                          {s.token}
                        </td>
                        <td className="py-2.5 px-3">{s.posCount}</td>
                        <td className="py-2.5 px-3">{s.negCount}</td>
                        <td
                          className={`py-2.5 px-3 ${
                            s.isZeroInPos ? 'text-rose-600 font-bold' : 'text-slate-600'
                          }`}
                        >
                          {s.pGivenPosZeroAlpha.toFixed(5)}
                        </td>
                        <td className="py-2.5 px-3 text-emerald-700 font-semibold">
                          {s.pGivenPos.toFixed(5)}
                        </td>
                        <td
                          className={`py-2.5 px-3 ${
                            s.isZeroInNeg ? 'text-rose-600 font-bold' : 'text-slate-600'
                          }`}
                        >
                          {s.pGivenNegZeroAlpha.toFixed(5)}
                        </td>
                        <td className="py-2.5 px-3 text-emerald-700 font-semibold">
                          {s.pGivenNeg.toFixed(5)}
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          {hasZero ? (
                            alpha > 0 ? (
                              <span className="text-emerald-700 font-medium">
                                ✓ 已由 α 注入先验修复
                              </span>
                            ) : (
                              <span className="text-rose-700 font-semibold">
                                ⚠ 零概率坍缩触发!
                              </span>
                            )
                          ) : (
                            <span className="text-slate-500">常规词频估计</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {smoothSlice === 'conservation' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              为什么分子加 α，分母必须加 α · |V|？（全概率归一守恒证明切片）
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              初学者常问：为什么不能只在分子加 α？因为条件概率分布必须满足<strong>柯尔莫哥洛夫公理归一化约束</strong>：词表中所有 |V| 个词元在给定类别 Y 下的概率之和必须恒等于 1：
            </p>
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 font-mono-tabular text-sm space-y-2 text-slate-800">
              <div className="py-1">
                <KatexMath
                  math="\sum_{i=1}^{|V|} P(X_i \mid Y) = \sum_{i=1}^{|V|} \frac{N_{Y, i} + \alpha}{N_Y + \alpha \cdot |V|} = \frac{\sum_{i=1}^{|V|} N_{Y, i} + \alpha \cdot |V|}{N_Y + \alpha \cdot |V|} = \frac{N_Y + \alpha \cdot |V|}{N_Y + \alpha \cdot |V|} \equiv 1"
                  block
                />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div className="p-4 rounded-lg border border-slate-200 bg-[#F8FAFC]">
                <div className="text-xs text-slate-500">当 α → 0 时</div>
                <div className="text-sm font-semibold text-slate-900 mt-1">
                  退化为极大似然估计 (MLE)
                </div>
                <p className="text-xs text-slate-600 mt-1">
                  完全信赖观测频次比 N_&#123;Y,i&#125;/N_Y，方差大，对未登录词零容忍。
                </p>
              </div>
              <div className="p-4 rounded-lg border border-slate-200 bg-[#F8FAFC]">
                <div className="text-xs text-slate-500">当 α = 1 时</div>
                <div className="text-sm font-semibold text-slate-900 mt-1">
                  拉普拉斯加一平滑 (Add-1)
                </div>
                <p className="text-xs text-slate-600 mt-1">
                  等价于在观测前先假定词表中每个词各出现过 1 次（均匀 Dirichlet 先验）。
                </p>
              </div>
              <div className="p-4 rounded-lg border border-slate-200 bg-[#F8FAFC]">
                <div className="text-xs text-slate-500">当 α → +∞ 时</div>
                <div className="text-sm font-semibold text-slate-900 mt-1">
                  趋近于均匀分布 1 / |V|
                </div>
                <p className="text-xs text-slate-600 mt-1">
                  所有特征词的条件概率趋于相等 1/|V|，似然比趋于 1，分类器退化为仅靠先验 P(Y) 盲猜。
                </p>
              </div>
            </div>
          </div>
        )}

        {smoothSlice === 'alpha_curve' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              平滑系数 α ∈ [0, 5] 连续切片下高频词与零频词概率演化对比
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-2.5 pt-2">
              {alphaSamples.map((aVal) => {
                const denom = inference.totalPosTokens + aVal * inference.vocabSize;
                const highFreqToken = scenario.vocab[0];
                const pHigh = denom > 0 ? (highFreqToken.posCount + aVal) / denom : 0;
                const pZero = denom > 0 ? (0 + aVal) / denom : 0;
                const isCurrent = Math.abs(alpha - aVal) < 0.08;
                return (
                  <button
                    key={aVal}
                    onClick={() => setAlpha(aVal)}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      isCurrent
                        ? 'bg-slate-900 text-white border-slate-900'
                        : 'bg-slate-50 text-slate-800 border-slate-200 hover:border-slate-400'
                    }`}
                  >
                    <div className="text-xs font-mono-tabular font-semibold">
                      α = {aVal}
                    </div>
                    <div className="mt-2 text-[11px] opacity-80">高频「{highFreqToken.token}」</div>
                    <div className="text-xs font-mono-tabular font-medium">
                      {(pHigh * 100).toFixed(2)}%
                    </div>
                    <div className="mt-1.5 text-[11px] opacity-80">零频词 P</div>
                    <div className="text-xs font-mono-tabular font-medium">
                      {(pZero * 100).toFixed(3)}%
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <span>模块 04</span>
            <span aria-hidden="true">·</span>
            <span>贝叶斯网络 DAG 拓扑</span>
            <span aria-hidden="true">·</span>
            <span>对数似然权重桑基流</span>
            <span aria-hidden="true">·</span>
            <span>正交投影与共现畸变</span>
          </div>
          <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
            特征条件独立性空间演播：贝叶斯网络 DAG、权重桑基流与共现畸变
          </h2>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg self-start">
          <button
            onClick={() => setIndepSlice('dag_sankey')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
              indepSlice === 'dag_sankey'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            切片一：贝叶斯网络 DAG 与桑基流
          </button>
          <button
            onClick={() => {
              setIndepSlice('orthogonal_proj');
              setCorrelationFactor(0.0);
            }}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
              indepSlice === 'orthogonal_proj'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            切片二：正交边缘独立投影 (r=0)
          </button>
          <button
            onClick={() => {
              setIndepSlice('collinear_inflation');
              setCorrelationFactor(scenario.correlatedHighlight.r);
              setFeatureX(scenario.correlatedHighlight.tokenA);
              setFeatureY(scenario.correlatedHighlight.tokenB);
            }}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
              indepSlice === 'collinear_inflation'
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            切片三：强共现膨胀畸变 (r={scenario.correlatedHighlight.r})
          </button>
        </div>
      </div>

      {indepSlice === 'dag_sankey' ? (
        <BayesianDagSankey
          scenario={scenario}
          inference={inference}
          alpha={alpha}
          inputText={inputText}
          onOpenAiModal={onOpenAiModal}
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-white border border-slate-200/90 rounded-xl p-4">
          <canvas
            ref={canvas2dRef}
            className="w-full h-[410px] rounded-lg border border-slate-200/70 block"
          />
        </div>

        <div className="lg:col-span-4 bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col justify-between space-y-5">
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-slate-900 border-b border-slate-100 pb-2.5">
              二维特征子空间投影轴与相关性调节
            </h3>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-slate-600 mb-1">横轴特征 X₁</label>
                <select
                  value={featureX}
                  onChange={(e) => setFeatureX(e.target.value)}
                  className="w-full text-xs border border-slate-200 rounded-lg px-2.5 py-2 bg-slate-50 text-slate-900"
                >
                  {scenario.vocab.map((v) => (
                    <option key={v.token} value={v.token}>
                      {v.token} (χ²={v.chiSquare})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-slate-600 mb-1">纵轴特征 X₂</label>
                <select
                  value={featureY}
                  onChange={(e) => setFeatureY(e.target.value)}
                  className="w-full text-xs border border-slate-200 rounded-lg px-2.5 py-2 bg-slate-50 text-slate-900"
                >
                  {scenario.vocab.map((v) => (
                    <option key={v.token} value={v.token}>
                      {v.token} (χ²={v.chiSquare})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-700 mb-1">
                <span>特征条件相关系数 r(X₁, X₂ | Y)</span>
                <span className="font-mono-tabular font-semibold">{correlationFactor.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.95}
                step={0.05}
                value={correlationFactor}
                onChange={(e) => setCorrelationFactor(Number(e.target.value))}
                className="w-full accent-slate-900 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                <span>r=0.0 (完全条件独立)</span>
                <span>r=0.95 (同义词完全共线)</span>
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-semibold text-slate-800">
                典型共现陷阱：{scenario.correlatedHighlight.tokenA} + {scenario.correlatedHighlight.tokenB}
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                {scenario.correlatedHighlight.explanation}
              </p>
            </div>
          </div>

          <button
            onClick={() =>
              onOpenAiModal(
                `在「${scenario.name}」场景中，特征词「${featureX}」与「${featureY}」存在相关性 r=${correlationFactor.toFixed(
                  2
                )}，请评估其对朴素贝叶斯决策边界的影响并给出卡方特征选择建议。`
              )
            }
            className="w-full py-2.5 px-4 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors"
          >
            使用 AI 诊断当前子空间条件独立性违背度
          </button>
        </div>
      </div>
      )}
    </div>
  );
};
