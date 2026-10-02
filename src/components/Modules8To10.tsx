import React, { useState } from 'react';
import {
  Download,
  Upload,
  FileText,
  FileSpreadsheet,
  Printer,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Sliders,
  BookOpen,
  Layers,
  Eye,
  Copy,
  Check,
  Database,
  Sparkles,
  BarChart3,
  FolderDown,
} from 'lucide-react';
import { BAYES_CASES, BayesCaseScenario } from '../data/bayesCases';
import {
  BayesInferenceResult,
  evaluateDataset,
} from '../utils/bayesEngine';
import { KatexMath } from './KatexMath';

interface Modules8To10Props {
  activeModule: 8 | 9 | 10;
  scenario: BayesCaseScenario;
  inference: BayesInferenceResult;
  alpha: number;
  setAlpha: (val: number) => void;
  filterStopwords: boolean;
  setFilterStopwords: (val: boolean) => void;
  customPriorPos: number | undefined;
  setCustomPriorPos: (val: number | undefined) => void;
  inputText: string;
  setInputText: (val: string) => void;
}

export const Modules8To10: React.FC<Modules8To10Props> = ({
  activeModule,
  scenario,
  inference,
  alpha,
  setAlpha,
  filterStopwords,
  setFilterStopwords,
  customPriorPos,
  setCustomPriorPos,
  inputText,
  setInputText,
}) => {
  const [pipelineStage, setPipelineStage] = useState<1 | 2 | 3 | 4>(1);
  const [reportSlice, setReportSlice] = useState<
    'report_preview' | 'cases_download' | 'confusion_roc' | 'vocab_matrix' | 'csv_upload'
  >('report_preview');
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0.5);
  const [copiedReport, setCopiedReport] = useState<boolean>(false);
  const [customCsvSamples, setCustomCsvSamples] = useState<
    Array<{ id: string; text: string; actualClass: 1 | 0 }>
  >([]);
  const [csvStatusMsg, setCsvStatusMsg] = useState<string>('');

  const [knowledgeSlice, setKnowledgeSlice] = useState<1 | 2 | 3 | 4 | 5 | 6>(1);
  const [simVocabDim, setSimVocabDim] = useState<number>(20);
  const [gaussianX, setGaussianX] = useState<number>(0.72);

  const downloadTextFile = (filename: string, content: string, mime = 'text/plain;charset=utf-8') => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportVocabCsv = () => {
    const header = 'Token,PosCount,NegCount,P_Given_Pos,P_Given_Neg,ChiSquare,MutualInfo\n';
    const denomPos = inference.totalPosTokens + alpha * inference.vocabSize;
    const denomNeg = inference.totalNegTokens + alpha * inference.vocabSize;
    const rows = scenario.vocab
      .map((v) => {
        const pPos = ((v.posCount + alpha) / denomPos).toFixed(6);
        const pNeg = ((v.negCount + alpha) / denomNeg).toFixed(6);
        return `"${v.token}",${v.posCount},${v.negCount},${pPos},${pNeg},${v.chiSquare},${v.mutualInfo}`;
      })
      .join('\n');
    downloadTextFile(`naive_bayes_vocab_${scenario.id}_alpha_${alpha.toFixed(2)}.csv`, '\uFEFF' + header + rows, 'text/csv;charset=utf-8');
  };

  const handleDownloadCaseDataset = (caseId: BayesCaseScenario['id']) => {
    const targetCase = BAYES_CASES[caseId];
    const header = 'ID,Text,ActualClass,LabelName,Note\n';
    const rows = targetCase.samples
      .map(
        (s) =>
          `"${s.id}","${s.text.replace(/"/g, '""')}",${s.actualClass},"${
            s.actualClass === 1 ? targetCase.posLabel : targetCase.negLabel
          }","${(s.note || '').replace(/"/g, '""')}"`
      )
      .join('\n');
    downloadTextFile(`dataset_${caseId}_raw.csv`, '\uFEFF' + header + rows, 'text/csv;charset=utf-8');
  };

  const handleExportDatasetCsv = () => {
    handleDownloadCaseDataset(scenario.id);
  };

  const handleDownloadCaseVocab = (caseId: BayesCaseScenario['id']) => {
    const targetCase = BAYES_CASES[caseId];
    const header = 'Token,PosCount,NegCount,ChiSquare,MutualInfo,CorrelatedWith,CorrelationCoeff\n';
    const rows = targetCase.vocab
      .map((v) => {
        return `"${v.token}",${v.posCount},${v.negCount},${v.chiSquare},${v.mutualInfo},"${v.correlatedWith || ''}",${v.correlationCoeff || 0}`;
      })
      .join('\n');
    downloadTextFile(`vocab_${caseId}_features.csv`, '\uFEFF' + header + rows, 'text/csv;charset=utf-8');
  };

  const handleDownloadCaseJson = (caseId: BayesCaseScenario['id']) => {
    const targetCase = BAYES_CASES[caseId];
    downloadTextFile(`case_${caseId}_package.json`, JSON.stringify(targetCase, null, 2), 'application/json;charset=utf-8');
  };

  const handleDownloadAllCasesCorpus = () => {
    const header = 'CaseID,CaseName,ID,Text,ActualClass,LabelName,Note\n';
    const allRows: string[] = [];
    Object.values(BAYES_CASES).forEach((c) => {
      c.samples.forEach((s) => {
        allRows.push(
          `"${c.id}","${c.name}","${s.id}","${s.text.replace(/"/g, '""')}",${s.actualClass},"${
            s.actualClass === 1 ? c.posLabel : c.negLabel
          }","${(s.note || '').replace(/"/g, '""')}"`
        );
      });
    });
    downloadTextFile('all_4_cases_corpus_benchmark.csv', '\uFEFF' + header + allRows.join('\n'), 'text/csv;charset=utf-8');
  };

  const handleDownloadAllCasesJson = () => {
    downloadTextFile('all_4_cases_benchmark_full.json', JSON.stringify(BAYES_CASES, null, 2), 'application/json;charset=utf-8');
  };

  // Structured Full-Process Report matching the 6 steps of the 全流程导引:
  const buildFullProcessMarkdownReport = () => {
    const evalMetrics = evaluateDataset(scenario, alpha, decisionThreshold, customCsvSamples);
    const dateStr = new Date().toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    });

    return `# 朴素贝叶斯文本过滤实验室 · 全流程实验诊断与评估报告

> **报告元信息**：生成日期：${dateStr} | 实验场景：${scenario.name} (${scenario.badgeText}) | 平滑系数 α=${alpha.toFixed(2)} | 判别阈值 T=${decisionThreshold.toFixed(2)}

---

## 【第一部分 · 语料基准与类先验分布】(Scenario Benchmark & Prior Distributions)
- **业务场景定位**：${scenario.subtitle}
- **正类定义 (Class 1)**：${scenario.posLabel} · 训练集基准先验 P(Y=1) = ${(scenario.priorPos * 100).toFixed(1)}% (N_pos = ${scenario.totalPosDocs})
- **负类定义 (Class 0)**：${scenario.negLabel} · 训练集基准先验 P(Y=0) = ${(scenario.priorNeg * 100).toFixed(1)}% (N_neg = ${scenario.totalNegDocs})
- **当前注入先验**：P(Y=1) = ${inference.priorPos.toFixed(4)}, P(Y=0) = ${inference.priorNeg.toFixed(4)}
- **场景全局特征词表规模 (|V|)**：${inference.vocabSize} 个特征词元

---

## 【第二部分 · 文本清洗与最大匹配分词词元流】(Text Preprocessing & Tokenization Stream)
- **原始待测输入文本**：「${inputText}」
- **分词算法实现**：前向最大正向匹配 (Maximum Forward Matching, MFM)
- **切分所得词元流序列**：${inference.rawTokens.map((t) => `\`${t}\``).join(' / ')} (共 ${inference.rawTokens.length} 个原始词元)
- **高频停用词过滤状态**：${filterStopwords ? '已开启 (Enabled)' : '已关闭 (Disabled)'}
- **被识别并过滤的无区分度停用词**：${
      inference.removedStopwords.length > 0
        ? inference.removedStopwords.map((t) => `\`${t}\``).join('、')
        : '无停用词'
    } (共 ${inference.removedStopwords.length} 个)
- **有效进入推理的特征词元序列**：${inference.steps.map((s) => `\`${s.token}\``).join('、')} (共 ${inference.steps.length} 个特征)

---

## 【第三部分 · 词袋 (BoW) 与特征向量化空间】(Bag-of-Words & Sparse Feature Vector)
- **高维特征空间向量维度**：${inference.vocabSize} 维稀疏空间
- **当前文本非零特征命中数**：${inference.steps.length} 维 (稀疏度 ${(100 - (inference.steps.length / inference.vocabSize) * 100).toFixed(1)}%)
- **词频统计 (TF) 与逆文档频率 (IDF) 权重分布明细**：

| 特征词元 (Token) | 文本内词频 (TF) | 逆文档频率 (IDF) | 卡方显著性 χ² | 互信息 (MI) | 特征状态 |
| :--- | :---: | :---: | :---: | :---: | :--- |
${inference.steps
  .map(
    (s) =>
      `| **${s.token}** | ${s.countInDoc} | ${s.tfidfWeight.toFixed(3)} | ${s.chiSquare.toFixed(2)} | ${s.mutualInfo.toFixed(4)} | ${
        s.isOOV
          ? '未登录词 (OOV)'
          : s.isZeroInPos || s.isZeroInNeg
          ? '零频词 (Zero-Freq)'
          : '正常词袋特征'
      } |`
  )
  .join('\n')}

---

## 【第四部分 · 平滑参数与条件概率估计矩阵】(Laplace Smoothing & Conditional Probabilities)
- **平滑算法类型**：拉普拉斯平滑 (Laplace / Lidstone Additive Smoothing)
- **平滑系数 α 设定值**：α = ${alpha.toFixed(2)} (Dirichlet 先验虚拟计数)
- **平滑估计条件概率求解公式**：
  - P(X_i | Y=1) = (N_{1,i} + α) / (N_1 + α * |V|)
  - P(X_i | Y=0) = (N_{0,i} + α) / (N_0 + α * |V|)
- **未登录词 (OOV) 与零频词修复状态**：${
      inference.zeroFreqTokensFound.length > 0
        ? `成功修复 ${inference.zeroFreqTokensFound.map((t) => `\`${t}\``).join('、')} 导致的“零概率陷阱”`
        : '全部特征词在正负类训练集均有登录，未触发零频乘积塌陷'
    }
- **各特征词平滑条件概率数值表**：

| 词元 (Token) | 正类条件概率 P(X_i|1) | 负类条件概率 P(X_i|0) | 似然比 LLR | 极性偏向判定 |
| :--- | :---: | :---: | :---: | :--- |
${inference.steps
  .map(
    (s) =>
      `| **${s.token}** | ${s.pGivenPos.toFixed(5)} | ${s.pGivenNeg.toFixed(5)} | ${s.logLikelihoodRatio >= 0 ? '+' : ''}${s.logLikelihoodRatio.toFixed(3)} | ${
        s.logLikelihoodRatio > 0.3
          ? '强正向偏好'
          : s.logLikelihoodRatio < -0.3
          ? '强负向偏好'
          : '中性平衡'
      } |`
  )
  .join('\n')}

---

## 【第五部分 · 对数似然评估与代数推导流水线】(Log-Likelihood Accumulation & Algebraic Derivation)
- **对数防下溢累加方程**：
  ln P(Y=c | X) = ln P(Y=c) + ∑ c_i * ln P(X_i | Y=c) - ln P(X)
- **正类联合对数似然求解**：
  - 类先验对数：ln P(Y=1) = ${inference.logPriorPos.toFixed(4)} nats
  - 特征条件对数似然累加：∑ ln P(X_i | Y=1) = ${(inference.totalLogPos - inference.logPriorPos).toFixed(4)} nats
  - 正类联合对数似然总和：**${inference.totalLogPos.toFixed(4)} nats**
- **负类联合对数似然求解**：
  - 类先验对数：ln P(Y=0) = ${inference.logPriorNeg.toFixed(4)} nats
  - 特征条件对数似然累加：∑ ln P(X_i | Y=0) = ${(inference.totalLogNeg - inference.logPriorNeg).toFixed(4)} nats
  - 负类联合对数似然总和：**${inference.totalLogNeg.toFixed(4)} nats**
- **对数奇数比边际差值 (Log-Odds Margin Δ)**：
  Δ = ln [P(Y=1|X) / P(Y=0|X)] = ${inference.totalLogPos.toFixed(4)} - (${inference.totalLogNeg.toFixed(4)}) = **${(inference.totalLogPos - inference.totalLogNeg).toFixed(4)} nats**

---

## 【第六部分 · MAP 判决输出与全量基准评估】(MAP Decision & Dataset Evaluation Metrics)
- **最大后验决策法则 (MAP Rule)**：
  c* = argmax_{c ∈ {0, 1}} P(Y=c | X)
- **当前样本最终分类决策**：**${inference.predictedLabel}**
- **后验归一化概率置信度**：
  - 正类后验置信度 P(Y=1 | X)：**${(inference.posteriorPos * 100).toFixed(2)}%**
  - 负类后验置信度 P(Y=0 | X)：**${(inference.posteriorNeg * 100).toFixed(2)}%**
- **场景基准测试集评估指标 (判决阈值 T = ${decisionThreshold.toFixed(2)})**：
  - **二分类混淆矩阵**：
    - 真正例 (True Positive, TP)：${evalMetrics.tp}
    - 假正例 (False Positive, FP / 误报)：${evalMetrics.fp}
    - 真反例 (True Negative, TN)：${evalMetrics.tn}
    - 假反例 (False Negative, FN / 漏报)：${evalMetrics.fn}
  - **核心量化评估指标**：
    - 准确率 (Accuracy)：**${(evalMetrics.accuracy * 100).toFixed(2)}%**
    - 精确率 (Precision)：**${(evalMetrics.precision * 100).toFixed(2)}%**
    - 召回率 (Recall)：**${(evalMetrics.recall * 100).toFixed(2)}%**
    - F1 综合调和均值：**${(evalMetrics.f1 * 100).toFixed(2)}%**
    - 受试者工作特征曲线面积 (ROC-AUC)：**${evalMetrics.auc.toFixed(4)}**

---

## 【第七部分 (附录诊断) · 特征条件独立性诊断与卡方优化建议】(Attribute Independence Deviation & Chi-Square Prescription)
- **特征条件独立性违背偏离指数**：**${inference.independenceDeviationScore} / 100**
- **检出的强相关共现违背词对**：${
      inference.correlatedPairsFound.length > 0
        ? inference.correlatedPairsFound
            .map((p) => `「${p.pair}」(皮尔逊相关 r=${p.r}，产生对数奇数膨胀 +${p.inflationLogOdds.toFixed(2)} nats)`)
            .join('；')
        : '未检出显著强共现违背词对，特征间正交性保持优良。'
    }
- **工业级优化处方建议**：
  1. 若共现词对 Pearson 相关系数 r > 0.70，建议使用 Bigram 短语合并或通过卡方检验保留最高分特征；
  2. 对停用词及无区分度词元进行前置过滤，避免稀释类条件似然分布；
  3. 平滑系数建议保持在 0.5 ~ 1.5 之间，以平衡零频修复与保持原始先验敏感度。
`;
  };

  const handleExportMarkdownReport = () => {
    const md = buildFullProcessMarkdownReport();
    downloadTextFile(`naive_bayes_full_report_${scenario.id}.md`, md, 'text/markdown;charset=utf-8');
  };

  const handleCopyReportMarkdown = () => {
    const md = buildFullProcessMarkdownReport();
    navigator.clipboard.writeText(md);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2000);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = String(evt.target?.result || '');
      const lines = text
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
      const parsed: Array<{ id: string; text: string; actualClass: 1 | 0 }> = [];
      lines.forEach((line, idx) => {
        if (idx === 0 && (line.toLowerCase().includes('text') || line.includes('文本'))) return;
        const parts = line.split(',');
        if (parts.length >= 2) {
          const rawLabel = parts[parts.length - 1].replace(/"/g, '').trim();
          const labelNum: 1 | 0 =
            rawLabel === '1' || rawLabel.toLowerCase() === 'spam' || rawLabel.includes('正') ? 1 : 0;
          const docText = parts
            .slice(0, parts.length - 1)
            .join(',')
            .replace(/^"|"$/g, '');
          if (docText) {
            parsed.push({
              id: `CSV-${idx}`,
              text: docText,
              actualClass: labelNum,
            });
          }
        }
      });
      if (parsed.length > 0) {
        setCustomCsvSamples(parsed);
        setCsvStatusMsg(`成功载入自定义 CSV 数据集（共 ${parsed.length} 条文本样本）！`);
      } else {
        setCsvStatusMsg('未能解析有效行，请确保 CSV 每行格式为：文本内容,1或0');
      }
    };
    reader.readAsText(file);
  };

  if (activeModule === 8) {
    return (
      <div className="space-y-6">
        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 08</span>
              <span aria-hidden="true">·</span>
              <span>四阶端到端流水线切片</span>
              <span aria-hidden="true">·</span>
              <span>从原始字符流到后验决策单</span>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
              文本 → 分词向量化 → 概率推导 → 决策全流程导引
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportMarkdownReport}
              className="px-3.5 py-2 text-xs font-medium bg-slate-900 text-white hover:bg-slate-800 rounded-lg flex items-center gap-1.5 transition-colors whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" /> 导出当前样本决策报告 (.md)
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {[
            {
              stage: 1 as const,
              title: '01. 文本清洗与分词切片',
              desc: `切分出 ${inference.rawTokens.length} 个原始词元，过滤 ${inference.removedStopwords.length} 个停用词`,
            },
            {
              stage: 2 as const,
              title: '02. BoW 与 TF-IDF 向量化',
              desc: `构建 ${inference.steps.length} 维非零稀疏向量与 IDF 逆文档权重`,
            },
            {
              stage: 3 as const,
              title: '03. 先验与平滑条件概率',
              desc: `注入 α=${alpha.toFixed(2)} 计算 P(Xᵢ|Y=1) 与 P(Xᵢ|Y=0)`,
            },
            {
              stage: 4 as const,
              title: '04. 对数似然评估与决策',
              desc: `输出后验 P(正类|X)=${(inference.posteriorPos * 100).toFixed(1)}% 并生成报告`,
            },
          ].map((item) => {
            const active = pipelineStage === item.stage;
            return (
              <button
                key={item.stage}
                onClick={() => setPipelineStage(item.stage)}
                className={`p-4 rounded-xl border text-left transition-all ${
                  active
                    ? 'bg-white border-slate-900 ring-1 ring-slate-900 shadow-xs'
                    : 'bg-white border-slate-200/90 hover:border-slate-400'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-semibold text-slate-900">
                  <span>{item.title}</span>
                  {active && <span className="text-sky-700">● 当前切片</span>}
                </div>
                <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">{item.desc}</p>
              </button>
            );
          })}
        </div>

        {pipelineStage === 1 && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
              <div>
                <h3 className="text-base font-semibold text-slate-900">
                  第一阶段切片：最大正向匹配分词与高频停用词去噪
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  切换下方停用词过滤开关，观察“的、今天、我们”等无区分度高频词对特征流的影响
                </p>
              </div>
              <button
                onClick={() => setFilterStopwords(!filterStopwords)}
                className={`px-3.5 py-2 text-xs font-medium rounded-lg border transition-colors whitespace-nowrap ${
                  filterStopwords
                    ? 'bg-emerald-50 text-emerald-900 border-emerald-300'
                    : 'bg-amber-50 text-amber-900 border-amber-300'
                }`}
              >
                {filterStopwords
                  ? '✓ 停用词过滤：已开启（推荐）'
                  : '⚠ 停用词过滤：已关闭（停用词混入概率连乘）'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="text-xs font-semibold text-slate-700">
                  有效入模特征词序列（共 {inference.activeTokens.length} 词次）：
                </div>
                <div className="flex flex-wrap gap-2">
                  {inference.activeTokens.map((tok, i) => (
                    <span
                      key={`${tok}-${i}`}
                      className="px-2.5 py-1 bg-white border border-slate-200 rounded-md text-xs font-medium text-slate-900"
                    >
                      {tok}
                    </span>
                  ))}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="text-xs font-semibold text-slate-700">
                  已拦截的高频停用词与标点噪点（{inference.removedStopwords.length} 项）：
                </div>
                {inference.removedStopwords.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {inference.removedStopwords.map((tok, i) => (
                      <span
                        key={`${tok}-${i}`}
                        className="px-2.5 py-1 bg-slate-200/70 text-slate-500 line-through rounded-md text-xs"
                      >
                        {tok}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    {filterStopwords
                      ? '当前文本未包含停用词（可尝试在顶部输入框加入“今天、的、我们”测试过滤效果）。'
                      : '停用词过滤已关闭，所有词元均直接进入概率连乘！'}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {pipelineStage === 2 && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              第二阶段切片：词袋模型（Bag-of-Words）词频统计与 TF-IDF 权重矩阵
            </h3>
            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="w-full text-left text-xs font-mono-tabular">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3 font-sans">特征词元 Xᵢ</th>
                    <th className="py-2.5 px-3">文档内词频 BoW(Xᵢ)</th>
                    <th className="py-2.5 px-3">TF-IDF 归一化权重</th>
                    <th className="py-2.5 px-3">卡方区分度 χ²</th>
                    <th className="py-2.5 px-3 font-sans">词表状态</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {inference.steps.map((s) => (
                    <tr key={s.token} className="hover:bg-slate-50">
                      <td className="py-2 px-3 font-sans font-semibold text-slate-900">
                        {s.token}
                      </td>
                      <td className="py-2 px-3 font-semibold">{s.countInDoc}</td>
                      <td className="py-2 px-3 text-sky-700">{s.tfidfWeight.toFixed(4)}</td>
                      <td className="py-2 px-3">{s.chiSquare.toFixed(2)}</td>
                      <td className="py-2 px-3 font-sans">
                        {s.isOOV ? (
                          <span className="text-amber-700 font-medium">未登录词 (OOV)</span>
                        ) : (
                          <span className="text-emerald-700">已收录特征词</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {pipelineStage === 3 && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              第三阶段切片：先验概率 P(Y) 与平滑条件似然 P(Xᵢ | Y) 映射表
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-rose-50/50 border border-rose-200 font-mono-tabular text-xs space-y-1">
                <div className="font-sans font-semibold text-rose-900">
                  正类（{scenario.posLabel}）先验与分母参数
                </div>
                <div>先验 P(Y=1) = {inference.priorPos.toFixed(4)} (ln P = {inference.logPriorPos.toFixed(4)})</div>
                <div>
                  平滑分母 N₁ + α|V| = {inference.totalPosTokens} + {alpha.toFixed(2)} × {inference.vocabSize} ={' '}
                  {(inference.totalPosTokens + alpha * inference.vocabSize).toFixed(2)}
                </div>
              </div>
              <div className="p-4 rounded-xl bg-sky-50/50 border border-sky-200 font-mono-tabular text-xs space-y-1">
                <div className="font-sans font-semibold text-sky-900">
                  负类（{scenario.negLabel}）先验与分母参数
                </div>
                <div>先验 P(Y=0) = {inference.priorNeg.toFixed(4)} (ln P = {inference.logPriorNeg.toFixed(4)})</div>
                <div>
                  平滑分母 N₀ + α|V| = {inference.totalNegTokens} + {alpha.toFixed(2)} × {inference.vocabSize} ={' '}
                  {(inference.totalNegTokens + alpha * inference.vocabSize).toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        )}

        {pipelineStage === 4 && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              第四阶段切片：对数似然累加汇总与 MAP 最终分类判决单
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono-tabular">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-sans text-slate-500">正类对数似然总和 ℓ(Y=1)</div>
                <div className="text-xl font-semibold text-rose-700 mt-1">
                  {inference.totalLogPos.toFixed(4)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  归一化后验：{(inference.posteriorPos * 100).toFixed(2)}%
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-sans text-slate-500">负类对数似然总和 ℓ(Y=0)</div>
                <div className="text-xl font-semibold text-sky-700 mt-1">
                  {inference.totalLogNeg.toFixed(4)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  归一化后验：{(inference.posteriorNeg * 100).toFixed(2)}%
                </div>
              </div>
              <div className="p-4 rounded-xl bg-emerald-50/70 border border-emerald-300">
                <div className="text-xs font-sans text-emerald-800">MAP 最优分类决策输出</div>
                <div className="text-lg font-sans font-bold text-emerald-950 mt-1">
                  {inference.predictedLabel}
                </div>
                <div className="text-xs text-emerald-800 mt-1">
                  对数奇数比差 Δ = {(inference.totalLogPos - inference.totalLogNeg).toFixed(3)}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (activeModule === 9) {
    const evalMetrics = evaluateDataset(scenario, alpha, decisionThreshold, customCsvSamples);
    const allCasesList = Object.values(BAYES_CASES);

    return (
      <div className="space-y-6">
        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 09</span>
              <span aria-hidden="true">·</span>
              <span>四大案例原始语料全量下载</span>
              <span aria-hidden="true">·</span>
              <span>全流程导引 6 阶诊断报告预览与导出</span>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
              数据集下载与分类报告导出引擎
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleCopyReportMarkdown}
              className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center gap-1.5 whitespace-nowrap transition-colors"
              title="复制 6 大阶段 Markdown 格式报告至剪贴板"
            >
              {copiedReport ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-emerald-700">已复制报告源码</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>复制报告源码</span>
                </>
              )}
            </button>
            <button
              onClick={handleExportMarkdownReport}
              className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center gap-1.5 whitespace-nowrap transition-colors"
              title="导出 6 大阶段 Markdown 结构化报告文件"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>导出报告 (.md)</span>
            </button>
            <button
              onClick={handleExportVocabCsv}
              className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center gap-1.5 whitespace-nowrap transition-colors"
              title="导出当前案例词表平滑条件概率矩阵 CSV"
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span>导出词表矩阵 (.csv)</span>
            </button>
            <button
              onClick={() => window.print()}
              className="px-3.5 py-1.5 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-white rounded-lg flex items-center gap-1.5 whitespace-nowrap shadow-xs transition-colors"
              title="直接调用系统打印机打印或保存为 PDF 报告"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>打印 / 导出 PDF</span>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-lg w-fit no-print">
          <button
            onClick={() => setReportSlice('report_preview')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              reportSlice === 'report_preview'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Eye className="w-3.5 h-3.5 text-sky-700" />
            <span>切片一：全流程报告实时预览 (6大阶段报告)</span>
          </button>
          <button
            onClick={() => setReportSlice('cases_download')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              reportSlice === 'cases_download'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Database className="w-3.5 h-3.5 text-emerald-700" />
            <span>切片二：四大案例原始数据下载 (各案例独立选项)</span>
          </button>
          <button
            onClick={() => setReportSlice('confusion_roc')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              reportSlice === 'confusion_roc'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5 text-indigo-700" />
            <span>切片三：混淆矩阵与 ROC/AUC 曲线</span>
          </button>
          <button
            onClick={() => setReportSlice('vocab_matrix')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              reportSlice === 'vocab_matrix'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-amber-700" />
            <span>切片四：词表条件概率矩阵预览</span>
          </button>
          <button
            onClick={() => setReportSlice('csv_upload')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              reportSlice === 'csv_upload'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Upload className="w-3.5 h-3.5 text-rose-700" />
            <span>切片五：上传自定义 CSV 语料集</span>
          </button>
        </div>

        {/* ==================================================================== */}
        {/* SLICE 1: REPORT PREVIEW (ORGANIZED INTO 6+ PARTS MATCHING PIPELINE)   */}
        {/* ==================================================================== */}
        {reportSlice === 'report_preview' && (
          <div className="bg-white border border-slate-200/90 rounded-2xl p-6 sm:p-8 space-y-8 shadow-xs">
            {/* Report Header */}
            <div className="border-b border-slate-200 pb-6 flex flex-col md:flex-row md:items-start md:justify-between gap-4">
              <div>
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 text-xs font-medium mb-2.5">
                  <FileText className="w-3.5 h-3.5 text-sky-700" />
                  <span>国家实验室标准学术格式 · 全流程分类推导报告</span>
                </div>
                <h3 className="text-2xl font-bold text-slate-900 font-serif-title tracking-tight">
                  朴素贝叶斯文本过滤实验室 · 全流程实验诊断与评估报告
                </h3>
                <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
                  严格按照「文本清洗 → 分词向量化 → 概率推导 → 决策输出」全流程导引 6 大阶梯组织编排
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs font-mono-tabular space-y-1 self-start min-w-[220px]">
                <div className="text-slate-500 flex justify-between">
                  <span>当前场景：</span>
                  <span className="font-semibold text-slate-900">{scenario.name}</span>
                </div>
                <div className="text-slate-500 flex justify-between">
                  <span>平滑系数 α：</span>
                  <span className="font-semibold text-slate-900">{alpha.toFixed(2)}</span>
                </div>
                <div className="text-slate-500 flex justify-between">
                  <span>判别阈值 T：</span>
                  <span className="font-semibold text-slate-900">{decisionThreshold.toFixed(2)}</span>
                </div>
                <div className="text-slate-500 flex justify-between">
                  <span>停用词过滤：</span>
                  <span className={filterStopwords ? 'text-emerald-700 font-medium' : 'text-slate-600'}>
                    {filterStopwords ? '已开启' : '已关闭'}
                  </span>
                </div>
              </div>
            </div>

            {/* SECTION 1: SCENARIO BENCHMARK & PRIOR */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  01
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第一部分 · 语料基准与类先验分布 (Scenario Benchmark & Prior Distributions)
                </h4>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono-tabular">
                <div className="p-4 rounded-xl bg-rose-50/60 border border-rose-200">
                  <div className="text-[11px] font-sans text-rose-800 font-medium">正类 Class 1 定位与先验</div>
                  <div className="text-lg font-bold text-rose-950 mt-1">{scenario.posLabel}</div>
                  <div className="mt-1 text-slate-600">
                    P(Y=1) = <strong className="text-rose-700">{inference.priorPos.toFixed(4)}</strong> ({scenario.totalPosDocs}篇)
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-sky-50/60 border border-sky-200">
                  <div className="text-[11px] font-sans text-sky-800 font-medium">负类 Class 0 定位与先验</div>
                  <div className="text-lg font-bold text-sky-950 mt-1">{scenario.negLabel}</div>
                  <div className="mt-1 text-slate-600">
                    P(Y=0) = <strong className="text-sky-700">{inference.priorNeg.toFixed(4)}</strong> ({scenario.totalNegDocs}篇)
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-sans text-slate-600 font-medium">特征词表基数 |V|</div>
                  <div className="text-lg font-bold text-slate-900 mt-1">{inference.vocabSize} 个词元</div>
                  <div className="mt-1 text-slate-500">训练语料总收录词袋空间</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-sans text-slate-600 font-medium">类分布平衡度 (Pos/Neg)</div>
                  <div className="text-lg font-bold text-slate-900 mt-1">
                    {(inference.priorPos / inference.priorNeg).toFixed(2)} : 1
                  </div>
                  <div className="mt-1 text-slate-500">
                    {inference.priorPos === inference.priorNeg ? '完全平衡先验' : '偏斜非均匀先验'}
                  </div>
                </div>
              </div>
            </section>

            {/* SECTION 2: TEXT CLEANING & TOKENIZATION STREAM */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  02
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第二部分 · 文本清洗与最大匹配分词词元流 (Text Preprocessing & Tokenization Stream)
                </h4>
              </div>

              <div className="p-4 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3 text-xs">
                <div>
                  <span className="font-semibold text-slate-800">原始待测输入文本：</span>
                  <span className="bg-white px-2.5 py-1 rounded border border-slate-300 font-sans text-slate-900 ml-1">
                    {inputText}
                  </span>
                </div>

                <div className="space-y-1.5">
                  <div className="text-slate-600 flex items-center gap-2">
                    <span>前向最大正向匹配 (MFM) 切分所得词元序列 ({inference.rawTokens.length} 个)：</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 font-mono-tabular">
                    {inference.rawTokens.map((t, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-800 font-medium"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-200/80 flex flex-wrap items-center gap-4 text-slate-600">
                  <div>
                    <span>过滤高频无信息停用词：</span>
                    {inference.removedStopwords.length > 0 ? (
                      <span className="text-amber-700 font-semibold font-mono-tabular ml-1">
                        {inference.removedStopwords.join('、')} ({inference.removedStopwords.length}个)
                      </span>
                    ) : (
                      <span className="text-slate-400 font-mono-tabular ml-1">无</span>
                    )}
                  </div>
                  <div>
                    <span>保留有效特征词元数：</span>
                    <strong className="text-emerald-700 font-mono-tabular ml-1">
                      {inference.steps.length} 个
                    </strong>
                  </div>
                </div>
              </div>
            </section>

            {/* SECTION 3: BAG-OF-WORDS & SPARSITY */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  03
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第三部分 · 词袋 (BoW) 与特征向量化空间 (Bag-of-Words & Sparse Feature Vector)
                </h4>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left text-xs font-mono-tabular">
                  <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3 font-sans">特征词元 (Token)</th>
                      <th className="py-2.5 px-3">文档内词频 (TF)</th>
                      <th className="py-2.5 px-3">逆文档频率权重 (IDF)</th>
                      <th className="py-2.5 px-3">卡方显著度 χ²</th>
                      <th className="py-2.5 px-3">互信息量 (MI)</th>
                      <th className="py-2.5 px-3 font-sans">词表特征登录状态</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {inference.steps.map((s) => (
                      <tr key={s.token} className="hover:bg-slate-50">
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">
                          {s.token}
                        </td>
                        <td className="py-2 px-3 font-bold">{s.countInDoc}</td>
                        <td className="py-2 px-3 text-sky-700">{s.tfidfWeight.toFixed(3)}</td>
                        <td className="py-2 px-3">
                          {s.chiSquare.toFixed(2)}
                        </td>
                        <td className="py-2 px-3 text-emerald-700">
                          {s.mutualInfo.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 font-sans">
                          {s.isOOV ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] bg-rose-50 text-rose-700 font-medium">
                              未登录词 (OOV)
                            </span>
                          ) : s.isZeroInPos || s.isZeroInNeg ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] bg-amber-50 text-amber-700 font-medium">
                              单类零频词
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] bg-emerald-50 text-emerald-700 font-medium">
                              词表收录良性
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* SECTION 4: LAPLACE SMOOTHING & CONDITIONAL PROBABILITIES */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  04
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第四部分 · 平滑参数与条件概率估计矩阵 (Laplace Smoothing & Conditional Probabilities)
                </h4>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs font-mono-tabular space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <span className="font-sans font-semibold text-slate-800">平滑估计求解方程：</span>
                    <code>P(Xᵢ | Y=c) = (N(c,i) + α) / (N_c + α * |V|)</code>
                  </div>
                  <div className="text-slate-600">
                    当前参数：<strong>α = {alpha.toFixed(2)}</strong> | 正类分母={inference.totalPosTokens + alpha * inference.vocabSize} | 负类分母={inference.totalNegTokens + alpha * inference.vocabSize}
                  </div>
                </div>
                <div className="text-slate-600">
                  零频陷阱诊断：
                  {inference.zeroFreqTokensFound.length > 0 ? (
                    <span className="text-rose-700 font-semibold ml-1">
                      检出未登录/零频词元 [{inference.zeroFreqTokensFound.join('、')}]，经 α={alpha.toFixed(2)} 平滑后获得基底概率，避免了联合概率下溢归零。
                    </span>
                  ) : (
                    <span className="text-emerald-700 font-medium ml-1">
                      当前待测词元均登录于训练语料库，未产生零频乘积塌陷。
                    </span>
                  )}
                </div>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left text-xs font-mono-tabular">
                  <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3 font-sans">特征词元</th>
                      <th className="py-2.5 px-3 text-rose-700">P(Xᵢ | 正类, α)</th>
                      <th className="py-2.5 px-3 text-sky-700">P(Xᵢ | 负类, α)</th>
                      <th className="py-2.5 px-3">对数似然比 LLR</th>
                      <th className="py-2.5 px-3 font-sans">类极性偏向判定</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {inference.steps.map((s) => (
                      <tr key={s.token} className="hover:bg-slate-50">
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">
                          {s.token}
                        </td>
                        <td className="py-2 px-3 text-rose-700">{s.pGivenPos.toFixed(5)}</td>
                        <td className="py-2 px-3 text-sky-700">{s.pGivenNeg.toFixed(5)}</td>
                        <td className="py-2 px-3 font-bold text-slate-900">
                          {s.logLikelihoodRatio >= 0 ? '+' : ''}
                          {s.logLikelihoodRatio.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 font-sans">
                          {s.logLikelihoodRatio > 0.4 ? (
                            <span className="text-rose-700 font-medium">强偏正类 ({scenario.posLabel})</span>
                          ) : s.logLikelihoodRatio < -0.4 ? (
                            <span className="text-sky-700 font-medium">强偏负类 ({scenario.negLabel})</span>
                          ) : (
                            <span className="text-slate-500">中性均衡特征</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* SECTION 5: LOG-LIKELIHOOD ACCUMULATION & ALGEBRAIC DERIVATION */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  05
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第五部分 · 对数似然评估与代数推导流水线 (Log-Likelihood Accumulation & Algebraic Derivation)
                </h4>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono-tabular text-xs">
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
                  <div className="font-sans font-semibold text-slate-700">正类联合对数似然 ln P(Y=1, X)</div>
                  <div className="text-xl font-bold text-rose-700 mt-1">
                    {inference.totalLogPos.toFixed(4)} <span className="text-xs font-normal text-slate-500">nats</span>
                  </div>
                  <div className="text-slate-500 pt-1 border-t border-slate-200">
                    ln P(Y=1) [{inference.logPriorPos.toFixed(3)}] + ∑ ln P(Xᵢ|1) [{(inference.totalLogPos - inference.logPriorPos).toFixed(3)}]
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
                  <div className="font-sans font-semibold text-slate-700">负类联合对数似然 ln P(Y=0, X)</div>
                  <div className="text-xl font-bold text-sky-700 mt-1">
                    {inference.totalLogNeg.toFixed(4)} <span className="text-xs font-normal text-slate-500">nats</span>
                  </div>
                  <div className="text-slate-500 pt-1 border-t border-slate-200">
                    ln P(Y=0) [{inference.logPriorNeg.toFixed(3)}] + ∑ ln P(Xᵢ|0) [{(inference.totalLogNeg - inference.logPriorNeg).toFixed(3)}]
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900 text-white space-y-1.5">
                  <div className="font-sans font-semibold text-slate-300">对数奇数比边际差值 (Log-Odds Margin Δ)</div>
                  <div className="text-xl font-bold text-emerald-400 mt-1">
                    {(inference.totalLogPos - inference.totalLogNeg).toFixed(4)}{' '}
                    <span className="text-xs font-normal text-slate-400">nats</span>
                  </div>
                  <div className="text-slate-400 pt-1 border-t border-slate-800">
                    Δ = ℓ(Pos) - ℓ(Neg) &gt; 0 倾向正类，反之倾向负类
                  </div>
                </div>
              </div>
            </section>

            {/* SECTION 6: MAP DECISION & DATASET EVALUATION METRICS */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  06
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第六部分 · MAP 判决输出与全量基准评估 (MAP Decision & Dataset Evaluation Metrics)
                </h4>
              </div>

              {/* MAP Decision Banner */}
              <div className="p-4 rounded-xl bg-emerald-50/80 border border-emerald-300 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div>
                  <div className="text-xs text-emerald-800 font-semibold font-sans">
                    最大后验决策法则输出 (Maximum A Posteriori Rule)
                  </div>
                  <div className="text-xl font-bold text-emerald-950 mt-0.5">
                    最终分类判决：{inference.predictedLabel}
                  </div>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono-tabular">
                  <div className="text-rose-800">
                    正类后验置信度：<strong>{(inference.posteriorPos * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="text-sky-800">
                    负类后验置信度：<strong>{(inference.posteriorNeg * 100).toFixed(2)}%</strong>
                  </div>
                </div>
              </div>

              {/* Confusion Matrix and Metrics Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                <div className="lg:col-span-6 p-4 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3">
                  <div className="flex justify-between items-center text-xs font-semibold text-slate-800 font-sans border-b border-slate-200 pb-2">
                    <span>测试语料混淆矩阵 (Confusion Matrix)</span>
                    <span className="font-mono-tabular text-slate-500 font-normal">
                      阈值 T = {decisionThreshold.toFixed(2)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono-tabular">
                    <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
                      <div className="text-[11px] font-sans text-emerald-800">真正例 (TP)</div>
                      <div className="text-xl font-bold text-emerald-900 mt-0.5">{evalMetrics.tp}</div>
                      <div className="text-[10px] text-emerald-700">实际正类 · 预测正类</div>
                    </div>
                    <div className="p-3 rounded-lg bg-rose-50 border border-rose-200">
                      <div className="text-[11px] font-sans text-rose-800">假正例 (FP / 误报)</div>
                      <div className="text-xl font-bold text-rose-900 mt-0.5">{evalMetrics.fp}</div>
                      <div className="text-[10px] text-rose-700">实际负类 · 误判正类</div>
                    </div>
                    <div className="p-3 rounded-lg bg-amber-50 border border-amber-200">
                      <div className="text-[11px] font-sans text-amber-800">假反例 (FN / 漏报)</div>
                      <div className="text-xl font-bold text-amber-900 mt-0.5">{evalMetrics.fn}</div>
                      <div className="text-[10px] text-amber-700">实际正类 · 漏判负类</div>
                    </div>
                    <div className="p-3 rounded-lg bg-sky-50 border border-sky-200">
                      <div className="text-[11px] font-sans text-sky-800">真反例 (TN)</div>
                      <div className="text-xl font-bold text-sky-900 mt-0.5">{evalMetrics.tn}</div>
                      <div className="text-[10px] text-sky-700">实际负类 · 预测负类</div>
                    </div>
                  </div>
                </div>

                <div className="lg:col-span-6 p-4 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3 flex flex-col justify-between">
                  <div className="text-xs font-semibold text-slate-800 font-sans border-b border-slate-200 pb-2">
                    全量综合量化指标 (Quantitative Evaluation)
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-center font-mono-tabular">
                    <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                      <div className="text-[11px] font-sans text-slate-500">准确率 Acc</div>
                      <div className="text-base font-bold text-slate-900 mt-0.5">
                        {(evalMetrics.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                      <div className="text-[11px] font-sans text-slate-500">精确率 Prec</div>
                      <div className="text-base font-bold text-slate-900 mt-0.5">
                        {(evalMetrics.precision * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                      <div className="text-[11px] font-sans text-slate-500">召回率 Rec</div>
                      <div className="text-base font-bold text-slate-900 mt-0.5">
                        {(evalMetrics.recall * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                      <div className="text-[11px] font-sans text-slate-500">F1 调和均值</div>
                      <div className="text-base font-bold text-emerald-700 mt-0.5">
                        {(evalMetrics.f1 * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white border border-slate-200 col-span-2 sm:col-span-2">
                      <div className="text-[11px] font-sans text-slate-500">ROC-AUC 面积</div>
                      <div className="text-base font-bold text-sky-700 mt-0.5">
                        {evalMetrics.auc.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* SECTION 7: ATTRIBUTE INDEPENDENCE & CHI-SQUARE PRESCRIPTION */}
            <section className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="w-6 h-6 rounded-md bg-slate-900 text-white text-xs font-bold flex items-center justify-center font-mono-tabular">
                  07
                </span>
                <h4 className="text-base font-semibold text-slate-900 font-serif-title">
                  第七部分 (附录诊断) · 特征条件独立性违背诊断与优化建议 (Attribute Independence Deviation & Chi-Square Prescription)
                </h4>
              </div>

              <div className="p-4 rounded-xl bg-amber-50/60 border border-amber-200 text-xs font-mono-tabular space-y-2">
                <div className="flex justify-between items-center text-amber-900 font-sans font-semibold">
                  <span>条件独立性假设违背偏离指数：</span>
                  <span className="font-mono-tabular text-sm font-bold text-amber-950">
                    {inference.independenceDeviationScore} / 100
                  </span>
                </div>
                <div className="text-slate-700 leading-relaxed font-sans">
                  <strong>检出的强共现关联词对：</strong>
                  {inference.correlatedPairsFound.length > 0 ? (
                    inference.correlatedPairsFound.map((p, idx) => (
                      <span key={idx} className="ml-1 text-rose-800 font-medium font-mono-tabular">
                        「{p.pair}」(r={p.r}, 产生对数似然虚假膨胀 +{p.inflationLogOdds.toFixed(2)} nats)；
                      </span>
                    ))
                  ) : (
                    <span className="text-emerald-700 font-medium ml-1">
                      未检出高相关共现词对，特征正交性维持较好。
                    </span>
                  )}
                </div>
                <div className="text-slate-600 font-sans pt-1 border-t border-amber-200/60">
                  <strong>专家建议：</strong>
                  对共现相关系数高于 0.70 的词对推荐进行短语级 Bigram 向量聚合，或基于卡方检验 (Chi-Square χ²) 进行 Top-K 特征筛选降维，抑制后验概率虚假向 0% 或 100% 极化。
                </div>
              </div>
            </section>

            {/* Report Footer Action Toolbar */}
            <div className="pt-6 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3 no-print">
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyReportMarkdown}
                  className="px-4 py-2 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-white rounded-lg flex items-center gap-1.5 transition-colors"
                >
                  {copiedReport ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedReport ? '已成功复制 Markdown 源码' : '复制完整报告 Markdown 源码'}</span>
                </button>
                <button
                  onClick={handleExportMarkdownReport}
                  className="px-4 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center gap-1.5 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>导出为 .md 文件</span>
                </button>
                <button
                  onClick={() => window.print()}
                  className="px-4 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center gap-1.5 transition-colors"
                >
                  <Printer className="w-3.5 h-3.5" />
                  <span>打印报告</span>
                </button>
              </div>

              <button
                onClick={() => setReportSlice('cases_download')}
                className="text-xs font-medium text-sky-700 hover:underline flex items-center gap-1"
              >
                前往切片二下载四大案例原始数据集 <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* SLICE 2: 4 CASES RAW DATASET DOWNLOAD (EXPLICIT OPTION FOR EACH CASE) */}
        {/* ==================================================================== */}
        {reportSlice === 'cases_download' && (
          <div className="space-y-6">
            <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h3 className="text-base font-semibold text-slate-900 font-serif-title">
                  四大经典场景案例原始数据集独立下载与全量包
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  每个案例均提供标准格式的原始样本语料 (.csv)、特征词表与卡方表 (.csv) 及全量 JSON 数据包 (.json)
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2 self-start">
                <button
                  onClick={handleDownloadAllCasesCorpus}
                  className="px-3.5 py-2 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-white rounded-lg flex items-center gap-1.5 whitespace-nowrap shadow-xs transition-colors"
                >
                  <FolderDown className="w-3.5 h-3.5" />
                  <span>一键打包下载四大案例全部原始语料 (.csv)</span>
                </button>
                <button
                  onClick={handleDownloadAllCasesJson}
                  className="px-3.5 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center gap-1.5 whitespace-nowrap transition-colors"
                >
                  <Database className="w-3.5 h-3.5 text-slate-700" />
                  <span>全量 JSON 数据包</span>
                </button>
              </div>
            </div>

            {/* 4 Cards for the 4 Cases */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {allCasesList.map((c) => {
                const isCurrent = c.id === scenario.id;
                return (
                  <div
                    key={c.id}
                    className={`bg-white rounded-xl border p-5 flex flex-col justify-between space-y-4 transition-all ${
                      isCurrent
                        ? 'border-slate-900 ring-1 ring-slate-900 shadow-xs'
                        : 'border-slate-200/90 hover:border-slate-300'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between text-xs text-slate-500">
                        <span className="font-mono-tabular">{c.badgeText}</span>
                        {isCurrent && (
                          <span className="text-emerald-700 font-medium">● 实验室当前载入</span>
                        )}
                      </div>
                      <h4 className="text-base font-semibold text-slate-900 mt-1">{c.name}</h4>
                      <p className="text-xs text-slate-600 mt-1 leading-relaxed">{c.subtitle}</p>

                      <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-3 gap-2 text-xs font-mono-tabular text-slate-600">
                        <div>
                          <span className="text-slate-400 block text-[10px]">基准样本</span>
                          <strong>{c.samples.length} 条测试文本</strong>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">特征词表</span>
                          <strong>{c.vocab.length} 个词元</strong>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">先验比例</span>
                          <strong>{(c.priorPos * 100).toFixed(0)}% : {(c.priorNeg * 100).toFixed(0)}%</strong>
                        </div>
                      </div>
                    </div>

                    <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center gap-2">
                      <button
                        onClick={() => handleDownloadCaseDataset(c.id)}
                        className="flex-1 min-w-[130px] px-3 py-1.5 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-white rounded-lg flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
                        title={`下载${c.name}的标准测试集 CSV 文件（含ID、Text、ActualClass、LabelName）`}
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>下载原始样本语料 (.csv)</span>
                      </button>
                      <button
                        onClick={() => handleDownloadCaseVocab(c.id)}
                        className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
                        title={`下载${c.name}特征词词频、卡方检验与互信息表格`}
                      >
                        <FileSpreadsheet className="w-3.5 h-3.5" />
                        <span>词表与卡方 (.csv)</span>
                      </button>
                      <button
                        onClick={() => handleDownloadCaseJson(c.id)}
                        className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
                        title={`下载${c.name}全量 JSON 配置`}
                      >
                        <FileText className="w-3.5 h-3.5" />
                        <span>JSON 数据</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {reportSlice === 'confusion_roc' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-6 bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <h3 className="text-base font-semibold text-slate-900">
                  二分类混淆矩阵 (Confusion Matrix)
                </h3>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-500">判别阈值 T = {decisionThreshold.toFixed(2)}</span>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={decisionThreshold}
                    onChange={(e) => setDecisionThreshold(Number(e.target.value))}
                    className="w-24 accent-slate-900 cursor-pointer"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 font-mono-tabular">
                <div className="p-4 rounded-xl bg-emerald-50/70 border border-emerald-200">
                  <div className="text-xs font-sans text-emerald-800">
                    真正例 True Positive (TP)
                  </div>
                  <div className="text-2xl font-bold text-emerald-900 mt-1">
                    {evalMetrics.tp}
                  </div>
                  <div className="text-[11px] font-sans text-emerald-700 mt-0.5">
                    实际正类 · 预测正类
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-rose-50/70 border border-rose-200">
                  <div className="text-xs font-sans text-rose-800">
                    假正例 False Positive (FP / 误拦)
                  </div>
                  <div className="text-2xl font-bold text-rose-900 mt-1">
                    {evalMetrics.fp}
                  </div>
                  <div className="text-[11px] font-sans text-rose-700 mt-0.5">
                    实际负类 · 误判为正类
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200">
                  <div className="text-xs font-sans text-amber-800">
                    假反例 False Negative (FN / 漏报)
                  </div>
                  <div className="text-2xl font-bold text-amber-900 mt-1">
                    {evalMetrics.fn}
                  </div>
                  <div className="text-[11px] font-sans text-amber-700 mt-0.5">
                    实际正类 · 漏判为负类
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-sky-50/70 border border-sky-200">
                  <div className="text-xs font-sans text-sky-800">
                    真反例 True Negative (TN)
                  </div>
                  <div className="text-2xl font-bold text-sky-900 mt-1">
                    {evalMetrics.tn}
                  </div>
                  <div className="text-[11px] font-sans text-sky-700 mt-0.5">
                    实际负类 · 预测负类
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2 pt-2 text-center font-mono-tabular">
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-sans text-slate-500">准确率 Acc</div>
                  <div className="text-sm font-semibold text-slate-900 mt-0.5">
                    {(evalMetrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-sans text-slate-500">精确率 Prec</div>
                  <div className="text-sm font-semibold text-slate-900 mt-0.5">
                    {(evalMetrics.precision * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-sans text-slate-500">召回率 Rec</div>
                  <div className="text-sm font-semibold text-slate-900 mt-0.5">
                    {(evalMetrics.recall * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-[11px] font-sans text-slate-500">F1 调和均值</div>
                  <div className="text-sm font-semibold text-emerald-700 mt-0.5">
                    {(evalMetrics.f1 * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-white border border-slate-200/90 rounded-xl p-6 flex flex-col justify-between">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">
                    受试者工作特征曲线 (ROC Curve) 与 AUC 面积
                  </h3>
                  <p className="text-xs text-slate-500">
                    横轴：假正例率 FPR · 纵轴：真正例率 TPR
                  </p>
                </div>
                <span className="font-mono-tabular text-sm font-semibold text-emerald-700">
                  AUC = {evalMetrics.auc.toFixed(4)}
                </span>
              </div>

              <div className="my-4 flex items-center justify-center">
                <svg viewBox="0 0 340 240" className="w-full max-w-[420px] h-auto overflow-visible">
                  {[0, 0.25, 0.5, 0.75, 1].map((g) => {
                    const x = 45 + g * 260;
                    const y = 200 - g * 175;
                    return (
                      <g key={g}>
                        <line x1={x} y1={25} x2={x} y2={200} stroke="#E2E8F0" strokeWidth="1" />
                        <line x1={45} y1={y} x2={305} y2={y} stroke="#E2E8F0" strokeWidth="1" />
                        <text x={x - 8} y={216} fontSize="9" fill="#64748B">
                          {g.toFixed(2)}
                        </text>
                        <text x={18} y={y + 3} fontSize="9" fill="#64748B">
                          {g.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}

                  <line
                    x1={45}
                    y1={200}
                    x2={305}
                    y2={25}
                    stroke="#94A3B8"
                    strokeDasharray="4 4"
                    strokeWidth="1.2"
                  />

                  <path
                    d={`M 45 200 L 45 55 Q 85 28, 305 25 L 305 200 Z`}
                    fill="rgba(16, 185, 129, 0.12)"
                  />
                  <path
                    d={`M 45 200 L 45 55 Q 85 28, 305 25`}
                    fill="none"
                    stroke="#059669"
                    strokeWidth="2.5"
                  />
                </svg>
              </div>

              <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
                <span>平滑系数 α = {alpha.toFixed(2)}</span>
                <button
                  onClick={handleExportDatasetCsv}
                  className="text-sky-700 hover:underline font-medium"
                >
                  下载当前测试语料 CSV 模板 →
                </button>
              </div>
            </div>
          </div>
        )}

        {reportSlice === 'vocab_matrix' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-900">
                全词表拉普拉斯平滑条件概率矩阵 (α = {alpha.toFixed(2)})
              </h3>
              <button
                onClick={handleExportVocabCsv}
                className="px-3 py-1.5 text-xs font-medium bg-slate-900 text-white rounded-lg"
              >
                一键下载此矩阵 (.csv)
              </button>
            </div>
            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="w-full text-left text-xs font-mono-tabular">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2 px-3 font-sans">词元</th>
                    <th className="py-2 px-3">N(正类)</th>
                    <th className="py-2 px-3">N(负类)</th>
                    <th className="py-2 px-3">P(Xᵢ | 正类, α)</th>
                    <th className="py-2 px-3">P(Xᵢ | 负类, α)</th>
                    <th className="py-2 px-3">ln [P(Xᵢ|1)/P(Xᵢ|0)]</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {scenario.vocab.map((v) => {
                    const p1 =
                      (v.posCount + alpha) /
                      (inference.totalPosTokens + alpha * inference.vocabSize);
                    const p0 =
                      (v.negCount + alpha) /
                      (inference.totalNegTokens + alpha * inference.vocabSize);
                    const llr = Math.log(Math.max(p1, 1e-12) / Math.max(p0, 1e-12));
                    return (
                      <tr key={v.token} className="hover:bg-slate-50">
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">
                          {v.token}
                        </td>
                        <td className="py-2 px-3">{v.posCount}</td>
                        <td className="py-2 px-3">{v.negCount}</td>
                        <td className="py-2 px-3 text-rose-700">{p1.toFixed(5)}</td>
                        <td className="py-2 px-3 text-sky-700">{p0.toFixed(5)}</td>
                        <td className="py-2 px-3 font-semibold">
                          {llr >= 0 ? '+' : ''}
                          {llr.toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {reportSlice === 'csv_upload' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              上传自定义文本 CSV 语料集进行批量贝叶斯评估
            </h3>
            <p className="text-xs text-slate-600">
              支持标准 UTF-8 CSV 文件，每行格式为 <code>文本内容,类别标签(1或0)</code>。上传后将立即重算混淆矩阵与 ROC/AUC 曲线。
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <label className="px-4 py-2.5 text-xs font-medium bg-slate-900 text-white rounded-lg cursor-pointer hover:bg-slate-800 inline-flex items-center gap-2">
                <Upload className="w-4 h-4" />
                <span>选择本地 CSV 文件上传</span>
                <input type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" />
              </label>
              <button
                onClick={handleExportDatasetCsv}
                className="px-4 py-2.5 text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200 rounded-lg"
              >
                下载标准示例 CSV 模板
              </button>
              {customCsvSamples.length > 0 && (
                <button
                  onClick={() => {
                    setCustomCsvSamples([]);
                    setCsvStatusMsg('已恢复默认内置基准测试集。');
                  }}
                  className="px-3 py-2 text-xs text-rose-700 hover:underline"
                >
                  恢复默认测试集
                </button>
              )}
            </div>
            {csvStatusMsg && (
              <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-900">
                {csvStatusMsg}
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  const fullJointParams = Math.pow(2, simVocabDim) - 1;
  const naiveLinearParams = simVocabDim;

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <span>模块 10</span>
            <span aria-hidden="true">·</span>
            <span>六大核心理论与工程实践切片</span>
            <span aria-hidden="true">·</span>
            <span>交互式概念解构</span>
          </div>
          <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
            朴素贝叶斯机理与文本挖掘知识导引（六大切片全解）
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-lg">
          {[
            { id: 1 as const, label: '切片一：三大模型变体' },
            { id: 2 as const, label: '切片二：适用条件与降维' },
            { id: 3 as const, label: '切片三：三大致命陷阱' },
            { id: 4 as const, label: '切片四：误区与停用词' },
            { id: 5 as const, label: '切片五：贝叶斯学习机制' },
            { id: 6 as const, label: '切片六：工业应用场景' },
          ].map((s) => (
            <button
              key={s.id}
              onClick={() => setKnowledgeSlice(s.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                knowledgeSlice === s.id
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {knowledgeSlice === 1 && (
        <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-6">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-semibold text-slate-900">
              【切片一：三大模型变体机理差异】MultinomialNB · GaussianNB · BernoulliNB
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              不同特征数据分布决定了似然函数 P(Xᵢ | Y) 的参数形式，切勿在离散稀疏词频上误用 GaussianNB
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div className="p-5 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3">
              <div className="text-xs font-mono-tabular text-rose-700 font-semibold">
                01. 多项式模型 (MultinomialNB)
              </div>
              <h4 className="text-base font-semibold text-slate-900">
                适用于离散词频计数与 TF-IDF 加权
              </h4>
              <div className="p-3 bg-white rounded-lg border border-slate-200 font-mono-tabular text-xs">
                <KatexMath math="P(X_i \mid Y) = \frac{N_{Y, i} + \alpha}{N_Y + \alpha \cdot |V|}" block />
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                将文档看作从词袋中抽样 L 次的多项式分布实验。高频出现的关键特征词按频次 cᵢ 倍增对数似然权重 cᵢ · ln P(Xᵢ | Y)，是中长文本分类的首选基座。
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3">
              <div className="text-xs font-mono-tabular text-sky-700 font-semibold">
                02. 高斯模型 (GaussianNB)
              </div>
              <h4 className="text-base font-semibold text-slate-900">
                适用于连续实数特征与稠密向量
              </h4>
              <div className="p-3 bg-white rounded-lg border border-slate-200 font-mono-tabular text-xs">
                <KatexMath math="P(x_i \mid Y) = \frac{1}{\sqrt{2\pi\sigma_{Y,i}^2}} \exp\left(-\frac{(x_i - \mu_{Y,i})^2}{2\sigma_{Y,i}^2}\right)" block />
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                假设每个连续特征在给定类别下服从正态分布 N(μ_&#123;Y,i&#125;, σ²_&#123;Y,i&#125;)。适用于文本长度、发信时间间隔、大写字母占比或稠密 Embedding 降维特征。
              </p>
              <div className="pt-1">
                <div className="flex justify-between text-[11px] text-slate-500">
                  <span>模拟连续特征值 x = {gaussianX.toFixed(2)}</span>
                  <span className="font-mono-tabular">
                    高斯密度 = {(Math.exp(-Math.pow(gaussianX - 0.7, 2) / 0.08) / Math.sqrt(2 * Math.PI * 0.04)).toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={gaussianX}
                  onChange={(e) => setGaussianX(Number(e.target.value))}
                  className="w-full accent-slate-900 mt-1"
                />
              </div>
            </div>

            <div className="p-5 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3">
              <div className="text-xs font-mono-tabular text-emerald-700 font-semibold">
                03. 伯努利模型 (BernoulliNB)
              </div>
              <h4 className="text-base font-semibold text-slate-900">
                适用于二值化 (0/1) 词汇存在性
              </h4>
              <div className="p-3 bg-white rounded-lg border border-slate-200 font-mono-tabular text-xs">
                <KatexMath math="P(X \mid Y) = \prod_{i=1}^d P(w_i \mid Y)^{b_i} \left(1 - P(w_i \mid Y)\right)^{1 - b_i}" block />
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                只关心某词在文档中“是否出现”（bᵢ ∈ &#123;0, 1&#125;），不计重复次数；且<strong>显式将“未出现的词” (1 - P(wᵢ | Y)) 纳入连乘</strong>，在超短短信与标题党识别中表现优异。
              </p>
            </div>
          </div>
        </div>
      )}

      {knowledgeSlice === 2 && (
        <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-semibold text-slate-900">
              【切片二：适用条件与空间表示】高维稀疏文本空间与指数级降维计算器
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              为什么词表维度 |V| = 50,000 时，复杂全概率模型彻底失效，而朴素贝叶斯依然毫秒级收敛？
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
            <div className="md:col-span-6 space-y-3">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>调节词表特征维度 d = |V|</span>
                <span className="font-mono-tabular font-semibold">{simVocabDim} 维</span>
              </div>
              <input
                type="range"
                min={5}
                max={40}
                step={1}
                value={simVocabDim}
                onChange={(e) => setSimVocabDim(Number(e.target.value))}
                className="w-full accent-slate-900 cursor-pointer"
              />
              <p className="text-xs text-slate-600 leading-relaxed">
                若不作条件独立假设，刻画 d 个二值特征的完整联合条件分布 P(X₁, …, X_d | Y) 需要估计 2^d - 1 个独立参数；引入条件独立假设后，参数量骤降为线性级 d 个！
              </p>
            </div>

            <div className="md:col-span-6 grid grid-cols-2 gap-4 font-mono-tabular">
              <div className="p-4 rounded-xl bg-rose-50/60 border border-rose-200">
                <div className="text-xs font-sans text-rose-800">无独立假设联合分布参数量</div>
                <div className="text-lg font-bold text-rose-900 mt-1">
                  2^{simVocabDim} - 1 = {fullJointParams.toLocaleString()}
                </div>
                <div className="text-[11px] font-sans text-rose-700 mt-1">
                  维数灾难：样本量远不足以覆盖
                </div>
              </div>
              <div className="p-4 rounded-xl bg-emerald-50/60 border border-emerald-200">
                <div className="text-xs font-sans text-emerald-800">朴素贝叶斯边缘参数量</div>
                <div className="text-lg font-bold text-emerald-900 mt-1">
                  O(d) = {naiveLinearParams} 个参数
                </div>
                <div className="text-[11px] font-sans text-emerald-700 mt-1">
                  高维稀疏文本线性对数加权求和
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {knowledgeSlice === 3 && (
        <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-semibold text-slate-900">
              【切片三：三大致命训练陷阱】零频否决 · 共现膨胀 · 先验偏置
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-2.5">
              <div className="text-xs font-semibold text-rose-700">
                陷阱一 · 未加平滑导致未登录词一票否决
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                当 α = 0 时，只要测试集中出现哪怕 1 个训练集未收录的新词（如新型诈骗词汇“量子币”），P(X_new | Y) = 0 会瞬间乘穿整个连乘式 ∏ P(Xᵢ | Y) = 0。
              </p>
              <button
                onClick={() => setAlpha(0)}
                className="px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md hover:bg-slate-100 font-medium"
              >
                点击体验 α=0 零概率坍缩
              </button>
            </div>

            <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-2.5">
              <div className="text-xs font-semibold text-amber-700">
                陷阱二 · 特征高度相关（如“人工智能”与“AI”）
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                当“人工智能”与“AI”同时出现时，真实信息量仅相当于 1 个概念，但朴素贝叶斯将其视作两个独立事件相乘，导致对数似然比翻倍膨胀，后验概率严重失真。
              </p>
              <div className="text-xs font-mono-tabular text-amber-900 bg-amber-100/60 px-2.5 py-1.5 rounded">
                当前文本独立性违背度：{inference.independenceDeviationScore} / 100
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-2.5">
              <div className="text-xs font-semibold text-sky-700">
                陷阱三 · 类别不平衡歪曲先验概率 P(Y)
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                若正负样本比例为 1:99，先验项 ln P(Y=1) - ln P(Y=0) = -4.59 会产生巨大负偏置。拖动下方滑块实时体验先验倾斜对当前文本后验决策的扭曲：
              </p>
              <div>
                <div className="flex justify-between text-[11px] font-mono-tabular">
                  <span>设定先验 P(正类) = {(inference.priorPos * 100).toFixed(0)}%</span>
                  <button
                    onClick={() => setCustomPriorPos(undefined)}
                    className="text-sky-700 hover:underline font-sans"
                  >
                    复位默认
                  </button>
                </div>
                <input
                  type="range"
                  min={0.02}
                  max={0.98}
                  step={0.02}
                  value={inference.priorPos}
                  onChange={(e) => setCustomPriorPos(Number(e.target.value))}
                  className="w-full accent-slate-900 mt-1"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {knowledgeSlice === 4 && (
        <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-semibold text-slate-900">
              【切片四：误区警示与诊断陷阱】概率极化失真 vs 排序高效性 · 停用词污染
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-5 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3">
              <h4 className="text-sm font-semibold text-slate-900">
                1. 为什么绝对概率值不准（常趋于 0.9999 或 0.0001），但分类准确率依然极高？
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                如 Pedro Domingos 经典论文所证明：强条件独立假设会使对数似然向两端过度发散（因此直接把 predict_proba 当作真实校准概率是常见误区，若需校准需配合 <code>CalibratedClassifierCV</code> Isotonic/Platt 缩放）。但只要后验概率在 <strong>0.5 决策阈值两侧的相对大小顺序（Rank Order）</strong> 正确，0-1 分类损失就为零！
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-3">
              <h4 className="text-sm font-semibold text-slate-900">
                2. 未处理停用词如何破坏条件概率分布？
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                在多项式模型分母 N_Y + α|V| 中，如果未剔除“的、了、是、今天”等极高频停用词，N_Y 会被停用词稀释膨胀数倍，导致真正具有区分度的低频关键词条件概率 P(Xᵢ | Y) 被大幅压低，信噪比恶化。
              </p>
            </div>
          </div>
        </div>
      )}

      {knowledgeSlice === 5 && (
        <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-semibold text-slate-900">
              【切片五：贝叶斯学习机制】先验信念 → 似然证据 → 后验增量更新 (Online Learning)
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-mono-tabular text-slate-500">机制 01 · 生成式建模本质</div>
              <div className="text-sm font-semibold text-slate-900 py-1">
                联合分布建模 <KatexMath math="P(X, Y) = P(Y) \cdot P(X \mid Y)" />
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                不同于逻辑回归/SVM 直接拟合判别边界 P(Y | X)，朴素贝叶斯先为每个类别单独建立“词汇生成模型” P(X | Y)，再由贝叶斯公式反转求得后验。
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-mono-tabular text-slate-500">机制 02 · O(1) 增量在线学习</div>
              <div className="text-sm font-semibold text-slate-900">
                流式更新词频计数器 (partial_fit)
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                当新标注样本到达时，无需像神经网络那样重新反向传播训练整个数据集，只需在对应类别词频矩阵 N_&#123;Y,i&#125; 上执行一次加法即可完成模型热更新！
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-xs font-mono-tabular text-slate-500">机制 03 · 共轭先验解释</div>
              <div className="text-sm font-semibold text-slate-900">
                Dirichlet-Multinomial 共轭对偶
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                拉普拉斯平滑参数 α 在贝叶斯统计学中等价于对称狄利克雷先验 Dir(α, …, α) 的超参数，实现了频率派计数向贝叶斯后验期望的平滑过渡。
              </p>
            </div>
          </div>
        </div>
      )}

      {knowledgeSlice === 6 && (
        <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-5">
          <div className="border-b border-slate-100 pb-3">
            <h3 className="text-base font-semibold text-slate-900">
              【切片六：应用场景】工业界高吞吐与小样本冷启动四大黄金阵地
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                tag: '场景 01 · 毫秒级网关',
                title: '垃圾短信与钓鱼邮件第一道防线',
                desc: '在千万级 QPS 运营商网关中作为一级粗排过滤器，以微秒级延迟拦截 95% 显性垃圾流量，减轻下游大模型负担。',
              },
              {
                tag: '场景 02 · 冷启动利器',
                title: '小样本工单与医疗分诊路由',
                desc: '如 Ng & Jordan (2001) 经典结论：在训练样本极少时，生成式朴素贝叶斯以 O(log n) 速率收敛，显著优于判别式模型。',
              },
              {
                tag: '场景 03 · 实时流计算',
                title: '舆情监控与影评情感极性流',
                desc: '支持 partial_fit 在线增量更新词表，用户点击“标记为垃圾邮件”后毫秒级更新个人专属贝叶斯先验与似然表。',
              },
              {
                tag: '场景 04 · 高维基准线',
                title: '多类别新闻与专利自动归档',
                desc: '面对上百个学科分类与十万维专业术语词表，配合 TF-IDF 与卡方特征选择，提供高可解释性的白盒词权重依据。',
              },
            ].map((card) => (
              <div
                key={card.title}
                className="p-4 rounded-xl bg-[#F8FAFC] border border-slate-200 space-y-2"
              >
                <div className="text-xs font-mono-tabular text-sky-700">{card.tag}</div>
                <h4 className="text-sm font-semibold text-slate-900">{card.title}</h4>
                <p className="text-xs text-slate-600 leading-relaxed">{card.desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
