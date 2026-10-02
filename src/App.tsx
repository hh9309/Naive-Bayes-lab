import React, { useState, useMemo } from 'react';
import { BAYES_CASES, BayesCaseScenario } from './data/bayesCases';
import { runBayesInference } from './utils/bayesEngine';
import { Modules1To4 } from './components/Modules1To4';
import { Modules5To6 } from './components/Modules5To6';
import { Modules8To10 } from './components/Modules8To10';
import { AiDiagnosticModal } from './components/AiDiagnosticModal';

const TEN_MODULES = [
  { id: 1, code: '1. 贝叶斯代数', subtitle: '后验公式与对数防下溢' },
  { id: 2, code: '2. 词频对数', subtitle: '3D 手势旋转概率柱群' },
  { id: 3, code: '3. 平滑切片', subtitle: '拉普拉斯 α 零概率修复' },
  { id: 4, code: '4. 条件独立性', subtitle: '贝叶斯网络DAG与桑基流' },
  { id: 5, code: '5. 四大案例', subtitle: '短信/影评/新闻/钓鱼库' },
  { id: 6, code: '6. 代码引擎', subtitle: 'Sklearn 与 NumPy 手写' },
  { id: 7, code: '7. AI对话窗口', subtitle: '强相关违背与卡方随诊' },
  { id: 8, code: '8. 全流程导引', subtitle: '分词→向量→推导→决策' },
  { id: 9, code: '9. 数据报告', subtitle: 'CSV上传与ROC/AUC导出' },
  { id: 10, code: '10. 知识导引', subtitle: '六大机理与陷阱切片' },
] as const;

export default function App() {
  const [activeModule, setActiveModule] = useState<number>(1);
  const [scenarioId, setScenarioId] = useState<BayesCaseScenario['id']>('spam');
  const [alpha, setAlpha] = useState<number>(1.0);
  const [filterStopwords, setFilterStopwords] = useState<boolean>(true);
  const [customPriorPos, setCustomPriorPos] = useState<number | undefined>(undefined);
  const [inputText, setInputText] = useState<string>(BAYES_CASES.spam.defaultTestText);

  const [isOpenFloatingAi, setIsOpenFloatingAi] = useState<boolean>(false);
  const [externalAiPrompt, setExternalAiPrompt] = useState<string | undefined>(undefined);

  const scenario = BAYES_CASES[scenarioId];

  const handleSelectScenario = (newId: BayesCaseScenario['id']) => {
    setScenarioId(newId);
    setCustomPriorPos(undefined);
    setInputText(BAYES_CASES[newId].defaultTestText);
  };

  const inference = useMemo(() => {
    return runBayesInference(inputText, scenario, alpha, filterStopwords, customPriorPos);
  }, [inputText, scenario, alpha, filterStopwords, customPriorPos]);

  const handleTriggerAiModal = (prompt?: string) => {
    if (prompt) {
      setExternalAiPrompt(prompt);
    }
    if (activeModule !== 7) {
      setIsOpenFloatingAi(true);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F6F7F9] text-[#1E293B]">
      <header className="bg-white border-b border-slate-200/90 px-6 py-3.5 flex items-center justify-between sticky top-0 z-30 no-print">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveModule(1);
          }}
          className="text-lg font-bold tracking-tight text-slate-900 font-serif-title whitespace-nowrap"
        >
          朴素贝叶斯文本过滤实验室
        </a>

        <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-600">
          <button
            onClick={() => setActiveModule(1)}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
              activeModule <= 4 ? 'text-slate-900 underline underline-offset-4' : ''
            }`}
          >
            代数与空间演播
          </button>
          <button
            onClick={() => setActiveModule(5)}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
              activeModule === 5 ? 'text-slate-900 underline underline-offset-4' : ''
            }`}
          >
            四大实战案例
          </button>
          <button
            onClick={() => setActiveModule(6)}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
              activeModule === 6 ? 'text-slate-900 underline underline-offset-4' : ''
            }`}
          >
            代码引擎
          </button>
          <button
            onClick={() => setActiveModule(8)}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
              activeModule === 8 || activeModule === 9
                ? 'text-slate-900 underline underline-offset-4'
                : ''
            }`}
          >
            全流程与报告
          </button>
          <button
            onClick={() => setActiveModule(10)}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap ${
              activeModule === 10 ? 'text-slate-900 underline underline-offset-4' : ''
            }`}
          >
            六大知识切片
          </button>
        </nav>

        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setActiveModule(7)}
            className="px-3.5 py-1.5 text-xs font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap"
          >
            AI 独立性随诊
          </button>
        </div>
      </header>

      <main className="max-w-[1380px] w-full mx-auto px-4 sm:px-6 py-6 flex-1 space-y-6">
        <section className="bg-white border border-slate-200/90 rounded-xl overflow-hidden shadow-2xs no-print">
          <div className="px-5 py-2.5 bg-slate-50/90 border-b border-slate-200/90 flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-semibold text-slate-800 tracking-wide">
              朴素贝叶斯与垃圾文本过滤实验室 · “10大”核心模块矩阵切片切换器
            </span>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span>当前案例：{scenario.name}</span>
              <span aria-hidden="true">·</span>
              <span>平滑系数 α = {alpha.toFixed(2)}</span>
              <span aria-hidden="true">·</span>
              <span>点击任一模块切片即刻切换实验台</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 divide-x divide-y divide-slate-200/80">
            {TEN_MODULES.map((mod) => {
              const isSelected = activeModule === mod.id;
              return (
                <button
                  key={mod.id}
                  onClick={() => setActiveModule(mod.id)}
                  className={`p-3.5 text-left transition-all flex flex-col justify-between ${
                    isSelected
                      ? 'bg-slate-900 text-white'
                      : 'bg-white text-slate-800 hover:bg-slate-50/90'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold whitespace-nowrap">
                      {mod.code}
                    </span>
                    {isSelected && (
                      <span className="text-[10px] font-mono-tabular text-sky-300">
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <span
                    className={`text-[11px] mt-1 truncate ${
                      isSelected ? 'text-slate-300' : 'text-slate-500'
                    }`}
                  >
                    {mod.subtitle}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="bg-white border border-slate-200/90 rounded-xl p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
          <div className="lg:col-span-7 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
            <select
              value={scenarioId}
              onChange={(e) => handleSelectScenario(e.target.value as BayesCaseScenario['id'])}
              aria-label="选择实验场景案例"
              className="text-xs font-medium bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-slate-900 shrink-0 cursor-pointer"
            >
              <option value="spam">案例1：垃圾短信过滤</option>
              <option value="sentiment">案例2：影评情感识别</option>
              <option value="news">案例3：新闻主题归类</option>
              <option value="phishing">案例4：钓鱼邮件检测</option>
            </select>

            <div className="flex-1 flex items-center bg-[#F8FAFC] border border-slate-200 rounded-lg px-3 py-1.5">
              <span className="text-xs text-slate-400 mr-2 whitespace-nowrap">待测文本:</span>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full text-xs text-slate-900 bg-transparent focus:outline-none"
                placeholder="输入任意中文文本实时推导贝叶斯后验概率..."
              />
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-wrap items-center justify-between gap-3 pt-2 lg:pt-0 border-t lg:border-t-0 lg:border-l border-slate-100 lg:pl-4">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 whitespace-nowrap">平滑 α:</span>
              <input
                type="range"
                min={0}
                max={5}
                step={0.1}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-20 accent-slate-900 cursor-pointer"
              />
              <span className="text-xs font-mono-tabular font-semibold text-slate-800 w-8">
                {alpha.toFixed(1)}
              </span>
            </div>

            <div className="flex items-center gap-3 text-xs font-mono-tabular">
              <div>
                <span className="text-slate-500 font-sans">正类后验: </span>
                <strong className="text-rose-700">
                  {(inference.posteriorPos * 100).toFixed(2)}%
                </strong>
              </div>
              <span className="text-slate-300">|</span>
              <div>
                <span className="text-slate-500 font-sans">判决: </span>
                <strong
                  className={
                    inference.predictedClass === 1 ? 'text-rose-700' : 'text-sky-700'
                  }
                >
                  {inference.predictedLabel.split(' ')[0]}
                </strong>
              </div>
            </div>
          </div>
        </section>

        {(activeModule === 1 || activeModule === 2 || activeModule === 3 || activeModule === 4) && (
          <Modules1To4
            activeModule={activeModule}
            scenario={scenario}
            inference={inference}
            alpha={alpha}
            setAlpha={setAlpha}
            inputText={inputText}
            setInputText={setInputText}
            onOpenAiModal={handleTriggerAiModal}
          />
        )}

        {(activeModule === 5 || activeModule === 6) && (
          <Modules5To6
            activeModule={activeModule}
            scenario={scenario}
            onSelectScenario={handleSelectScenario}
            inference={inference}
            alpha={alpha}
            setAlpha={setAlpha}
            inputText={inputText}
            setInputText={setInputText}
            onJumpToModule={(mod) => setActiveModule(mod)}
          />
        )}

        {activeModule === 7 && (
          <AiDiagnosticModal
            isOpenFloating={isOpenFloatingAi}
            setIsOpenFloating={setIsOpenFloatingAi}
            isDedicatedModule7={true}
            scenario={scenario}
            inference={inference}
            alpha={alpha}
            inputText={inputText}
            externalPrompt={externalAiPrompt}
            clearExternalPrompt={() => setExternalAiPrompt(undefined)}
          />
        )}

        {(activeModule === 8 || activeModule === 9 || activeModule === 10) && (
          <Modules8To10
            activeModule={activeModule}
            scenario={scenario}
            inference={inference}
            alpha={alpha}
            setAlpha={setAlpha}
            filterStopwords={filterStopwords}
            setFilterStopwords={setFilterStopwords}
            customPriorPos={customPriorPos}
            setCustomPriorPos={setCustomPriorPos}
            inputText={inputText}
            setInputText={setInputText}
          />
        )}
      </main>

      {activeModule !== 7 && (
        <AiDiagnosticModal
          isOpenFloating={isOpenFloatingAi}
          setIsOpenFloating={setIsOpenFloatingAi}
          isDedicatedModule7={false}
          scenario={scenario}
          inference={inference}
          alpha={alpha}
          inputText={inputText}
          externalPrompt={externalAiPrompt}
          clearExternalPrompt={() => setExternalAiPrompt(undefined)}
        />
      )}

      <footer className="bg-white border-t border-slate-200/80 py-4 px-6 mt-10 no-print">
        <div className="max-w-[1380px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <div>
            朴素贝叶斯文本过滤实验室 (Naive Bayes Lab) · 交互式教学与算法演练平台
          </div>
          <div className="flex items-center gap-3">
            <span>支持拉普拉斯平滑切片</span>
            <span>·</span>
            <span>3D 词频似然演播</span>
            <span>·</span>
            <span>卡方 χ² 与互信息特征选择</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
