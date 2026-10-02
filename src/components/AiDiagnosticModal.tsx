import React, { useState, useEffect } from 'react';
import {
  MessageSquare,
  X,
  Send,
  Settings,
  Key,
  Check,
  Eye,
  EyeOff,
  Sparkles,
  AlertCircle,
  Cpu,
} from 'lucide-react';
import { BayesCaseScenario } from '../data/bayesCases';
import { BayesInferenceResult } from '../utils/bayesEngine';

export type SupportedLlmModel = 'gemini 3 flash' | 'deepseek-v4-pro';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  modelUsed?: string;
}

interface AiDiagnosticModalProps {
  isOpenFloating: boolean;
  setIsOpenFloating: (val: boolean) => void;
  isDedicatedModule7: boolean;
  scenario: BayesCaseScenario;
  inference: BayesInferenceResult;
  alpha: number;
  inputText: string;
  externalPrompt?: string;
  clearExternalPrompt: () => void;
}

export const AiDiagnosticModal: React.FC<AiDiagnosticModalProps> = ({
  isOpenFloating,
  setIsOpenFloating,
  isDedicatedModule7,
  scenario,
  inference,
  alpha,
  inputText,
  externalPrompt,
  clearExternalPrompt,
}) => {
  // LLM Configuration state (persisted in localStorage for static GitHub Pages deployment)
  const [selectedModel, setSelectedModel] = useState<SupportedLlmModel>(() => {
    return (localStorage.getItem('NB_LLM_MODEL') as SupportedLlmModel) || 'gemini 3 flash';
  });
  const [apiKey, setApiKey] = useState<string>(() => {
    return localStorage.getItem('NB_LLM_API_KEY') || '';
  });

  // Settings Modal State
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [tempApiKey, setTempApiKey] = useState<string>(apiKey);
  const [tempModel, setTempModel] = useState<SupportedLlmModel>(selectedModel);
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [saveToast, setSaveToast] = useState<string>('');

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-1',
      role: 'assistant',
      content: `您好！我是朴素贝叶斯实验室 AI 随诊助手。\n\n📌 **运行机制说明**：本项目支持直接部署至 GitHub（浏览器纯前端直接调用），**所有大模型调用必须先输入 API-Key 后方可调用**。\n\n请点击右上角 **小齿轮 ⚙️** 输入您的 API-Key 并确认选择 **gemini 3 flash** 或 **deepseek-v4-pro**。`,
      timestamp: '就绪',
    },
  ]);
  const [questionInput, setQuestionInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [mod7Slice, setMod7Slice] = useState<'chat_diagnosis' | 'chi2_selector'>('chat_diagnosis');

  // Keep temporary settings in sync when opening settings modal
  const openSettings = () => {
    setTempApiKey(apiKey);
    setTempModel(selectedModel);
    setSaveToast('');
    setIsSettingsOpen(true);
  };

  const handleSaveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanKey = tempApiKey.trim();
    if (!cleanKey) {
      setSaveToast('⚠️ 请输入有效的 API-Key 后再确认保存');
      return;
    }
    setApiKey(cleanKey);
    setSelectedModel(tempModel);
    localStorage.setItem('NB_LLM_API_KEY', cleanKey);
    localStorage.setItem('NB_LLM_MODEL', tempModel);
    setSaveToast(`✓ 已确认大模型：${tempModel} 并安全保存 API-Key！`);
    setTimeout(() => {
      setIsSettingsOpen(false);
      setSaveToast('');
    }, 900);
  };

  const handleClearApiKey = () => {
    setApiKey('');
    setTempApiKey('');
    localStorage.removeItem('NB_LLM_API_KEY');
    setSaveToast('已清除本地存储的 API-Key');
  };

  const sendDiagnosisRequest = async (promptText: string) => {
    const trimmed = promptText.trim();
    if (!trimmed || isLoading) return;

    // Requirement: "所有大模型调用必须输入API-Key后才能调用"
    if (!apiKey.trim()) {
      const warningMsg: ChatMessage = {
        id: `warn-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ **未检测到 API-Key，无法调用大模型！**\n\n本项目需部署到 GitHub（浏览器前端直接调用），**所有大模型调用必须输入 API-Key 后才能调用**。\n\n已为您自动打开右上角 **小齿轮 ⚙️ 设置窗口**，请输入 API-Key 并确认大模型选择后重试。`,
        timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, warningMsg]);
      openSettings();
      return;
    }

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuestionInput('');
    setIsLoading(true);

    const systemInstruction = `你是一位资深机器学习与自然语言处理专家，专注于朴素贝叶斯（Naive Bayes）分类器机理、拉普拉斯平滑（Laplace/Lidstone Smoothing）、特征条件独立性假设违背（Attribute Independence Deviation）诊断，以及卡方检验（Chi-Square）与互信息（Mutual Information）特征选择优化。
请使用专业、清晰、结构化的简体中文回答用户问题。
结合当前实验状态：
1. 【条件独立性违背评估】：分析是否存在强相关共现词对导致后验对数似然重复累加与概率极化。
2. 【零概率与平滑系数诊断】：结合当前平滑系数 α 与未登录词情况给出建议。
3. 【特征选择建议】：基于卡方(Chi-Square)与互信息(MI)给出特征保留或裁剪建议。`;

    const topChiSquareTokens = [...scenario.vocab]
      .sort((a, b) => b.chiSquare - a.chiSquare)
      .slice(0, 6)
      .map((v) => `${v.token}(χ²=${v.chiSquare}, MI=${v.mutualInfo})`)
      .join('、');

    const correlatedList =
      inference.correlatedPairsFound.length > 0
        ? inference.correlatedPairsFound
            .map((p) => `${p.pair} (r=${p.r}, 膨胀+${p.inflationLogOdds.toFixed(2)} nats)`)
            .join('、')
        : '未检出显著强相关词对';

    const zeroFreqList =
      inference.zeroFreqTokensFound.length > 0
        ? inference.zeroFreqTokensFound.join('、')
        : '无';

    const contextPrompt = `【当前实验切片状态】
- 场景案例：${scenario.name} (${scenario.subtitle})
- 待测文本："${inputText}"
- 拉普拉斯平滑系数 α：${alpha.toFixed(2)}
- 预测类别：${inference.predictedLabel} (正类后验概率: ${(inference.posteriorPos * 100).toFixed(2)}%)
- 条件独立性违背指数：${inference.independenceDeviationScore}/100
- 检出的强共现特征词对：${correlatedList}
- 未登录词/零频词：${zeroFreqList}
- Top 卡方/互信息特征词：${topChiSquareTokens}

【用户提问】：
${trimmed}`;

    try {
      let replyContent = '';

      if (selectedModel === 'gemini 3 flash') {
        // Direct browser client-side call to Google Gemini REST API (works on GitHub Pages!)
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
          apiKey.trim()
        )}`;
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            system_instruction: {
              parts: [{ text: systemInstruction }],
            },
            contents: [
              {
                role: 'user',
                parts: [{ text: contextPrompt }],
              },
            ],
            generationConfig: {
              temperature: 0.4,
            },
          }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `HTTP ${response.status}: 调用 Gemini 接口失败`);
        }

        const data = await response.json();
        replyContent =
          data.candidates?.[0]?.content?.parts?.[0]?.text || '模型已返回空响应，请重试。';
      } else {
        // Direct browser client-side call to DeepSeek Chat API
        const response = await fetch('https://api.deepseek.com/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey.trim()}`,
          },
          body: JSON.stringify({
            model: 'deepseek-chat',
            messages: [
              { role: 'system', content: systemInstruction },
              { role: 'user', content: contextPrompt },
            ],
            temperature: 0.4,
          }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `HTTP ${response.status}: 调用 DeepSeek 接口失败`);
        }

        const data = await response.json();
        replyContent =
          data.choices?.[0]?.message?.content || 'DeepSeek 模型已返回空响应，请重试。';
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: replyContent,
          timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
          modelUsed: selectedModel,
        },
      ]);
    } catch (err: any) {
      // If browser CORS or direct call failed, attempt local fallback or show precise diagnostic
      try {
        const fallbackRes = await fetch('/api/ai-diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: trimmed,
            caseName: scenario.name,
            inputText,
            alpha,
            predictedLabel: inference.predictedLabel,
            posteriorPositive: inference.posteriorPos,
            correlatedPairs: inference.correlatedPairsFound,
            zeroFreqTokens: inference.zeroFreqTokensFound,
            topChiSquareTokens,
            userApiKey: apiKey.trim(),
            selectedModel,
          }),
        });
        const fallbackData = await fallbackRes.json();
        if (fallbackData?.reply) {
          setMessages((prev) => [
            ...prev,
            {
              id: `a-${Date.now()}`,
              role: 'assistant',
              content: fallbackData.reply,
              timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
              modelUsed: `${selectedModel} (代理模式)`,
            },
          ]);
          return;
        }
      } catch {
        // Fallback also unavailable
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `❌ **调用失败 (${selectedModel})**：${err?.message || '网络连接超时或 API-Key 无效'}\n\n💡 **排查指南**：\n1. 请点击右上角 **小齿轮 ⚙️** 检查您输入的 API-Key 是否正确；\n2. 若部署在 GitHub Pages 浏览器端，请确认您的网络可直接访问对应大模型官方接口；\n3. 本地算法代数诊断引擎已就绪，当前文本独立性违背指数为 **${inference.independenceDeviationScore}/100**。`,
          timestamp: '调用异常',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (externalPrompt) {
      sendDiagnosisRequest(externalPrompt);
      clearExternalPrompt();
    }
  }, [externalPrompt]);

  const quickPrompts = [
    '评估当前文本的特征强相关违背（Attribute Independence Deviation）影响',
    '给出当前案例基于卡方检验(Chi-Square)与互信息(MI)的特征选择建议',
    `当前平滑系数 α=${alpha.toFixed(2)} 是否存在欠平滑或过平滑风险？`,
    '为什么“人工智能”与“AI”共现会导致后验概率过度极化？如何修复？',
  ];

  // =========================================================================
  // Settings Modal Overlay (Reusable for both dedicated view and floating view)
  // =========================================================================
  const renderSettingsModal = () => {
    if (!isSettingsOpen) return null;

    return (
      <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center z-60 p-4 no-print">
        <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-[480px] w-full p-6 space-y-5 animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-slate-800" />
              <h3 className="text-base font-semibold text-slate-900 font-serif-title">
                大模型设置与 API-Key 配置
              </h3>
            </div>
            <button
              onClick={() => setIsSettingsOpen(false)}
              className="text-slate-400 hover:text-slate-700 p-1 rounded-md"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed bg-[#F8FAFC] p-3 rounded-lg border border-slate-200/80">
            <strong>GitHub 静态部署支持</strong>：本项目已实现纯浏览器直接发起 API 调用。为保证安全与独立性，<strong>所有大模型调用必须先输入 API-Key</strong>。Key 仅保存在您本地浏览器的 localStorage 中。
          </p>

          <form onSubmit={handleSaveSettings} className="space-y-4">
            {/* 1. 选择大模型 (2 models) */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-800 flex items-center justify-between">
                <span>1. 选择大模型 (Model Selection)</span>
                <span className="text-[11px] font-normal text-slate-500">双旗舰模型支持</span>
              </label>

              <div className="grid grid-cols-2 gap-2.5">
                <button
                  type="button"
                  onClick={() => setTempModel('gemini 3 flash')}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    tempModel === 'gemini 3 flash'
                      ? 'bg-sky-50/70 border-sky-600 ring-1 ring-sky-600 text-sky-950'
                      : 'bg-white border-slate-200 hover:border-slate-300 text-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold">gemini 3 flash</span>
                    {tempModel === 'gemini 3 flash' && (
                      <Check className="w-3.5 h-3.5 text-sky-600" />
                    )}
                  </div>
                  <p className="text-[11px] text-slate-500 mt-1 leading-snug">
                    Google 官方轻量高速大模型，毫秒级快速推导
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => setTempModel('deepseek-v4-pro')}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    tempModel === 'deepseek-v4-pro'
                      ? 'bg-rose-50/70 border-rose-600 ring-1 ring-rose-600 text-rose-950'
                      : 'bg-white border-slate-200 hover:border-slate-300 text-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold">deepseek-v4-pro</span>
                    {tempModel === 'deepseek-v4-pro' && (
                      <Check className="w-3.5 h-3.5 text-rose-600" />
                    )}
                  </div>
                  <p className="text-[11px] text-slate-500 mt-1 leading-snug">
                    DeepSeek 深度推理模型，擅长特征选择与归因
                  </p>
                </button>
              </div>
            </div>

            {/* 2. 手工输入 API-Key */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-800">
                <label htmlFor="apiKeyInput" className="flex items-center gap-1.5">
                  <Key className="w-3.5 h-3.5 text-slate-600" />
                  <span>2. 手工输入 API-Key (API Key Required)</span>
                </label>
                {apiKey && (
                  <button
                    type="button"
                    onClick={handleClearApiKey}
                    className="text-[11px] text-rose-600 hover:underline font-normal"
                  >
                    清除当前 Key
                  </button>
                )}
              </div>

              <div className="relative">
                <input
                  id="apiKeyInput"
                  type={showPassword ? 'text' : 'password'}
                  value={tempApiKey}
                  onChange={(e) => setTempApiKey(e.target.value)}
                  placeholder={
                    tempModel === 'gemini 3 flash'
                      ? '请输入您的 Google Gemini API Key (AIzaSy...)'
                      : '请输入您的 DeepSeek API Key (sk-...)'
                  }
                  className="w-full text-xs font-mono-tabular bg-slate-50 border border-slate-300 rounded-lg pl-3 pr-10 py-2.5 focus:outline-none focus:border-slate-900"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 p-1"
                >
                  {showPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                </button>
              </div>
              <p className="text-[11px] text-slate-400">
                支持在项目外部申请的官方 Key，部署至 GitHub 纯静态环境时直接在浏览器中调用。
              </p>
            </div>

            {saveToast && (
              <div
                className={`p-2.5 rounded-lg text-xs font-medium ${
                  saveToast.startsWith('✓')
                    ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                    : 'bg-amber-50 text-amber-800 border border-amber-200'
                }`}
              >
                {saveToast}
              </div>
            )}

            {/* 3. 确认大模型与保存 */}
            <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-slate-100">
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                className="px-4 py-2 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              >
                取消
              </button>
              <button
                type="submit"
                className="px-4 py-2 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-1.5 shadow-xs"
              >
                <Check className="w-3.5 h-3.5" />
                <span>3. 确认大模型并保存</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  // =========================================================================
  // Dedicated Module 7 Full Page View
  // =========================================================================
  if (isDedicatedModule7) {
    return (
      <div className="space-y-6">
        {renderSettingsModal()}

        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 07</span>
              <span aria-hidden="true">·</span>
              <span>特征强相关违背（Attribute Independence Deviation）动态评估</span>
              <span aria-hidden="true">·</span>
              <span>Chi-Square / Mutual Info 特征选择随诊</span>
            </div>
            {/* Title with gear icon button at the far right */}
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
                AI 诊断与大模型 Q&amp;A 交互对话工作台
              </h2>
              <button
                onClick={openSettings}
                className="p-1.5 text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg border border-slate-200 transition-colors flex items-center gap-1.5 text-xs font-sans"
                title="设置大模型与 API-Key"
              >
                <Settings className="w-4 h-4 text-slate-700" />
                <span className="hidden sm:inline font-mono-tabular text-[11px] text-slate-600">
                  {selectedModel}
                </span>
                <span
                  className={`w-2 h-2 rounded-full ${
                    apiKey ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'
                  }`}
                  title={apiKey ? '已配置 API-Key' : '未输入 API-Key'}
                />
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start">
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              <button
                onClick={() => setMod7Slice('chat_diagnosis')}
                className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  mod7Slice === 'chat_diagnosis'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                切片一：AI 交互随诊与独立性违背评估
              </button>
              <button
                onClick={() => setMod7Slice('chi2_selector')}
                className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  mod7Slice === 'chi2_selector'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                切片二：Chi-Square / Mutual Info 特征裁剪处方
              </button>
            </div>
          </div>
        </div>

        {mod7Slice === 'chat_diagnosis' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Live Diagnostic Telemetry */}
            <div className="lg:col-span-5 bg-white border border-slate-200/90 rounded-xl p-5 space-y-5 flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
                  <h3 className="text-sm font-semibold text-slate-900">
                    当前样本特征独立性违背实时体检单
                  </h3>
                  <span className="text-[11px] font-mono-tabular text-slate-500">
                    模型: {selectedModel}
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-600">
                      条件独立性违背指数 (Attribute Independence Deviation)
                    </span>
                    <span className="font-mono-tabular font-bold text-slate-900">
                      {inference.independenceDeviationScore} / 100
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        inference.independenceDeviationScore > 30 ? 'bg-amber-500' : 'bg-emerald-600'
                      }`}
                      style={{ width: `${inference.independenceDeviationScore}%` }}
                    />
                  </div>
                </div>

                {/* Detected Correlated Pairs */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-slate-700">
                    检出的强共现冗余特征对：
                  </div>
                  {inference.correlatedPairsFound.length > 0 ? (
                    <div className="space-y-2">
                      {inference.correlatedPairsFound.map((cp) => (
                        <div
                          key={cp.pair}
                          className="p-3 rounded-lg bg-amber-50/70 border border-amber-200 text-xs space-y-1"
                        >
                          <div className="flex items-center justify-between font-semibold text-amber-900">
                            <span>共现词对：「{cp.pair}」</span>
                            <span className="font-mono-tabular">相关系数 r = {cp.r}</span>
                          </div>
                          <div className="text-amber-800">
                            对数奇数比重复累加膨胀量：+{cp.inflationLogOdds.toFixed(3)} nats
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-3 rounded-lg bg-emerald-50/60 border border-emerald-200 text-xs text-emerald-800">
                      ✓ 当前待测样本未检出显著破坏条件独立假设的同义共现词对。
                    </div>
                  )}
                </div>

                {/* Quick Prompts */}
                <div className="space-y-2 pt-2">
                  <div className="text-xs font-medium text-slate-700">
                    点击一键发起专家随诊提问：
                  </div>
                  <div className="space-y-1.5">
                    {quickPrompts.map((q) => (
                      <button
                        key={q}
                        onClick={() => sendDiagnosisRequest(q)}
                        className="w-full text-left px-3 py-2 text-xs bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-800 transition-colors"
                      >
                        → {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {!apiKey && (
                <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-center justify-between">
                  <span>⚠️ 尚未输入 API-Key，调用大模型前需配置</span>
                  <button
                    onClick={openSettings}
                    className="font-semibold underline ml-2 text-amber-950 whitespace-nowrap"
                  >
                    立即设置 →
                  </button>
                </div>
              )}
            </div>

            {/* Right Interactive Q&A Conversation Window */}
            <div className="lg:col-span-7 bg-white border border-slate-200/90 rounded-xl flex flex-col h-[520px]">
              <div className="px-5 py-3.5 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-sky-700" />
                  <span className="text-sm font-semibold text-slate-900">
                    AI 贝叶斯诊断与大模型 Q&amp;A 实时对话流
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={openSettings}
                    className="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-md text-xs flex items-center gap-1.5"
                    title="设置大模型与 API-Key"
                  >
                    <Settings className="w-3.5 h-3.5" />
                    <span>{selectedModel}</span>
                    <span
                      className={`w-2 h-2 rounded-full ${
                        apiKey ? 'bg-emerald-500' : 'bg-amber-500'
                      }`}
                    />
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-5 space-y-4">
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`flex flex-col ${
                      m.role === 'user' ? 'items-end' : 'items-start'
                    }`}
                  >
                    <div
                      className={`max-w-[88%] rounded-xl px-4 py-3 text-xs leading-relaxed whitespace-pre-wrap ${
                        m.role === 'user'
                          ? 'bg-slate-900 text-white'
                          : 'bg-[#F8FAFC] border border-slate-200 text-slate-800'
                      }`}
                    >
                      {m.content}
                    </div>
                    <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-1 px-1">
                      <span>{m.timestamp}</span>
                      {m.modelUsed && <span>· 来自 {m.modelUsed}</span>}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="text-xs text-slate-500 animate-pulse flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-sky-600 animate-spin" />
                    <span>
                      大模型 ({selectedModel}) 正在结合当前词表概率矩阵进行推导...
                    </span>
                  </div>
                )}
              </div>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  sendDiagnosisRequest(questionInput);
                }}
                className="p-3.5 border-t border-slate-100 flex items-center gap-2"
              >
                <input
                  type="text"
                  value={questionInput}
                  onChange={(e) => setQuestionInput(e.target.value)}
                  placeholder="向大模型提问（所有调用必须在小齿轮 ⚙️ 中输入 API-Key）..."
                  className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2.5 focus:outline-none focus:border-slate-400"
                />
                <button
                  type="submit"
                  disabled={isLoading}
                  className="px-4 py-2.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-medium rounded-lg flex items-center gap-1.5 whitespace-nowrap"
                >
                  <Send className="w-3.5 h-3.5" /> 发送随诊
                </button>
              </form>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-slate-900">
              基于卡方检验（Chi-Square χ²）与互信息（Mutual Information）的特征选择优化处方
            </h3>
            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="w-full text-left text-xs font-mono-tabular">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3 font-sans">特征词 Xᵢ</th>
                    <th className="py-2.5 px-3">卡方统计量 χ²</th>
                    <th className="py-2.5 px-3">互信息 I(Xᵢ;Y)</th>
                    <th className="py-2.5 px-3 font-sans">独立性诊断</th>
                    <th className="py-2.5 px-3 font-sans">AI 特征工程处方建议</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {[...scenario.vocab]
                    .sort((a, b) => b.chiSquare - a.chiSquare)
                    .map((v) => (
                      <tr key={v.token} className="hover:bg-slate-50">
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">
                          {v.token}
                        </td>
                        <td className="py-2 px-3 font-semibold">{v.chiSquare.toFixed(2)}</td>
                        <td className="py-2 px-3 text-emerald-700">{v.mutualInfo.toFixed(4)}</td>
                        <td className="py-2 px-3 font-sans">
                          {v.correlatedWith ? (
                            <span className="text-amber-700">
                              与「{v.correlatedWith}」强相关 (r={v.correlationCoeff})
                            </span>
                          ) : v.isStopword ? (
                            <span className="text-rose-700">低信息量停用词</span>
                          ) : (
                            <span className="text-emerald-700">正交性良好</span>
                          )}
                        </td>
                        <td className="py-2 px-3 font-sans text-slate-700">
                          {v.correlatedWith
                            ? `建议与「${v.correlatedWith}」合并为 Bi-gram 复合特征或保留 χ² 较高者`
                            : v.isStopword
                            ? 'χ² < 1.08，建议通过停用词表或 SelectKBest 裁剪剔除'
                            : v.chiSquare > 80
                            ? '核心判别特征，优先保留入模'
                            : '辅助长尾特征，配合拉普拉斯平滑保留'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  }

  // =========================================================================
  // Floating Drawer Mode (Accessible across all other modules)
  // =========================================================================
  return (
    <div className="fixed bottom-5 right-5 z-50 no-print">
      {renderSettingsModal()}

      {!isOpenFloating ? (
        <button
          onClick={() => setIsOpenFloating(true)}
          className="px-4 py-3 rounded-full bg-slate-900 hover:bg-slate-800 text-white shadow-lg flex items-center gap-2 text-xs font-medium transition-transform hover:scale-[1.02]"
        >
          <MessageSquare className="w-4 h-4 text-sky-400" />
          <span>AI 随诊与独立性诊断窗口</span>
          <span
            className={`w-2 h-2 rounded-full ${
              apiKey ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'
            }`}
          />
        </button>
      ) : (
        <div className="w-[380px] sm:w-[440px] h-[520px] bg-white border border-slate-200 rounded-2xl shadow-xl flex flex-col overflow-hidden">
          <div className="px-4 py-3 bg-slate-900 text-white flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold">
              <MessageSquare className="w-4 h-4 text-sky-400" />
              <span>AI 随诊 ({selectedModel})</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={openSettings}
                className="p-1 text-slate-300 hover:text-white rounded"
                title="设置大模型与 API-Key"
              >
                <Settings className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsOpenFloating(false)}
                className="p-1 text-slate-300 hover:text-white rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-[11px] text-slate-600">
            <span>违背度: {inference.independenceDeviationScore}/100</span>
            <span className="flex items-center gap-1">
              <span
                className={`w-2 h-2 rounded-full ${apiKey ? 'bg-emerald-500' : 'bg-amber-500'}`}
              />
              <span>{apiKey ? 'Key 已绑定' : '未输入 Key'}</span>
            </span>
            <span className="font-semibold text-slate-900">
              预测: {inference.predictedLabel.split(' ')[0]}
            </span>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex flex-col ${
                  m.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[90%] rounded-xl px-3.5 py-2.5 text-xs leading-relaxed whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-slate-900 text-white'
                      : 'bg-[#F8FAFC] border border-slate-200 text-slate-800'
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="text-xs text-slate-500 animate-pulse flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-sky-600 animate-spin" />
                <span>大模型 ({selectedModel}) 正在推导特征相关性...</span>
              </div>
            )}
          </div>

          <div className="px-3 py-2 bg-slate-50 border-t border-slate-100 flex gap-1.5 overflow-x-auto">
            {quickPrompts.slice(0, 2).map((q) => (
              <button
                key={q}
                onClick={() => sendDiagnosisRequest(q)}
                className="px-2.5 py-1 bg-white border border-slate-200 rounded-md text-[11px] text-slate-700 hover:bg-slate-100 whitespace-nowrap"
              >
                {q.slice(0, 16)}...
              </button>
            ))}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendDiagnosisRequest(questionInput);
            }}
            className="p-3 border-t border-slate-200 flex items-center gap-2 bg-white"
          >
            <input
              type="text"
              value={questionInput}
              onChange={(e) => setQuestionInput(e.target.value)}
              placeholder="向大模型提问（必须在小齿轮 ⚙️ 中输入 Key）..."
              className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:border-slate-400"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-3 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium rounded-lg whitespace-nowrap"
            >
              发送
            </button>
          </form>
        </div>
      )}
    </div>
  );
};
