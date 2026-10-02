import React, { useState } from 'react';
import {
  Copy,
  Check,
  Terminal,
  Play,
  FileCode2,
  ArrowRight,
  BarChart3,
  Table as TableIcon,
  Activity,
} from 'lucide-react';
import { BAYES_CASES, BayesCaseScenario } from '../data/bayesCases';
import { BayesInferenceResult, runBayesInference } from '../utils/bayesEngine';

interface Modules5To6Props {
  activeModule: 5 | 6;
  scenario: BayesCaseScenario;
  onSelectScenario: (id: BayesCaseScenario['id']) => void;
  inference: BayesInferenceResult;
  alpha: number;
  setAlpha: (val: number) => void;
  inputText: string;
  setInputText: (val: string) => void;
  onJumpToModule: (mod: number) => void;
}

export const Modules5To6: React.FC<Modules5To6Props> = ({
  activeModule,
  scenario,
  onSelectScenario,
  inference,
  alpha,
  setAlpha,
  inputText,
  setInputText,
  onJumpToModule,
}) => {
  const [caseSlice, setCaseSlice] = useState<'samples' | 'vocab_chi2'>('samples');
  const [codeSlice, setCodeSlice] = useState<'numpy_scratch' | 'sklearn_multinomial' | 'sklearn_gaussian'>('numpy_scratch');
  const [copied, setCopied] = useState<boolean>(false);
  const [runCount, setRunCount] = useState<number>(1);
  const [activeOutputTab, setActiveOutputTab] = useState<'chart' | 'table' | 'log'>('chart');
  const [isRunningSim, setIsRunningSim] = useState<boolean>(false);

  const handleRunCode = () => {
    setIsRunningSim(true);
    setTimeout(() => {
      setRunCount((c) => c + 1);
      setIsRunningSim(false);
    }, 240);
  };

  if (activeModule === 5) {
    const allCases = Object.values(BAYES_CASES);

    return (
      <div className="space-y-6">
        <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
              <span>模块 05</span>
              <span aria-hidden="true">·</span>
              <span>四大行业经典文本分类语料</span>
              <span aria-hidden="true">·</span>
              <span>一键切换全局同步</span>
            </div>
            <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
              四大文本分类实战案例库：垃圾短信 · 影评情感 · 新闻归类 · 钓鱼邮件
            </h2>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg self-start">
            <button
              onClick={() => setCaseSlice('samples')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                caseSlice === 'samples'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片一：经典测试样本集 ({scenario.samples.length}条)
            </button>
            <button
              onClick={() => setCaseSlice('vocab_chi2')}
              className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                caseSlice === 'vocab_chi2'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              切片二：特征词卡方 χ² 与互信息表
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {allCases.map((c) => {
            const isSelected = c.id === scenario.id;
            return (
              <button
                key={c.id}
                onClick={() => onSelectScenario(c.id)}
                className={`text-left p-5 rounded-xl border transition-all flex flex-col justify-between ${
                  isSelected
                    ? 'bg-white border-slate-900 ring-1 ring-slate-900 shadow-xs'
                    : 'bg-white border-slate-200/90 hover:border-slate-400'
                }`}
              >
                <div>
                  <div className="text-xs text-slate-500 flex items-center justify-between">
                    <span>{c.badgeText}</span>
                    {isSelected && (
                      <span className="text-emerald-700 font-medium">● 当前激活</span>
                    )}
                  </div>
                  <h3 className="text-base font-semibold text-slate-900 mt-1.5">
                    {c.name}
                  </h3>
                  <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">
                    {c.subtitle}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs font-mono-tabular text-slate-600">
                  <span>先验 P(正)={(c.priorPos * 100).toFixed(0)}%</span>
                  <span>词表 |V|={c.vocab.length}</span>
                </div>
              </button>
            );
          })}
        </div>

        {caseSlice === 'samples' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-base font-semibold text-slate-900">
                  「{scenario.name}」基准验证样本切片（点击任意一行载入全实验室推理）
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  包含常规高置信度样本、跨界特征冲突边界样本，以及未登录词（OOV）压力测试样本
                </p>
              </div>
              <button
                onClick={() => onJumpToModule(8)}
                className="px-3.5 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg flex items-center gap-1 self-start whitespace-nowrap"
              >
                前往模块 8 查看分词到决策全流程 <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {scenario.samples.map((samp) => {
                const res = runBayesInference(samp.text, scenario, alpha, true);
                const isActive = inputText === samp.text;
                const isCorrect = res.predictedClass === samp.actualClass;

                return (
                  <div
                    key={samp.id}
                    onClick={() => setInputText(samp.text)}
                    className={`p-4 rounded-xl border cursor-pointer transition-all ${
                      isActive
                        ? 'bg-slate-50 border-slate-900'
                        : 'bg-white border-slate-200 hover:border-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono-tabular font-semibold text-slate-700">
                          {samp.id}
                        </span>
                        <span>·</span>
                        <span>{samp.note}</span>
                      </div>
                      <span
                        className={`font-medium ${
                          isCorrect ? 'text-emerald-700' : 'text-amber-700'
                        }`}
                      >
                        {isCorrect ? '✓ 预测一致' : '▲ 边界样本'}
                      </span>
                    </div>

                    <div className="text-sm font-medium text-slate-900 leading-snug my-2">
                      “{samp.text}”
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100 text-xs font-mono-tabular">
                      <span className="text-slate-500 font-sans">
                        真实类别：
                        <strong className="text-slate-800">
                          {samp.actualClass === 1
                            ? scenario.posLabel.split(' ')[0]
                            : scenario.negLabel.split(' ')[0]}
                        </strong>
                      </span>
                      <span
                        className={
                          res.predictedClass === 1 ? 'text-rose-700 font-semibold' : 'text-sky-700 font-semibold'
                        }
                      >
                        P(正类|X) = {(res.posteriorPos * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {caseSlice === 'vocab_chi2' && (
          <div className="bg-white border border-slate-200/90 rounded-xl p-6 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <h3 className="text-base font-semibold text-slate-900">
                「{scenario.name}」核心词表概率矩阵与特征选择指标（χ² 卡方值 / 互信息 MI）
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                按卡方统计量 χ²(Xᵢ, Y) 排序，直观展示哪些词具有最强类别区分能力，哪些词属于强共现或低信息量停用词
              </p>
            </div>

            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="w-full text-left text-xs font-mono-tabular">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3 font-sans">特征词元</th>
                    <th className="py-2.5 px-3">正类频次 N₁,ᵢ</th>
                    <th className="py-2.5 px-3">负类频次 N₀,ᵢ</th>
                    <th className="py-2.5 px-3">卡方统计量 χ²</th>
                    <th className="py-2.5 px-3">互信息 I(Xᵢ;Y)</th>
                    <th className="py-2.5 px-3 font-sans">共现相关特征</th>
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
                        <td className="py-2 px-3 text-rose-700">{v.posCount}</td>
                        <td className="py-2 px-3 text-sky-700">{v.negCount}</td>
                        <td className="py-2 px-3 font-semibold text-slate-900">
                          {v.chiSquare.toFixed(2)}
                        </td>
                        <td className="py-2 px-3 text-emerald-700">
                          {v.mutualInfo.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 font-sans text-slate-600">
                          {v.correlatedWith ? (
                            <span className="text-amber-700">
                              与「{v.correlatedWith}」共现 (r={v.correlationCoeff})
                            </span>
                          ) : v.isStopword ? (
                            <span className="text-slate-400">高频停用词（建议过滤）</span>
                          ) : v.posCount === 0 && v.negCount === 0 ? (
                            <span className="text-rose-700">未登录词 OOV (N=0)</span>
                          ) : (
                            <span>独立良好</span>
                          )}
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

  // ============================================================================
  // MODULE 6: PYTHON / SCIKIT-LEARN 代码引擎
  // ============================================================================
  // Construct clean self-contained Python code snippets with English titles, legends, and axes:
  const trainingSampleTexts = scenario.samples.map((s) => `    "${s.text}"`).join(',\n');
  const trainingSampleLabels = scenario.samples.map((s) => s.actualClass).join(', ');

  const numpyScratchCode = `"""
Naive Bayes Text Classifier - Built from Scratch with NumPy & Matplotlib
Fully self-contained script: ready to copy and run standalone via: python script.py
All figure titles, legends, and axis labels use pure English for maximum cross-platform compatibility.
"""
import numpy as np
import matplotlib.pyplot as plt

class ScratchMultinomialNB:
    def __init__(self, alpha: float = ${alpha.toFixed(2)}):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.vocab_ = {}

    def fit(self, texts, labels):
        # 1. Build vocabulary dictionary
        unique_tokens = set()
        for t in texts:
            for token in t.split():
                unique_tokens.add(token)
        self.vocab_ = {w: i for i, w in enumerate(sorted(unique_tokens))}
        vocab_size = len(self.vocab_)
        
        # 2. Convert raw texts to Bag-of-Words count matrix
        X_counts = np.zeros((len(texts), vocab_size), dtype=np.float64)
        for row, t in enumerate(texts):
            for token in t.split():
                if token in self.vocab_:
                    X_counts[row, self.vocab_[token]] += 1.0

        y = np.array(labels)
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        self.class_log_prior_ = np.zeros(len(self.classes_))
        self.feature_log_prob_ = np.zeros((len(self.classes_), vocab_size))

        for idx, c in enumerate(self.classes_):
            X_c = X_counts[y == c]
            # Prior Log-Probability: ln P(Y=c)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            # Laplace Smoothing: P(X_i | Y=c) = (N_{c,i} + alpha) / (N_c + alpha * |V|)
            token_counts = X_c.sum(axis=0) + self.alpha
            total_counts_smoothed = token_counts.sum()
            self.feature_log_prob_[idx] = np.log(token_counts / total_counts_smoothed)
        return self

    def predict_log_proba(self, query_text: str) -> np.ndarray:
        # Convert query to count vector
        x_vec = np.zeros((1, len(self.vocab_)))
        for token in query_text.split():
            if token in self.vocab_:
                x_vec[0, self.vocab_[token]] += 1.0
        
        # Vectorized linear log-likelihood accumulation
        jll = x_vec @ self.feature_log_prob_.T + self.class_log_prior_
        # Log-Sum-Exp normalization to prevent float underflow
        max_log = np.max(jll, axis=1, keepdims=True)
        log_prob_norm = max_log + np.log(np.sum(np.exp(jll - max_log), axis=1, keepdims=True))
        return jll - log_prob_norm

    def predict_proba(self, query_text: str) -> np.ndarray:
        return np.exp(self.predict_log_proba(query_text))

# ----------------- Execution & Plotting Demo -----------------
if __name__ == "__main__":
    # Self-contained dataset from current scenario: ${scenario.name}
    train_docs = [
${trainingSampleTexts}
    ]
    train_labels = [${trainingSampleLabels}]
    
    # Train Scratch Model
    model = ScratchMultinomialNB(alpha=${alpha.toFixed(2)})
    model.fit(train_docs, train_labels)

    # Inference query from active lab state
    test_query = "${inputText}"
    probs = model.predict_proba(test_query)[0]
    pred = int(np.argmax(probs))

    print("=" * 60)
    print("NAIVE BAYES SCRATCH CLASSIFIER RESULT")
    print(f"Scenario: ${scenario.name} (alpha = ${alpha.toFixed(2)})")
    print(f"Input Text: {test_query}")
    print(f"Predicted Class: {pred} ({'${scenario.posLabel}' if pred == 1 else '${scenario.negLabel}'})")
    print(f"Posterior Probabilities: Class 0 = {probs[0]:.4f}, Class 1 = {probs[1]:.4f}")
    print("=" * 60)

    # English Visualization: Feature Conditional Probability Comparison
    sample_tokens = list(model.vocab_.keys())[:8]
    indices = [model.vocab_[t] for t in sample_tokens]
    p_class0 = np.exp(model.feature_log_prob_[0, indices])
    p_class1 = np.exp(model.feature_log_prob_[1, indices])

    x_pos = np.arange(len(sample_tokens))
    bar_width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - bar_width/2, p_class0, width=bar_width, color="#0284C7", label="Class 0 (Negative/Ham)")
    plt.bar(x_pos + bar_width/2, p_class1, width=bar_width, color="#E11D48", label="Class 1 (Positive/Spam)")
    
    plt.title("Token Conditional Probability Distribution P(X|Y)", fontsize=13, fontweight="bold")
    plt.xlabel("Vocabulary Feature Tokens (Indexed)", fontsize=11)
    plt.ylabel("Smoothed Probability P(X_i | Y)", fontsize=11)
    plt.xticks(x_pos, [f"T{i+1}" for i in range(len(sample_tokens))])
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()`;

  const sklearnMultinomialCode = `"""
Scikit-Learn Text Classification Pipeline with Chi-Square Feature Selection
Fully self-contained script: ready to copy and run standalone via: python script.py
All figure titles, legends, and axis labels use pure English for maximum cross-platform compatibility.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

train_texts = [
${trainingSampleTexts}
]
train_labels = np.array([${trainingSampleLabels}])

# Construct complete NLP Pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('chi2_select', SelectKBest(score_func=chi2, k=min(10, len(train_texts)))),
    ('nb_classifier', MultinomialNB(alpha=${alpha.toFixed(2)}, fit_prior=True))
])

# Fit pipeline on training corpus
pipeline.fit(train_texts, train_labels)

# Active test query inference
test_sample = ["${inputText}"]
pred_class = pipeline.predict(test_sample)[0]
pred_proba = pipeline.predict_proba(test_sample)[0]

print("=" * 60)
print("SCIKIT-LEARN PIPELINE EVALUATION")
print(f"Alpha: ${alpha.toFixed(2)} | Predicted Class: {pred_class}")
print(f"Probabilities: Class 0 = {pred_proba[0]:.4f}, Class 1 = {pred_proba[1]:.4f}")
print("=" * 60)

# Evaluate on training batch for metrics display
y_pred_batch = pipeline.predict(train_texts)
print(classification_report(train_labels, y_pred_batch, target_names=["Class 0", "Class 1"]))

# English Visualization: Confusion Matrix Plot
cm = confusion_matrix(train_labels, y_pred_batch)
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center", size=14, weight="bold")

plt.title("Confusion Matrix Evaluation", fontsize=12, pad=16, weight="bold")
fig.colorbar(cax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Class 0 (Neg)", "Class 1 (Pos)"])
ax.set_yticklabels(["Class 0 (Neg)", "Class 1 (Pos)"])
plt.xlabel("Predicted Class Label", fontsize=10)
plt.ylabel("True Class Label", fontsize=10)
plt.tight_layout()
plt.show()`;

  const sklearnGaussianCode = `"""
GaussianNB & BernoulliNB Comparative Text Experiment
Fully self-contained script: ready to copy and run standalone via: python script.py
All figure titles, legends, and axis labels use pure English for maximum cross-platform compatibility.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB

corpus = [
${trainingSampleTexts}
]
labels = np.array([${trainingSampleLabels}])

# Vectorize into binary/dense formats
vectorizer = CountVectorizer(binary=True)
X_binary = vectorizer.fit_transform(corpus).toarray()

# 1. BernoulliNB on Binary Occurrence
bnb = BernoulliNB(alpha=${alpha.toFixed(2)})
bnb.fit(X_binary, labels)

# 2. GaussianNB on Simulated Continuous Sentence Length & Density Features
dense_features = np.column_stack([
    X_binary.sum(axis=1),
    np.array([len(t) for t in corpus], dtype=np.float64) / 10.0
])
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(dense_features, labels)

print("BernoulliNB Feature Log Prob Shape:", bnb.feature_log_prob_.shape)
print("GaussianNB Class Means (Theta):", gnb.theta_)

# English Visualization: Gaussian Feature Density Curves
x_grid = np.linspace(dense_features[:, 0].min() - 1, dense_features[:, 0].max() + 1, 100)
mu0, std0 = gnb.theta_[0, 0], np.sqrt(gnb.var_[0, 0])
mu1, std1 = gnb.theta_[1, 0], np.sqrt(gnb.var_[1, 0])

pdf0 = (1 / (std0 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu0) / std0) ** 2)
pdf1 = (1 / (std1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu1) / std1) ** 2)

plt.figure(figsize=(8, 4.5))
plt.plot(x_grid, pdf0, color="#0284C7", linewidth=2.2, label=f"Class 0 Normal PDF (mu={mu0:.2f})")
plt.plot(x_grid, pdf1, color="#E11D48", linewidth=2.2, label=f"Class 1 Normal PDF (mu={mu1:.2f})")
plt.title("Gaussian Class Feature Probability Density Estimation", fontsize=12, fontweight="bold")
plt.xlabel("Continuous Text Density Feature Value", fontsize=10)
plt.ylabel("Probability Density Function P(x|Y)", fontsize=10)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()`;

  const activeCode =
    codeSlice === 'numpy_scratch'
      ? numpyScratchCode
      : codeSlice === 'sklearn_multinomial'
      ? sklearnMultinomialCode
      : sklearnGaussianCode;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const displayTokens = inference.steps.slice(0, 6);

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <span>模块 06</span>
            <span aria-hidden="true">·</span>
            <span>NumPy 从零手写源码</span>
            <span aria-hidden="true">·</span>
            <span>Scikit-Learn 工业级 Pipeline</span>
          </div>
          <h2 className="text-xl font-semibold text-slate-900 font-serif-title">
            Python / Scikit-Learn 代码引擎：实时绑定实验参数与代数输出
          </h2>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg self-start">
          <button
            onClick={() => setCodeSlice('numpy_scratch')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
              codeSlice === 'numpy_scratch'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            切片一：NumPy 从零手写完整推导
          </button>
          <button
            onClick={() => setCodeSlice('sklearn_multinomial')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
              codeSlice === 'sklearn_multinomial'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            切片二：MultinomialNB + χ² 特征选择
          </button>
          <button
            onClick={() => setCodeSlice('sklearn_gaussian')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
              codeSlice === 'sklearn_gaussian'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            切片三：GaussianNB 与 BernoulliNB
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Code Editor Panel */}
        <div className="lg:col-span-7 bg-[#0F172A] text-slate-100 rounded-xl overflow-hidden border border-slate-800 flex flex-col">
          <div className="px-4 py-3 bg-[#1E293B] border-b border-slate-700/80 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-mono-tabular text-slate-300">
              <FileCode2 className="w-4 h-4 text-sky-400" />
              <span>
                {codeSlice === 'numpy_scratch'
                  ? 'scratch_naive_bayes.py'
                  : codeSlice === 'sklearn_multinomial'
                  ? 'sklearn_multinomial_pipeline.py'
                  : 'sklearn_gaussian_bernoulli.py'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRunCode}
                disabled={isRunningSim}
                className="px-3 py-1.5 text-xs font-medium bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-md flex items-center gap-1.5 transition-colors whitespace-nowrap shadow-xs"
              >
                <Play className="w-3.5 h-3.5" />
                <span>运行代码</span>
              </button>
              <button
                onClick={handleCopy}
                className="px-3 py-1.5 text-xs font-medium bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-md flex items-center gap-1.5 transition-colors whitespace-nowrap"
                title="复制完整自包含代码，可直接在项目外部终端 python script.py 运行"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? '已复制源码（可外部直接运行）' : '复制源码'}
              </button>
            </div>
          </div>

          <pre className="p-5 text-xs font-mono-tabular leading-relaxed overflow-x-auto text-slate-200 flex-1 max-h-[580px]">
            <code>{activeCode}</code>
          </pre>

          <div className="px-4 py-2.5 bg-[#1E293B]/70 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
            <span>提示：源码内嵌纯英文标题、图例与坐标轴，复制后可在任意外部环境直接运行</span>
            <span className="font-mono-tabular">Python 3.8+ / NumPy / Scikit-Learn</span>
          </div>
        </div>

        {/* Live Output Window with Chart, Table, and Terminal */}
        <div className="lg:col-span-5 bg-white border border-slate-200/90 rounded-xl p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                <Terminal className="w-4 h-4 text-slate-700" />
                <span>输出窗口（运行结果与图表展示）</span>
              </div>
              <span className="text-xs font-mono-tabular text-emerald-700">
                #RUN {runCount} · EXIT 0
              </span>
            </div>

            {/* Output Sub-Tabs */}
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              <button
                onClick={() => setActiveOutputTab('chart')}
                className={`flex-1 py-1.5 text-xs font-medium rounded-md flex items-center justify-center gap-1.5 transition-colors ${
                  activeOutputTab === 'chart'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5 text-sky-700" />
                <span>英文图表 (Figure)</span>
              </button>
              <button
                onClick={() => setActiveOutputTab('table')}
                className={`flex-1 py-1.5 text-xs font-medium rounded-md flex items-center justify-center gap-1.5 transition-colors ${
                  activeOutputTab === 'table'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <TableIcon className="w-3.5 h-3.5 text-emerald-700" />
                <span>指标表格 (Table)</span>
              </button>
              <button
                onClick={() => setActiveOutputTab('log')}
                className={`flex-1 py-1.5 text-xs font-medium rounded-md flex items-center justify-center gap-1.5 transition-colors ${
                  activeOutputTab === 'log'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Activity className="w-3.5 h-3.5 text-rose-700" />
                <span>控制台日志 (Stdout)</span>
              </button>
            </div>

            {/* TAB 1: Real-time Rendered Figure with English titles, legends, and axes */}
            {activeOutputTab === 'chart' && (
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
                <div className="text-center font-mono-tabular text-xs font-semibold text-slate-800">
                  Token Conditional Probability Distribution P(X|Y)
                </div>

                <div className="h-[210px] w-full flex items-end justify-between gap-2 px-2 pt-4 pb-2 border-b border-l border-slate-300 bg-white rounded-lg">
                  {displayTokens.map((tok) => {
                    const maxP = Math.max(
                      0.04,
                      ...displayTokens.map((t) => Math.max(t.pGivenPos, t.pGivenNeg))
                    );
                    const hPosPercent = Math.min(100, Math.max(8, (tok.pGivenPos / maxP) * 85));
                    const hNegPercent = Math.min(100, Math.max(8, (tok.pGivenNeg / maxP) * 85));

                    return (
                      <div key={tok.token} className="flex-1 flex flex-col items-center h-full justify-end group">
                        <div className="w-full flex items-end justify-center gap-1 h-[145px]">
                          <div
                            style={{ height: `${hNegPercent}%` }}
                            className="w-1/2 bg-[#0284C7] rounded-t-xs transition-all duration-300 relative"
                            title={`Class 0 P(${tok.token}|0) = ${tok.pGivenNeg.toFixed(4)}`}
                          />
                          <div
                            style={{ height: `${hPosPercent}%` }}
                            className="w-1/2 bg-[#E11D48] rounded-t-xs transition-all duration-300 relative"
                            title={`Class 1 P(${tok.token}|1) = ${tok.pGivenPos.toFixed(4)}`}
                          />
                        </div>
                        <span className="text-[10px] font-sans text-slate-700 truncate max-w-[42px] mt-1.5" title={tok.token}>
                          {tok.token}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div className="flex items-center justify-between text-[11px] font-mono-tabular text-slate-500 px-1">
                  <span>Y-axis: Probability P(X_i | Y)</span>
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1 text-[#0284C7]">
                      <span className="w-2.5 h-2.5 bg-[#0284C7] rounded-xs inline-block" />
                      Class 0 (Neg)
                    </span>
                    <span className="flex items-center gap-1 text-[#E11D48]">
                      <span className="w-2.5 h-2.5 bg-[#E11D48] rounded-xs inline-block" />
                      Class 1 (Pos)
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: Execution Metrics Table */}
            {activeOutputTab === 'table' && (
              <div className="border border-slate-200 rounded-xl overflow-hidden text-xs font-mono-tabular">
                <table className="w-full text-left">
                  <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                    <tr>
                      <th className="py-2 px-3 font-sans">评估维度 (Metric)</th>
                      <th className="py-2 px-3">数值 (Value)</th>
                      <th className="py-2 px-3 font-sans">说明 (Description)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-800">Prior P(Y=1)</td>
                      <td className="py-2 px-3 text-rose-700 font-semibold">{inference.priorPos.toFixed(4)}</td>
                      <td className="py-2 px-3 font-sans text-slate-500">正类先验概率 (Class 1)</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-800">Prior P(Y=0)</td>
                      <td className="py-2 px-3 text-sky-700 font-semibold">{inference.priorNeg.toFixed(4)}</td>
                      <td className="py-2 px-3 font-sans text-slate-500">负类先验概率 (Class 0)</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-800">Laplace Alpha</td>
                      <td className="py-2 px-3 text-emerald-700 font-semibold">{alpha.toFixed(2)}</td>
                      <td className="py-2 px-3 font-sans text-slate-500">平滑参数 (Dirichlet Prior)</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-800">Post P(Y=1|X)</td>
                      <td className="py-2 px-3 text-rose-700 font-semibold">{(inference.posteriorPos * 100).toFixed(2)}%</td>
                      <td className="py-2 px-3 font-sans text-slate-500">当前样本预测后验概率</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-800">Log-Odds Diff</td>
                      <td className="py-2 px-3 font-semibold text-slate-900">
                        {(inference.totalLogPos - inference.totalLogNeg).toFixed(3)}
                      </td>
                      <td className="py-2 px-3 font-sans text-slate-500">对数奇数比差值 Δ</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            {/* TAB 3: Terminal Log (Stdout) */}
            {activeOutputTab === 'log' && (
              <div className="bg-slate-900 text-slate-100 rounded-xl p-4 font-mono-tabular text-xs space-y-2 overflow-x-auto max-h-[260px]">
                <div className="text-slate-400">
                  &gt;&gt;&gt; python scratch_naive_bayes.py --scenario="{scenario.id}" --alpha={alpha.toFixed(2)}
                </div>
                <div>
                  [Model] ScratchMultinomialNB initialized with alpha={alpha.toFixed(2)}
                </div>
                <div>
                  [Vocab] Active tokens: {JSON.stringify(inference.steps.map((s) => s.token))}
                </div>
                <div className="text-amber-300">
                  [JLL] log_likelihood_pos={inference.totalLogPos.toFixed(4)} | log_likelihood_neg={inference.totalLogNeg.toFixed(4)}
                </div>
                <div className="text-emerald-400 font-semibold">
                  [Posterior] P(Y=1|X)={(inference.posteriorPos * 100).toFixed(2)}% | P(Y=0|X)={(inference.posteriorNeg * 100).toFixed(2)}%
                </div>
                <div className="text-sky-300 pt-1 border-t border-slate-700">
                  [Decision] 最终分类判决: {inference.predictedLabel}
                </div>
              </div>
            )}

            {/* Alpha adjustment within output window */}
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>实时调节 Python 代码参数 alpha</span>
                <span className="font-mono-tabular font-semibold">alpha = {alpha.toFixed(2)}</span>
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
          </div>

          <div className="text-xs text-slate-500 leading-relaxed bg-[#F8FAFC] p-3 rounded-lg border border-slate-200/80">
            <strong>运行与复制说明：</strong>点击「运行代码」可在项目内直接执行推导并刷新上方图表与指标；点击「复制源码」可获取完整自包含脚本，粘贴至本地终端 <code>python script.py</code> 即可独立运行。
          </div>
        </div>
      </div>
    </div>
  );
};
