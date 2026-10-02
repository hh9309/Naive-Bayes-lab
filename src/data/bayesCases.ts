export interface VocabEntry {
  token: string;
  posCount: number;
  negCount: number;
  chiSquare: number;
  mutualInfo: number;
  isStopword?: boolean;
  correlatedWith?: string;
  correlationCoeff?: number;
}

export interface SampleDoc {
  id: string;
  text: string;
  tokens: string[];
  actualClass: 1 | 0;
  note?: string;
}

export interface BayesCaseScenario {
  id: 'spam' | 'sentiment' | 'news' | 'phishing';
  name: string;
  subtitle: string;
  badgeText: string;
  posLabel: string;
  negLabel: string;
  priorPos: number;
  priorNeg: number;
  totalPosDocs: number;
  totalNegDocs: number;
  defaultTestText: string;
  defaultFeaturePair: [string, string];
  correlatedHighlight: {
    tokenA: string;
    tokenB: string;
    r: number;
    explanation: string;
  };
  vocab: VocabEntry[];
  samples: SampleDoc[];
}

export const STOPWORDS_SET = new Set([
  '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '请问', '今天', '我们'
]);

export const BAYES_CASES: Record<BayesCaseScenario['id'], BayesCaseScenario> = {
  spam: {
    id: 'spam',
    name: '垃圾短信过滤',
    subtitle: '电信运营商高频营销与诈骗短消息实时拦截',
    badgeText: '场景一 · 电信反垃圾',
    posLabel: '垃圾短信 (Spam)',
    negLabel: '正常通信 (Ham)',
    priorPos: 0.35,
    priorNeg: 0.65,
    totalPosDocs: 350,
    totalNegDocs: 650,
    defaultTestText: '恭喜您获得免费抽奖特权点击链接立即领取现金大奖今天退订回T',
    defaultFeaturePair: ['免费', '中奖'],
    correlatedHighlight: {
      tokenA: '免费',
      tokenB: '领取',
      r: 0.84,
      explanation: '营销短信中“免费”与“领取”共现率高达84%，强条件独立假设会令二者对数似然重复叠加，使后验概率向 99.9% 极化。'
    },
    vocab: [
      { token: '免费', posCount: 142, negCount: 6, chiSquare: 198.4, mutualInfo: 0.312, correlatedWith: '领取', correlationCoeff: 0.84 },
      { token: '中奖', posCount: 118, negCount: 2, chiSquare: 185.2, mutualInfo: 0.295, correlatedWith: '大奖', correlationCoeff: 0.79 },
      { token: '领取', posCount: 126, negCount: 9, chiSquare: 164.8, mutualInfo: 0.268, correlatedWith: '免费', correlationCoeff: 0.84 },
      { token: '链接', posCount: 95, negCount: 8, chiSquare: 121.5, mutualInfo: 0.214, correlatedWith: '点击', correlationCoeff: 0.88 },
      { token: '点击', posCount: 104, negCount: 11, chiSquare: 126.1, mutualInfo: 0.221, correlatedWith: '链接', correlationCoeff: 0.88 },
      { token: '现金', posCount: 89, negCount: 7, chiSquare: 115.3, mutualInfo: 0.198 },
      { token: '退订', posCount: 112, negCount: 0, chiSquare: 189.6, mutualInfo: 0.304 },
      { token: '特权', posCount: 74, negCount: 4, chiSquare: 101.2, mutualInfo: 0.176 },
      { token: '大奖', posCount: 82, negCount: 3, chiSquare: 118.7, mutualInfo: 0.205, correlatedWith: '中奖', correlationCoeff: 0.79 },
      { token: '恭喜', posCount: 68, negCount: 14, chiSquare: 69.4, mutualInfo: 0.132 },
      { token: '会议', posCount: 3, negCount: 145, chiSquare: 78.9, mutualInfo: 0.154 },
      { token: '报告', posCount: 2, negCount: 132, chiSquare: 72.4, mutualInfo: 0.141 },
      { token: '明天', posCount: 8, negCount: 118, chiSquare: 51.2, mutualInfo: 0.108 },
      { token: '项目', posCount: 1, negCount: 128, chiSquare: 73.1, mutualInfo: 0.146 },
      { token: '快递', posCount: 12, negCount: 96, chiSquare: 34.6, mutualInfo: 0.082 },
      { token: '今天', posCount: 45, negCount: 82, chiSquare: 0.12, mutualInfo: 0.002, isStopword: true },
      { token: '的', posCount: 210, negCount: 390, chiSquare: 0.01, mutualInfo: 0.0001, isStopword: true },
      { token: '量子币', posCount: 0, negCount: 0, chiSquare: 0.0, mutualInfo: 0.0 }
    ],
    samples: [
      { id: 'S-01', text: '恭喜您获得免费抽奖特权点击链接立即领取现金大奖退订回T', tokens: ['恭喜', '免费', '特权', '点击', '链接', '领取', '现金', '大奖', '退订'], actualClass: 1, note: '典型高共现博彩营销短信' },
      { id: 'S-02', text: '双十一会员专享免费领取百元现金红包点击链接激活特权', tokens: ['免费', '领取', '现金', '点击', '链接', '特权'], actualClass: 1, note: '电商诱导点击垃圾短信' },
      { id: 'S-03', text: '恭喜中奖您的手机号获得特等奖现金大奖请点击链接领取', tokens: ['恭喜', '中奖', '现金', '大奖', '点击', '链接', '领取'], actualClass: 1, note: '欺诈中奖通知' },
      { id: 'S-04', text: '明天上午十点项目评审会议请准备好季度数据报告', tokens: ['明天', '项目', '会议', '报告'], actualClass: 0, note: '日常研发办公协同通知' },
      { id: 'S-05', text: '您的顺丰快递已到达丰巢快递柜请今天凭取件码领取', tokens: ['快递', '今天', '领取'], actualClass: 0, note: '含“领取”触发词的正常物流短信（易误判边界样本）' },
      { id: 'S-06', text: '项目进度报告已发送至邮箱明天会议我们一起讨论', tokens: ['项目', '报告', '明天', '会议', '我们'], actualClass: 0, note: '正常团队沟通短信' },
      { id: 'S-07', text: '内部特权通道免费空投量子币点击链接立即提现现金', tokens: ['特权', '免费', '量子币', '点击', '链接', '现金'], actualClass: 1, note: '含未登录词“量子币”的新型诈骗短信' },
      { id: 'S-08', text: '恭喜团队项目报告获得年度创新大奖明天会议表彰', tokens: ['恭喜', '项目', '报告', '大奖', '明天', '会议'], actualClass: 0, note: '含“恭喜/大奖”的正向办公通知（考验多特征权衡）' }
    ]
  },

  sentiment: {
    id: 'sentiment',
    name: '影评情感倾向识别',
    subtitle: '院线电影观众长短评正向赞誉与负向吐槽自动极性判别',
    badgeText: '场景二 · 情感极性分析',
    posLabel: '正向好评 (Positive)',
    negLabel: '负向差评 (Negative)',
    priorPos: 0.52,
    priorNeg: 0.48,
    totalPosDocs: 520,
    totalNegDocs: 480,
    defaultTestText: '导演叙事极其惊艳演员演技炸裂剧情紧凑感人强烈推荐二刷',
    defaultFeaturePair: ['惊艳', '烂片'],
    correlatedHighlight: {
      tokenA: '演技',
      tokenB: '惊艳',
      r: 0.78,
      explanation: '口碑影评中“演技”与“惊艳”高度绑定出现，独立性假设下会产生双倍正向极性增益。'
    },
    vocab: [
      { token: '惊艳', posCount: 165, negCount: 5, chiSquare: 158.4, mutualInfo: 0.284, correlatedWith: '演技', correlationCoeff: 0.78 },
      { token: '推荐', posCount: 182, negCount: 9, chiSquare: 164.2, mutualInfo: 0.291 },
      { token: '感人', posCount: 138, negCount: 8, chiSquare: 121.6, mutualInfo: 0.228 },
      { token: '紧凑', posCount: 119, negCount: 6, chiSquare: 106.5, mutualInfo: 0.199 },
      { token: '演技', posCount: 154, negCount: 42, chiSquare: 68.3, mutualInfo: 0.135, correlatedWith: '惊艳', correlationCoeff: 0.78 },
      { token: '神作', posCount: 96, negCount: 2, chiSquare: 92.8, mutualInfo: 0.178 },
      { token: '二刷', posCount: 88, negCount: 0, chiSquare: 89.4, mutualInfo: 0.172 },
      { token: '烂片', posCount: 2, negCount: 174, chiSquare: 184.6, mutualInfo: 0.318, correlatedWith: '敷衍', correlationCoeff: 0.82 },
      { token: '尴尬', posCount: 6, negCount: 148, chiSquare: 146.2, mutualInfo: 0.264 },
      { token: '拖沓', posCount: 4, negCount: 132, chiSquare: 134.8, mutualInfo: 0.245 },
      { token: '敷衍', posCount: 3, negCount: 126, chiSquare: 130.1, mutualInfo: 0.239, correlatedWith: '烂片', correlationCoeff: 0.82 },
      { token: '退票', posCount: 0, negCount: 94, chiSquare: 104.5, mutualInfo: 0.195 },
      { token: '催眠', posCount: 1, negCount: 86, chiSquare: 92.1, mutualInfo: 0.174 },
      { token: '剧情', posCount: 140, negCount: 135, chiSquare: 0.08, mutualInfo: 0.001 },
      { token: '的', posCount: 260, negCount: 245, chiSquare: 0.01, mutualInfo: 0.0001, isStopword: true },
      { token: '赛博美学', posCount: 0, negCount: 0, chiSquare: 0.0, mutualInfo: 0.0 }
    ],
    samples: [
      { id: 'M-01', text: '导演镜头惊艳演技全员在线剧情紧凑感人强烈推荐二刷神作', tokens: ['惊艳', '演技', '剧情', '紧凑', '感人', '推荐', '二刷', '神作'], actualClass: 1, note: '高分口碑影评' },
      { id: 'M-02', text: '全片节奏拖沓台词尴尬特效敷衍年度史诗级烂片想退票', tokens: ['拖沓', '尴尬', '敷衍', '烂片', '退票'], actualClass: 0, note: '典型负向差评' },
      { id: 'M-03', text: '虽然剧情前半段略显拖沓但结局反转惊艳非常感人值得推荐', tokens: ['剧情', '拖沓', '惊艳', '感人', '推荐'], actualClass: 1, note: '转折型好评（含“拖沓”负向词）' },
      { id: 'M-04', text: '冲着演员演技去的奈何剧本太敷衍全程尴尬催眠不推荐', tokens: ['演技', '敷衍', '尴尬', '催眠', '推荐'], actualClass: 0, note: '含“演技/推荐”字眼的否定差评' },
      { id: 'M-05', text: '赛博美学镜头惊艳配乐紧凑绝对值得二刷', tokens: ['赛博美学', '惊艳', '紧凑', '二刷'], actualClass: 1, note: '含未登录词“赛博美学”的影评' },
      { id: 'M-06', text: '剧情敷衍烂片催眠退票', tokens: ['剧情', '敷衍', '烂片', '催眠', '退票'], actualClass: 0, note: '短文本强负向吐槽' }
    ]
  },

  news: {
    id: 'news',
    name: '新闻主题自动归类',
    subtitle: '高维稀疏资讯流中“前沿人工智能科技”与“宏观资本财经”自动路由',
    badgeText: '场景三 · 新闻主题路由',
    posLabel: '前沿科技 (Tech/AI)',
    negLabel: '宏观财经 (Finance)',
    priorPos: 0.48,
    priorNeg: 0.52,
    totalPosDocs: 480,
    totalNegDocs: 520,
    defaultTestText: '大模型与人工智能AI芯片算力架构突破推动开源算法生态演进',
    defaultFeaturePair: ['人工智能', '利率'],
    correlatedHighlight: {
      tokenA: '人工智能',
      tokenB: 'AI',
      r: 0.92,
      explanation: '新闻稿中“人工智能”与“AI”几乎作为同义词成对出现（r=0.92），是教材切片三中最经典的“特征高度相关导致概率膨胀”案例。'
    },
    vocab: [
      { token: '人工智能', posCount: 176, negCount: 8, chiSquare: 189.5, mutualInfo: 0.324, correlatedWith: 'AI', correlationCoeff: 0.92 },
      { token: 'AI', posCount: 184, negCount: 11, chiSquare: 191.2, mutualInfo: 0.329, correlatedWith: '人工智能', correlationCoeff: 0.92 },
      { token: '大模型', posCount: 152, negCount: 4, chiSquare: 168.4, mutualInfo: 0.296, correlatedWith: '算力', correlationCoeff: 0.81 },
      { token: '算力', posCount: 138, negCount: 6, chiSquare: 147.8, mutualInfo: 0.265, correlatedWith: '大模型', correlationCoeff: 0.81 },
      { token: '芯片', posCount: 145, negCount: 19, chiSquare: 124.6, mutualInfo: 0.228 },
      { token: '算法', posCount: 128, negCount: 5, chiSquare: 138.9, mutualInfo: 0.251 },
      { token: '开源', posCount: 104, negCount: 2, chiSquare: 116.2, mutualInfo: 0.214 },
      { token: '利率', posCount: 2, negCount: 168, chiSquare: 161.4, mutualInfo: 0.289, correlatedWith: '央行', correlationCoeff: 0.86 },
      { token: '央行', posCount: 1, negCount: 154, chiSquare: 149.8, mutualInfo: 0.272, correlatedWith: '利率', correlationCoeff: 0.86 },
      { token: '债券', posCount: 0, negCount: 136, chiSquare: 135.2, mutualInfo: 0.248 },
      { token: '通胀', posCount: 3, negCount: 142, chiSquare: 131.6, mutualInfo: 0.241 },
      { token: '财报', posCount: 24, negCount: 148, chiSquare: 92.4, mutualInfo: 0.175 },
      { token: '汇率', posCount: 1, negCount: 118, chiSquare: 114.2, mutualInfo: 0.212 },
      { token: '今天', posCount: 62, negCount: 68, chiSquare: 0.02, mutualInfo: 0.0004, isStopword: true },
      { token: '室温超导', posCount: 0, negCount: 0, chiSquare: 0.0, mutualInfo: 0.0 }
    ],
    samples: [
      { id: 'N-01', text: '新一代开源大模型算法发布人工智能AI算力芯片需求激增', tokens: ['开源', '大模型', '算法', '人工智能', 'AI', '算力', '芯片'], actualClass: 1, note: '含“人工智能+AI”强共现词对的科技新闻' },
      { id: 'N-02', text: '央行宣布下调基准利率抑制通胀压力国债债券收益率企稳', tokens: ['央行', '利率', '通胀', '债券'], actualClass: 0, note: '典型宏观货币政策财经报道' },
      { id: 'N-03', text: '芯片巨头最新季度财报营收超预期人工智能算力中心订单饱满', tokens: ['芯片', '财报', '人工智能', '算力'], actualClass: 1, note: '科技股财报交叉新闻（含“财报”财经特征）' },
      { id: 'N-04', text: '外汇市场汇率波动加剧央行释放流动性稳定债券与利率预期', tokens: ['汇率', '央行', '债券', '利率'], actualClass: 0, note: '金融外汇市场快讯' },
      { id: 'N-05', text: '室温超导材料突破或将重塑下一代AI芯片与算力架构', tokens: ['室温超导', 'AI', '芯片', '算力'], actualClass: 1, note: '含未登录词“室温超导”的前沿科技快讯' },
      { id: 'N-06', text: '量化基金引入AI算法分析财报与通胀利率走势', tokens: ['AI', '算法', '财报', '通胀', '利率'], actualClass: 0, note: '金融科技跨界新闻（多维特征拉锯）' }
    ]
  },

  phishing: {
    id: 'phishing',
    name: '高风险钓鱼邮件检测',
    subtitle: '企业安全网关针对伪造财务汇款、账号冻结与木马附件的精准识别',
    badgeText: '场景四 · 企业安全网关',
    posLabel: '高危钓鱼 (Phishing)',
    negLabel: '合规邮件 (Legitimate)',
    priorPos: 0.25,
    priorNeg: 0.75,
    totalPosDocs: 250,
    totalNegDocs: 750,
    defaultTestText: '紧急通知您的企业邮箱账号异常即将冻结请立即登录安全链接验证密码',
    defaultFeaturePair: ['冻结', '纪要'],
    correlatedHighlight: {
      tokenA: '冻结',
      tokenB: '验证',
      r: 0.87,
      explanation: '社工钓鱼邮件常利用“账号冻结”恐吓诱导“立即验证”，二者强共现（r=0.87）构成核心高危信号。'
    },
    vocab: [
      { token: '冻结', posCount: 112, negCount: 2, chiSquare: 248.6, mutualInfo: 0.352, correlatedWith: '验证', correlationCoeff: 0.87 },
      { token: '验证', posCount: 124, negCount: 11, chiSquare: 236.4, mutualInfo: 0.338, correlatedWith: '冻结', correlationCoeff: 0.87 },
      { token: '密码', posCount: 108, negCount: 9, chiSquare: 208.1, mutualInfo: 0.305, correlatedWith: '账号', correlationCoeff: 0.83 },
      { token: '紧急', posCount: 96, negCount: 14, chiSquare: 168.2, mutualInfo: 0.256 },
      { token: '异常', posCount: 92, negCount: 16, chiSquare: 152.9, mutualInfo: 0.238 },
      { token: '账号', posCount: 104, negCount: 18, chiSquare: 174.5, mutualInfo: 0.264, correlatedWith: '密码', correlationCoeff: 0.83 },
      { token: '汇款', posCount: 78, negCount: 5, chiSquare: 162.3, mutualInfo: 0.249 },
      { token: '附件', posCount: 64, negCount: 72, chiSquare: 24.8, mutualInfo: 0.058 },
      { token: '纪要', posCount: 1, negCount: 165, chiSquare: 68.4, mutualInfo: 0.142 },
      { token: '周报', posCount: 0, negCount: 158, chiSquare: 66.8, mutualInfo: 0.139 },
      { token: '排期', posCount: 2, negCount: 142, chiSquare: 56.2, mutualInfo: 0.118 },
      { token: '审批', posCount: 8, negCount: 134, chiSquare: 41.5, mutualInfo: 0.092 },
      { token: '请问', posCount: 32, negCount: 95, chiSquare: 0.01, mutualInfo: 0.0002, isStopword: true },
      { token: '零日漏洞补丁', posCount: 0, negCount: 0, chiSquare: 0.0, mutualInfo: 0.0 }
    ],
    samples: [
      { id: 'P-01', text: '紧急通知您的企业账号异常即将冻结请立即验证密码', tokens: ['紧急', '账号', '异常', '冻结', '验证', '密码'], actualClass: 1, note: '典型伪造IT管理员钓鱼邮件' },
      { id: 'P-02', text: '财务紧急汇款指令请查收加密附件并立即安排跨行汇款', tokens: ['紧急', '汇款', '附件'], actualClass: 1, note: 'BEC高管欺诈汇款钓鱼邮件' },
      { id: 'P-03', text: '附件是本周研发团队周报与下周迭代排期会议纪要请查阅', tokens: ['附件', '周报', '排期', '纪要'], actualClass: 0, note: '正常研发周报邮件（含“附件”中性词）' },
      { id: 'P-04', text: '请审批下季度采购预算排期表会议纪要已同步在附件', tokens: ['审批', '排期', '纪要', '附件'], actualClass: 0, note: '正常OA审批流通知' },
      { id: 'P-05', text: '紧急安全告警请点击附件安装零日漏洞补丁并验证账号密码', tokens: ['紧急', '附件', '零日漏洞补丁', '验证', '账号', '密码'], actualClass: 1, note: '含未登录词“零日漏洞补丁”的木马钓鱼邮件' },
      { id: 'P-06', text: '监控系统发现服务器日志异常请运维同学结合周报排期排查', tokens: ['异常', '周报', '排期'], actualClass: 0, note: '含“异常”字眼的正常运维办公邮件' }
    ]
  }
};
