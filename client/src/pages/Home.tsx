import { useState } from "react";
import { ChevronDown, Code2, Layers, Zap, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ComponentInfo {
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  details: {
    init: string[];
    forward: string[];
    keyPoint: string;
  };
}

const components: ComponentInfo[] = [
  {
    name: "LayerNorm",
    description: "레이어 정규화 (Layer Normalization)",
    icon: <Layers className="w-6 h-6" />,
    color: "from-blue-500 to-blue-600",
    details: {
      init: [
        "self.eps = 1e-5 (수치 안정성)",
        "self.scale = nn.Parameter(torch.ones(emb_dim))",
        "self.shift = nn.Parameter(torch.zeros(emb_dim))",
      ],
      forward: [
        "mean = x.mean(dim=-1, keepdim=True)",
        "var = x.var(dim=-1, keepdim=True, unbiased=False)",
        "norm_x = (x-mean) / sqrt(var+eps)",
        "return scale * norm_x + shift",
      ],
      keyPoint: "마지막 차원 기준으로 정규화하고 학습 가능한 scale/shift 적용",
    },
  },
  {
    name: "FeedForward",
    description: "피드포워드 신경망 (Feed-Forward Network)",
    icon: <Zap className="w-6 h-6" />,
    color: "from-purple-500 to-purple-600",
    details: {
      init: [
        "nn.Linear(emb_dim, 4*emb_dim) - 차원 확장",
        "GELU() - 활성화 함수",
        "nn.Linear(4*emb_dim, emb_dim) - 차원 복원",
      ],
      forward: ["return self.layers(x)"],
      keyPoint: "차원 확장 → 비선형 활성화 → 차원 복원의 3단계 구조",
    },
  },
  {
    name: "TransformerBlock",
    description: "트랜스포머 블록 (Transformer Block)",
    icon: <Brain className="w-6 h-6" />,
    color: "from-emerald-500 to-emerald-600",
    details: {
      init: [
        "self.att = MultiHeadAttention(...)",
        "self.ff = FeedForward(cfg)",
        "self.norm1, self.norm2 = LayerNorm(...)",
        "self.drop_shortcut = nn.Dropout(...)",
      ],
      forward: [
        "① norm → attention → dropout → add (잔차연결)",
        "② norm → feedforward → dropout → add (잔차연결)",
      ],
      keyPoint:
        "정규화-기능-드롭아웃-잔차연결 패턴이 2번 반복되는 구조",
    },
  },
  {
    name: "GPTModel",
    description: "GPT 모델 전체 아키텍처 (GPT Model Architecture)",
    icon: <Code2 className="w-6 h-6" />,
    color: "from-orange-500 to-orange-600",
    details: {
      init: [
        "self.tok_emb = nn.Embedding(vocab_size, emb_dim)",
        "self.pos_emb = nn.Embedding(context_length, emb_dim)",
        "self.trf_block = nn.Sequential(TransformerBlock × n_layers)",
        "self.final_norm = LayerNorm(emb_dim)",
        "self.out_head = nn.Linear(emb_dim, vocab_size)",
      ],
      forward: [
        "tok_embeds = self.tok_emb(in_idx)",
        "pos_embeds = self.pos_emb(torch.arange(seq_len))",
        "x = tok_embeds + pos_embeds",
        "x = self.drop_emb(x)",
        "x = self.trf_block(x) × n_layers",
        "x = self.final_norm(x)",
        "logits = self.out_head(x)",
      ],
      keyPoint:
        "임베딩 결합 → 드롭아웃 → 여러 블록 통과 → 정규화 → 로짓 출력",
    },
  },
];

export default function Home() {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Code2 className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              GPT 실습 코드 학습 가이드
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            트랜스포머 기반 GPT 모델의 핵심 컴포넌트를 단계별로 학습하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Learning Path Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-900 mb-6">
            📚 학습 경로
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {components.map((comp, idx) => (
              <div key={idx} className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-slate-200 to-slate-300 rounded-lg blur opacity-50" />
                <div className="relative bg-white rounded-lg p-4 border border-slate-200 hover:border-slate-300 transition-colors">
                  <div className="text-2xl mb-2">{comp.icon}</div>
                  <h3 className="font-semibold text-slate-900 mb-1">
                    {comp.name}
                  </h3>
                  <p className="text-xs text-slate-600">{comp.description}</p>
                  <div className="mt-3 flex items-center text-xs font-medium text-slate-500">
                    Step {idx + 1}
                    <div className="ml-auto">
                      {idx < 3 && (
                        <ChevronDown className="w-4 h-4 rotate-[-90deg]" />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Detailed Component Breakdown */}
        <section>
          <h2 className="text-2xl font-bold text-slate-900 mb-6">
            🔍 상세 분석
          </h2>
          <div className="space-y-4">
            {components.map((comp, idx) => (
              <Card
                key={idx}
                className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                  expandedIndex === idx
                    ? `border-slate-300 shadow-lg bg-white`
                    : "border-slate-200 bg-slate-50"
                }`}
                onClick={() =>
                  setExpandedIndex(expandedIndex === idx ? null : idx)
                }
              >
                {/* Header */}
                <div
                  className={`bg-gradient-to-r ${comp.color} p-6 text-white cursor-pointer`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="text-3xl">{comp.icon}</div>
                      <div>
                        <h3 className="text-xl font-bold">{comp.name}</h3>
                        <p className="text-sm opacity-90">
                          {comp.description}
                        </p>
                      </div>
                    </div>
                    <ChevronDown
                      className={`w-6 h-6 transition-transform duration-300 ${
                        expandedIndex === idx ? "rotate-180" : ""
                      }`}
                    />
                  </div>
                </div>

                {/* Expandable Content */}
                {expandedIndex === idx && (
                  <div className="p-6 space-y-6 bg-white">
                    {/* Key Point Highlight */}
                    <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                      <p className="text-sm font-semibold text-blue-900 mb-1">
                        💡 핵심 포인트
                      </p>
                      <p className="text-sm text-blue-800">
                        {comp.details.keyPoint}
                      </p>
                    </div>

                    {/* __init__ Method */}
                    <div>
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                        <span className="text-lg">🔧</span>
                        __init__ (초기화)
                      </h4>
                      <div className="bg-slate-900 text-slate-100 rounded-lg p-4 font-mono text-sm space-y-1 overflow-x-auto">
                        {comp.details.init.map((line, i) => (
                          <div key={i} className="text-slate-300">
                            <span className="text-slate-500">{"→"}</span>{" "}
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* forward Method */}
                    <div>
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                        <span className="text-lg">⚡</span>
                        forward (순전파)
                      </h4>
                      <div className="bg-slate-900 text-slate-100 rounded-lg p-4 font-mono text-sm space-y-1 overflow-x-auto">
                        {comp.details.forward.map((line, i) => (
                          <div key={i} className="text-slate-300">
                            <span className="text-slate-500">{i + 1}.</span>{" "}
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </section>

        {/* Memory Tips */}
        <section className="mt-12 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl p-8 border border-amber-200">
          <h2 className="text-2xl font-bold text-amber-900 mb-4">
            🧠 암기 팁
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h3 className="font-semibold text-amber-900">
                ✓ 작은 부품부터 큰 구조로
              </h3>
              <p className="text-sm text-amber-800">
                LayerNorm → FeedForward → TransformerBlock → GPTModel 순서로
                학습하면 각 부품의 역할을 명확하게 이해할 수 있습니다.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-amber-900">
                ✓ 패턴 인식하기
              </h3>
              <p className="text-sm text-amber-800">
                TransformerBlock의 "norm-att-drop-add / norm-ff-drop-add" 패턴과
                GPTModel의 "tok+pos → drop → blocks → norm → head" 패턴을
                반복해서 암기하세요.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-amber-900">
                ✓ __init__과 forward 분리
              </h3>
              <p className="text-sm text-amber-800">
                __init__은 "무엇을 정의하는가", forward는 "어떻게 데이터가
                흐르는가"라고 생각하면 구조가 명확합니다.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-amber-900">
                ✓ 코드 직접 작성해보기
              </h3>
              <p className="text-sm text-amber-800">
                각 클래스를 직접 손으로 작성해보면서 학습하면 코드 구조가
                자연스럽게 암기됩니다.
              </p>
            </div>
          </div>
        </section>

        {/* Architecture Diagram */}
        <section className="mt-12">
          <h2 className="text-2xl font-bold text-slate-900 mb-6">
            📊 전체 아키텍처
          </h2>
          <div className="bg-white rounded-xl p-8 border border-slate-200 overflow-x-auto">
            <div className="min-w-max">
              <div className="flex items-center justify-between gap-4 text-center">
                {/* Input */}
                <div className="flex flex-col items-center">
                  <div className="bg-blue-100 border-2 border-blue-500 rounded-lg px-4 py-3 font-semibold text-blue-900 whitespace-nowrap">
                    Input Tokens
                  </div>
                  <div className="text-slate-500 text-sm mt-1">in_idx</div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Token Embedding */}
                <div className="flex flex-col items-center">
                  <div className="bg-blue-100 border-2 border-blue-500 rounded-lg px-4 py-3 font-semibold text-blue-900 whitespace-nowrap">
                    Token Embedding
                  </div>
                  <div className="text-slate-500 text-sm mt-1">tok_emb</div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Position Embedding */}
                <div className="flex flex-col items-center">
                  <div className="bg-purple-100 border-2 border-purple-500 rounded-lg px-4 py-3 font-semibold text-purple-900 whitespace-nowrap">
                    Position Embedding
                  </div>
                  <div className="text-slate-500 text-sm mt-1">pos_emb</div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Combine */}
                <div className="flex flex-col items-center">
                  <div className="bg-emerald-100 border-2 border-emerald-500 rounded-lg px-4 py-3 font-semibold text-emerald-900 whitespace-nowrap">
                    Combine
                  </div>
                  <div className="text-slate-500 text-sm mt-1">tok + pos</div>
                </div>
              </div>

              {/* Second Row */}
              <div className="mt-8 flex items-center justify-between gap-4 text-center">
                {/* Dropout */}
                <div className="flex flex-col items-center">
                  <div className="bg-orange-100 border-2 border-orange-500 rounded-lg px-4 py-3 font-semibold text-orange-900 whitespace-nowrap">
                    Dropout
                  </div>
                  <div className="text-slate-500 text-sm mt-1">drop_emb</div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Transformer Blocks */}
                <div className="flex flex-col items-center">
                  <div className="bg-emerald-100 border-2 border-emerald-500 rounded-lg px-4 py-3 font-semibold text-emerald-900 whitespace-nowrap">
                    Transformer Blocks
                  </div>
                  <div className="text-slate-500 text-sm mt-1">
                    × n_layers
                  </div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Final Norm */}
                <div className="flex flex-col items-center">
                  <div className="bg-blue-100 border-2 border-blue-500 rounded-lg px-4 py-3 font-semibold text-blue-900 whitespace-nowrap">
                    Final LayerNorm
                  </div>
                  <div className="text-slate-500 text-sm mt-1">final_norm</div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Output Head */}
                <div className="flex flex-col items-center">
                  <div className="bg-purple-100 border-2 border-purple-500 rounded-lg px-4 py-3 font-semibold text-purple-900 whitespace-nowrap">
                    Output Head
                  </div>
                  <div className="text-slate-500 text-sm mt-1">out_head</div>
                </div>

                {/* Arrow */}
                <div className="text-2xl text-slate-400">→</div>

                {/* Output */}
                <div className="flex flex-col items-center">
                  <div className="bg-purple-100 border-2 border-purple-500 rounded-lg px-4 py-3 font-semibold text-purple-900 whitespace-nowrap">
                    Logits
                  </div>
                  <div className="text-slate-500 text-sm mt-1">output</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="mt-12 text-center">
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-8 text-white">
            <h2 className="text-2xl font-bold mb-3">
              🚀 이제 코드를 직접 작성해보세요!
            </h2>
            <p className="mb-6 text-blue-100 max-w-2xl mx-auto">
              위의 구조를 이해했다면, 실제로 PyTorch를 사용하여 각 클래스를
              구현해보세요. 직접 손으로 코드를 작성하면서 학습하면 더욱 깊이
              있는 이해가 가능합니다.
            </p>
            <Button
              size="lg"
              className="bg-white text-purple-600 hover:bg-blue-50 font-semibold"
            >
              PyTorch 공식 문서 보기
            </Button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | 트랜스포머 기반 언어 모델 학습
          </p>
        </div>
      </footer>
    </div>
  );
}
