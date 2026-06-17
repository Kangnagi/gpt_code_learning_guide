import { useState } from "react";
import { ChevronDown, Copy, Check, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useLocation } from "wouter";

interface ActivationInfo {
  title: string;
  subtitle: string;
  description: string;
  code: string;
  formula?: string;
  advantages: string[];
  disadvantages?: string[];
}

const activationFunctions: ActivationInfo[] = [
  {
    title: "ReLU (Rectified Linear Unit)",
    subtitle: "전통적인 활성화 함수",
    description:
      "ReLU는 오랫동안 신경망의 표준 활성화 함수로 사용되었습니다. 간단하고 계산이 빠르지만 몇 가지 한계가 있습니다.",
    formula: "ReLU(x) = max(0, x)",
    code: `import torch
import torch.nn as nn

# ReLU 활성화 함수
relu = nn.ReLU()

# 입력 데이터
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU 적용
output = relu(x)
print(f"입력: {x}")
print(f"ReLU 출력: {output}")
# 출력: tensor([0., 0., 0., 1., 2.])

# 시각화
import matplotlib.pyplot as plt
x_range = torch.linspace(-3, 3, 100)
y_relu = torch.relu(x_range)

plt.figure(figsize=(8, 4))
plt.plot(x_range.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Activation Function')
plt.legend()
plt.show()`,
    advantages: [
      "계산이 매우 빠름 (단순 max 연산)",
      "구현이 간단함",
      "희소성(Sparsity) 유도 - 일부 뉴런만 활성화",
    ],
    disadvantages: [
      "Dead ReLU 문제: x < 0일 때 그래디언트가 0이 되어 뉴런이 학습하지 않음",
      "x = 0에서 미분 불가능 (Sharp corner)",
      "음수 영역에서 완전히 비활성화",
    ],
  },
  {
    title: "Leaky ReLU",
    subtitle: "ReLU의 개선 버전",
    description:
      "Leaky ReLU는 음수 영역에서도 작은 그래디언트를 유지하여 Dead ReLU 문제를 완화합니다.",
    formula: "Leaky ReLU(x) = max(αx, x), 여기서 α ≈ 0.01",
    code: `import torch
import torch.nn as nn

# Leaky ReLU 활성화 함수
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# 입력 데이터
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Leaky ReLU 적용
output = leaky_relu(x)
print(f"입력: {x}")
print(f"Leaky ReLU 출력: {output}")
# 출력: tensor([-0.0200, -0.0100,  0.0000,  1.0000,  2.0000])

# 비교
import matplotlib.pyplot as plt
x_range = torch.linspace(-3, 3, 100)
y_relu = torch.relu(x_range)
y_leaky_relu = torch.nn.functional.leaky_relu(x_range, negative_slope=0.01)

plt.figure(figsize=(10, 4))
plt.plot(x_range.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.plot(x_range.numpy(), y_leaky_relu.numpy(), label='Leaky ReLU', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('Output')
plt.title('ReLU vs Leaky ReLU')
plt.legend()
plt.show()`,
    advantages: [
      "음수 영역에서도 작은 그래디언트 유지",
      "Dead ReLU 문제 완화",
      "ReLU보다 안정적인 학습",
    ],
    disadvantages: [
      "α 값을 선택해야 하는 하이퍼파라미터",
      "ReLU보다 계산량 약간 증가",
    ],
  },
  {
    title: "GELU (Gaussian Error Linear Unit)",
    subtitle: "현대 LLM의 표준 활성화 함수",
    description:
      "GELU는 정규 분포의 누적 분포 함수(CDF)를 활용한 매끄러운 활성화 함수입니다. GPT-2, GPT-3, BERT 등 최신 모델에서 표준으로 사용됩니다.",
    formula:
      "GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))",
    code: `import torch
import torch.nn as nn

# GELU 활성화 함수
gelu = nn.GELU()

# 입력 데이터
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# GELU 적용
output = gelu(x)
print(f"입력: {x}")
print(f"GELU 출력: {output}")
# 출력: tensor([-0.0455, -0.1588,  0.0000,  0.8413,  1.9545])

# 비교: ReLU vs GELU
import matplotlib.pyplot as plt
x_range = torch.linspace(-3, 3, 100)
y_relu = torch.relu(x_range)
y_gelu = torch.nn.functional.gelu(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range.numpy(), y_relu.numpy(), label='ReLU', linewidth=2)
plt.plot(x_range.numpy(), y_gelu.numpy(), label='GELU', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('Output')
plt.title('ReLU vs GELU')
plt.legend(fontsize=12)
plt.show()

# FeedForward 네트워크에서 GELU 사용
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),  # GELU 활성화
            nn.Linear(4 * emb_dim, emb_dim),
        )
    
    def forward(self, x):
        return self.layers(x)

# 사용 예시
ff = FeedForward(emb_dim=768)
input_tensor = torch.randn(6, 768)  # (seq_len, emb_dim)
output = ff(input_tensor)
print(f"입력 형태: {input_tensor.shape}")
print(f"출력 형태: {output.shape}")`,
    advantages: [
      "모든 지점에서 미분 가능 (매끄러운 곡선)",
      "음수 영역에서도 작은 그래디언트 유지",
      "Dead ReLU 문제 없음",
      "더 정교한 최적화 성능",
      "최신 LLM의 표준으로 채택됨",
    ],
    disadvantages: [
      "ReLU보다 계산량 더 많음",
      "복잡한 수식으로 인한 구현 복잡도",
    ],
  },
  {
    title: "SiLU (Sigmoid Linear Unit) / Swish",
    subtitle: "또 다른 현대 활성화 함수",
    description:
      "SiLU는 시그모이드 함수를 사용한 활성화 함수로, GELU와 유사한 성능을 보입니다.",
    formula: "SiLU(x) = x × σ(x), 여기서 σ는 시그모이드 함수",
    code: `import torch
import torch.nn as nn

# SiLU 활성화 함수
silu = nn.SiLU()

# 입력 데이터
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# SiLU 적용
output = silu(x)
print(f"입력: {x}")
print(f"SiLU 출력: {output}")

# 비교: GELU vs SiLU
import matplotlib.pyplot as plt
x_range = torch.linspace(-3, 3, 100)
y_gelu = torch.nn.functional.gelu(x_range)
y_silu = torch.nn.functional.silu(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range.numpy(), y_gelu.numpy(), label='GELU', linewidth=2)
plt.plot(x_range.numpy(), y_silu.numpy(), label='SiLU', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('Output')
plt.title('GELU vs SiLU')
plt.legend(fontsize=12)
plt.show()`,
    advantages: [
      "GELU와 유사한 성능",
      "계산이 GELU보다 약간 빠름",
      "모든 지점에서 미분 가능",
    ],
    disadvantages: [
      "GELU보다 덜 알려짐",
      "여전히 ReLU보다 계산 비용 높음",
    ],
  },
];

export default function ActivationFunctions() {
  const [, setLocation] = useLocation();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">🔥</span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
              Activation Functions 상세 분석
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            ReLU부터 GELU까지 활성화 함수의 진화 과정을 학습하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Info Box */}
        <div className="bg-orange-50 border-l-4 border-orange-500 p-6 rounded-lg mb-8">
          <h2 className="font-semibold text-orange-900 mb-2">
            💡 활성화 함수란?
          </h2>
          <p className="text-orange-800 text-sm">
            활성화 함수는 신경망에 비선형성을 도입하여 복잡한 패턴을 학습할 수 있게 합니다.
            선택하는 활성화 함수는 모델의 성능과 학습 안정성에 큰 영향을 미칩니다.
          </p>
        </div>

        {/* Activation Functions */}
        <div className="space-y-4">
          {activationFunctions.map((func, idx) => (
            <Card
              key={idx}
              className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                expandedIndex === idx
                  ? "border-orange-300 shadow-lg bg-white"
                  : "border-slate-200 bg-slate-50"
              }`}
              onClick={() =>
                setExpandedIndex(expandedIndex === idx ? null : idx)
              }
            >
              {/* Header */}
              <div
                className={`p-6 text-white cursor-pointer ${
                  idx === 0
                    ? "bg-gradient-to-r from-blue-500 to-cyan-600"
                    : idx === 1
                      ? "bg-gradient-to-r from-purple-500 to-pink-600"
                      : idx === 2
                        ? "bg-gradient-to-r from-orange-500 to-red-600"
                        : "bg-gradient-to-r from-green-500 to-emerald-600"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-bold">{func.title}</h3>
                    <p className="text-white/80 text-sm mt-1">
                      {func.subtitle}
                    </p>
                  </div>
                  <ChevronDown
                    className={`w-6 h-6 transition-transform duration-300 flex-shrink-0 ${
                      expandedIndex === idx ? "rotate-180" : ""
                    }`}
                  />
                </div>
              </div>

              {/* Expandable Content */}
              {expandedIndex === idx && (
                <div className="p-6 space-y-6 bg-white">
                  {/* Description */}
                  <div>
                    <h4 className="font-semibold text-slate-900 mb-2">
                      📌 개요
                    </h4>
                    <p className="text-slate-700 leading-relaxed">
                      {func.description}
                    </p>
                  </div>

                  {/* Formula */}
                  {func.formula && (
                    <div className="bg-slate-100 p-4 rounded-lg">
                      <h4 className="font-semibold text-slate-900 mb-2">
                        📐 수학 공식
                      </h4>
                      <p className="font-mono text-slate-800 text-center text-lg">
                        {func.formula}
                      </p>
                    </div>
                  )}

                  {/* Code Block */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-slate-900">
                        💻 코드
                      </h4>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation();
                          copyToClipboard(func.code, idx);
                        }}
                        className="gap-2"
                      >
                        {copiedIndex === idx ? (
                          <>
                            <Check className="w-4 h-4" />
                            복사됨
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4" />
                            복사
                          </>
                        )}
                      </Button>
                    </div>
                    <div className="bg-slate-900 text-slate-100 rounded-lg p-4 font-mono text-sm overflow-x-auto whitespace-pre-wrap break-words">
                      {func.code}
                    </div>
                  </div>

                  {/* Advantages */}
                  <div>
                    <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                      <span className="text-lg">✅</span>
                      장점
                    </h4>
                    <div className="space-y-2">
                      {func.advantages.map((adv, advIdx) => (
                        <div
                          key={advIdx}
                          className="bg-green-50 border-l-4 border-green-400 p-3 rounded text-sm text-slate-700"
                        >
                          <p className="leading-relaxed">• {adv}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Disadvantages */}
                  {func.disadvantages && func.disadvantages.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                        <span className="text-lg">⚠️</span>
                        단점
                      </h4>
                      <div className="space-y-2">
                        {func.disadvantages.map((dis, disIdx) => (
                          <div
                            key={disIdx}
                            className="bg-red-50 border-l-4 border-red-400 p-3 rounded text-sm text-slate-700"
                          >
                            <p className="leading-relaxed">• {dis}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </Card>
          ))}
        </div>

        {/* Comparison Section */}
        <section className="mt-12 bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-8 border border-orange-200">
          <h2 className="text-2xl font-bold text-orange-900 mb-6">
            📊 활성화 함수 비교
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-orange-300">
                  <th className="text-left py-3 px-4 font-semibold text-orange-900">
                    함수
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-orange-900">
                    계산 속도
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-orange-900">
                    학습 안정성
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-orange-900">
                    사용 시기
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-orange-100 hover:bg-orange-100/50">
                  <td className="py-3 px-4 font-mono">ReLU</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">CNN, 일반 신경망</td>
                </tr>
                <tr className="border-b border-orange-100 hover:bg-orange-100/50">
                  <td className="py-3 px-4 font-mono">Leaky ReLU</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">GAN, 깊은 네트워크</td>
                </tr>
                <tr className="border-b border-orange-100 hover:bg-orange-100/50">
                  <td className="py-3 px-4 font-mono">GELU</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">Transformer, LLM</td>
                </tr>
                <tr className="hover:bg-orange-100/50">
                  <td className="py-3 px-4 font-mono">SiLU</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">최신 LLM</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Navigation */}
        <section className="mt-12 flex gap-4 flex-wrap">
          <Button
            onClick={() => setLocation("/")}
            variant="outline"
            className="gap-2"
          >
            ← 홈으로 돌아가기
          </Button>
          <Button
            onClick={() => setLocation("/attention-mechanism")}
            className="gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
          >
            Attention Mechanism 보기
            <ArrowRight className="w-4 h-4" />
          </Button>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | Activation Functions 상세 분석
          </p>
        </div>
      </footer>
    </div>
  );
}
