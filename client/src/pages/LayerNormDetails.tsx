import { useState } from "react";
import { ChevronDown, Copy, Check, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useLocation } from "wouter";

interface NormalizationStep {
  title: string;
  description: string;
  code: string;
  explanation: string;
  keyPoints: string[];
}

const normalizationSteps: NormalizationStep[] = [
  {
    title: "1단계: 정규화의 필요성 이해",
    description: "왜 신경망에서 정규화가 필요한지 알아봅시다.",
    code: `import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 정규화 전: 다양한 범위의 활성화 값
print("=== 정규화 전 ===")
x = torch.tensor([
    [0.001, 0.002, 0.0015],  # 매우 작은 값
    [100.0, 200.0, 150.0],    # 매우 큰 값
    [0.5, 0.3, 0.7],          # 중간 값
])

print(f"입력 데이터:\\n{x}")
print(f"평균: {x.mean(dim=-1)}")
print(f"분산: {x.var(dim=-1)}")

# 문제점: 그래디언트 불안정성
print("\\n=== 문제점 ===")
print("• 층이 깊어질수록 활성화 값의 분포가 요동침")
print("• Gradient Vanishing (그래디언트 소실)")
print("• Gradient Exploding (그래디언트 폭주)")
print("• 학습이 느려지거나 불안정해짐")`,
    explanation:
      "신경망이 깊어질수록 각 층의 활성화 값 분포가 달라져서 학습이 불안정해집니다. 이를 해결하기 위해 정규화가 필요합니다.",
    keyPoints: [
      "활성화 값의 범위가 다양하면 옵티마이저가 적절한 학습률을 설정하기 어려움",
      "일부 층은 매우 민감하고 일부 층은 변화가 없는 불균형 발생",
      "정규화로 모든 층의 입력을 일정한 범위로 조정하여 안정적인 학습 동역학 확보",
    ],
  },
  {
    title: "2단계: 표준화 (Standardization) 과정",
    description: "데이터를 평균 0, 분산 1로 변환하는 표준화 과정입니다.",
    code: `import torch

# 입력 데이터
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

print(f"원본 데이터:\\n{x}")
print(f"원본 평균: {x.mean(dim=-1, keepdim=True)}")
print(f"원본 분산: {x.var(dim=-1, keepdim=True, unbiased=False)}")

# 1단계: 평균 계산 (마지막 차원 기준)
mean = x.mean(dim=-1, keepdim=True)
print(f"\\n평균:\\n{mean}")

# 2단계: 분산 계산
var = x.var(dim=-1, keepdim=True, unbiased=False)
print(f"분산:\\n{var}")

# 3단계: 표준화
eps = 1e-5  # 분모가 0이 되는 것을 방지
x_normalized = (x - mean) / torch.sqrt(var + eps)

print(f"\\n정규화된 데이터:\\n{x_normalized}")
print(f"정규화 후 평균: {x_normalized.mean(dim=-1, keepdim=True)}")
print(f"정규화 후 분산: {x_normalized.var(dim=-1, keepdim=True, unbiased=False)}")`,
    explanation:
      "표준화는 (x - 평균) / 표준편차 공식으로 데이터를 평균 0, 분산 1의 분포로 변환합니다.",
    keyPoints: [
      "dim=-1: 마지막 차원(피처 차원)을 기준으로 정규화",
      "keepdim=True: 계산 후 차원을 유지하여 브로드캐스팅 가능",
      "eps (epsilon): 분모가 0이 되는 것을 방지하는 작은 상수 (보통 1e-5)",
      "결과: Zero-centered 데이터로 변환 (평균=0, 분산=1)",
    ],
  },
  {
    title: "3단계: Scale과 Shift 파라미터",
    description:
      "학습 가능한 scale(γ)과 shift(β) 파라미터를 추가합니다.",
    code: `import torch
import torch.nn as nn

# 정규화된 데이터
x_normalized = torch.tensor([
    [-1.2247, 0.0000, 1.2247],
    [-1.2247, 0.0000, 1.2247],
    [-1.2247, 0.0000, 1.2247],
])

emb_dim = 3

# 1. 학습 가능한 파라미터 초기화
scale = nn.Parameter(torch.ones(emb_dim))  # γ (gamma)
shift = nn.Parameter(torch.zeros(emb_dim))  # β (beta)

print(f"Scale (γ): {scale}")
print(f"Shift (β): {shift}")

# 2. Scale과 Shift 적용
# y = γ * x_normalized + β
y = scale * x_normalized + shift

print(f"\\nScale과 Shift 적용 후:\\n{y}")

# 3. 학습 과정에서 파라미터 업데이트
# 모델이 학습하면서 scale과 shift가 최적의 값으로 조정됨
print(f"\\n=== Scale과 Shift의 역할 ===")
print("• Scale (γ): 정규화된 데이터의 분산을 조절")
print("• Shift (β): 정규화된 데이터의 평균 위치를 이동")
print("• 모델이 정규화의 정도를 자동으로 조절 가능")
print("• 필요하면 정규화를 완전히 되돌릴 수도 있음 (γ=1, β=0일 때)")`,
    explanation:
      "Scale과 Shift 파라미터를 통해 모델이 정규화의 정도를 학습하면서 조절할 수 있습니다.",
    keyPoints: [
      "Scale (γ): 분산 조절 파라미터, 초기값 1",
      "Shift (β): 평균 이동 파라미터, 초기값 0",
      "공식: y = γ * x_norm + β",
      "이 파라미터들은 모델 학습 중에 그래디언트로 업데이트됨",
      "모델이 정규화가 필요 없다고 판단하면 γ→1, β→0으로 수렴 가능",
    ],
  },
  {
    title: "4단계: dim 파라미터 이해",
    description: "dim 파라미터가 정규화 방향에 미치는 영향을 이해합니다.",
    code: `import torch

# 2D 텐서 (행렬)
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print("원본 데이터:")
print(x)
print(f"형태: {x.shape}  # (2, 3) - 2행 3열")

print("\\n=== dim=0 (행 방향) ===")
print("각 열을 기준으로 정규화")
mean_dim0 = x.mean(dim=0, keepdim=True)
var_dim0 = x.var(dim=0, keepdim=True, unbiased=False)
x_norm_dim0 = (x - mean_dim0) / torch.sqrt(var_dim0 + 1e-5)
print(f"평균 (dim=0): {mean_dim0}")
print(f"정규화 결과:\\n{x_norm_dim0}")

print("\\n=== dim=1 또는 dim=-1 (열 방향) ===")
print("각 행을 기준으로 정규화")
mean_dim1 = x.mean(dim=-1, keepdim=True)
var_dim1 = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm_dim1 = (x - mean_dim1) / torch.sqrt(var_dim1 + 1e-5)
print(f"평균 (dim=-1): {mean_dim1}")
print(f"정규화 결과:\\n{x_norm_dim1}")

print("\\n=== 3D 텐서 예시 ===")
# (배치, 시퀀스, 임베딩)
x_3d = torch.randn(2, 4, 3)
print(f"3D 텐서 형태: {x_3d.shape}  # (batch=2, seq=4, emb=3)")

# LayerNorm은 보통 마지막 차원을 기준으로 정규화
mean_3d = x_3d.mean(dim=-1, keepdim=True)
print(f"LayerNorm 평균 형태: {mean_3d.shape}  # (2, 4, 1)")`,
    explanation:
      "dim 파라미터는 정규화를 수행할 차원을 지정합니다. LayerNorm은 보통 마지막 차원(피처 차원)을 기준으로 정규화합니다.",
    keyPoints: [
      "dim=0: 행 방향으로 정규화 (각 열의 통계 계산)",
      "dim=-1: 마지막 차원 방향으로 정규화 (각 행의 통계 계산)",
      "LayerNorm: 보통 dim=-1 사용 (임베딩 차원 기준)",
      "BatchNorm: 보통 dim=0 사용 (배치 차원 기준)",
    ],
  },
  {
    title: "5단계: LayerNorm 구현",
    description: "완전한 LayerNorm 클래스를 구현합니다.",
    code: `import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization 구현
    
    입력: (batch_size, seq_len, emb_dim) 또는 (seq_len, emb_dim)
    출력: 동일한 형태
    """
    def __init__(self, emb_dim):
        super().__init__()
        # 1. 정규화 파라미터
        self.eps = 1e-5  # 분모가 0이 되는 것을 방지
        
        # 2. 학습 가능한 파라미터
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        """
        Args:
            x: 입력 텐서 (..., emb_dim)
        
        Returns:
            정규화된 텐서 (동일한 형태)
        """
        # 1. 마지막 차원을 기준으로 평균 계산
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. 마지막 차원을 기준으로 분산 계산
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 3. 정규화
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. Scale과 Shift 적용
        output = self.scale * x_normalized + self.shift
        
        return output

# 사용 예시
emb_dim = 768
layer_norm = LayerNorm(emb_dim)

# 입력 데이터
x = torch.randn(2, 4, 768)  # (batch, seq_len, emb_dim)

# LayerNorm 적용
output = layer_norm(x)

print(f"입력 형태: {x.shape}")
print(f"출력 형태: {output.shape}")
print(f"출력 평균: {output.mean(dim=-1)}")  # 거의 0에 가까움
print(f"출력 분산: {output.var(dim=-1)}")  # 거의 1에 가까움

# PyTorch 내장 LayerNorm과 비교
pytorch_ln = nn.LayerNorm(emb_dim)
pytorch_output = pytorch_ln(x)
print(f"\\nPyTorch LayerNorm 출력과 유사: {torch.allclose(output, pytorch_output, atol=1e-5)}")`,
    explanation:
      "LayerNorm은 입력을 정규화하고 학습 가능한 scale과 shift를 적용합니다.",
    keyPoints: [
      "입력의 마지막 차원을 기준으로 정규화",
      "모든 배치와 시퀀스 요소에 대해 독립적으로 정규화",
      "학습 가능한 파라미터로 정규화 정도 조절",
      "출력의 평균 ≈ 0, 분산 ≈ 1",
    ],
  },
  {
    title: "6단계: TransformerBlock에서의 LayerNorm",
    description:
      "Transformer 블록에서 LayerNorm이 어떻게 사용되는지 알아봅시다.",
    code: `import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_normalized + self.shift

class TransformerBlock(nn.Module):
    """
    Pre-Normalization 구조를 사용하는 Transformer 블록
    """
    def __init__(self, emb_dim):
        super().__init__()
        # 정규화 레이어
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        
        # 어텐션과 피드포워드는 생략 (여기서는 정규화에 집중)
        # self.attention = MultiHeadAttention(...)
        # self.ff = FeedForward(...)
    
    def forward(self, x):
        """
        Pre-Normalization 구조:
        1. 정규화 → 어텐션 → 잔차연결
        2. 정규화 → 피드포워드 → 잔차연결
        """
        # ===== 첫 번째 서브 블록: 어텐션 =====
        # 1. 입력을 저장 (잔차연결용)
        shortcut = x
        
        # 2. 정규화 (Pre-Normalization)
        x = self.norm1(x)
        
        # 3. 어텐션 적용 (생략)
        # x = self.attention(x)
        
        # 4. 잔차연결
        x = x + shortcut
        
        # ===== 두 번째 서브 블록: 피드포워드 =====
        # 1. 입력을 저장
        shortcut = x
        
        # 2. 정규화
        x = self.norm2(x)
        
        # 3. 피드포워드 적용 (생략)
        # x = self.ff(x)
        
        # 4. 잔차연결
        x = x + shortcut
        
        return x

# 사용 예시
emb_dim = 768
block = TransformerBlock(emb_dim)

x = torch.randn(2, 4, 768)  # (batch, seq_len, emb_dim)
output = block(x)

print(f"입력 형태: {x.shape}")
print(f"출력 형태: {output.shape}")
print(f"\\n=== LayerNorm의 역할 ===")
print("• Pre-Normalization: 각 서브 블록 이전에 정규화")
print("• 어텐션 이전: 입력을 정규화하여 안정적인 어텐션 계산")
print("• 피드포워드 이전: 입력을 정규화하여 안정적인 변환")
print("• 결과: 깊은 네트워크에서도 안정적인 학습")`,
    explanation:
      "Transformer에서는 Pre-Normalization 구조로 각 서브 블록 이전에 LayerNorm을 적용합니다.",
    keyPoints: [
      "Pre-Normalization: 어텐션/피드포워드 이전에 정규화",
      "Post-Normalization: 어텐션/피드포워드 이후에 정규화 (구형 구조)",
      "Pre-Normalization이 더 안정적인 학습을 제공",
      "각 서브 블록마다 독립적인 LayerNorm 사용",
    ],
  },
];

export default function LayerNormDetails() {
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">⚙️</span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
              Layer Normalization 상세 분석
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            정규화의 필요성부터 구현까지 단계별로 학습하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Info Box */}
        <div className="bg-green-50 border-l-4 border-green-500 p-6 rounded-lg mb-8">
          <h2 className="font-semibold text-green-900 mb-2">
            💡 Layer Normalization이란?
          </h2>
          <p className="text-green-800 text-sm">
            Layer Normalization은 신경망의 각 층의 입력을 정규화하여 학습을 안정화하고 가속화합니다.
            Transformer와 최신 LLM에서 필수적인 컴포넌트입니다.
          </p>
        </div>

        {/* Normalization Steps */}
        <div className="space-y-4">
          {normalizationSteps.map((step, idx) => (
            <Card
              key={idx}
              className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                expandedIndex === idx
                  ? "border-green-300 shadow-lg bg-white"
                  : "border-slate-200 bg-slate-50"
              }`}
              onClick={() =>
                setExpandedIndex(expandedIndex === idx ? null : idx)
              }
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-green-500 to-emerald-600 p-6 text-white cursor-pointer">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold">{step.title}</h3>
                  <ChevronDown
                    className={`w-6 h-6 transition-transform duration-300 flex-shrink-0 ${
                      expandedIndex === idx ? "rotate-180" : ""
                    }`}
                  />
                </div>
                <p className="text-green-100 text-sm mt-1">{step.description}</p>
              </div>

              {/* Expandable Content */}
              {expandedIndex === idx && (
                <div className="p-6 space-y-6 bg-white">
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
                          copyToClipboard(step.code, idx);
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
                      {step.code}
                    </div>
                  </div>

                  {/* Explanation */}
                  <div>
                    <h4 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                      <span className="text-lg">📖</span>
                      설명
                    </h4>
                    <p className="text-slate-700 leading-relaxed">
                      {step.explanation}
                    </p>
                  </div>

                  {/* Key Points */}
                  <div>
                    <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                      <span className="text-lg">🔑</span>
                      핵심 포인트
                    </h4>
                    <div className="space-y-2">
                      {step.keyPoints.map((point, pointIdx) => (
                        <div
                          key={pointIdx}
                          className="bg-emerald-50 border-l-4 border-emerald-400 p-3 rounded text-sm text-slate-700"
                        >
                          <p className="leading-relaxed">• {point}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </Card>
          ))}
        </div>

        {/* Comparison Section */}
        <section className="mt-12 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-8 border border-green-200">
          <h2 className="text-2xl font-bold text-green-900 mb-6">
            📊 정규화 방식 비교
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-green-300">
                  <th className="text-left py-3 px-4 font-semibold text-green-900">
                    방식
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-green-900">
                    정규화 기준
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-green-900">
                    사용 시기
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-green-900">
                    특징
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-green-100 hover:bg-green-100/50">
                  <td className="py-3 px-4 font-mono">Batch Norm</td>
                  <td className="py-3 px-4">배치 차원 (dim=0)</td>
                  <td className="py-3 px-4">CNN, 일반 신경망</td>
                  <td className="py-3 px-4">배치 크기에 의존</td>
                </tr>
                <tr className="border-b border-green-100 hover:bg-green-100/50">
                  <td className="py-3 px-4 font-mono">Layer Norm</td>
                  <td className="py-3 px-4">피처 차원 (dim=-1)</td>
                  <td className="py-3 px-4">Transformer, RNN</td>
                  <td className="py-3 px-4">배치 크기 독립적</td>
                </tr>
                <tr className="border-b border-green-100 hover:bg-green-100/50">
                  <td className="py-3 px-4 font-mono">Instance Norm</td>
                  <td className="py-3 px-4">채널 차원</td>
                  <td className="py-3 px-4">스타일 변환</td>
                  <td className="py-3 px-4">채널별 정규화</td>
                </tr>
                <tr className="hover:bg-green-100/50">
                  <td className="py-3 px-4 font-mono">Group Norm</td>
                  <td className="py-3 px-4">채널 그룹</td>
                  <td className="py-3 px-4">작은 배치</td>
                  <td className="py-3 px-4">배치 크기 독립적</td>
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
            onClick={() => setLocation("/activation-functions")}
            className="gap-2 bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700"
          >
            Activation Functions 보기
            <ArrowRight className="w-4 h-4" />
          </Button>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | Layer Normalization 상세 분석
          </p>
        </div>
      </footer>
    </div>
  );
}
