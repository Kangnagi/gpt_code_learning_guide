import { useState } from "react";
import { ChevronDown, Copy, Check, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useLocation } from "wouter";

interface AttentionStep {
  title: string;
  description: string;
  code: string;
  explanation: string;
  keyPoints: string[];
}

const attentionSteps: AttentionStep[] = [
  {
    title: "1단계: 기본 개념 - 입력 데이터 준비",
    description: "Self-Attention의 첫 번째 단계로 입력 데이터를 준비합니다.",
    code: `import torch
import torch.nn as nn

# 입력 데이터: 6개의 토큰, 각 토큰은 3차원 벡터
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts (x^3)
    [0.22, 0.58, 0.33],  # with (x^4)
    [0.77, 0.25, 0.10],  # one (x^5)
    [0.05, 0.80, 0.55]   # step (x^6)
])

print(f"입력 형태: {inputs.shape}")  # torch.Size([6, 3])`,
    explanation:
      "Self-Attention은 시퀀스의 각 토큰이 다른 모든 토큰과의 관계를 학습하는 메커니즘입니다. 먼저 입력 데이터를 준비합니다.",
    keyPoints: [
      "입력: (시퀀스 길이, 임베딩 차원) = (6, 3)",
      "각 행은 하나의 토큰을 나타냄",
      "모든 토큰이 동일한 차원의 벡터로 표현됨",
    ],
  },
  {
    title: "2단계: 간단한 유사도 계산 (Dot Product)",
    description: "두 토큰 간의 유사도를 내적(Dot Product)으로 계산합니다.",
    code: `# 2번째 토큰(index 1)을 기준으로 다른 모든 토큰과의 유사도 계산
query = inputs[1]  # [0.55, 0.87, 0.66]

# 각 토큰과의 내적 계산
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)

print(f"어텐션 점수: {attn_scores_2}")
# 출력: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])`,
    explanation:
      "내적을 통해 두 벡터의 유사도를 계산합니다. 값이 클수록 두 토큰이 더 유사함을 의미합니다.",
    keyPoints: [
      "내적(Dot Product): 두 벡터의 방향 유사도를 측정",
      "큰 값 = 유사한 토큰, 작은 값 = 다른 토큰",
      "모든 토큰 쌍에 대해 반복 계산 가능",
    ],
  },
  {
    title: "3단계: 소프트맥스로 확률 변환",
    description: "점수를 확률 분포로 변환하여 가중치를 계산합니다.",
    code: `# 1. 수동으로 소프트맥스 계산
attn_scores_2_exp = torch.exp(attn_scores_2)
attn_weights_2_naive = attn_scores_2_exp / attn_scores_2_exp.sum()

print(f"어텐션 가중치 (수동): {attn_weights_2_naive}")
# 출력: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

# 2. PyTorch 내장 함수 사용 (권장)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print(f"어텐션 가중치 (PyTorch): {attn_weights_2}")
print(f"합계: {attn_weights_2.sum()}")  # 1.0`,
    explanation:
      "소프트맥스 함수는 점수를 0~1 사이의 확률로 변환하며, 모든 확률의 합은 1이 됩니다.",
    keyPoints: [
      "소프트맥스: exp(x) / sum(exp(x))",
      "모든 가중치의 합 = 1.0",
      "큰 점수는 높은 확률로, 작은 점수는 낮은 확률로 변환",
    ],
  },
  {
    title: "4단계: 가중치 합산으로 컨텍스트 벡터 생성",
    description: "가중치를 적용하여 컨텍스트 벡터를 생성합니다.",
    code: `# 1. 컨텍스트 벡터 초기화
context_vec_2 = torch.zeros(query.shape)

# 2. 가중치를 적용한 입력 벡터의 합산
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(f"컨텍스트 벡터: {context_vec_2}")
# 출력: tensor([0.4419, 0.6515, 0.5683])

# 또는 행렬 연산으로 한 번에 계산
context_vec_2_matrix = attn_weights_2 @ inputs
print(f"컨텍스트 벡터 (행렬): {context_vec_2_matrix}")`,
    explanation:
      "각 입력 벡터에 계산된 가중치를 곱하고 모두 더하면, 주변 토큰들의 정보가 집약된 새로운 표현이 됩니다.",
    keyPoints: [
      "가중치 합산: Σ(weight_i × input_i)",
      "결과는 원래 입력과 동일한 차원",
      "주변 토큰들의 정보가 반영된 새로운 표현",
    ],
  },
  {
    title: "5단계: 전체 시퀀스에 대한 어텐션 계산",
    description: "모든 토큰에 대해 어텐션을 계산합니다.",
    code: `# 1. 전체 어텐션 점수 행렬 생성 (행렬 곱셈)
attn_scores = inputs @ inputs.T
print(f"어텐션 점수 행렬 형태: {attn_scores.shape}")  # torch.Size([6, 6])

# 2. 전체 어텐션 가중치 행렬 생성
attn_weights = torch.softmax(attn_scores, dim=-1)
print(f"어텐션 가중치 행렬:\\n{attn_weights}")

# 3. 모든 컨텍스트 벡터 생성
all_context_vecs = attn_weights @ inputs
print(f"모든 컨텍스트 벡터 형태: {all_context_vecs.shape}")  # torch.Size([6, 3])`,
    explanation:
      "행렬 연산을 사용하면 모든 토큰에 대한 어텐션을 효율적으로 계산할 수 있습니다.",
    keyPoints: [
      "입력 @ 입력.T = 모든 토큰 쌍의 내적",
      "softmax(dim=-1): 각 행을 정규화하여 확률 분포 생성",
      "결과: (시퀀스 길이, 시�ence 길이) 행렬",
    ],
  },
  {
    title: "6단계: 학습 가능한 가중치 추가 (Self-Attention v1)",
    description: "Query, Key, Value 행렬을 추가하여 학습 가능한 Self-Attention을 구현합니다.",
    code: `class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # 1. 학습 가능한 가중치 행렬 초기화
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        # 2. 입력을 Q, K, V 공간으로 투영
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        
        # 3. 어텐션 점수 계산 (Scaled Dot-Product)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        # 4. 컨텍스트 벡터 생성
        context_vec = attn_weights @ values
        return context_vec

# 사용 예시
d_in = 3  # 입력 차원
d_out = 2  # 출력 차원
sa = SelfAttention_v1(d_in, d_out)
output = sa(inputs)
print(f"출력 형태: {output.shape}")  # torch.Size([6, 2])`,
    explanation:
      "학습 가능한 Q, K, V 행렬을 추가하면 모델이 각 입력에 대해 최적의 표현을 학습할 수 있습니다.",
    keyPoints: [
      "W_query, W_key, W_value: 학습 가능한 파라미터",
      "Scaled Dot-Product: attn_scores / sqrt(d_k)로 스케일링",
      "스케일링은 그래디언트 안정성 향상",
    ],
  },
  {
    title: "7단계: Scaled Dot-Product Attention",
    description:
      "스케일링을 추가하여 더 안정적인 어텐션을 구현합니다.",
    code: `# Scaled Dot-Product Attention 공식:
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

def scaled_dot_product_attention(queries, keys, values):
    """
    Scaled Dot-Product Attention 계산
    
    Args:
        queries: (seq_len, d_k)
        keys: (seq_len, d_k)
        values: (seq_len, d_v)
    
    Returns:
        context_vec: (seq_len, d_v)
    """
    # 1. 어텐션 점수 계산
    attn_scores = queries @ keys.T
    
    # 2. 스케일링 (sqrt(d_k)로 나누기)
    d_k = keys.shape[-1]
    attn_scores_scaled = attn_scores / (d_k ** 0.5)
    
    # 3. 소프트맥스로 확률 변환
    attn_weights = torch.softmax(attn_scores_scaled, dim=-1)
    
    # 4. 컨텍스트 벡터 생성
    context_vec = attn_weights @ values
    
    return context_vec, attn_weights

# 사용 예시
d_k = 3
queries = inputs
keys = inputs
values = inputs
context, weights = scaled_dot_product_attention(queries, keys, values)
print(f"컨텍스트 벡터 형태: {context.shape}")`,
    explanation:
      "sqrt(d_k)로 스케일링하면 어텐션 점수의 분산이 안정화되어 학습이 더 잘 됩니다.",
    keyPoints: [
      "스케일링 인자: 1 / sqrt(d_k)",
      "큰 d_k에서 점수가 너무 커지는 것을 방지",
      "소프트맥스의 그래디언트가 더 안정적",
    ],
  },
  {
    title: "8단계: Multi-Head Attention",
    description: "여러 개의 어텐션 헤드를 병렬로 실행합니다.",
    code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        
        # 1. Q, K, V 선형 변환 (전체 차원)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # 2. 출력 선형 변환
        self.out_proj = nn.Linear(d_out, d_out)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Q, K, V 계산
        queries = self.W_query(x)  # (batch, seq_len, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # 2. 여러 헤드로 분할
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 3. 각 헤드에서 어텐션 계산
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 4. 컨텍스트 벡터 생성
        context = torch.matmul(attn_weights, values)
        
        # 5. 헤드 결합
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_out)
        
        # 6. 출력 변환
        output = self.out_proj(context)
        
        return output

# 사용 예시
mha = MultiHeadAttention(d_in=3, d_out=4, num_heads=2)
batch_inputs = inputs.unsqueeze(0)  # (1, 6, 3) - 배치 차원 추가
output = mha(batch_inputs)
print(f"출력 형태: {output.shape}")  # (1, 6, 4)`,
    explanation:
      "Multi-Head Attention은 여러 개의 어텐션 헤드를 병렬로 실행하여 다양한 관점에서 관계를 학습합니다.",
    keyPoints: [
      "num_heads개의 독립적인 어텐션 계산",
      "각 헤드는 d_out/num_heads 차원에서 작동",
      "모든 헤드의 결과를 연결(concatenate)하여 최종 출력",
      "더 풍부한 표현 학습 가능",
    ],
  },
];

export default function AttentionMechanism() {
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">⚡</span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
              Attention Mechanism 상세 분석
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            Self-Attention부터 Multi-Head Attention까지 단계별로 학습하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Info Box */}
        <div className="bg-cyan-50 border-l-4 border-cyan-500 p-6 rounded-lg mb-8">
          <h2 className="font-semibold text-cyan-900 mb-2">
            💡 Attention Mechanism이란?
          </h2>
          <p className="text-cyan-800 text-sm">
            Attention은 시퀀스의 각 토큰이 다른 모든 토큰과의 관계를 학습하는 메커니즘입니다.
            Query(쿼리), Key(키), Value(값)의 상호작용을 통해 문맥을 이해합니다.
          </p>
        </div>

        {/* Attention Steps */}
        <div className="space-y-4">
          {attentionSteps.map((step, idx) => (
            <Card
              key={idx}
              className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                expandedIndex === idx
                  ? "border-cyan-300 shadow-lg bg-white"
                  : "border-slate-200 bg-slate-50"
              }`}
              onClick={() =>
                setExpandedIndex(expandedIndex === idx ? null : idx)
              }
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-cyan-500 to-blue-600 p-6 text-white cursor-pointer">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-bold">{step.title}</h3>
                    <p className="text-cyan-100 text-sm mt-1">
                      {step.description}
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
                  {/* Code Block */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-slate-900">
                        📝 코드
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
                          className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded text-sm text-slate-700"
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

        {/* Summary Section */}
        <section className="mt-12 bg-gradient-to-r from-cyan-50 to-blue-50 rounded-xl p-8 border border-cyan-200">
          <h2 className="text-2xl font-bold text-cyan-900 mb-6">
            🎯 Attention의 진화 과정
          </h2>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-cyan-500 text-white flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <h3 className="font-semibold text-cyan-900">
                  Simplified Self-Attention
                </h3>
                <p className="text-sm text-cyan-800">
                  기본 내적 기반 유사도 계산
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <h3 className="font-semibold text-blue-900">
                  Self-Attention
                </h3>
                <p className="text-sm text-blue-800">
                  학습 가능한 Q, K, V 행렬 추가
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <h3 className="font-semibold text-purple-900">
                  Scaled Dot-Product Attention
                </h3>
                <p className="text-sm text-purple-800">
                  sqrt(d_k)로 스케일링하여 안정성 향상
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-500 text-white flex items-center justify-center font-bold">
                4
              </div>
              <div>
                <h3 className="font-semibold text-indigo-900">
                  Multi-Head Attention
                </h3>
                <p className="text-sm text-indigo-800">
                  여러 헤드로 병렬 처리하여 표현력 극대화
                </p>
              </div>
            </div>
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
            onClick={() => setLocation("/full-code")}
            className="gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
          >
            전체 코드 분석 보기
            <ArrowRight className="w-4 h-4" />
          </Button>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | Attention Mechanism 상세 분석
          </p>
        </div>
      </footer>
    </div>
  );
}
