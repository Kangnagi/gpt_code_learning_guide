import { useState } from "react";
import { ChevronDown, Copy, Check, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useLocation } from "wouter";

interface ConfigSection {
  title: string;
  description: string;
  code: string;
  explanation: string;
  keyPoints: string[];
}

const configSections: ConfigSection[] = [
  {
    title: "1단계: GPT-2 하이퍼파라미터 이해",
    description: "GPT-2 모델의 핵심 설정값들을 알아봅시다.",
    code: `# GPT-2 Configuration Parameters
# 모델의 아키텍처와 동작을 정의하는 핵심 하이퍼파라미터

class GPT2Config:
    """GPT-2 모델의 설정값"""
    
    # 1. 어휘 크기 (Vocabulary Size)
    vocab_size = 50_257
    # BPE(Byte Pair Encoding) 토크나이저가 사용하는 단어 사전 크기
    
    # 2. 컨텍스트 길이 (Context Length)
    context_length = 1024
    # 모델이 한 번에 처리할 수 있는 최대 토큰 수
    # 위치 임베딩(Positional Embedding)으로 제한됨
    
    # 3. 임베딩 차원 (Embedding Dimension)
    emb_dim = 768
    # 각 토큰이 변환되는 벡터의 차원
    # 모델의 "너비"를 결정
    
    # 4. 어텐션 헤드 수 (Number of Attention Heads)
    n_heads = 12
    # Multi-Head Attention에서 병렬로 실행되는 어텐션의 개수
    # emb_dim은 n_heads로 나누어떨어져야 함
    # 각 헤드의 차원: emb_dim / n_heads = 768 / 12 = 64
    
    # 5. Transformer 블록 수 (Number of Layers)
    n_layers = 12
    # 모델의 "깊이"를 결정
    # 더 많은 블록 = 더 깊은 모델 = 더 복잡한 패턴 학습 가능
    
    # 6. 드롭아웃 비율 (Dropout Rate)
    drop_rate = 0.1
    # 0.1 = 10% 확률로 뉴런을 무작위로 비활성화
    # 과적합(Overfitting) 방지
    
    # 7. QKV 바이어스 (Query, Key, Value Bias)
    qkv_bias = False
    # Multi-Head Attention의 선형 변환에 바이어스 추가 여부
    # GPT-2는 바이어스를 사용하지 않음

print(f"총 파라미터 수 (대략): {GPT2Config.vocab_size * GPT2Config.emb_dim + GPT2Config.emb_dim * GPT2Config.emb_dim * 4 * GPT2Config.n_layers}")`,
    explanation:
      "GPT-2의 하이퍼파라미터는 모델의 크기, 성능, 학습 속도를 결정합니다. 각 파라미터의 의미와 역할을 이해하는 것이 중요합니다.",
    keyPoints: [
      "vocab_size: 50,257 - BPE 토크나이저의 어휘 크기",
      "context_length: 1024 - 최대 입력 시퀀스 길이",
      "emb_dim: 768 - 임베딩 벡터의 차원 (모델의 너비)",
      "n_heads: 12 - 멀티헤드 어텐션의 헤드 수",
      "n_layers: 12 - Transformer 블록의 개수 (모델의 깊이)",
      "drop_rate: 0.1 - 드롭아웃 비율 (10%)",
      "qkv_bias: False - 어텐션 선형층의 바이어스 미사용",
    ],
  },
  {
    title: "2단계: GPT-2 모델 크기 비교",
    description: "GPT-2의 다양한 크기 버전들을 비교합니다.",
    code: `# GPT-2 모델 크기별 설정값 비교

gpt2_configs = {
    "Small": {
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "parameters": "125M"
    },
    "Medium": {
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "parameters": "355M"
    },
    "Large": {
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "parameters": "774M"
    },
    "XL": {
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "parameters": "1.5B"
    }
}

# 설정값 비교
import pandas as pd

df = pd.DataFrame(gpt2_configs).T
print(df)

# 각 버전의 특징
print("\\n=== GPT-2 모델 크기별 특징 ===")
print("Small (125M):")
print("  • 개인 PC에서 실행 가능")
print("  • 빠른 학습과 추론")
print("  • 기본 학습용으로 적합")
print()
print("Medium (355M):")
print("  • 더 나은 성능")
print("  • 적절한 크기와 성능의 균형")
print()
print("Large (774M):")
print("  • 고성능 모델")
print("  • GPU 메모리 충분히 필요")
print()
print("XL (1.5B):")
print("  • 최고 성능")
print("  • 고사양 GPU 필수")`,
    explanation:
      "GPT-2는 다양한 크기로 제공되며, 각 버전은 임베딩 차원, 헤드 수, 레이어 수를 조정하여 만들어집니다.",
    keyPoints: [
      "Small: 125M 파라미터 - 학습용 기본 모델",
      "Medium: 355M 파라미터 - 균형잡힌 성능",
      "Large: 774M 파라미터 - 고성능 모델",
      "XL: 1.5B 파라미터 - 최고 성능",
      "더 큰 모델 = 더 많은 파라미터 = 더 나은 성능 (하지만 더 많은 메모리 필요)",
    ],
  },
  {
    title: "3단계: 데이터 흐름 (Token IDs to Logits)",
    description: "입력부터 출력까지의 텐서 형태 변화를 추적합니다.",
    code: `import torch
import torch.nn as nn

# GPT-2 설정
vocab_size = 50_257
context_length = 1024
emb_dim = 768
batch_size = 8

print("=== Token IDs to Logits 데이터 흐름 ===\\n")

# 1단계: 입력 (Token IDs)
token_ids = torch.randint(0, vocab_size, (batch_size, context_length))
print(f"1. Input (Token IDs)")
print(f"   형태: {token_ids.shape}")
print(f"   의미: (배치 크기, 시퀀스 길이)")
print(f"   예: {token_ids[0, :10]}\\n")

# 2단계: Token Embedding
token_embedding = nn.Embedding(vocab_size, emb_dim)
x_emb = token_embedding(token_ids)
print(f"2. Token Embedding")
print(f"   형태: {x_emb.shape}")
print(f"   의미: (배치, 시퀀스, 임베딩 차원)")
print(f"   변환: 각 토큰이 {emb_dim}차원 벡터로 변환\\n")

# 3단계: Positional Embedding 추가
pos_embedding = nn.Embedding(context_length, emb_dim)
positions = torch.arange(context_length).unsqueeze(0).expand(batch_size, -1)
x_pos = pos_embedding(positions)
x = x_emb + x_pos
print(f"3. Positional Embedding 추가")
print(f"   형태: {x.shape}")
print(f"   의미: 토큰 임베딩 + 위치 임베딩")
print(f"   목적: 토큰의 순서 정보 추가\\n")

# 4단계: Transformer 블록 통과 (생략, 형태 유지)
print(f"4. Transformer 블록 통과 (12개)")
print(f"   형태: {x.shape}")
print(f"   변화: 형태는 유지, 내용은 문맥 정보로 재인코딩\\n")

# 5단계: Output Linear Layer
output_linear = nn.Linear(emb_dim, vocab_size)
logits = output_linear(x)
print(f"5. Output Linear Layer")
print(f"   형태: {logits.shape}")
print(f"   의미: (배치, 시퀀스, 어휘 크기)")
print(f"   설명: 각 토큰 위치에서 다음 토큰의 확률\\n")

# 6단계: Softmax로 확률 변환
probabilities = torch.softmax(logits, dim=-1)
print(f"6. Softmax 변환")
print(f"   형태: {probabilities.shape}")
print(f"   의미: 각 위치에서 어휘의 확률 분포")
print(f"   합계: {probabilities.sum(dim=-1)[0, 0]:.4f} (≈ 1.0)\\n")

# 7단계: 다음 토큰 선택
next_token_probs = probabilities[0, -1, :]  # 마지막 위치의 확률
next_token_id = torch.argmax(next_token_probs).item()
print(f"7. 다음 토큰 선택")
print(f"   다음 토큰 ID: {next_token_id}")
print(f"   확률: {next_token_probs[next_token_id]:.4f}\\n")

print("=== 요약 ===")
print(f"입력: (배치={batch_size}, 시퀀스={context_length})")
print(f"출력: (배치={batch_size}, 시퀀스={context_length}, 어휘={vocab_size})")`,
    explanation:
      "GPT 모델의 데이터는 토큰 ID에서 시작하여 임베딩, Transformer 블록을 거쳐 최종적으로 로짓(확률)으로 변환됩니다.",
    keyPoints: [
      "입력: Token IDs (정수) → (batch, seq_len)",
      "Embedding: 각 토큰을 벡터로 변환 → (batch, seq_len, emb_dim)",
      "Positional Embedding: 위치 정보 추가",
      "Transformer 블록: 형태 유지, 내용 변환",
      "Output: 로짓 생성 → (batch, seq_len, vocab_size)",
      "Softmax: 로짓을 확률로 변환",
    ],
  },
  {
    title: "4단계: Weight Tying (가중치 공유)",
    description: "임베딩과 출력 레이어의 가중치를 공유하는 방식입니다.",
    code: `import torch
import torch.nn as nn

# GPT-2 설정
vocab_size = 50_257
emb_dim = 768

print("=== Weight Tying 개념 ===\\n")

# 일반적인 방식 (가중치 공유 없음)
print("1. 일반적인 방식 (가중치 공유 없음)")
print(f"   Token Embedding 가중치: ({vocab_size}, {emb_dim})")
print(f"   Output Linear 가중치: ({emb_dim}, {vocab_size})")
print(f"   총 파라미터: {vocab_size * emb_dim + emb_dim * vocab_size}")
print(f"   = {vocab_size * emb_dim * 2}\\n")

# Weight Tying 방식
print("2. Weight Tying 방식 (가중치 공유)")
print(f"   Token Embedding 가중치: ({vocab_size}, {emb_dim})")
print(f"   Output Linear 가중치: Token Embedding의 전치 행렬 사용")
print(f"   총 파라미터: {vocab_size * emb_dim}")
print(f"   절감: {vocab_size * emb_dim} (약 50%)\\n")

# 수학적 설명
print("=== 수학적 설명 ===\\n")
print("Token Embedding (입력):")
print("  X_emb = V × W_emb")
print("  여기서 V는 one-hot 벡터, W_emb는 (vocab_size, emb_dim)\\n")

print("Output Linear Layer (일반):")
print("  Logits = H × W_out")
print("  여기서 W_out은 (emb_dim, vocab_size)\\n")

print("Output Linear Layer (Weight Tying):")
print("  Logits = H × W_emb^T")
print("  W_emb^T는 (emb_dim, vocab_size) - 임베딩 행렬의 전치\\n")

# 코드 구현
print("=== 코드 구현 ===\\n")

class GPTWithWeightTying(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        # 공유할 임베딩 가중치
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(1024, emb_dim)
        
        # Transformer 블록 (생략)
        # self.transformer_blocks = ...
        
        # 출력 레이어: 임베딩 가중치의 전치 사용
        # Logits = H @ self.token_emb.weight.T
    
    def forward(self, token_ids):
        # 임베딩
        x = self.token_emb(token_ids)
        # ... Transformer 블록 처리 ...
        # 출력: H @ self.token_emb.weight.T
        logits = x @ self.token_emb.weight.T
        return logits

# 파라미터 계산
model = GPTWithWeightTying(vocab_size, emb_dim)
total_params = sum(p.numel() for p in model.parameters())
print(f"Weight Tying 적용 모델의 파라미터: {total_params:,}")
print(f"임베딩 파라미터: {vocab_size * emb_dim:,}")`,
    explanation:
      "Weight Tying은 토큰 임베딩 레이어와 출력 선형 레이어의 가중치를 공유하여 파라미터 수를 줄이는 기법입니다.",
    keyPoints: [
      "임베딩 가중치: (vocab_size, emb_dim) = (50257, 768)",
      "출력 가중치: 임베딩 가중치의 전치 (emb_dim, vocab_size)",
      "파라미터 절감: 약 3,800만 개 (50%)",
      "성능: 가중치 공유해도 성능 저하 없음",
      "수식: Logits = H × W_emb^T",
      "이유: 입력과 출력의 어휘 공간이 동일하므로 같은 가중치 사용 가능",
    ],
  },
];

export default function GPT2Config() {
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">⚙️</span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              GPT-2 Configuration & Weight Tying
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            GPT-2의 하이퍼파라미터와 가중치 공유 기법을 학습하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Info Box */}
        <div className="bg-purple-50 border-l-4 border-purple-500 p-6 rounded-lg mb-8">
          <h2 className="font-semibold text-purple-900 mb-2">
            💡 GPT-2 Configuration이란?
          </h2>
          <p className="text-purple-800 text-sm">
            GPT-2의 하이퍼파라미터는 모델의 크기, 성능, 학습 속도를 결정합니다.
            Weight Tying은 파라미터 수를 줄이면서도 성능을 유지하는 기법입니다.
          </p>
        </div>

        {/* Config Sections */}
        <div className="space-y-4">
          {configSections.map((section, idx) => (
            <Card
              key={idx}
              className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                expandedIndex === idx
                  ? "border-purple-300 shadow-lg bg-white"
                  : "border-slate-200 bg-slate-50"
              }`}
              onClick={() =>
                setExpandedIndex(expandedIndex === idx ? null : idx)
              }
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-purple-500 to-pink-600 p-6 text-white cursor-pointer">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold">{section.title}</h3>
                  <ChevronDown
                    className={`w-6 h-6 transition-transform duration-300 flex-shrink-0 ${
                      expandedIndex === idx ? "rotate-180" : ""
                    }`}
                  />
                </div>
                <p className="text-purple-100 text-sm mt-1">
                  {section.description}
                </p>
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
                          copyToClipboard(section.code, idx);
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
                      {section.code}
                    </div>
                  </div>

                  {/* Explanation */}
                  <div>
                    <h4 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                      <span className="text-lg">📖</span>
                      설명
                    </h4>
                    <p className="text-slate-700 leading-relaxed">
                      {section.explanation}
                    </p>
                  </div>

                  {/* Key Points */}
                  <div>
                    <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                      <span className="text-lg">🔑</span>
                      핵심 포인트
                    </h4>
                    <div className="space-y-2">
                      {section.keyPoints.map((point, pointIdx) => (
                        <div
                          key={pointIdx}
                          className="bg-purple-50 border-l-4 border-purple-400 p-3 rounded text-sm text-slate-700"
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
        <section className="mt-12 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-8 border border-purple-200">
          <h2 className="text-2xl font-bold text-purple-900 mb-6">
            📊 GPT-2 파라미터 계산
          </h2>
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border border-purple-200">
              <h3 className="font-semibold text-purple-900 mb-2">
                GPT-2 Small (125M)
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-slate-600">임베딩 차원</p>
                  <p className="font-mono font-bold">768</p>
                </div>
                <div>
                  <p className="text-slate-600">어텐션 헤드</p>
                  <p className="font-mono font-bold">12</p>
                </div>
                <div>
                  <p className="text-slate-600">레이어 수</p>
                  <p className="font-mono font-bold">12</p>
                </div>
                <div>
                  <p className="text-slate-600">총 파라미터</p>
                  <p className="font-mono font-bold">~125M</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg border border-purple-200">
              <h3 className="font-semibold text-purple-900 mb-2">
                파라미터 구성
              </h3>
              <ul className="text-sm space-y-2 text-slate-700">
                <li>
                  • Token Embedding: 50,257 × 768 = 38,597,376
                </li>
                <li>• Positional Embedding: 1,024 × 768 = 786,432</li>
                <li>
                  • Transformer Blocks (12개): ~85,000,000
                </li>
                <li>• Output Linear: 768 × 50,257 (Weight Tying 사용)</li>
              </ul>
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
            onClick={() => setLocation("/layer-norm-details")}
            className="gap-2 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
          >
            Layer Normalization 보기
            <ArrowRight className="w-4 h-4" />
          </Button>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | GPT-2 Configuration & Weight Tying
          </p>
        </div>
      </footer>
    </div>
  );
}
