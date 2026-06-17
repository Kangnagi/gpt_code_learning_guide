import { useState } from "react";
import { ChevronDown, Copy, Check, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useLocation } from "wouter";

interface GenerationSection {
  title: string;
  description: string;
  code: string;
  explanation: string;
  keyPoints: string[];
}

const generationSections: GenerationSection[] = [
  {
    title: "1단계: Dummy GPT 모델 개념",
    description: "복잡한 구조를 단순화하여 데이터 흐름을 검증합니다.",
    code: `import torch
import torch.nn as nn

print("=== Dummy GPT 모델의 목적 ===\\n")
print("1. 전체 아키텍처의 데이터 흐름 검증")
print("2. 텐서 차원 문제 조기 발견")
print("3. 입출력 형태 확인")
print("4. 실제 구현 전 설계 검증\\n")

class DummyGPT(nn.Module):
    """
    실제 Transformer 블록 없이 입출력 형태만 맞춘 더미 모델
    """
    def __init__(self, vocab_size, context_length, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        
        # 1. 임베딩 레이어
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        
        # 2. 출력 레이어
        self.output_linear = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # 1. 토큰 임베딩
        x = self.token_emb(token_ids)  # (batch, seq_len, emb_dim)
        
        # 2. 위치 임베딩 추가
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(positions)
        
        # 3. 여기서는 Transformer 블록 생략
        # 실제로는 여러 Transformer 블록을 통과
        # for block in self.transformer_blocks:
        #     x = block(x)
        
        # 4. 출력 레이어
        logits = self.output_linear(x)  # (batch, seq_len, vocab_size)
        
        return logits

# 사용 예시
vocab_size = 50_257
context_length = 1024
emb_dim = 768
batch_size = 4

model = DummyGPT(vocab_size, context_length, emb_dim)

# 입력 생성
token_ids = torch.randint(0, vocab_size, (batch_size, context_length))

# 순전파
logits = model(token_ids)

print(f"입력 형태: {token_ids.shape}")
print(f"출력 형태: {logits.shape}")
print(f"예상 출력: (batch={batch_size}, seq_len={context_length}, vocab={vocab_size})")
print(f"실제 출력: {logits.shape}")
print(f"✓ 형태 검증 완료!")`,
    explanation:
      "Dummy 모델은 실제 Transformer 블록 없이 임베딩과 출력 레이어만 포함하여 데이터 흐름을 빠르게 검증합니다.",
    keyPoints: [
      "입출력 형태만 맞춘 간단한 모델",
      "Transformer 블록 생략 (성능 무관)",
      "데이터 흐름 검증에 집중",
      "텐서 차원 문제 조기 발견",
      "실제 구현 전 설계 검증",
      "빠른 디버깅 가능",
    ],
  },
  {
    title: "2단계: 텍스트 생성 기본 원리",
    description: "GPT가 한 번에 한 토큰씩 텍스트를 생성하는 방식입니다.",
    code: `import torch
import torch.nn as nn

print("=== 텍스트 생성의 기본 원리 ===\\n")

# 간단한 GPT 모델 (실제로는 더 복잡함)
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, token_ids):
        x = self.embedding(token_ids)
        logits = self.output(x)
        return logits

# 모델 초기화
vocab_size = 50_257
emb_dim = 768
model = SimpleGPT(vocab_size, emb_dim)
model.eval()

# 토크나이저 (실제로는 BPE 토크나이저 사용)
def tokenize(text):
    """텍스트를 토큰 ID로 변환 (간단한 예시)"""
    return torch.tensor([1, 2, 3], dtype=torch.long)  # 실제로는 복잡함

def detokenize(token_id):
    """토큰 ID를 텍스트로 변환"""
    token_to_word = {1: "Hello", 2: "world", 3: "!"}
    return token_to_word.get(token_id, "[UNK]")

# 텍스트 생성 프로세스
print("=== 단계별 텍스트 생성 ===\\n")

# 초기 입력
prompt = "Hello"
token_ids = tokenize(prompt)
print(f"초기 입력: {prompt}")
print(f"토큰 ID: {token_ids}\\n")

# 반복적으로 토큰 생성
max_new_tokens = 5
generated_tokens = []

for step in range(max_new_tokens):
    print(f"--- 스텝 {step + 1} ---")
    
    # 1. 모델에 입력
    with torch.no_grad():
        logits = model(token_ids.unsqueeze(0))  # (1, seq_len, vocab_size)
    
    # 2. 마지막 위치의 로짓 추출
    last_logits = logits[0, -1, :]  # (vocab_size,)
    
    # 3. 소프트맥스로 확률 변환
    probs = torch.softmax(last_logits, dim=-1)
    
    # 4. 다음 토큰 선택 (greedy 방식)
    next_token_id = torch.argmax(probs).item()
    
    # 5. 생성된 토큰 저장
    generated_tokens.append(next_token_id)
    
    # 6. 다음 입력에 추가
    token_ids = torch.cat([token_ids, torch.tensor([next_token_id])])
    
    print(f"생성된 토큰 ID: {next_token_id}")
    print(f"생성된 단어: {detokenize(next_token_id)}")
    print(f"확률: {probs[next_token_id]:.4f}\\n")

print("=== 생성 완료 ===")
print(f"생성된 토큰: {generated_tokens}")`,
    explanation:
      "GPT는 초기 입력부터 시작하여 한 번에 한 토큰씩 생성합니다. 각 단계에서 모델은 현재까지의 시퀀스를 입력받아 다음 토큰의 확률을 계산합니다.",
    keyPoints: [
      "초기 입력(프롬프트)에서 시작",
      "모델이 다음 토큰의 확률 분포 계산",
      "확률이 가장 높은 토큰 선택 (Greedy)",
      "선택된 토큰을 입력에 추가",
      "최대 토큰 수 또는 종료 토큰까지 반복",
      "결과: 프롬프트 + 생성된 텍스트",
    ],
  },
  {
    title: "3단계: 다양한 샘플링 전략",
    description: "텍스트 생성의 다양성을 조절하는 방법들입니다.",
    code: `import torch
import torch.nn as nn

print("=== 텍스트 생성 샘플링 전략 ===\\n")

# 로짓 (모델 출력)
logits = torch.tensor([2.0, 1.0, 0.5, 0.1])  # 예시
probs = torch.softmax(logits, dim=-1)

print(f"로짓: {logits}")
print(f"확률: {probs}\\n")

# 1. Greedy Decoding
print("1. Greedy Decoding (탐욕적 선택)")
print("   설명: 확률이 가장 높은 토큰 선택")
greedy_token = torch.argmax(probs).item()
print(f"   선택된 토큰: {greedy_token}")
print(f"   확률: {probs[greedy_token]:.4f}")
print("   장점: 결정적, 빠름")
print("   단점: 다양성 없음, 반복적\\n")

# 2. Temperature Sampling
print("2. Temperature Sampling")
print("   설명: 온도 파라미터로 확률 분포 조절")
temperatures = [0.5, 1.0, 2.0]
for temp in temperatures:
    scaled_logits = logits / temp
    scaled_probs = torch.softmax(scaled_logits, dim=-1)
    print(f"   Temperature={temp}: {scaled_probs}")
print("   • Temperature < 1: 확률 분포 더 뾰족 (결정적)")
print("   • Temperature = 1: 원본 확률 분포")
print("   • Temperature > 1: 확률 분포 더 완만 (다양성)\\n")

# 3. Top-k Sampling
print("3. Top-k Sampling")
print("   설명: 확률이 높은 k개 토큰만 고려")
k = 2
top_k_probs, top_k_indices = torch.topk(probs, k)
print(f"   Top-{k} 토큰: {top_k_indices}")
print(f"   Top-{k} 확률: {top_k_probs}")
print(f"   정규화된 확률: {top_k_probs / top_k_probs.sum()}")
print("   장점: 말이 안 되는 토큰 제거")
print("   단점: 고정된 k 값\\n")

# 4. Top-p (Nucleus) Sampling
print("4. Top-p (Nucleus) Sampling")
print("   설명: 누적 확률이 p를 초과할 때까지의 토큰 고려")
p = 0.9
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum_probs = torch.cumsum(sorted_probs, dim=0)
mask = cumsum_probs <= p
print(f"   누적 확률: {cumsum_probs}")
print(f"   마스크 (p={p}): {mask}")
print(f"   선택된 토큰: {sorted_indices[mask]}")
print("   장점: 동적으로 다양성 조절")
print("   단점: 계산 비용 약간 증가\\n")

# 5. Beam Search
print("5. Beam Search")
print("   설명: 여러 경로를 동시에 탐색")
print("   예: beam_width=3")
print("   • 각 스텝에서 상위 3개 경로 유지")
print("   • 최종적으로 가장 확률이 높은 경로 선택")
print("   장점: 전역 최적 해에 가까움")
print("   단점: 계산 비용 높음\\n")

print("=== 샘플링 전략 비교 ===")
print("Greedy: 빠름, 다양성 낮음")
print("Temperature: 간단함, 직관적")
print("Top-k: 말이 안 되는 토큰 제거")
print("Top-p: 동적 다양성 조절")
print("Beam Search: 높은 품질, 느림")`,
    explanation:
      "텍스트 생성 시 다양한 샘플링 전략을 사용하여 생성 텍스트의 다양성과 품질을 조절할 수 있습니다.",
    keyPoints: [
      "Greedy: 확률 최대 토큰 선택 (결정적)",
      "Temperature: 확률 분포 조절 (0.5~2.0)",
      "Top-k: 상위 k개 토큰만 고려",
      "Top-p: 누적 확률 p까지의 토큰 고려",
      "Beam Search: 여러 경로 동시 탐색",
      "선택: 속도 vs 품질의 트레이드오프",
    ],
  },
  {
    title: "4단계: 실제 텍스트 생성 구현",
    description: "완전한 텍스트 생성 함수를 구현합니다.",
    code: `import torch
import torch.nn as nn

class GPTTextGenerator:
    """GPT 모델을 사용한 텍스트 생성"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        seed: int = None
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 초기 프롬프트 텍스트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도 (0.5~2.0)
            top_k: Top-k 샘플링 (None이면 미사용)
            top_p: Top-p 샘플링 (None이면 미사용)
            seed: 난수 시드
        
        Returns:
            생성된 텍스트
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # 1. 프롬프트 토큰화
        token_ids = self.tokenizer.encode(prompt)
        token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        token_ids = token_ids.to(self.device)
        
        # 2. 토큰 생성 반복
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 모델 순전파
                logits = self.model(token_ids)
                
                # 마지막 위치의 로짓
                next_logits = logits[0, -1, :] / temperature
                
                # Top-k 필터링
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits[top_k_indices] = top_k_logits
                
                # Top-p 필터링
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=0)
                    sorted_indices_to_remove = cumsum > top_p
                    sorted_indices_to_remove[0] = False
                    next_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
                
                # 확률 계산
                probs = torch.softmax(next_logits, dim=-1)
                
                # 토큰 샘플링
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token.item())
                
                # 입력에 추가
                token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)
                
                # 종료 토큰 확인
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # 3. 토큰을 텍스트로 변환
        all_tokens = token_ids[0].tolist()
        generated_text = self.tokenizer.decode(all_tokens)
        
        return generated_text

# 사용 예시
print("=== 텍스트 생성 예시 ===\\n")

# 모델과 토크나이저 로드 (실제 코드)
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 생성기 초기화
# generator = GPTTextGenerator(model, tokenizer)

# 다양한 설정으로 생성
# prompts = [
#     "The future of AI is",
#     "Once upon a time",
#     "In the beginning"
# ]

# for prompt in prompts:
#     print(f"프롬프트: {prompt}")
#     
#     # 결정적 생성 (Greedy)
#     text1 = generator.generate(prompt, max_new_tokens=50, temperature=0.1)
#     print(f"Greedy: {text1}\\n")
#     
#     # 창의적 생성 (Temperature + Top-p)
#     text2 = generator.generate(prompt, max_new_tokens=50, temperature=0.8, top_p=0.9)
#     print(f"Creative: {text2}\\n")`,
    explanation:
      "완전한 텍스트 생성 함수는 프롬프트 토큰화, 반복적 토큰 생성, 다양한 샘플링 전략을 포함합니다.",
    keyPoints: [
      "프롬프트를 토큰 ID로 변환",
      "최대 토큰 수만큼 반복 생성",
      "Temperature로 확률 분포 조절",
      "Top-k/Top-p로 토큰 필터링",
      "Multinomial sampling으로 토큰 선택",
      "종료 토큰에서 생성 중단",
      "토큰을 텍스트로 디코딩",
    ],
  },
];

export default function TextGeneration() {
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">✨</span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent">
              Text Generation & Dummy Model
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            텍스트 생성의 원리와 다양한 샘플링 전략을 학습하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Info Box */}
        <div className="bg-indigo-50 border-l-4 border-indigo-500 p-6 rounded-lg mb-8">
          <h2 className="font-semibold text-indigo-900 mb-2">
            💡 Text Generation이란?
          </h2>
          <p className="text-indigo-800 text-sm">
            GPT는 초기 프롬프트부터 시작하여 한 번에 한 토큰씩 텍스트를 생성합니다.
            다양한 샘플링 전략을 사용하여 생성 텍스트의 품질과 다양성을 조절할 수 있습니다.
          </p>
        </div>

        {/* Generation Sections */}
        <div className="space-y-4">
          {generationSections.map((section, idx) => (
            <Card
              key={idx}
              className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                expandedIndex === idx
                  ? "border-indigo-300 shadow-lg bg-white"
                  : "border-slate-200 bg-slate-50"
              }`}
              onClick={() =>
                setExpandedIndex(expandedIndex === idx ? null : idx)
              }
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-indigo-500 to-blue-600 p-6 text-white cursor-pointer">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold">{section.title}</h3>
                  <ChevronDown
                    className={`w-6 h-6 transition-transform duration-300 flex-shrink-0 ${
                      expandedIndex === idx ? "rotate-180" : ""
                    }`}
                  />
                </div>
                <p className="text-indigo-100 text-sm mt-1">
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
                          className="bg-indigo-50 border-l-4 border-indigo-400 p-3 rounded text-sm text-slate-700"
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
        <section className="mt-12 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-xl p-8 border border-indigo-200">
          <h2 className="text-2xl font-bold text-indigo-900 mb-6">
            📊 샘플링 전략 비교
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-indigo-300">
                  <th className="text-left py-3 px-4 font-semibold text-indigo-900">
                    전략
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-indigo-900">
                    속도
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-indigo-900">
                    다양성
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-indigo-900">
                    품질
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-indigo-900">
                    사용 시기
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-indigo-100 hover:bg-indigo-100/50">
                  <td className="py-3 px-4 font-mono">Greedy</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">빠른 추론</td>
                </tr>
                <tr className="border-b border-indigo-100 hover:bg-indigo-100/50">
                  <td className="py-3 px-4 font-mono">Temperature</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">기본 설정</td>
                </tr>
                <tr className="border-b border-indigo-100 hover:bg-indigo-100/50">
                  <td className="py-3 px-4 font-mono">Top-k</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">품질 중시</td>
                </tr>
                <tr className="border-b border-indigo-100 hover:bg-indigo-100/50">
                  <td className="py-3 px-4 font-mono">Top-p</td>
                  <td className="py-3 px-4">⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">권장</td>
                </tr>
                <tr className="hover:bg-indigo-100/50">
                  <td className="py-3 px-4 font-mono">Beam Search</td>
                  <td className="py-3 px-4">⭐</td>
                  <td className="py-3 px-4">⭐</td>
                  <td className="py-3 px-4">⭐⭐⭐⭐⭐</td>
                  <td className="py-3 px-4">최고 품질</td>
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
            onClick={() => setLocation("/gpt2-config")}
            className="gap-2 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700"
          >
            GPT-2 Configuration 보기
            <ArrowRight className="w-4 h-4" />
          </Button>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | Text Generation & Dummy Model
          </p>
        </div>
      </footer>
    </div>
  );
}
