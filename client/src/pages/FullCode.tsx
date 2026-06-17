import { useState } from "react";
import { ChevronDown, Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface CodeSection {
  title: string;
  code: string;
  explanation: string;
  details?: string[];
}

const codeSections: CodeSection[] = [
  {
    title: "1. LayerNorm 클래스 - 초기화",
    code: `class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))`,
    explanation:
      "LayerNorm 클래스의 초기화 메서드입니다. 정규화에 필요한 파라미터들을 정의합니다.",
    details: [
      "eps (epsilon): 분모가 0이 되는 것을 방지하기 위한 작은 상수로 1e-5로 설정",
      "scale: 정규화된 값에 곱해지는 학습 가능한 파라미터, 초기값은 모두 1",
      "shift: 정규화된 값에 더해지는 학습 가능한 파라미터, 초기값은 모두 0",
      "이 두 파라미터는 모델이 학습하면서 최적의 정규화 방식을 찾을 수 있게 합니다",
    ],
  },
  {
    title: "2. LayerNorm 클래스 - 순전파",
    code: `    def forward(self, x):
        # x의 마지막 차원을 기준으로 평균 계산
        mean = x.mean(dim=-1, keepdim=True)
        
        # x의 마지막 차원을 기준으로 분산 계산
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 정규화: (x - mean) / sqrt(var + eps)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # 학습 가능한 scale과 shift 적용
        return self.scale * norm_x + self.shift`,
    explanation:
      "LayerNorm의 순전파 메서드로, 입력 텐서를 정규화하고 학습 가능한 파라미터를 적용합니다.",
    details: [
      "dim=-1: 마지막 차원(피처 차원)을 기준으로 정규화",
      "keepdim=True: 계산 후 차원을 유지하여 브로드캐스팅이 가능하게 함",
      "unbiased=False: 편향된 분산을 사용 (딥러닝에서 표준)",
      "정규화 공식: (x - mean) / sqrt(var + eps)로 평균 0, 분산 1의 분포로 변환",
      "마지막에 scale * norm_x + shift를 적용하여 모델이 정규화의 정도를 조절할 수 있게 함",
    ],
  },
  {
    title: "3. FeedForward 클래스 - 초기화",
    code: `class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # 첫 번째 선형 변환: emb_dim → 4*emb_dim (차원 확장)
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            
            # GELU 활성화 함수 (비선형성 도입)
            GELU(),
            
            # 두 번째 선형 변환: 4*emb_dim → emb_dim (차원 복원)
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )`,
    explanation:
      "FeedForward 네트워크의 초기화로, 3개의 레이어를 순차적으로 연결합니다.",
    details: [
      "첫 번째 Linear: 입력 차원을 4배로 확장하여 더 많은 표현력 확보",
      "GELU: Gaussian Error Linear Unit 활성화 함수로 비선형성 도입",
      "두 번째 Linear: 확장된 차원을 다시 원래 차원으로 복원",
      "이 구조는 'Attention is All You Need' 논문에서 제안한 표준 구조",
      "4배 확장은 경험적으로 좋은 성능을 보이는 비율로 알려져 있음",
    ],
  },
  {
    title: "4. FeedForward 클래스 - 순전파",
    code: `    def forward(self, x):
        # self.layers에 정의된 모든 레이어를 순차적으로 통과
        return self.layers(x)`,
    explanation:
      "FeedForward의 순전파는 매우 간단합니다. 입력을 nn.Sequential에 정의된 레이어들을 통과시킵니다.",
    details: [
      "nn.Sequential은 입력을 첫 번째 레이어부터 마지막 레이어까지 순차적으로 통과시킴",
      "각 레이어의 출력이 다음 레이어의 입력이 됨",
      "최종 출력은 원래 임베딩 차원과 동일한 크기",
    ],
  },
  {
    title: "5. TransformerBlock 클래스 - 초기화",
    code: `class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 멀티 헤드 어텐션 모듈
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        # 피드포워드 네트워크
        self.ff = FeedForward(cfg)
        
        # 첫 번째 정규화 (어텐션 이전)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        
        # 두 번째 정규화 (피드포워드 이전)
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # 잔차 연결에 적용되는 드롭아웃
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])`,
    explanation:
      "TransformerBlock은 어텐션과 피드포워드 두 개의 서브 블록으로 구성됩니다.",
    details: [
      "att: 멀티 헤드 어텐션으로 시퀀스 내 토큰들 간의 관계를 학습",
      "ff: 피드포워드 네트워크로 각 토큰에 대한 비선형 변환 수행",
      "norm1, norm2: Pre-Normalization 구조로 각 서브 블록 이전에 정규화",
      "drop_shortcut: 잔차 연결의 출력에 드롭아웃을 적용하여 정규화 효과",
    ],
  },
  {
    title: "6. TransformerBlock 클래스 - 순전파 (어텐션 부분)",
    code: `    def forward(self, x):
        # ===== 첫 번째 서브 블록: 어텐션 =====
        
        # 잔차 연결을 위해 원래 입력 저장
        shortcut = x
        
        # 1단계: 정규화
        x = self.norm1(x)
        
        # 2단계: 멀티 헤드 어텐션 적용
        x = self.att(x)
        
        # 3단계: 드롭아웃 적용
        x = self.drop_shortcut(x)
        
        # 4단계: 잔차 연결 (원래 입력과 더하기)
        x = x + shortcut`,
    explanation:
      "첫 번째 서브 블록에서는 정규화 → 어텐션 → 드롭아웃 → 잔차연결 순서로 진행됩니다.",
    details: [
      "shortcut: 원래 입력을 저장하여 나중에 더할 수 있게 함",
      "정규화를 먼저 수행하는 것을 Pre-Normalization이라고 함 (최신 구조)",
      "어텐션 후 드롭아웃으로 일부 뉴런을 무작위로 비활성화하여 과적합 방지",
      "잔차 연결(x + shortcut)은 그래디언트가 더 잘 흐르게 하고 학습을 안정화",
    ],
  },
  {
    title: "7. TransformerBlock 클래스 - 순전파 (피드포워드 부분)",
    code: `        # ===== 두 번째 서브 블록: 피드포워드 =====
        
        # 잔차 연결을 위해 현재 입력 저장
        shortcut = x
        
        # 1단계: 정규화
        x = self.norm2(x)
        
        # 2단계: 피드포워드 네트워크 적용
        x = self.ff(x)
        
        # 3단계: 드롭아웃 적용
        x = self.drop_shortcut(x)
        
        # 4단계: 잔차 연결
        x = x + shortcut
        
        return x`,
    explanation:
      "두 번째 서브 블록도 첫 번째와 동일한 패턴을 따릅니다: 정규화 → 기능 → 드롭아웃 → 잔차연결",
    details: [
      "이번 shortcut은 어텐션 서브 블록의 출력을 저장",
      "norm2를 사용하여 피드포워드 이전에 다시 정규화",
      "피드포워드는 각 토큰에 독립적으로 적용되는 비선형 변환",
      "마지막 잔차 연결로 정보 손실을 최소화하고 학습 안정성 향상",
      "이 패턴의 반복이 트랜스포머의 강력함의 핵심",
    ],
  },
  {
    title: "8. GPTModel 클래스 - 초기화 (임베딩)",
    code: `class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 토큰 임베딩: 토큰 인덱스를 임베딩 벡터로 변환
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # 위치 임베딩: 시퀀스 내 위치를 임베딩 벡터로 변환
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # 임베딩 출력에 적용되는 드롭아웃
        self.drop_emb = nn.Dropout(cfg["drop_rate"])`,
    explanation:
      "GPTModel의 초기화에서 먼저 임베딩 레이어들을 정의합니다.",
    details: [
      "tok_emb: 어휘 크기만큼의 임베딩 벡터를 학습하여 각 토큰을 표현",
      "pos_emb: 시퀀스 내 위치 정보를 임베딩으로 인코딩",
      "토큰 임베딩과 위치 임베딩을 더하여 위치 정보를 포함한 표현 생성",
      "drop_emb: 임베딩 단계에서의 드롭아웃으로 초기 정규화",
    ],
  },
  {
    title: "9. GPTModel 클래스 - 초기화 (트랜스포머 블록)",
    code: `        # 여러 개의 TransformerBlock을 순차적으로 연결
        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # 최종 정규화 (모든 블록 이후)
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # 출력 헤드: 임베딩을 어휘 크기의 로짓으로 변환
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)`,
    explanation:
      "트랜스포머 블록 스택, 최종 정규화, 그리고 출력 헤드를 정의합니다.",
    details: [
      "n_layers개의 TransformerBlock을 생성하여 깊이 있는 모델 구성",
      "리스트 컴프리헨션으로 동일한 구조의 블록을 여러 개 생성",
      "*를 사용하여 리스트를 nn.Sequential의 인자로 언팩",
      "final_norm: 모든 트랜스포머 블록을 통과한 후 최종 정규화",
      "out_head: 최종 임베딩을 다음 토큰 예측을 위한 로짓으로 변환",
      "bias=False: 출력 헤드에서 바이어스를 사용하지 않음 (선택사항)",
    ],
  },
  {
    title: "10. GPTModel 클래스 - 순전파 (임베딩 결합)",
    code: `    def forward(self, in_idx):
        # in_idx의 형태: (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape
        
        # 1단계: 토큰 임베딩
        # 입력: (batch_size, seq_len)
        # 출력: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # 2단계: 위치 임베딩
        # torch.arange(seq_len): [0, 1, 2, ..., seq_len-1]
        # 출력: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # 3단계: 토큰 임베딩과 위치 임베딩을 더하기
        # 브로드캐스팅으로 자동으로 (batch_size, seq_len, emb_dim) 형태로 확장
        x = tok_embeds + pos_embeds`,
    explanation:
      "순전파의 첫 번째 단계로 토큰과 위치 임베딩을 결합합니다.",
    details: [
      "in_idx: 토큰 인덱스 텐서로 (배치 크기, 시퀀스 길이) 형태",
      "tok_emb(in_idx): 각 토큰 인덱스를 해당하는 임베딩 벡터로 변환",
      "torch.arange(seq_len): 0부터 seq_len-1까지의 위치 인덱스 생성",
      "device=in_idx.device: GPU/CPU 일치를 위해 동일한 디바이스에 생성",
      "두 임베딩을 더하여 위치 정보를 포함한 토큰 표현 생성",
    ],
  },
  {
    title: "11. GPTModel 클래스 - 순전파 (드롭아웃 및 블록 통과)",
    code: `        # 4단계: 임베딩 드롭아웃
        # 일부 뉴런을 무작위로 비활성화하여 과적합 방지
        x = self.drop_emb(x)
        
        # 5단계: 모든 TransformerBlock을 순차적으로 통과
        # 각 블록은 어텐션과 피드포워드를 수행
        # 입력과 출력 형태: (batch_size, seq_len, emb_dim)
        x = self.trf_block(x)
        
        # 6단계: 최종 정규화
        # 모든 트랜스포머 블록을 통과한 후 정규화
        x = self.final_norm(x)`,
    explanation:
      "임베딩을 드롭아웃한 후 모든 트랜스포머 블록을 통과시킵니다.",
    details: [
      "drop_emb: 임베딩 단계에서의 드롭아웃으로 초기 정규화",
      "trf_block: n_layers개의 TransformerBlock이 순차적으로 실행",
      "각 블록은 이전 블록의 출력을 입력으로 받음",
      "깊이가 깊어질수록 더 추상적이고 고수준의 특징을 학습",
      "final_norm: 최종 정규화로 출력의 안정성 향상",
    ],
  },
  {
    title: "12. GPTModel 클래스 - 순전파 (출력)",
    code: `        # 7단계: 출력 헤드를 통해 로짓 생성
        # 입력: (batch_size, seq_len, emb_dim)
        # 출력: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)
        
        # 최종 출력: 각 위치에서 다음 토큰에 대한 확률 분포
        return logits`,
    explanation:
      "최종적으로 로짓을 생성하여 반환합니다. 이 로짓은 다음 토큰 예측에 사용됩니다.",
    details: [
      "out_head: 임베딩 벡터를 어휘 크기의 로짓으로 변환",
      "로짓: 소프트맥스 이전의 원시 점수로, 각 토큰의 상대적 확률을 나타냄",
      "출력 형태: (batch_size, seq_len, vocab_size)",
      "학습 시: 로짓에 소프트맥스를 적용하여 확률로 변환 후 크로스 엔트로피 손실 계산",
      "추론 시: 로짓에 소프트맥스를 적용하여 다음 토큰의 확률 분포 획득",
    ],
  },
];

export default function FullCode() {
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <span className="text-white font-bold">{"</>"}</span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              전체 코드 분석
            </h1>
          </div>
          <p className="text-slate-600 ml-13">
            GPT 모델의 전체 코드를 단계별로 살펴보고 각 부분의 역할을 이해하세요
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Info Box */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg mb-8">
          <h2 className="font-semibold text-blue-900 mb-2">💡 학습 팁</h2>
          <p className="text-blue-800 text-sm">
            아래의 각 섹션을 클릭하면 코드와 함께 상세한 설명을 볼 수 있습니다.
            코드 블록의 오른쪽 위에 있는 복사 버튼을 클릭하면 코드를 복사할 수
            있습니다.
          </p>
        </div>

        {/* Code Sections */}
        <div className="space-y-4">
          {codeSections.map((section, idx) => (
            <Card
              key={idx}
              className={`overflow-hidden border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                expandedIndex === idx
                  ? "border-blue-300 shadow-lg bg-white"
                  : "border-slate-200 bg-slate-50"
              }`}
              onClick={() =>
                setExpandedIndex(expandedIndex === idx ? null : idx)
              }
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-6 text-white cursor-pointer">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold">{section.title}</h3>
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
                  {/* Code Block */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-slate-900">📝 코드</h4>
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

                  {/* Details */}
                  {section.details && section.details.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                        <span className="text-lg">🔍</span>
                        상세 설명
                      </h4>
                      <div className="space-y-2">
                        {section.details.map((detail, detailIdx) => (
                          <div
                            key={detailIdx}
                            className="bg-amber-50 border-l-4 border-amber-400 p-3 rounded text-sm text-slate-700"
                          >
                            <p className="leading-relaxed">{detail}</p>
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

        {/* Summary Section */}
        <section className="mt-12 bg-gradient-to-r from-emerald-50 to-blue-50 rounded-xl p-8 border border-emerald-200">
          <h2 className="text-2xl font-bold text-emerald-900 mb-6">
            🎯 전체 흐름 요약
          </h2>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-500 text-white flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <h3 className="font-semibold text-emerald-900">
                  입력 토큰 인덱스
                </h3>
                <p className="text-sm text-emerald-800">
                  모델에 입력되는 토큰의 인덱스 (예: [1, 5, 23, 45])
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <h3 className="font-semibold text-blue-900">
                  임베딩 + 위치 정보
                </h3>
                <p className="text-sm text-blue-800">
                  토큰 임베딩과 위치 임베딩을 결합하여 위치 정보를 포함한 표현 생성
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <h3 className="font-semibold text-purple-900">
                  트랜스포머 블록 통과
                </h3>
                <p className="text-sm text-purple-800">
                  여러 개의 TransformerBlock을 순차적으로 통과하며 특징 학습
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-orange-500 text-white flex items-center justify-center font-bold">
                4
              </div>
              <div>
                <h3 className="font-semibold text-orange-900">
                  로짓 생성 및 출력
                </h3>
                <p className="text-sm text-orange-800">
                  최종 임베딩을 어휘 크기의 로짓으로 변환하여 다음 토큰 예측
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Concepts */}
        <section className="mt-12">
          <h2 className="text-2xl font-bold text-slate-900 mb-6">
            🔑 핵심 개념 정리
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="p-6 border-2 border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-3">
                정규화 (Normalization)
              </h3>
              <p className="text-sm text-slate-700">
                LayerNorm은 각 샘플의 피처 차원을 정규화하여 학습을 안정화합니다.
                평균 0, 분산 1로 변환한 후 학습 가능한 scale과 shift를 적용합니다.
              </p>
            </Card>
            <Card className="p-6 border-2 border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-3">
                잔차 연결 (Residual Connection)
              </h3>
              <p className="text-sm text-slate-700">
                x + shortcut 형태로 원래 입력을 더하여 그래디언트가 더 잘 흐르게
                하고 깊은 네트워크의 학습을 안정화합니다.
              </p>
            </Card>
            <Card className="p-6 border-2 border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-3">
                위치 임베딩 (Positional Embedding)
              </h3>
              <p className="text-sm text-slate-700">
                시퀀스 내 토큰의 위치 정보를 임베딩으로 인코딩하여 모델이 토큰의
                순서를 이해할 수 있게 합니다.
              </p>
            </Card>
            <Card className="p-6 border-2 border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-3">
                어텐션 (Attention)
              </h3>
              <p className="text-sm text-slate-700">
                시퀀스 내 모든 토큰 쌍 사이의 관계를 학습하여 각 토큰이 다른
                토큰들로부터 정보를 얻을 수 있게 합니다.
              </p>
            </Card>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-300 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm">
            GPT 실습 코드 학습 가이드 © 2024 | 전체 코드 분석
          </p>
        </div>
      </footer>
    </div>
  );
}
