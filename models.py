"""
Korean Grammar RAG System - LLM Model Wrappers (A100 최적화 버전)
태스크별 최적화된 Hugging Face 모델들을 관리하는 클래스들
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# A100 최적화를 위한 imports
try:
    from utils import MemoryManager
except ImportError:
    class MemoryManager:
        @staticmethod
        def check_memory_status():
            return {'usage_ratio': 0.5}
        @staticmethod
        def clear_gpu_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# bitsandbytes 완전 비활성화
BITSANDBYTES_AVAILABLE = False

class ModelConfig:
    """A100 최적화 모델 설정 관리"""
    
    def __init__(self):
        # A100 최적화 설정
        self.device_map = "auto"
        self.torch_dtype = torch.float16  # bfloat16 대신 float16 사용
        self.low_cpu_mem_usage = True
        self.load_in_8bit = True  # 8비트 양자화 사용
        self.max_memory = {0: "35GB"}  # A100의 80% 사용
        self.trust_remote_code = True
    
    # PyTorch 데이터 타입
    TORCH_DTYPE = torch.float16
    
    # 양자화 설정 - 완전히 비활성화
    QUANTIZATION_CONFIG = None
    
    # 생성 파라미터
    GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": None,  # 모델별로 설정
        "eos_token_id": None,  # 모델별로 설정
    }

class QueryRewriter:
    """쿼리 재작성 및 HyDE 구현"""

    def __init__(self):
        self.model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.config = ModelConfig()

    def load_model(self):
        """A100 최적화 모델 로딩"""
        if self.is_loaded:
            return

        print(f"🔄 Loading Query Rewriter: {self.model_name}")

        try:
            # 메모리 체크
            MemoryManager.check_memory_status()
            
            # 이전 모델 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            MemoryManager.clear_gpu_memory()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                load_in_8bit=self.config.load_in_8bit,
                max_memory=self.config.max_memory,
                trust_remote_code=self.config.trust_remote_code
            )

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print(f"✅ {self.model_name} 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.is_loaded = False
            MemoryManager.clear_gpu_memory()

    def rewrite_query(self, question):
        """쿼리 재작성 및 확장"""
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded:
            return question  # 로딩 실패 시 원본 반환

        prompt = f"""다음 한국어 어문 규범 질문을 다양한 표현으로 확장해 주세요.
가능한 표현을 중괄호 {{선택1/선택2}} 형식으로 묶어 출력하세요.

질문: {question}

확장된 질문:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 원래 프롬프트 제거하고 답변만 추출
            expanded_query = response.split("확장된 질문:")[-1].strip()

            return expanded_query if expanded_query else question
            
        except Exception as e:
            print(f"❌ Query rewriting 실패: {e}")
            return question

class KoreanEmbedder:
    """한국어 문장 임베딩"""

    def __init__(self):
        self.model_name = "jhgan/ko-sbert-sts"
        self.model = None
        self.is_loaded = False
        self.config = ModelConfig()

    def load_model(self):
        """A100 최적화 모델 로딩"""
        if self.is_loaded:
            return

        print(f"🔄 Loading Korean Embedder: {self.model_name}")
        try:
            # 메모리 체크
            MemoryManager.check_memory_status()
            
            # 이전 모델 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            MemoryManager.clear_gpu_memory()
            
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            print(f"✅ {self.model_name} 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.is_loaded = False
            MemoryManager.clear_gpu_memory()

    def encode(self, texts):
        """텍스트 임베딩"""
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded:
            # 임베딩 실패 시 랜덤 벡터 반환
            import numpy as np
            if isinstance(texts, str):
                return torch.tensor(np.random.random((1, 768)))
            return torch.tensor(np.random.random((len(texts), 768)))

        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings
        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            import numpy as np
            return torch.tensor(np.random.random((len(texts), 768)))

class RankRAGModel:
    """RankRAG: 컨텍스트 랭킹 + 답변 생성 통합"""

    def __init__(self):
        self.model_name = "dnotitia/Llama-DNA-1.0-8B-Instruct"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.config = ModelConfig()

    def load_model(self):
        """A100 최적화 모델 로딩"""
        if self.is_loaded:
            return

        print(f"🔄 Loading RankRAG Model: {self.model_name}")

        try:
            # 메모리 체크
            MemoryManager.check_memory_status()
            
            # 이전 모델 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            MemoryManager.clear_gpu_memory()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                load_in_8bit=self.config.load_in_8bit,
                max_memory=self.config.max_memory,
                trust_remote_code=self.config.trust_remote_code
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print(f"✅ {self.model_name} 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.is_loaded = False
            MemoryManager.clear_gpu_memory()

    def rank_and_generate(self, question, contexts, question_type="선택형"):
        """컨텍스트 랭킹 + 답변 생성"""
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded:
            # 모델 로딩 실패 시 기본 답변
            return f'"{question}에 대한 답변"이 옳다. 모델 로딩 실패로 인한 기본 답변입니다.'

        # 컨텍스트 포맷팅
        context_text = ""
        for i, ctx in enumerate(contexts[:5], 1):  # 최대 5개 컨텍스트
            context_text += f"{i}. {ctx.get('text', str(ctx))[:500]}...\n\n"

        prompt = f"""한국어 어문 규범 질문에 대해 주어진 컨텍스트를 분석하고 정확한 답변을 생성하세요.

질문 유형: {question_type}
질문: {question}

참조 컨텍스트:
{context_text}

각 컨텍스트의 관련도를 평가하고, 가장 중요한 컨텍스트를 활용해 다음 형식으로 답변하세요:

답변 형식: "{{정답}}이/가 옳다. {{상세한 이유와 설명}}"

답변:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,  # 정확성을 위해 낮은 temperature
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("답변:")[-1].strip()

            return answer if answer else "답변 생성에 실패했습니다."
            
        except Exception as e:
            print(f"❌ RankRAG 답변 생성 실패: {e}")
            return f'"{question}에 대한 답변"이 옳다. 답변 생성 중 오류가 발생했습니다.'

class GuidedRankSelector:
    """LLM Guided Rank Selection - 컨텍스트 중요도 설명"""

    def __init__(self):
        self.model_name = "KRAFTON/KORani-v3-13B"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.config = ModelConfig()

    def load_model(self):
        """A100 최적화 모델 로딩"""
        if self.is_loaded:
            return

        print(f"🔄 Loading Guided Rank Selector: {self.model_name}")

        try:
            # 메모리 체크
            MemoryManager.check_memory_status()
            
            # 이전 모델 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            MemoryManager.clear_gpu_memory()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                load_in_8bit=self.config.load_in_8bit,
                max_memory=self.config.max_memory,
                trust_remote_code=self.config.trust_remote_code
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print(f"✅ {self.model_name} 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.is_loaded = False
            MemoryManager.clear_gpu_memory()

    def explain_context_ranking(self, question, contexts):
        """컨텍스트 중요도 설명 생성"""
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded:
            return "모델 로딩 실패로 인해 컨텍스트 중요도 분석을 수행할 수 없습니다."

        context_list = ""
        for i, ctx in enumerate(contexts[:3], 1):  # 최대 3개 분석
            context_list += f"{i}. {ctx.get('text', str(ctx))[:300]}...\n\n"

        prompt = f"""다음 한국어 어문 규범 질문에 대해 주어진 컨텍스트들의 중요도를 평가하고 설명해주세요.

질문: {question}

컨텍스트 목록:
{context_list}

각 컨텍스트의 중요도를 평가하고 그 이유를 설명하세요:

평가 기준:
- 질문과의 직접적 관련성
- 어문 규범 지식의 정확성
- 답변 생성에 필요한 정보 포함도

출력 형식:
컨텍스트 1 - 중요도: [높음/중간/낮음], 이유: [구체적 설명]
컨텍스트 2 - 중요도: [높음/중간/낮음], 이유: [구체적 설명]
컨텍스트 3 - 중요도: [높음/중간/낮음], 이유: [구체적 설명]

평가 결과:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1536)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            explanation = response.split("평가 결과:")[-1].strip()

            return explanation if explanation else "컨텍스트 중요도 분석 결과를 생성할 수 없습니다."
            
        except Exception as e:
            print(f"❌ 컨텍스트 중요도 분석 실패: {e}")
            return "컨텍스트 중요도 분석 중 오류가 발생했습니다."

class FinalAnswerGenerator:
    """최종 답변 및 설명 생성"""

    def __init__(self):
        self.model_name = "yanolja/EEVE-Korean-10.8B-v1.0"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.config = ModelConfig()

    def load_model(self):
        """A100 최적화 모델 로딩"""
        if self.is_loaded:
            return

        print(f"🔄 Loading Final Answer Generator: {self.model_name}")

        try:
            # 메모리 체크
            MemoryManager.check_memory_status()
            
            # 이전 모델 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            MemoryManager.clear_gpu_memory()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                load_in_8bit=self.config.load_in_8bit,
                max_memory=self.config.max_memory,
                trust_remote_code=self.config.trust_remote_code
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print(f"✅ {self.model_name} 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.is_loaded = False
            MemoryManager.clear_gpu_memory()

    def generate_final_answer(self, question, question_type, selected_contexts, context_explanation):
        """최종 답변 생성"""
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded:
            return f'"{question}에 대한 답변"이 옳다. 모델 로딩 실패로 인한 기본 답변입니다.'

        contexts_text = ""
        for i, ctx in enumerate(selected_contexts, 1):
            contexts_text += f"- {ctx.get('text', str(ctx))[:200]}...\n"

        prompt = f"""한국어 어문 규범 전문가로서 다음 질문에 정확하고 상세한 답변을 제공해주세요.

질문 유형: {question_type}
질문: {question}

참조한 규범 지식:
{contexts_text}

컨텍스트 분석:
{context_explanation}

다음 형식으로 완전한 답변을 작성하세요:

1. 정답: "{{정확한 정답}}이/가 옳다."
2. 규범 근거: {{해당 어문 규범 조항과 원칙}}
3. 상세 설명: {{문법적 근거와 논리적 설명}}
4. 예시: {{적절한 예시 2-3개}}
5. 주의사항: {{자주 틀리는 표현이나 혼동 사례}}

답변:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.4,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("답변:")[-1].strip()

            return answer if answer else "최종 답변 생성에 실패했습니다."
            
        except Exception as e:
            print(f"❌ 최종 답변 생성 실패: {e}")
            return f'"{question}에 대한 답변"이 옳다. 최종 답변 생성 중 오류가 발생했습니다.'

# 프롬프트 템플릿 유틸리티
class PromptTemplates:
    """프롬프트 템플릿 관리"""

    @staticmethod
    def get_prompt(stage, **kwargs):
        """단계별 프롬프트 생성"""
        if stage == "rewrite":
            return f"""다음 질문을 다양한 표현으로 확장해 주세요:
질문: {kwargs['question']}

확장된 표현:"""

        elif stage == "rankrag":
            contexts = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(kwargs['contexts'])])
            return f"""질문: {kwargs['question']}

{contexts}

가장 중요한 컨텍스트를 선택하여 답변하세요."""

        elif stage == "guided_rank":
            contexts = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(kwargs['contexts'])])
            return f"""질문: {kwargs['question']}

각 컨텍스트의 중요도를 평가하고 설명하세요:
{contexts}"""

        elif stage == "final":
            return f"""질문: {kwargs['question']}

정확한 규범 기반 설명과 예시를 포함한 답변을 생성하세요."""

        return ""
