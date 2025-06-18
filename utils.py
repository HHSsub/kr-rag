"""
Korean Grammar RAG System - Utility Functions
한국어 어문 규범 RAG 시스템을 위한 유틸리티 함수들
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class KoreanTextProcessor:
    """한국어 텍스트 전처리"""

    @staticmethod
    def extract_options_from_question(question: str) -> List[str]:
        """질문에서 선택지 추출 (예: {옵션1/옵션2})"""
        pattern = r'{([^}]+)}'
        matches = re.findall(pattern, question)

        options = []
        for match in matches:
            if '/' in match:
                options.extend([opt.strip() for opt in match.split('/')])

        return options

    @staticmethod
    def expand_query_with_options(question: str) -> List[str]:
        """질문을 선택지로 확장"""
        options = KoreanTextProcessor.extract_options_from_question(question)
        expanded_queries = [question]

        # 각 선택지로 질문 확장
        for option in options:
            # 중괄호 부분을 각 선택지로 대체
            pattern = r'{[^}]+}'
            expanded_query = re.sub(pattern, option, question)
            if expanded_query not in expanded_queries:
                expanded_queries.append(expanded_query)

        return expanded_queries

    @staticmethod
    def extract_grammar_keywords(text: str) -> List[str]:
        """문법 관련 키워드 추출"""
        grammar_keywords = [
            '맞춤법', '띄어쓰기', '표준어', '문장부호', '외래어표기',
            '어간', '어미', '받침', '활용', '조사', '의존명사',
            '양성모음', '음성모음', '두음법칙', '사이시옷',
            '마침표', '쉼표', '물음표', '느낌표', '괄호', '따옴표'
        ]

        found_keywords = []
        for keyword in grammar_keywords:
            if keyword in text:
                found_keywords.append(keyword)

        return found_keywords

    @staticmethod
    def normalize_korean_text(text: str) -> str:
        """한국어 텍스트 정규화"""
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())

        # 특수 문자 정리 (필요한 것만 유지)
        text = re.sub(r'[^\w\s가-힣{}/.,;:!?""''()\[\]-]', '', text)

        return text

class HybridRetriever:
    """하이브리드 검색 (Dense + Sparse)"""

    def __init__(self, knowledge_chunks: List[Dict], embedder=None):
        self.knowledge_chunks = knowledge_chunks
        self.embedder = embedder
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_embeddings = None

        self._build_indices()

    def _build_indices(self):
        """검색 인덱스 구축"""
        try:
            # TF-IDF 인덱스 구축 (Sparse)
            texts = [chunk['text'] for chunk in self.knowledge_chunks]
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            # Dense 임베딩 인덱스 구축 (임베더가 있는 경우)
            if self.embedder:
                print("🔄 Building dense embeddings for knowledge chunks...")
                self.chunk_embeddings = self.embedder.encode(texts)
                print("✅ Dense embeddings built successfully")
        except Exception as e:
            print(f"❌ 인덱스 구축 실패: {e}")

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """TF-IDF 기반 sparse 검색"""
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # 상위 k개 인덱스와 점수 반환
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]

            return results
        except Exception as e:
            print(f"❌ Sparse 검색 실패: {e}")
            return []

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Dense 임베딩 기반 검색"""
        if not self.embedder or self.chunk_embeddings is None:
            return []

        try:
            query_embedding = self.embedder.encode([query])

            # 코사인 유사도 계산
            if isinstance(self.chunk_embeddings, torch.Tensor):
                similarities = torch.cosine_similarity(
                    query_embedding, self.chunk_embeddings, dim=1
                ).cpu().numpy()
            else:
                similarities = cosine_similarity(query_embedding, self.chunk_embeddings).flatten()

            # 상위 k개 인덱스와 점수 반환
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]

            return results
        except Exception as e:
            print(f"❌ Dense 검색 실패: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10, 
                     sparse_weight: float = 0.3, dense_weight: float = 0.7) -> List[Dict]:
        """하이브리드 검색 (Sparse + Dense 결합)"""
        # Sparse 검색
        sparse_results = self.sparse_search(query, top_k * 2)

        # Dense 검색
        dense_results = self.dense_search(query, top_k * 2)

        # 점수 정규화 및 결합
        combined_scores = {}

        # Sparse 점수 추가
        for idx, score in sparse_results:
            combined_scores[idx] = sparse_weight * score

        # Dense 점수 추가
        for idx, score in dense_results:
            if idx in combined_scores:
                combined_scores[idx] += dense_weight * score
            else:
                combined_scores[idx] = dense_weight * score

        # 최종 랭킹
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 k개 결과 반환
        final_results = []
        for idx, score in sorted_results[:top_k]:
            chunk = self.knowledge_chunks[idx].copy()
            chunk['retrieval_score'] = score
            final_results.append(chunk)

        return final_results

class MultiStageReranker:
    """다단계 재랭킹 시스템"""

    def __init__(self):
        self.category_weights = {
            '맞춤법': 1.2,
            '띄어쓰기': 1.1,
            '표준어': 1.0,
            '문장부호': 0.9,
            '외래어표기': 0.8,
            '문법': 1.0
        }

    def calculate_category_match_score(self, question: str, context: Dict) -> float:
        """카테고리 매칭 점수 계산"""
        try:
            question_keywords = KoreanTextProcessor.extract_grammar_keywords(question)
            context_category = context.get('category', '')
            context_keywords = KoreanTextProcessor.extract_grammar_keywords(context['text'])

            # 카테고리 직접 매칭
            category_score = 0.0
            if context_category in question or any(kw in context_category for kw in question_keywords):
                category_score = self.category_weights.get(context_category, 1.0)

            # 키워드 매칭 점수
            keyword_score = len(set(question_keywords) & set(context_keywords)) * 0.1

            return category_score + keyword_score
        except Exception as e:
            print(f"❌ 카테고리 매칭 점수 계산 실패: {e}")
            return 0.0

    def calculate_question_type_score(self, question_type: str, context: Dict) -> float:
        """질문 유형 매칭 점수"""
        try:
            if question_type == '선택형':
                # 선택형 질문에 대한 가중치
                if any(word in context['text'] for word in ['선택', '옳다', '바르다', '올바른']):
                    return 0.2
            elif question_type == '교정형':
                # 교정형 질문에 대한 가중치
                if any(word in context['text'] for word in ['교정', '고치', '바꾸', '수정']):
                    return 0.2

            return 0.0
        except Exception as e:
            print(f"❌ 질문 유형 점수 계산 실패: {e}")
            return 0.0

    def calculate_keyword_frequency_score(self, question: str, context: Dict) -> float:
        """키워드 빈도 점수"""
        try:
            question_words = set(question.split())
            context_words = context['text'].split()

            common_words = question_words & set(context_words)
            if not question_words:
                return 0.0

            frequency_score = len(common_words) / len(question_words)
            return frequency_score * 0.3
        except Exception as e:
            print(f"❌ 키워드 빈도 점수 계산 실패: {e}")
            return 0.0

    def rerank_contexts(self, question: str, question_type: str, contexts: List[Dict]) -> List[Dict]:
        """다단계 재랭킹 수행"""
        reranked_contexts = []

        for context in contexts:
            # 기본 검색 점수
            base_score = context.get('retrieval_score', 0.0)

            # 추가 점수 계산
            category_score = self.calculate_category_match_score(question, context)
            type_score = self.calculate_question_type_score(question_type, context)
            keyword_score = self.calculate_keyword_frequency_score(question, context)

            # 최종 점수 계산
            final_score = base_score + category_score + type_score + keyword_score

            context_copy = context.copy()
            context_copy['final_score'] = final_score
            context_copy['category_score'] = category_score
            context_copy['type_score'] = type_score
            context_copy['keyword_score'] = keyword_score

            reranked_contexts.append(context_copy)

        # 최종 점수로 정렬
        reranked_contexts.sort(key=lambda x: x['final_score'], reverse=True)

        return reranked_contexts

class EvaluationMetrics:
    """평가 지표 계산"""

    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """완전 일치 평가"""
        try:
            # 정답 부분만 추출 (첫 번째 문장)
            pred_answer = predicted.split('.')[0].strip() if '.' in predicted else predicted.strip()
            gt_answer = ground_truth.split('.')[0].strip() if '.' in ground_truth else ground_truth.strip()

            return pred_answer == gt_answer
        except Exception as e:
            print(f"❌ 완전 일치 평가 실패: {e}")
            return False

    @staticmethod
    def extract_correct_answer(text: str) -> str:
        """정답 부분 추출"""
        try:
            # "...이/가 옳다" 패턴으로 정답 추출
            patterns = [
                r'"([^"]+)"[이가] 옳다',
                r"'([^']+)'[이가] 옳다",
                r'([^.]+)[이가] 옳다'
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()

            return text.split('.')[0].strip()
        except Exception as e:
            print(f"❌ 정답 추출 실패: {e}")
            return text.split('.')[0].strip() if '.' in text else text.strip()

class DataLoader:
    """데이터 로딩 유틸리티"""

    @staticmethod
    def load_json_dataset(file_path: str) -> List[Dict]:
        """JSON 데이터셋 로딩"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ JSON 데이터셋 로딩 실패: {e}")
            return []

    @staticmethod
    def save_results(results: List[Dict], file_path: str):
        """결과 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 결과 저장 완료: {file_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

    @staticmethod
    def create_knowledge_chunks_from_data(train_data: List[Dict]) -> List[Dict]:
        """훈련 데이터에서 지식 청크 생성"""
        knowledge_chunks = []

        try:
            for i, item in enumerate(train_data):
                question = item['input']['question']
                answer = item['output']['answer']
                question_type = item['input']['question_type']

                # 답변에서 규범 지식 추출
                knowledge_text = f"{question} {answer}"

                # 카테고리 추출
                category = "기타"
                if any(word in knowledge_text for word in ['맞춤법', '철자', '어간', '어미']):
                    category = "맞춤법"
                elif any(word in knowledge_text for word in ['띄어쓰기', '띄어', '붙여']):
                    category = "띄어쓰기"
                elif any(word in knowledge_text for word in ['표준어', '표준', '사정']):
                    category = "표준어"
                elif any(word in knowledge_text for word in ['문장부호', '마침표', '쉼표']):
                    category = "문장부호"
                elif any(word in knowledge_text for word in ['외래어', '표기법']):
                    category = "외래어표기"

                chunk = {
                    'id': f"chunk_{i}",
                    'text': knowledge_text,
                    'category': category,
                    'question_type': question_type,
                    'source': 'training_data'
                }

                knowledge_chunks.append(chunk)

            print(f"✅ 지식 청크 생성 완료: {len(knowledge_chunks)}개")
            return knowledge_chunks
        except Exception as e:
            print(f"❌ 지식 청크 생성 실패: {e}")
            return []

class MemoryManager:
    """메모리 관리 유틸리티"""

    @staticmethod
    def clear_gpu_memory():
        """GPU 메모리 정리"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ GPU 메모리 정리 완료")
        except Exception as e:
            print(f"❌ GPU 메모리 정리 실패: {e}")

    @staticmethod
    def get_gpu_memory_info():
        """GPU 메모리 정보 반환"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            return "CUDA not available"
        except Exception as e:
            return f"GPU 메모리 정보 확인 실패: {e}"

    @staticmethod
    def optimize_model_loading():
        """모델 로딩 최적화"""
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("✅ 모델 로딩 최적화 완료")
        except Exception as e:
            print(f"❌ 모델 로딩 최적화 실패: {e}")

    @staticmethod
    def check_memory():
        """시스템 메모리 확인"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"시스템 메모리: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB 사용 중")
            print(MemoryManager.get_gpu_memory_info())
        except ImportError:
            print("psutil 패키지가 없어 시스템 메모리 확인 불가")
        except Exception as e:
            print(f"메모리 확인 실패: {e}")
