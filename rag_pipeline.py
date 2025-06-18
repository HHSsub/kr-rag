"""
Korean Grammar RAG System - Complete Pipeline Implementation
전체 RAG 파이프라인을 통합한 메인 시스템
"""

import json
import warnings
from typing import List, Dict, Any
try:
    from tqdm import tqdm
except ImportError:
    # tqdm이 없으면 기본 range 사용
    def tqdm(iterable, *args, **kwargs):
        return iterable

from models import (
    QueryRewriter, KoreanEmbedder, RankRAGModel, 
    GuidedRankSelector, FinalAnswerGenerator
)
from utils import (
    KoreanTextProcessor, HybridRetriever, MultiStageReranker,
    EvaluationMetrics, DataLoader, MemoryManager
)

warnings.filterwarnings('ignore')

class KoreanGrammarRAGSystem:
    """
    Complete Korean Grammar RAG System
    한국어 어문 규범 RAG 시스템 - 전체 파이프라인 통합
    """

    def __init__(self, enable_llm=True):
        """
        시스템 초기화

        Args:
            enable_llm (bool): 실제 LLM 사용 여부 (False시 템플릿 모드)
        """
        self.enable_llm = enable_llm

        # LLM 모델들 (지연 로딩)
        self.query_rewriter = QueryRewriter() if enable_llm else None
        self.embedder = KoreanEmbedder() if enable_llm else None
        self.rankrag_model = RankRAGModel() if enable_llm else None
        self.guided_selector = GuidedRankSelector() if enable_llm else None
        self.final_generator = FinalAnswerGenerator() if enable_llm else None

        # 검색 및 재랭킹 시스템
        self.hybrid_retriever = None
        self.reranker = MultiStageReranker()

        # 지식 베이스
        self.knowledge_chunks = []

        print(f"🚀 Korean Grammar RAG System initialized (LLM: {enable_llm})")

    def load_knowledge_base(self, train_data_path: str):
        """지식 베이스 구축"""
        print("📚 Loading knowledge base...")

        # 훈련 데이터 로드
        train_data = DataLoader.load_json_dataset(train_data_path)

        # 지식 청크 생성
        self.knowledge_chunks = DataLoader.create_knowledge_chunks_from_data(train_data)

        # 하이브리드 검색기 초기화
        embedder = self.embedder if self.enable_llm else None
        self.hybrid_retriever = HybridRetriever(self.knowledge_chunks, embedder)

        print(f"✅ Knowledge base loaded: {len(self.knowledge_chunks)} chunks")

        if self.enable_llm and self.embedder:
            print("🔄 Building dense embeddings...")
            # 임베딩 미리 로드
            self.embedder.load_model()

    def enhance_query(self, question: str) -> List[str]:
        """쿼리 향상 및 확장"""
        enhanced_queries = []

        # 1. 기본 쿼리 정규화
        normalized_query = KoreanTextProcessor.normalize_korean_text(question)
        enhanced_queries.append(normalized_query)

        # 2. 선택지 확장
        option_expanded = KoreanTextProcessor.expand_query_with_options(question)
        enhanced_queries.extend(option_expanded)

        # 3. LLM 기반 쿼리 재작성 (HyDE)
        if self.enable_llm and self.query_rewriter:
            try:
                llm_expanded = self.query_rewriter.rewrite_query(question)
                if llm_expanded and llm_expanded != question:
                    enhanced_queries.append(llm_expanded)
            except Exception as e:
                print(f"⚠️ Query rewriting failed: {e}")

        # 중복 제거
        unique_queries = list(dict.fromkeys(enhanced_queries))

        return unique_queries

    def retrieve_contexts(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        """하이브리드 검색으로 컨텍스트 검색"""
        if not self.hybrid_retriever:
            return []

        all_contexts = []

        # 각 확장된 쿼리로 검색
        for query in queries:
            contexts = self.hybrid_retriever.hybrid_search(query, top_k=top_k//len(queries) + 2)
            all_contexts.extend(contexts)

        # 중복 제거 (ID 기준)
        seen_ids = set()
        unique_contexts = []
        for ctx in all_contexts:
            if ctx['id'] not in seen_ids:
                unique_contexts.append(ctx)
                seen_ids.add(ctx['id'])

        # 검색 점수로 정렬
        unique_contexts.sort(key=lambda x: x.get('retrieval_score', 0), reverse=True)

        return unique_contexts[:top_k]

    def rerank_contexts(self, question: str, question_type: str, contexts: List[Dict]) -> List[Dict]:
        """다단계 재랭킹"""
        if not contexts:
            return []

        reranked = self.reranker.rerank_contexts(question, question_type, contexts)
        return reranked

    def rank_contexts_with_llm(self, question: str, contexts: List[Dict]) -> str:
        """LLM 기반 컨텍스트 랭킹 설명"""
        if not self.enable_llm or not self.guided_selector or not contexts:
            return "컨텍스트 분석을 위한 LLM이 로드되지 않았습니다."

        try:
            explanation = self.guided_selector.explain_context_ranking(question, contexts[:3])
            return explanation
        except Exception as e:
            print(f"⚠️ LLM guided ranking failed: {e}")
            return f"컨텍스트 랭킹 중 오류 발생: {str(e)}"

    def generate_answer_with_rankrag(self, question: str, question_type: str, contexts: List[Dict]) -> str:
        """RankRAG 모델로 답변 생성"""
        if not self.enable_llm or not self.rankrag_model or not contexts:
            return self._generate_template_answer(question, question_type, contexts)

        try:
            answer = self.rankrag_model.rank_and_generate(question, contexts, question_type)
            return answer
        except Exception as e:
            print(f"⚠️ RankRAG generation failed: {e}")
            return self._generate_template_answer(question, question_type, contexts)

    def generate_final_answer(self, question: str, question_type: str, 
                            selected_contexts: List[Dict], context_explanation: str) -> str:
        """최종 답변 생성"""
        if not self.enable_llm or not self.final_generator or not selected_contexts:
            return self._generate_template_answer(question, question_type, selected_contexts)

        try:
            answer = self.final_generator.generate_final_answer(
                question, question_type, selected_contexts, context_explanation
            )
            return answer
        except Exception as e:
            print(f"⚠️ Final answer generation failed: {e}")
            return self._generate_template_answer(question, question_type, selected_contexts)

    def _generate_template_answer(self, question: str, question_type: str, contexts: List[Dict]) -> str:
        """템플릿 기반 답변 생성 (LLM 없이)"""
        if not contexts:
            return f"질문에 대한 관련 정보를 찾을 수 없습니다. 질문: {question}"

        # 최고 점수 컨텍스트 사용
        best_context = contexts[0]

        # 간단한 템플릿 기반 답변
        if question_type == "선택형":
            # 선택지 추출 시도
            options = KoreanTextProcessor.extract_options_from_question(question)
            if options and len(options) >= 2:
                # 첫 번째 옵션을 정답으로 가정 (실제로는 더 복잡한 로직 필요)
                answer = f'"{options[0]}"이 옳다. {best_context["text"][:200]}...'
            else:
                answer = f"주어진 선택지 중 올바른 표현을 선택해야 합니다. {best_context['text'][:200]}..."
        else:  # 교정형
            answer = f"어문 규범에 맞게 교정이 필요합니다. {best_context['text'][:200]}..."

        return answer

    def process_question(self, question: str, question_type: str) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        results = {
            'question': question,
            'question_type': question_type,
            'enhanced_queries': [],
            'retrieved_contexts': [],
            'reranked_contexts': [],
            'context_explanation': '',
            'rankrag_answer': '',
            'final_answer': '',
            'processing_info': {}
        }

        try:
            # 1. 쿼리 향상
            print("🔄 Step 1: Query Enhancement")
            enhanced_queries = self.enhance_query(question)
            results['enhanced_queries'] = enhanced_queries

            # 2. 하이브리드 검색
            print("🔄 Step 2: Hybrid Retrieval")
            retrieved_contexts = self.retrieve_contexts(enhanced_queries, top_k=10)
            results['retrieved_contexts'] = retrieved_contexts

            # 3. 다단계 재랭킹
            print("🔄 Step 3: Multi-stage Reranking")
            reranked_contexts = self.rerank_contexts(question, question_type, retrieved_contexts)
            results['reranked_contexts'] = reranked_contexts

            # 4. LLM 기반 컨텍스트 랭킹 설명
            print("🔄 Step 4: LLM Guided Ranking")
            context_explanation = self.rank_contexts_with_llm(question, reranked_contexts[:3])
            results['context_explanation'] = context_explanation

            # 5. RankRAG 답변 생성
            print("🔄 Step 5: RankRAG Answer Generation")
            rankrag_answer = self.generate_answer_with_rankrag(
                question, question_type, reranked_contexts[:5]
            )
            results['rankrag_answer'] = rankrag_answer

            # 6. 최종 답변 생성
            print("🔄 Step 6: Final Answer Generation")
            final_answer = self.generate_final_answer(
                question, question_type, reranked_contexts[:3], context_explanation
            )
            results['final_answer'] = final_answer

            # 처리 정보
            results['processing_info'] = {
                'num_enhanced_queries': len(enhanced_queries),
                'num_retrieved_contexts': len(retrieved_contexts),
                'num_reranked_contexts': len(reranked_contexts),
                'top_context_score': reranked_contexts[0]['final_score'] if reranked_contexts else 0,
                'memory_info': MemoryManager.get_gpu_memory_info()
            }

            print("✅ Question processed successfully")

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            results['error'] = str(e)

        return results

    def evaluate_on_dataset(self, test_data_path: str, output_path: str = None, 
                           max_samples: int = None) -> Dict[str, float]:
        """데이터셋에서 평가 수행"""
        print(f"📊 Starting evaluation on dataset: {test_data_path}")

        # 테스트 데이터 로드
        test_data = DataLoader.load_json_dataset(test_data_path)

        if max_samples:
            test_data = test_data[:max_samples]

        results = []
        correct_predictions = 0
        total_predictions = len(test_data)

        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            question = sample['input']['question']
            question_type = sample['input']['question_type']
            ground_truth = sample.get('output', {}).get('answer', '')

            # 질문 처리
            result = self.process_question(question, question_type)

            # 최종 답변 선택 (RankRAG 또는 Final Answer)
            predicted_answer = result['final_answer'] or result['rankrag_answer']

            # 평가 (ground truth가 있는 경우)
            is_correct = False
            if ground_truth:
                is_correct = EvaluationMetrics.exact_match(predicted_answer, ground_truth)
                if is_correct:
                    correct_predictions += 1

            # 결과 저장
            sample_result = {
                'id': sample.get('id', i),
                'input': sample['input'],
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'processing_details': result
            }

            results.append(sample_result)

            # 메모리 정리 (주기적으로)
            if i % 10 == 0:
                MemoryManager.clear_gpu_memory()

        # 평가 지표 계산
        accuracy = correct_predictions / total_predictions if ground_truth else 0.0

        evaluation_metrics = {
            'accuracy': accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'average_contexts_used': sum(
                r['processing_details']['processing_info']['num_reranked_contexts'] 
                for r in results
            ) / len(results)
        }

        print(f"📊 Evaluation Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Total Samples: {total_predictions}")
        print(f"   Correct Predictions: {correct_predictions}")

        # 결과 저장
        if output_path:
            final_output = {
                'evaluation_metrics': evaluation_metrics,
                'predictions': results
            }
            DataLoader.save_results(final_output, output_path)
            print(f"💾 Results saved to: {output_path}")

        return evaluation_metrics

    def cleanup(self):
        """시스템 정리"""
        print("🧹 Cleaning up system resources...")
        MemoryManager.clear_gpu_memory()

        # 모델들 메모리에서 제거
        if hasattr(self, 'query_rewriter') and self.query_rewriter:
            self.query_rewriter.model = None
            self.query_rewriter.tokenizer = None

        if hasattr(self, 'rankrag_model') and self.rankrag_model:
            self.rankrag_model.model = None
            self.rankrag_model.tokenizer = None

        if hasattr(self, 'guided_selector') and self.guided_selector:
            self.guided_selector.model = None
            self.guided_selector.tokenizer = None

        if hasattr(self, 'final_generator') and self.final_generator:
            self.final_generator.model = None
            self.final_generator.tokenizer = None

        print("✅ Cleanup completed")

# 편의 함수들
def create_rag_system(enable_llm: bool = True) -> KoreanGrammarRAGSystem:
    """RAG 시스템 생성"""
    return KoreanGrammarRAGSystem(enable_llm=enable_llm)

def quick_test(system: KoreanGrammarRAGSystem, question: str, question_type: str = "선택형"):
    """빠른 테스트"""
    print(f"🧪 Testing question: {question}")
    result = system.process_question(question, question_type)
    print(f"📝 Answer: {result['final_answer'] or result['rankrag_answer']}")
    return result

# 메인 실행 함수
if __name__ == "__main__":
    # 시스템 생성 및 테스트
    rag_system = create_rag_system(enable_llm=True)

    # 지식 베이스 로드
    rag_system.load_knowledge_base('.//korean_language_rag_V1.0_train.json')

    # 테스트 질문
    test_question = "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다. 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."

    # 테스트 실행
    result = quick_test(rag_system, test_question, "선택형")

    # 정리
    rag_system.cleanup()
