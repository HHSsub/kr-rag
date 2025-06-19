"""
Korean Grammar RAG System - Main Execution Script (A100 최적화 버전)
한국어 어문 규범 RAG 시스템 메인 실행 스크립트

사용법:
    python main.py --mode demo                    # 데모 실행
    python main.py --mode evaluate --samples 10  # 평가 실행
    python main.py --mode test --enable_llm      # LLM 활성화 테스트
    python main.py --mode predict --data_path test.json --output_path predictions.json  # 예측 실행
"""

import argparse
import json
import time
import traceback
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# 로컬 모듈 임포트
from rag_pipeline import KoreanGrammarRAGSystem, create_rag_system, quick_test
from utils import DataLoader, MemoryManager, setup_a100_environment

def demo_mode():
    """데모 모드 - 몇 개 샘플 질문으로 시스템 테스트 (메모리 최적화)"""
    print("🎭 Demo Mode - Korean Grammar RAG System")
    print("=" * 60)

    # A100 환경 최적화
    if torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name():
        setup_a100_environment()

    rag_system = None
    try:
        # 시스템 생성 (LLM 비활성화로 빠른 테스트)
        print("🚀 Creating RAG System (Template Mode)...")
        rag_system = create_rag_system(enable_llm=False)

        # 지식 베이스 로드
        train_data_path = './/korean_language_rag_V1.0_train.json'
        if not Path(train_data_path).exists():
            print(f"❌ Training data not found: {train_data_path}")
            return

        rag_system.load_knowledge_base(train_data_path)

        # 데모 질문들
        demo_questions = [
            {
                "question": "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다. 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.",
                "type": "선택형"
            },
            {
                "question": "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"외출시에는 에어컨을 꼭 끕시다.\"",
                "type": "교정형"
            },
            {
                "question": "{검/껌}을 씹다. 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.",
                "type": "선택형"
            }
        ]

        # 각 질문 처리 (메모리 안전)
        for i, demo in enumerate(demo_questions, 1):
            print(f"\n📝 Demo Question {i}: {demo['type']}")
            print(f"Q: {demo['question']}")
            print("-" * 40)

            # 메모리 상태 체크
            MemoryManager.check_memory_status()

            start_time = time.time()
            
            # 메모리 안전 처리
            def process_demo():
                return rag_system.process_question(demo['question'], demo['type'])
            
            processing_time = time.time() - start_time

            try:
                result = rag_system.process_question_optimized(question_data)
            except Exception as e:
                print(f"❌ 질문 처리 실패: {e}")
                result = None
            
            if result:
                # 결과 저장
                sample_result = {
                    'id': item.get('id', f'sample_{i}'),
                    'input': item['input'],
                    'predicted_answer': result.get('predicted_answer') or "처리 실패",
                    'contexts_used': result.get('contexts_used', 0)
                }

            # 강제 메모리 정리
            MemoryManager.clear_gpu_memory(force=True)

        print(f"\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo 실행 중 오류: {e}")
        traceback.print_exc()
    finally:
        if rag_system:
            rag_system.cleanup()
        MemoryManager.clear_gpu_memory(force=True)

def evaluate_mode(max_samples=10, enable_llm=False):
    """평가 모드 - 데이터셋에서 성능 평가 (A100 최적화)"""
    print(f"📊 Evaluation Mode (LLM: {enable_llm}, Samples: {max_samples})")
    print("=" * 60)

    # A100 환경 최적화
    if torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name():
        setup_a100_environment()

    rag_system = None
    try:
        # 시스템 생성
        rag_system = create_rag_system(enable_llm=enable_llm)

        # 데이터 경로 확인
        train_data_path = './/korean_language_rag_V1.0_train.json'
        dev_data_path = './/korean_language_rag_V1.0_dev.json'

        if not Path(train_data_path).exists():
            print(f"❌ Training data not found: {train_data_path}")
            return

        if not Path(dev_data_path).exists():
            print(f"❌ Development data not found: {dev_data_path}")
            return

        # 지식 베이스 로드
        rag_system.load_knowledge_base(train_data_path)

        # 평가 실행 (메모리 최적화 버전)
        output_path = f'.//evaluation_results_llm_{enable_llm}.json'

        start_time = time.time()
        metrics = run_optimized_evaluation(
            rag_system, 
            dev_data_path, 
            output_path, 
            max_samples
        )
        total_time = time.time() - start_time

        # 결과 출력
        print(f"\n📈 Evaluation Results:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Correct: {metrics['correct_predictions']}/{metrics['total_samples']}")
        print(f"   Avg Contexts: {metrics['average_contexts_used']:.2f}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Time per Sample: {total_time/metrics['total_samples']:.2f}s")

    except Exception as e:
        print(f"❌ 평가 실행 중 오류: {e}")
        traceback.print_exc()
    finally:
        if rag_system:
            rag_system.cleanup()
        MemoryManager.clear_gpu_memory(force=True)

def run_optimized_evaluation(rag_system, data_path: str, output_path: str, max_samples: int) -> Dict:
    """A100 최적화 평가 실행"""
    print("🔄 메모리 최적화 평가 시작...")
    
    # 데이터 로드
    data = DataLoader.load_json_dataset(data_path)
    if not data:
        return {"accuracy": 0, "total_samples": 0, "correct_predictions": 0, "average_contexts_used": 0}
    
    # 샘플 수 제한
    if max_samples > 0:
        data = data[:max_samples]
    
    results = []
    correct_count = 0
    total_contexts = 0
    
    print(f"📋 처리할 샘플 수: {len(data)}")
    
    for i, item in enumerate(data):
        print(f"\n🔄 처리 중: {i+1}/{len(data)} - ID: {item.get('id', 'N/A')}")
        
        # 메모리 상태 체크 및 자동 정리
        MemoryManager.auto_cleanup_if_needed()
        
        try:
            # 안전한 질문 처리
            def process_item():
                question_data = item['input']
                return rag_system.process_question_optimized(question_data)
            
            result = MemoryManager.safe_model_operation(process_item)
            
            if result:
                # 결과 저장
                sample_result = {
                    'id': item.get('id', f'sample_{i}'),
                    'input': item['input'],
                    'predicted_answer': result.get('final_answer') or result.get('rankrag_answer') or "처리 실패",
                    'contexts_used': len(result.get('reranked_contexts', []))
                }
                
                results.append(sample_result)
                total_contexts += sample_result['contexts_used']
                
                # 정답 확인 (ground truth가 있는 경우)
                if 'output' in item:
                    predicted = sample_result['predicted_answer']
                    ground_truth = item['output'].get('answer', '')
                    if predicted and ground_truth:
                        from utils import EvaluationMetrics
                        if EvaluationMetrics.exact_match(predicted, ground_truth):
                            correct_count += 1
                
                print(f"✅ 처리 완료: ID {sample_result['id']}")
                
            else:
                # 실패한 경우 기본 결과 추가
                sample_result = {
                    'id': item.get('id', f'sample_{i}'),
                    'input': item['input'],
                    'predicted_answer': "메모리 부족으로 처리 실패",
                    'contexts_used': 0
                }
                results.append(sample_result)
                print(f"❌ 처리 실패: ID {sample_result['id']}")
        
        except Exception as e:
            print(f"❌ 샘플 처리 중 오류: {e}")
            # 오류 발생해도 결과 추가
            error_result = {
                'id': item.get('id', f'sample_{i}'),
                'input': item['input'],
                'predicted_answer': f"오류 발생: {str(e)[:100]}",
                'contexts_used': 0
            }
            results.append(error_result)
        
        # 중간 저장 (5개마다)
        if (i + 1) % 5 == 0:
            DataLoader.save_intermediate_results(results, output_path, i+1)
        
        # 강제 메모리 정리
        MemoryManager.clear_gpu_memory()
    
    # 최종 결과 계산
    metrics = {
        'accuracy': correct_count / len(results) if results else 0,
        'total_samples': len(results),
        'correct_predictions': correct_count,
        'average_contexts_used': total_contexts / len(results) if results else 0
    }
    
    # 전체 결과 저장
    final_results = {
        'evaluation_metrics': metrics,
        'predictions': results
    }
    
    DataLoader.save_results(final_results, output_path)
    print(f"💾 Results saved to: {output_path}")
    
    return metrics

def predict_mode(data_path: str, output_path: str, enable_llm: bool = True):
    """예측 모드 - 테스트 데이터에 대한 예측 생성"""
    print(f"🔮 Prediction Mode - {data_path} → {output_path}")
    print("=" * 60)
    
    # A100 환경 최적화
    if torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name():
        setup_a100_environment()
    
    rag_system = None
    try:
        # 입력 파일 확인
        if not Path(data_path).exists():
            print(f"❌ Input data not found: {data_path}")
            return
        
        # 시스템 생성
        rag_system = create_rag_system(enable_llm=enable_llm)
        
        # 지식 베이스 로드
        train_data_path = './/korean_language_rag_V1.0_train.json'
        if Path(train_data_path).exists():
            rag_system.load_knowledge_base(train_data_path)
        else:
            print(f"⚠️ Training data not found: {train_data_path}")
            print("    Creating minimal knowledge base...")
            # 최소 지식 베이스 생성
            create_minimal_knowledge_base(rag_system)
        
        # 예측 실행
        metrics = run_optimized_evaluation(rag_system, data_path, output_path, max_samples=0)
        
        print(f"\n✅ 예측 완료!")
        print(f"   처리된 샘플 수: {metrics['total_samples']}")
        print(f"   평균 컨텍스트 사용: {metrics['average_contexts_used']:.2f}")
        print(f"   결과 저장: {output_path}")
        
    except Exception as e:
        print(f"❌ 예측 실행 중 오류: {e}")
        traceback.print_exc()
    finally:
        if rag_system:
            rag_system.cleanup()
        MemoryManager.clear_gpu_memory(force=True)

def create_minimal_knowledge_base(rag_system):
    """최소한의 지식 베이스 생성"""
    minimal_chunks = [
        {
            'id': 'chunk_1',
            'text': '한국어 맞춤법에서 "먹이양"이 올바른 표현입니다. 한자어 "量"은 앞말이 고유어일 때 "양"이 됩니다.',
            'category': '맞춤법',
            'question_type': '선택형',
            'source': 'minimal'
        },
        {
            'id': 'chunk_2', 
            'text': '띄어쓰기에서 의존명사는 앞말과 띄어 써야 합니다. 예: 외출할 때에는',
            'category': '띄어쓰기',
            'question_type': '교정형',
            'source': 'minimal'
        },
        {
            'id': 'chunk_3',
            'text': '외래어 표기법에 따라 "껌"이 올바른 표현입니다. 영어 "gum"에서 온 외래어입니다.',
            'category': '외래어표기',
            'question_type': '선택형', 
            'source': 'minimal'
        }
    ]
    
    rag_system.knowledge_chunks = minimal_chunks
    from utils import HybridRetriever
    rag_system.hybrid_retriever = HybridRetriever(minimal_chunks, None)
    print("✅ 최소 지식 베이스 생성 완료")

def test_mode(enable_llm=True):
    """테스트 모드 - LLM 기능 테스트 (메모리 최적화)"""
    print(f"🧪 Test Mode (LLM: {enable_llm})")
    print("=" * 60)

    # A100 환경 최적화
    if torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name():
        setup_a100_environment()

    rag_system = None
    try:
        # 시스템 생성
        rag_system = create_rag_system(enable_llm=enable_llm)

        # 지식 베이스 로드
        train_data_path = './/korean_language_rag_V1.0_train.json'
        if Path(train_data_path).exists():
            rag_system.load_knowledge_base(train_data_path)
        else:
            print(f"⚠️ Training data not found, creating minimal knowledge base")
            create_minimal_knowledge_base(rag_system)

        # 테스트 질문
        test_question = "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다."

        print(f"🔬 Testing with question: {test_question}")
        print("-" * 40)

        if enable_llm:
            print("🤖 Testing LLM components...")

            # 각 LLM 컴포넌트 안전 테스트
            try:
                # 1. Query Rewriter 테스트
                print("\n1. Testing Query Rewriter...")
                if rag_system.query_rewriter:
                    def test_rewriter():
                        return rag_system.query_rewriter.rewrite_query(test_question)
                    
                    expanded = MemoryManager.safe_model_operation(test_rewriter)
                    if expanded:
                        print(f"   Original: {test_question}")
                        print(f"   Expanded: {expanded}")
                    else:
                        print("   ❌ Query Rewriter 테스트 실패")

                # 2. Embedder 테스트
                print("\n2. Testing Korean Embedder...")
                if rag_system.embedder:
                    def test_embedder():
                        return rag_system.embedder.encode([test_question])
                    
                    embeddings = MemoryManager.safe_model_operation(test_embedder)
                    if embeddings is not None:
                        shape = embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'
                        print(f"   Embedding shape: {shape}")
                    else:
                        print("   ❌ Embedder 테스트 실패")

                # 3. 전체 파이프라인 테스트
                print("\n3. Testing Full Pipeline...")
                def test_pipeline():
                    return rag_system.process_question(test_question, "선택형")
                
                result = MemoryManager.safe_model_operation(test_pipeline)
                if result:
                    answer = result.get('final_answer') or result.get('rankrag_answer') or "No answer"
                    print(f"   Final Answer: {answer}")
                else:
                    print("   ❌ 파이프라인 테스트 실패")

            except Exception as e:
                print(f"⚠️ LLM test failed: {e}")
                print("   This is expected if models are not available in this environment")

        else:
            print("📝 Testing Template Mode...")
            def test_template():
                return rag_system.process_question(test_question, "선택형")
            
            result = MemoryManager.safe_model_operation(test_template)
            if result:
                answer = result.get('final_answer') or result.get('rankrag_answer') or "No answer"
                print(f"   Template Answer: {answer}")
            else:
                print("   ❌ 템플릿 모드 테스트 실패")

        print(f"\n✅ Test completed!")

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        traceback.print_exc()
    finally:
        if rag_system:
            rag_system.cleanup()
        MemoryManager.clear_gpu_memory(force=True)

def show_system_info():
    """시스템 정보 표시 (A100 정보 포함)"""
    print("💻 System Information")
    print("=" * 60)

    # GPU 정보 (상세)
    MemoryManager.check_memory_status()
    
    # CUDA 정보
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            print(f"\n🎮 GPU 정보:")
            print(f"   장치명: {device_name}")
            print(f"   CUDA 버전: {torch.version.cuda}")
            print(f"   장치 수: {torch.cuda.device_count()}")
            
            if 'A100' in device_name:
                print(f"   ✅ A100 최적화 사용 가능!")
        else:
            print("❌ CUDA 사용 불가")
    except Exception as e:
        print(f"❌ GPU 정보 확인 실패: {e}")

    # 데이터 파일 확인
    data_files = [
        './/korean_language_rag_V1.0_train.json',
        './/korean_language_rag_V1.0_dev.json', 
        './/korean_language_rag_V1.0_test.json'
    ]

    print("\n📂 Data Files:")
    for file_path in data_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024  # KB
            print(f"   ✅ {Path(file_path).name} ({size:.1f} KB)")
        else:
            print(f"   ❌ {Path(file_path).name} (not found)")

    # 의존성 확인
    print("\n📦 Dependencies:")
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('sklearn', 'Scikit-learn'),
        ('numpy', 'NumPy')
    ]
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   ✅ {display_name}: {version}")
        except ImportError:
            print(f"   ❌ {display_name} not available")

def main():
    """메인 함수 (A100 최적화)"""
    parser = argparse.ArgumentParser(
        description="Korean Grammar RAG System (A100 Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode demo
    python main.py --mode evaluate --samples 5 --enable_llm
    python main.py --mode predict --data_path test.json --output_path predictions.json
    python main.py --mode test --enable_llm
    python main.py --mode info
        """
    )

    parser.add_argument(
        '--mode',
        choices=['demo', 'evaluate', 'test', 'predict', 'info'],
        default='demo',
        help='Execution mode'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples for evaluation (0 = all, default: 10)'
    )

    parser.add_argument(
        '--enable_llm',
        action='store_true',
        help='Enable LLM models (requires GPU and model downloads)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        help='Input data path for prediction mode'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str,
        help='Output path for prediction results'
    )

    args = parser.parse_args()

    print("🇰🇷 Korean Grammar RAG System (A100 Optimized)")
    print("=" * 60)

    try:
        if args.mode == 'demo':
            demo_mode()
        elif args.mode == 'evaluate':
            evaluate_mode(args.samples, args.enable_llm)
        elif args.mode == 'test':
            test_mode(args.enable_llm)
        elif args.mode == 'predict':
            if not args.data_path or not args.output_path:
                print("❌ predict 모드는 --data_path와 --output_path가 필요합니다")
                return
            predict_mode(args.data_path, args.output_path, args.enable_llm)
        elif args.mode == 'info':
            show_system_info()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        # 최종 정리
        MemoryManager.clear_gpu_memory(force=True)
        print("\n🧹 System cleanup completed")

if __name__ == "__main__":
    main()
