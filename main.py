"""
Korean Grammar RAG System - Main Execution Script
한국어 어문 규범 RAG 시스템 메인 실행 스크립트

사용법:
    python main.py --mode demo                    # 데모 실행
    python main.py --mode evaluate --samples 10  # 평가 실행
    python main.py --mode test --enable_llm      # LLM 활성화 테스트
"""

import argparse
import json
import time
from pathlib import Path

# 로컬 모듈 임포트
from rag_pipeline import KoreanGrammarRAGSystem, create_rag_system, quick_test
from utils import DataLoader, MemoryManager

def demo_mode():
    """데모 모드 - 몇 개 샘플 질문으로 시스템 테스트"""
    print("🎭 Demo Mode - Korean Grammar RAG System")
    print("=" * 60)

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

    # 각 질문 처리
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n📝 Demo Question {i}: {demo['type']}")
        print(f"Q: {demo['question']}")
        print("-" * 40)

        start_time = time.time()
        result = rag_system.process_question(demo['question'], demo['type'])
        processing_time = time.time() - start_time

        print(f"A: {result['final_answer'] or result['rankrag_answer']}")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📊 Contexts used: {len(result['reranked_contexts'])}")

        # 메모리 정리
        MemoryManager.clear_gpu_memory()

    print(f"\n✅ Demo completed successfully!")
    rag_system.cleanup()

def evaluate_mode(max_samples=10, enable_llm=False):
    """평가 모드 - 데이터셋에서 성능 평가"""
    print(f"📊 Evaluation Mode (LLM: {enable_llm}, Samples: {max_samples})")
    print("=" * 60)

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

    # 평가 실행
    output_path = f'.//evaluation_results_llm_{enable_llm}.json'

    start_time = time.time()
    metrics = rag_system.evaluate_on_dataset(
        dev_data_path, 
        output_path=output_path,
        max_samples=max_samples
    )
    total_time = time.time() - start_time

    # 결과 출력
    print(f"\n📈 Evaluation Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Correct: {metrics['correct_predictions']}/{metrics['total_samples']}")
    print(f"   Avg Contexts: {metrics['average_contexts_used']:.2f}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Time per Sample: {total_time/metrics['total_samples']:.2f}s")

    rag_system.cleanup()

def test_mode(enable_llm=True):
    """테스트 모드 - LLM 기능 테스트"""
    print(f"🧪 Test Mode (LLM: {enable_llm})")
    print("=" * 60)

    # 시스템 생성
    rag_system = create_rag_system(enable_llm=enable_llm)

    # 지식 베이스 로드
    train_data_path = './/korean_language_rag_V1.0_train.json'
    if Path(train_data_path).exists():
        rag_system.load_knowledge_base(train_data_path)
    else:
        print(f"⚠️ Training data not found, creating minimal knowledge base")
        # 최소한의 지식 베이스 생성
        rag_system.knowledge_chunks = [
            {
                'id': 'test_chunk_1',
                'text': '한국어 맞춤법에서 "먹이양"이 올바른 표현입니다. 한자어 "量"은 앞말이 고유어일 때 "양"이 됩니다.',
                'category': '맞춤법',
                'question_type': '선택형',
                'source': 'test'
            }
        ]
        from utils import HybridRetriever
        rag_system.hybrid_retriever = HybridRetriever(rag_system.knowledge_chunks, None)

    # 테스트 질문
    test_question = "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다."

    print(f"🔬 Testing with question: {test_question}")
    print("-" * 40)

    if enable_llm:
        print("🤖 Testing LLM components...")

        # 각 LLM 컴포넌트 테스트
        try:
            # 1. Query Rewriter 테스트
            print("\n1. Testing Query Rewriter...")
            if rag_system.query_rewriter:
                expanded = rag_system.query_rewriter.rewrite_query(test_question)
                print(f"   Original: {test_question}")
                print(f"   Expanded: {expanded}")

            # 2. Embedder 테스트
            print("\n2. Testing Korean Embedder...")
            if rag_system.embedder:
                embeddings = rag_system.embedder.encode([test_question])
                print(f"   Embedding shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")

            # 3. 전체 파이프라인 테스트
            print("\n3. Testing Full Pipeline...")
            result = rag_system.process_question(test_question, "선택형")
            print(f"   Final Answer: {result['final_answer'] or result['rankrag_answer']}")

        except Exception as e:
            print(f"⚠️ LLM test failed: {e}")
            print("   This is expected if models are not available in this environment")

    else:
        print("📝 Testing Template Mode...")
        result = rag_system.process_question(test_question, "선택형")
        print(f"   Template Answer: {result['final_answer'] or result['rankrag_answer']}")

    print(f"\n✅ Test completed!")
    rag_system.cleanup()

def show_system_info():
    """시스템 정보 표시"""
    print("💻 System Information")
    print("=" * 60)

    # GPU 정보
    gpu_info = MemoryManager.get_gpu_memory_info()
    print(f"GPU: {gpu_info}")

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
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("   ❌ PyTorch not available")

    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers not available")

    try:
        import sentence_transformers
        print(f"   ✅ Sentence Transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("   ❌ Sentence Transformers not available")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Korean Grammar RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode demo
    python main.py --mode evaluate --samples 5
    python main.py --mode test --enable_llm
    python main.py --mode info
        """
    )

    parser.add_argument(
        '--mode',
        choices=['demo', 'evaluate', 'test', 'info'],
        default='demo',
        help='Execution mode'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples for evaluation (default: 10)'
    )

    parser.add_argument(
        '--enable_llm',
        action='store_true',
        help='Enable LLM models (requires GPU and model downloads)'
    )

    args = parser.parse_args()

    print("🇰🇷 Korean Grammar RAG System")
    print("=" * 60)

    try:
        if args.mode == 'demo':
            demo_mode()
        elif args.mode == 'evaluate':
            evaluate_mode(args.samples, args.enable_llm)
        elif args.mode == 'test':
            test_mode(args.enable_llm)
        elif args.mode == 'info':
            show_system_info()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 최종 정리
        MemoryManager.clear_gpu_memory()
        print("\n🧹 System cleanup completed")

if __name__ == "__main__":
    main()
