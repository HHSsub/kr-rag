#!/usr/bin/env python3
"""
Korean Grammar RAG System - Lightweight Demo
PyTorch 없이도 작동하는 데모 스크립트
"""

import json
import time
import re
from pathlib import Path

class LightweightDemo:
    """PyTorch 의존성 없는 경량 데모"""

    def __init__(self):
        self.knowledge_chunks = []

    def load_knowledge_base(self, train_path):
        """지식 베이스 로드"""
        if not Path(train_path).exists():
            print(f"❌ Training data not found: {train_path}")
            return False

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        # 지식 청크 생성
        for i, item in enumerate(train_data):
            question = item['input']['question']
            answer = item['output']['answer']
            question_type = item['input']['question_type']

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

            self.knowledge_chunks.append(chunk)

        print(f"✅ Knowledge base loaded: {len(self.knowledge_chunks)} chunks")
        return True

    def extract_options_from_question(self, question):
        """질문에서 선택지 추출"""
        pattern = r'{([^}]+)}'
        matches = re.findall(pattern, question)

        options = []
        for match in matches:
            if '/' in match:
                options.extend([opt.strip() for opt in match.split('/')])

        return options

    def simple_search(self, question, top_k=5):
        """간단한 키워드 검색"""
        question_words = set(question.split())

        scored_chunks = []
        for chunk in self.knowledge_chunks:
            chunk_words = set(chunk['text'].split())

            # 키워드 매칭 점수
            common_words = question_words & chunk_words
            score = len(common_words) / len(question_words) if question_words else 0

            # 선택지 매칭 보너스
            options = self.extract_options_from_question(question)
            for option in options:
                if option in chunk['text']:
                    score += 0.3

            scored_chunks.append((chunk, score))

        # 점수로 정렬
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, score in scored_chunks[:top_k]]

    def generate_template_answer(self, question, question_type, contexts):
        """템플릿 기반 답변 생성"""
        if not contexts:
            return f"질문에 대한 관련 정보를 찾을 수 없습니다. 질문: {question}"

        best_context = contexts[0]

        if question_type == "선택형":
            options = self.extract_options_from_question(question)
            if options and len(options) >= 2:
                # 첫 번째 옵션을 정답으로 가정 (실제로는 더 복잡한 로직 필요)
                answer = f'"{options[0]}"이 옳다. {best_context["text"][len(question):300]}...'
            else:
                answer = f"주어진 선택지 중 올바른 표현을 선택해야 합니다. {best_context['text'][:300]}..."
        else:  # 교정형
            answer = f"어문 규범에 맞게 교정이 필요합니다. {best_context['text'][:300]}..."

        return answer

    def process_question(self, question, question_type):
        """질문 처리"""
        print(f"🔄 Processing: {question_type} question")

        # 1. 검색
        contexts = self.simple_search(question)
        print(f"🔍 Found {len(contexts)} relevant contexts")

        # 2. 답변 생성
        answer = self.generate_template_answer(question, question_type, contexts)

        return {
            'question': question,
            'question_type': question_type,
            'contexts_used': len(contexts),
            'answer': answer
        }

def main():
    """메인 함수"""
    print("🇰🇷 Korean Grammar RAG System - Lightweight Demo")
    print("=" * 60)
    print("⚠️  This is a lightweight demo that works without PyTorch")
    print("    For full LLM functionality, install dependencies and use main.py")
    print()

    # 시스템 생성
    demo = LightweightDemo()

    # 지식 베이스 로드
    train_path = './/korean_language_rag_V1.0_train.json'
    if not demo.load_knowledge_base(train_path):
        print("❌ Could not load knowledge base. Exiting.")
        return

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
    total_time = 0
    for i, demo_q in enumerate(demo_questions, 1):
        print(f"\n📝 Demo Question {i}: {demo_q['type']}")
        print(f"Q: {demo_q['question']}")
        print("-" * 50)

        start_time = time.time()
        result = demo.process_question(demo_q['question'], demo_q['type'])
        processing_time = time.time() - start_time
        total_time += processing_time

        print(f"A: {result['answer']}")
        print(f"⏱️  Processing time: {processing_time:.3f}s")
        print(f"📊 Contexts used: {result['contexts_used']}")

    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Total processing time: {total_time:.3f}s")
    print(f"⚡ Average time per question: {total_time/len(demo_questions):.3f}s")

    print("\n🚀 To run with full LLM functionality:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run: python main.py --mode demo --enable_llm")

if __name__ == "__main__":
    main()
