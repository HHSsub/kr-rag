#!/bin/bash

# Korean Grammar RAG System - Installation Script
echo "🇰🇷 Installing Korean Grammar RAG System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv korean_rag_env
source korean_rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (for RTX 4090)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install additional dependencies for Korean NLP
pip install konlpy soynlp

# Install the package
echo "⚙️ Installing Korean Grammar RAG System..."
pip install -e .

echo "✅ Installation completed!"
echo ""
echo "🚀 Quick Start:"
echo "   source korean_rag_env/bin/activate"
echo "   python main.py --mode demo"
echo ""
echo "📖 Full Usage:"
echo "   python main.py --mode evaluate --samples 10"
echo "   python main.py --mode test --enable_llm"
echo "   python main.py --mode info"
