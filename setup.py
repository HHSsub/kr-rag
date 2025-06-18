"""
Korean Grammar RAG System - Setup Script
"""

from setuptools import setup, find_packages

setup(
    name="korean-grammar-rag",
    version="1.0.0",
    description="State-of-the-art Korean Grammar RAG System with Hugging Face LLMs",
    author="Korean Grammar RAG Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "nltk>=3.8.0",
        "rouge-score>=0.1.2",
        "evaluate>=0.4.0",
        "huggingface-hub>=0.17.0",
        "safetensors>=0.3.0",
        "peft>=0.6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "korean-rag=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
