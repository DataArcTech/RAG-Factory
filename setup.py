from setuptools import setup, find_packages
from pathlib import Path

def get_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    with open(Path(__file__).parent / "requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="rag-factory",
    version="0.1.0",
    description="A factory for building advanced RAG (Retrieval-Augmented Generation) pipelines including GraphRAG and Multi-modal RAG",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Xiaojun WU",
    author_email="wuxiaojun@idea.edu.cn",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rag-factory=RAG-Factory.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)