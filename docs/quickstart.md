# RAG-Factory 快速入门

## 安装

```bash
git clone https://github.com/your-repo/rag-factory.git
cd rag-factory
pip install -e .
```

## 基本使用

1. 准备配置文件 `config.yaml` (参考examples目录)
2. 初始化RAG管道:

```python
from rag_factory import build_pipeline

pipeline = build_pipeline("config.yaml")
```

3. 执行查询:

```python
results = pipeline.query("你的问题")
print(results)
```

## 核心概念

- **GraphRAG**: 基于知识图谱的RAG实现
- **Multi-modal RAG**: 支持多模态数据的RAG
- **Pipeline Components**: 可插拔的组件设计