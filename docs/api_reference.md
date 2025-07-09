# API 参考

## 核心接口

### `build_pipeline(config_path: str) -> RAGPipeline`
```python
from rag_factory import build_pipeline

# 构建RAG管道
pipeline = build_pipeline("config.yaml")
```

### `RAGPipeline` 类
```python
class RAGPipeline:
    def query(self, question: str, **kwargs) -> dict:
        """执行查询并返回结果"""
        
    def add_documents(self, documents: List[Document]):
        """向知识库添加文档"""
        
    def visualize(self, output_path: str):
        """可视化知识图谱(仅GraphRAG)"""
```

## 主要模块

### 文档处理
```python
from rag_factory.documents import (
    PDFProcessor,
    WebPageExtractor,
    MultiModalLoader
)
```

### 图构造
```python
from rag_factory.graph_construction import (
    GraphRAGExtractor,
    HybridGraphBuilder
)
```

### 存储后端
```python
from rag_factory.storages import (
    Neo4jGraphStorage,
    LanceDBVectorStorage
)
```

## 回调接口
```python
from rag_factory.callbacks import (
    LoggingCallback,
    ProgressCallback,
    CustomCallback
)