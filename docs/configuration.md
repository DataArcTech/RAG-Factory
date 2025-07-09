# 配置指南

## 核心配置项

```yaml
# LLM配置
llm:
  provider: "openai"  # 或anthropic, cohere等
  model_name: "gpt-4"
  api_key: "your-api-key"

# 向量数据库配置
vector_db:
  type: "hnswlib"  # 或pinecone, weaviate等
  path: "./data/vector_index"

# 知识图谱配置
knowledge_graph:
  type: "neo4j"
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"
```

## 高级配置

### GraphRAG特定配置
```yaml
graph_rag:
  extraction_method: "hybrid"  # hybrid, ner或relation
  similarity_threshold: 0.75
  max_hop: 3
```

### 多模态RAG配置
```yaml
multimodal:
  image_processor: "clip"
  text_encoder: "bert"
```

## 环境变量
可通过环境变量覆盖配置:
```bash
export RAG_LLM_PROVIDER=anthropic
export RAG_LLM_MODEL=claude-2