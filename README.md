<div align="center">

<div style="margin: 20px 0;al">
  <img src="./assets/logo.png" width="120" height="120" alt="RAG-Factory Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 RAG-Factory: Advanced and Easy-Use RAG Pipelines
</div>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2411.06272" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/Arxiv-2411.06272-b31b1b.svg?logo=arXiv" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/DataArcTech/Golden-Touchstone" target="_blank" style="margin: 2px;">
    <img alt="github" src="https://img.shields.io/github/stars/DataArcTech/RAG-Factory.svg?style=social" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/IDEA-FinAI/TouchstoneGPT-7B-Instruct" target="_blank" style="margin: 2px;">
    <img alt="datasets" src="https://img.shields.io/badge/🤗-Datasets-yellow.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/IDEA-FinAI/TouchstoneGPT-7B-Instruct" target="_blank" style="margin: 2px;">
    <img alt="huggingface" src="https://img.shields.io/badge/🤗-Model-yellow.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

A factory for building advanced RAG (Retrieval-Augmented Generation) pipelines, including:

- Standard RAG implementations
- GraphRAG architectures 
- Multi-modal RAG systems

## Features

<div>
  <img src="./assets/knowledge_base_screenshot.png" alt="Example Knowledge Base Screenshot of RAG-Factory" width="800">
</div>

- Modular design for easy customization
- Support for various knowledge graph backends
- Integration with multiple LLM providers
- Configurable pipeline components

## Installation

```bash
pip install -e .
```

## Usage

```bash
python main.py --config examples/graphrag/config.yaml
```

## Examples

See the `examples/` directory for sample configurations and usage.

## Roadmap

### ✅ Implemented Features
- Vector RAG (基于Qdrant实现)
- Graph RAG (支持知识图检索)
- Lightweight SQLite Cache (轻量级缓存方案)

### 🚧 Planned Features
- Multi-modal RAG (多模态检索增强生成)
- ReAct QueryEngine (交互式查询引擎)
- Query Engineering:
  - Query Rewriting (查询重写)
  - Sub-Questions (子问题分解)
- Agentic RAG (智能工具选择优化性能)

## ⭐ Star History

<a href="https://star-history.com/#DataArcTech/RAG-Factory&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DataArcTech/RAG-Factory&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DataArcTech/RAG-Factory&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=DataArcTech/RAG-Factory&type=Date" />
 </picture>
</a>
<div align="center">
  <p>⭐ 如果这个项目对您有帮助，动动小手点亮Star吧！</p>
</div>

<!-- ## 🤝 Contribution

<div align="center">
  We thank all our contributors for their valuable contributions.
</div>

<div align="center">
  <a href="https://github.com/DataArcTech/RAG-Factory/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=DataArcTech/RAG-Factory" style="border-radius: 15px; box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);" />
  </a>
</div> -->

<!-- ## 📖 Citation

```python
@misc{wu2024goldentouchstonecomprehensivebilingual,
      title={Golden Touchstone: A Comprehensive Bilingual Benchmark for Evaluating Financial Large Language Models}, 
      author={Xiaojun Wu and Junxi Liu and Huanyi Su and Zhouchi Lin and Yiyan Qi and Chengjin Xu and Jiajun Su and Jiajie Zhong and Fuwei Wang and Saizhuo Wang and Fengrui Hua and Jia Li and Jian Guo},
      year={2024},
      eprint={2411.06272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.06272}, 
}
``` -->



