
#!/bin/bash

# Copyright (c) 2025, IDEA DataArc Team. All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

# input arguments of RAG solutions
solution=$1

echo "Running RAG solution: $solution"
if [ "$solution" == "naive_rag" ]; then
    echo "Starting NaiveRAG example..."
    python main.py --config examples/rag/config.yaml
elif [ "$solution" == "graph_rag" ]; then
    echo "Starting GraphRAG example..."
    python main.py --config examples/graphrag/config.yaml
elif [ "$solution" == "mm_rag" ]; then
    echo "Starting MultiModalRAG example..."
    python main.py --config examples/multimodal_rag/config.yaml
elif [ "$solution" == "ToG3" ]; then
    echo "Starting ToG3 example..."
    python main.py --config examples/ToG3/config.yaml
else
    echo "Unknown solution: $solution"
    exit 1
fi