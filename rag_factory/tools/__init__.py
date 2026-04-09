from llama_index.core.tools import FunctionTool
from .excute_sql import execute_sql
# from subgrapg_refinement import subgraph_refine
from .subgraph_refinement_v2 import subgraph_refine



execute_sql_tool = FunctionTool.from_defaults(fn=execute_sql)


subgraph_refine_tool = FunctionTool.from_defaults(fn=subgraph_refine)