# https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/
# https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook/
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from llama_index.llms.openai import OpenAI

llm = OpenAI()
question_gen = OpenAIQuestionGenerator.from_defaults(llm=llm)

if __name__ == "__main__":
    display_prompt_dict(question_gen.get_prompts())