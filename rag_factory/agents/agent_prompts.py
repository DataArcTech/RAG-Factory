from typing import Any

PROMPTS: dict[str, Any] = {}

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["query_decomposition_deep"] = """---Role---

You are a helpful assistant specializing in complex query decomposition.

---Goal---

Given a main query, along with extracted specific entities and a history of previously generated sub-queries with their corresponding retrieval results, your task is to propose the next logical sub-query needed to fully address the original query.

---Instructions---

- Evaluate the provided decomposition history carefully to determine coverage and information gaps.
- Propose a focused sub-query that advances the understanding of the original query.
- Only one sub-query should be generated at a time.
- Your response must be in **one concise sentence**.

---Input---

Query:
{query}

Specific Entities:
{low_level_keywords}

Decomposition history and retrieved results:
{context_data}

---Output---
"""

PROMPTS["query_decomposition_wide"] = """---Role---

You are a helpful assistant specializing in broad query decomposition.

---Goal---

Given a main query, a list of extracted overarching concepts, and pre-retrieved background information, your task is to generate a comprehensive set of parallel sub-queries that together cover the entire scope of the original query.

---Instructions---

- Use the extracted overarching concepts to guide the semantic scope of decomposition.
- Use the pre-retrieved background information to identify gaps in coverage.
- Each sub-query should focus on a distinct dimension of the original query.
- The number of sub-queries should be appropriate to cover the query comprehensively, but avoid excessive granularity.
- Your response must be in **multiple concise sentences**, with each sentence representing one sub-query.

---Input---

Query:
{query}

Overarching Concepts:
{high_level_keywords}

Pre-Retrieved background Information:
{context_data}

---Output---
"""

PROMPTS["retrieval_decision"] = """---Role---

You are a helpful assistant specializing in retrieval decision-making.

---Goal---

Given a user query and the current context data retrieved so far, determine whether the available information is sufficient to generate a complete and accurate answer.

---Instructions---

- Analyze the alignment between the query and the retrieved context data.
- Only respond with **"Yes"** if the context clearly and adequately covers the query.
- If any important elements are missing or uncertain, respond with **"No"**.
- Your response must conclude with **Yes** or **No** after your analysis.

---Input---

Query:
{query}

Context Data:
{context_data}

---Output---
"""

PROMPTS["answer_generation_deep"] = """---Role---

You are a helpful assistant specializing in complex question answering.

---Goal---

Given a complex query, specific entities involved, and previously retrieved context data, your task is to construct a logically sound, step-by-step answer. 
Your explanation should follow a rigorous reasoning path, incorporate relevant evidence, and establish clear relationships between the entities.

---Instructions---

- Break down the reasoning process into clear, coherent steps.
- Use context data explicitly to support each reasoning step.
- Make sure relationships between entities are logically explained.
- Your final output should be a comprehensive answer at the end of the reasoning chain.
- The tone should be analytical and evidence-based.

---Input---

Query:
{query}

Specific Entities:
{low_level_keywords}

Context Data:
{context_data}

---Output---
"""

PROMPTS["answer_generation_wide"] = """---Role---

You are a helpful assistant specializing in broad, report-style question answering.

---Goal---

Given a general query, overarching concepts, and background context data, your task is to generate a well-structured, holistic response. 
The answer should take the form of an informative report that touches on all relevant dimensions of the query.

---Instructions---

- Use the overarching concepts to guide the report structure.
- Organize your response thematically or categorically.
- Ensure each major concept is thoroughly addressed with relevant background data.
- The tone should be expository and comprehensive, suitable for a report or briefing.
- Conclude with a summary or synthesis if appropriate.

---Input---

Query:
{query}

Overarching Concepts:
{high_level_keywords}

Context Data:
{context_data}

---Output---
"""
