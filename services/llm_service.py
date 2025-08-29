from typing import Dict, Any
import os
import json
from structlog import get_logger
from groq import Groq

logger = get_logger()

class LLMService:
    def __init__(self):
        self.model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=groq_api_key)

    def build_prompt(self, summary: Dict, sample_rows: Any) -> str:
        columns = summary.get("columns", [])
        data_types = summary.get("data_types", {})
        missing = summary.get("missing_values", {})
        unique = summary.get("unique_values", {})

        col_info_lines = []
        for col in columns:
            dt = data_types.get(col, "unknown")
            miss = missing.get(col, 0)
            uniq = unique.get(col, 0)
            col_info_lines.append(f"- Column '{col}': type={dt}, missing={miss}, unique={uniq}")

        col_info_str = "\n".join(col_info_lines)
        sample_str = sample_rows.head(5).to_csv(index=False) if hasattr(sample_rows, "head") else ""

        prompt = f"""
You are an expert data scientist. Based on the dataset summary and sample data below, please provide:


1. Detailed preprocessing recommendations per column (imputation, encoding).
2. Suggested ML task type: classification, regression, or clustering.
3. Recommended ML models likely to perform well.


Dataset Summary:
Number of columns: {len(columns)}
{col_info_str}


Sample data (up to 5 rows):
{sample_str}


Respond with a JSON object including:
- recommended_preprocessing: dict of column to strategy
- recommended_task_type: string
- recommended_models: list of strings
- explanation: brief reasoning for your choices.


Begin:
"""
        return prompt.strip()

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                n=1,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error("Groq LLM API call failed", error=str(e))
            return {}

    def get_recommendations(self, summary: Dict, df_sample: Any) -> Dict[str, Any]:
        prompt = self.build_prompt(summary, df_sample)
        return self.call_llm(prompt)
