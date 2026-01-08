import sqlite3
import re
from utils.schema_loader import SchemaLoader
from generator.sql_generator import SQLGenerator

class SQLValidator:
    FORBIDDEN = re.compile(r"\b(drop|delete|update|insert|alter)\b", re.I)

    @staticmethod
    def validate(sql: str) -> None:
        if SQLValidator.FORBIDDEN.search(sql):
            raise ValueError("Unsafe SQL detected")
        if not sql.lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed")

class SafeExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, sql: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return columns, rows

class ResultSummarizer:
    @staticmethod
    def summarize(columns, rows) -> str:
        if not rows:
            return "No results found."

        summary = []
        for row in rows:
            summary.append(
                ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
            )
        return "\n".join(summary)

class SQLQAPipeline:
    def __init__(self, db_path: str, llm):
        self.schema = SchemaLoader(db_path).load_schema()
        self.generator = SQLGenerator(llm)
        self.executor = SafeExecutor(db_path)
        self.llm = llm  
        
    def run(self, question: str) -> str:
        sql = self.generator.generate(question, self.schema)
        SQLValidator.validate(sql)
        columns, rows = self.executor.execute(sql)
        summarized = ResultSummarizer.summarize(columns, rows)

        print("\n" + "=" * 60)
        print("ORIGINAL QUESTION")
        print("=" * 60)
        print(question)

        print("\n" + "=" * 60)
        print("GENERATED SQL")
        print("=" * 60)
        print(sql)

        print("\n" + "=" * 60)
        print("RAW RESULTS")
        print("=" * 60)
        print("Columns:", columns)
        print("Rows:")
        for r in rows:
            print(r)

        print("\n" + "=" * 60)
        print("SUMMARIZED RESULTS")
        print("=" * 60)
        print(summarized)

        final_prompt = f"""
You are a financial data analyst.

User question:
{question}

SQL result summary:
{summarized}

Based on the results above, provide a clear and concise answer
in natural language. Do not mention SQL.
"""

        final_answer = self.llm(final_prompt).strip()
        confidence_prompt = f"""
You are evaluating the reliability of an answer generated from SQL query results.

User question:
{question}

SQL result summary:
{summarized}

Final answer:
{final_answer}

Give a confidence score between 0 and 1 based on:
- Completeness of the data
- Directness of the answer
- Absence of ambiguity

Return ONLY a single number between 0 and 1.
"""

        confidence_raw = self.llm(confidence_prompt).strip()

        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except ValueError:
            confidence = 0.5

        print("\n" + "=" * 60)
        print("LLM FINAL ANSWER")
        print("=" * 60)
        print(final_answer)

        print("\nCONFIDENCE SCORE:", confidence)

        return {
            "answer": final_answer,
            "confidence": confidence,
            "context":summarized
        }