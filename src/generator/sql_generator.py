class SQLGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question: str, schema: dict) -> str:
        schema_str = "\n".join(
            f"{table}: {', '.join(cols)}" for table, cols in schema.items()
        )

        prompt = f"""
You are an expert SQL generator for SQLite.

Schema:
{schema_str}

User question:
{question}

Rules:
- Generate only ONE SQL query
- Use valid SQLite syntax
- ALWAYS qualify column names with table names or aliases (table.column)
- If a table is used, assign it a short alias and use it consistently
- Prefer single-table queries unless a JOIN is explicitly required
- Do NOT include explanations or comments
- Do NOT use DROP, DELETE, UPDATE, INSERT
- Only use tables and columns that exist in the schema above

SQL:
"""

        response = self.llm(prompt)
        sql = response.strip()

        if "```" in sql:
            sql = sql.split("```")[-1]

        sql = sql.replace("sql", "").strip()

        if ";" in sql:
            sql = sql.split(";")[0] + ";"

        return sql
