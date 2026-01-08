import sqlite3
from typing import Dict, List

class SchemaLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def load_schema(self) -> Dict[str, List[str]]:
        schema = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            for (table_name,) in tables:
                cols = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
                schema[table_name] = [col[1] for col in cols]
        return schema