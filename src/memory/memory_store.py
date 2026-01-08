# from typing import Dict, Any, List
# from langchain_classic.memory import ConversationBufferWindowMemory
# from langchain_core.messages import BaseMessage

# class MemoryStore:
#     def __init__(self, k: int = 5):
#         self.memory = ConversationBufferWindowMemory(k=k,return_messages=True,memory_key="history",input_key="input",output_key="output")
    
#     def save_turn(self, user_input: str, assistant_output: str) -> None:
#         self.memory.save_context(inputs={"input": user_input},outputs={"output": assistant_output})

#     def load(self) -> Dict[str, Any]:
#         return self.memory.load_memory_variables({})

#     def messages(self) -> List[BaseMessage]:
#         return self.memory.chat_memory.messages

#     def clear(self) -> None:
#         self.memory.clear()

'''
The above implementation is using deprecated packages and hence the memory store has been manually implemented below
'''

import time
from collections import deque
from typing import List, Dict


class MemoryStore:
    def __init__(self, k: int = 5):
        self.k = k
        self.buffer = deque(maxlen=k * 2)

    def add_user(self, text: str):
        self.buffer.append(
            {"role": "user", "content": text, "ts": time.time()}
        )

    def add_assistant(self, text: str):
        self.buffer.append(
            {"role": "assistant", "content": text, "ts": time.time()}
        )

    def get(self) -> List[Dict]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()