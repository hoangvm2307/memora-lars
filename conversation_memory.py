from typing import List, Dict


class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_interaction(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        return "\n".join(
            [
                f"Human: {interaction['question']}\nAI: {interaction['answer']}"
                for interaction in self.history
            ]
        )
