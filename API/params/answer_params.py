class AnswerParams:
    def __init__(self, query, context, prompt_type="default", quiz_count=5):
        self.query = query
        self.context = context
        self.prompt_type = prompt_type
        self.quiz_count = quiz_count
