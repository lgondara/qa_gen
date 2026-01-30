from .llm import LLMBackend

class Agent:
    def __init__(self, name: str, model_id: str, system_prompt: str):
        self.name = name
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.llm = LLMBackend()

    def generate_response(self, user_input: str, context: str = "") -> str:
        full_prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nTask:\n{user_input}"
        return self.llm.generate(self.model_id, full_prompt)

class CreatorAgent(Agent):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-32B-Instruct"):
        super().__init__(
            name="Creator",
            model_id=model_id,
            system_prompt="You are an expert dataset creator. Your task is to generate high-quality instruction-response pairs or QA pairs from the provided text. Ensure diversity, clarity, and correctness."
        )

    def create(self, source_text: str, feedback: str = None) -> str:
        prompt = f"Source Text:\n{source_text}\n"
        if feedback:
            prompt += f"\nPrevious Feedback to address:\n{feedback}\n\nPlease regenerate the data improving based on the feedback."
        else:
            prompt += "\nGenerate a high-quality instruction-response pair based on the text above."
        
        return self.generate_response(prompt)

class ReviewerAgent(Agent):
    def __init__(self, model_id: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        super().__init__(
            name="Reviewer",
            model_id=model_id,
            system_prompt="You are a strict data quality reviewer. Evaluate the generated instruction-response pair. Check for factual accuracy, formatting, clarity, and adherence to instructions. \nOutput format:\nSTATUS: [PASS/FAIL]\nFEEDBACK: [Detailed feedback if FAIL, or comments if PASS]"
        )

    def review(self, generated_data: str, source_text: str) -> dict:
        prompt = f"Source Text:\n{source_text}\n\nGenerated Data:\n{generated_data}\n\nEvaluate the quality."
        response = self.generate_response(prompt)
        
        # Simple parsing
        status = "FAIL"
        if "STATUS: PASS" in response:
            status = "PASS"
        
        return {
            "status": status,
            "feedback": response,
            "raw_output": response
        }

class FinalCheckerAgent(Agent):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-32B-Instruct"):
        super().__init__(
            name="FinalChecker",
            model_id=model_id,
            system_prompt="You are the final quality gate. formats the data into a clean JSON structure and performs a final sanity check. If the data is garbage, output 'DISCARD'. Otherwise, output the clean JSON."
        )

    def check_and_format(self, generated_data: str) -> str:
        prompt = f"Generated Data:\n{generated_data}\n\nVerify quality and format as JSON {{'instruction': ..., 'input': ..., 'output': ...}}."
        return self.generate_response(prompt)
