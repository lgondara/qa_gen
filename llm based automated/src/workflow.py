from .agents import CreatorAgent, ReviewerAgent, FinalCheckerAgent

class CurationWorkflow:
    def __init__(self, 
                 creator_model="Qwen/Qwen2.5-32B-Instruct",
                 reviewer_model="mistralai/Mistral-Nemo-Instruct-2407",
                 checker_model="Qwen/Qwen2.5-32B-Instruct",
                 max_retries=3):
        
        self.creator = CreatorAgent(model_id=creator_model)
        self.reviewer = ReviewerAgent(model_id=reviewer_model)
        self.checker = FinalCheckerAgent(model_id=checker_model)
        self.max_retries = max_retries

    def process_item(self, source_text: str) -> dict:
        print(f"--- Processing Item ---")
        
        # Initial Generation
        draft = self.creator.create(source_text)
        print(f"[Creator] Generated Draft")

        # Refinement Loop
        passed_review = False
        current_draft = draft
        
        for i in range(self.max_retries):
            review_result = self.reviewer.review(current_draft, source_text)
            print(f"[Reviewer] Round {i+1}: {review_result['status']}")
            
            if review_result['status'] == 'PASS':
                passed_review = True
                break
            
            # Refine
            print(f"[Creator] Refining based on feedback...")
            current_draft = self.creator.create(source_text, feedback=review_result['feedback'])

        if not passed_review:
            print("Max retries reached. Quality subpar. Discarding.")
            return None

        # Final Check
        print("[FinalChecker] Finalizing...")
        final_output = self.checker.check_and_format(current_draft)
        
        if "DISCARD" in final_output:
            print("Final checker rejected the item.")
            return None
            
        return final_output
