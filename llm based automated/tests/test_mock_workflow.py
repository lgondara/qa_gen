import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow import CurationWorkflow
from src.llm import LLMBackend

class TestWorkflow(unittest.TestCase):
    
    @patch('src.llm.LLMBackend')
    def test_workflow_pass_first_try(self, MockBackend):
        # Setup mock behavior
        mock_instance = MockBackend.return_value
        
        # Scenario: Creator -> Reviewer (Pass) -> Checker
        def side_effect(model_id, prompt, **kwargs):
            if "Creator" in model_id or "Source Text" in prompt:
                return "Generated QA Pair: Q: What is AI? A: Artificial Intelligence."
            elif "Reviewer" in model_id or "Reviewer" in prompt: # This might be fragile depending on how I instantiate
                return "STATUS: PASS\nLooks good."
            elif "FinalChecker" in model_id:
                return "{'instruction': 'Define AI', 'output': 'Artificial Intelligence'}"
            return ""
        
        mock_instance.generate.side_effect = side_effect
        
        workflow = CurationWorkflow()
        result = workflow.process_item("Define AI.")
        
        self.assertIsNotNone(result)
        print("Result:", result)
        self.assertTrue("output" in result)

    @patch('src.llm.LLMBackend')
    def test_workflow_retry_logic(self, MockBackend):
         # Setup mock behavior
        mock_instance = MockBackend.return_value
        
        # State to track retries
        self.review_count = 0
        
        def side_effect(model_id, prompt, **kwargs):
            if "Qwen" in model_id and "Previous Feedback" not in prompt:
                # First draft
                return "Bad Draft"
            
            if "Reviewer" in model_id or "Mistral" in model_id:
                if self.review_count == 0:
                    self.review_count += 1
                    return "STATUS: FAIL\nFEEDBACK: Too short."
                else:
                    return "STATUS: PASS\nMuch better."
            
            if "Qwen" in model_id and "Previous Feedback" in prompt:
                return "Better Draft"
                
            if "FinalChecker" in model_id:
                return "JSON OK"
            return ""
            
        mock_instance.generate.side_effect = side_effect
        
        workflow = CurationWorkflow(creator_model="Qwen", reviewer_model="Mistral", checker_model="Qwen")
        result = workflow.process_item("Context info.")
        
        self.assertEqual(self.review_count, 1)
        self.assertEqual(result, "JSON OK")

if __name__ == '__main__':
    unittest.main()
