import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

class LLMBackend:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMBackend, cls).__new__(cls)
            cls._instance.current_model = None
            cls._instance.current_tokenizer = None
            cls._instance.current_model_id = None
        return cls._instance

    def load_model(self, model_id: str):
        """Loads a model if it's not already loaded, unloading the previous one if necessary."""
        if self.current_model_id == model_id:
            return

        # Unload previous model
        if self.current_model is not None:
            print(f"Unloading {self.current_model_id}...")
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Loading {model_id}...")
        try:
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True
            )
            self.current_model_id = model_id
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise e

    def generate(self, model_id: str, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
        self.load_model(model_id)
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template if available, otherwise just use raw prompt
        if self.current_tokenizer.chat_template:
            text = self.current_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.current_tokenizer(text, return_tensors="pt").to(self.current_model.device)

        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.current_tokenizer.eos_token_id
            )

        # Decode directly
        response = self.current_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
