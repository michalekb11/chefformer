import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.shared.model.model import Chefformer
from configs.inference.settings import app_settings

class TextGenerationService:
    def __init__(self, model_checkpoint_path: str):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = Chefformer(model_settings=app_settings.model).to(self.device)
        
        # Load the weights from the .pth checkpoint
        if model_checkpoint_path:
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()

    def base_generate(self, prompt: str, temperature: float, top_p: float, top_k: int, max_tokens: int, stop_sequences: list[str] | None = None) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated_tokens = []

        # Disable gradient calculation for pure inference
        with torch.no_grad(): 
            for _ in range(max_tokens):
                # Forward Pass
                logits = self.model(input_ids)
                
                # Extract logits for the final token in the sequence
                next_token_logits = logits[0, -1, :]
                
                # Temperature Scaling
                if temperature != 1.0:
                    temperature = max(temperature, 1e-5) # Prevent division by zero
                    next_token_logits = next_token_logits / temperature
                
                # 3. Top-K Filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k, sorted=True)
                    min_top_k_value = top_k_values[-1]
                    # Mask out tail probabilities with negative infinity
                    next_token_logits[next_token_logits < min_top_k_value] = -float('Inf')
                    
                # Top-P (Nucleus) Filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Create boolean mask for tokens exceeding cumulative probability P
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift mask to ensure we keep the token that just crossed the threshold P
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    # Scatter mask back to original vocabulary indices
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Softmax and Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                # 6. Stop Condition Evaluation
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Prepare input for the next loop iteration (autoregression)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Prevent sequence from exceeding model's max context length
                if input_ids.shape[1] >= app_settings.model.max_context_length:
                    break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)