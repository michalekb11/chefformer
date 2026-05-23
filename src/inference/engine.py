import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from configs.inference.settings import app_settings
from loggers.console_logger import ConsoleLogger
import gc

logger = ConsoleLogger(__name__)

class TextGenerationService:
    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.model.gradient_checkpointing = False
        
    def update_model_weights(self, checkpoint_path: str) -> None:
        """
        Mutates the existing model weights in-place without reallocating 
        architectural memory space on the hardware accelerator.
        """
        logger.info(f"Updating runtime weights via checkpoint: {checkpoint_path}")
        try:
            # Load to host memory (CPU) first to eliminate driver contention
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Accommodate both raw state dicts and full training state dictionaries
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            # In-place injection
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("In-place weight mutation successful.")
        except Exception as e:
            logger.error(f"Critical failure loading weights from {checkpoint_path}: {str(e)}")
            raise e
        finally:
            # Defensive memory sanitization
            if 'checkpoint' in locals(): del checkpoint
            if 'state_dict' in locals(): del state_dict
            if self.device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

    def base_generate(self, prompt: str, temperature: float=1.0, top_p: float=1.0, top_k: int=-1, max_tokens: int=512, stop_sequences: list[str] | None=None) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []

        # Disable gradient calculation for pure inference
        with torch.no_grad(): 
            for _ in range(max_tokens):
                # Forward Pass
                # Requesting only the last logit significantly reduces peak memory usage
                logits = self.model(input_ids, last_token_only=True)
                
                # Extract logits for the final token in the sequence
                next_token_logits = logits[0, -1, :].clone()
                del logits # Explicitly delete the large tensor
                
                # Temperature Scaling
                if temperature != 1.0:
                    temperature = max(temperature, 1e-5) # Prevent division by zero
                    next_token_logits = next_token_logits / temperature
                
                # Top-K Filtering
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
                
                # Stop Condition Evaluation
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Prepare input for the next loop iteration (autoregression)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Prevent sequence from exceeding model's max context length
                if input_ids.shape[1] >= app_settings.model.max_context_length:
                    break

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        return output_text