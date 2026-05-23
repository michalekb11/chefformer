import json
from src.evaluation.metrics import calculate_ngram_diversity, calculate_self_bleu, calculate_lexical_validity
import re
import nltk
import argparse
from loggers.console_logger import ConsoleLogger
from src.inference.engine import TextGenerationService
import os
import torch
from transformers import AutoTokenizer
from configs.inference.settings import app_settings
from src.shared.model.model import Chefformer

logger = ConsoleLogger(__name__)

# Fixed prompts to track "learning"
PROMPTS = [
    # Famous continuation prompts
    "O say can you see\nBy the dawns early light\n",
    "Roses are red,\nViolets are blue,\n",

    # Structured list completion
    "January February March",

    # Code completion
    "class Dog:\n    def __init__(self, name):\n",

    # Few shot pattern introduction
    "2 -> 4\n3 -> 9\n4 -> 16\n5 ->",
    "apple = fruit\ncarrot = vegetable\nmilk = dairy\nsalmon =",

    # Long range coherence / reasoning
    "Alice is taller than Bob. Bob is taller than Charlie. Who is tallest?\nAnswer:",

    # Degeneration test
    "The the the the"
]

DIVERSITY_PROMPT = "Recipe name: Grilled chicken breast\nInstructions:"
NUM_SAMPLES_FOR_SELF_BLEU = 5

def run_checkpoint_evaluation(text_generator: TextGenerationService, output_file: str) -> None:
    # Load a common word set for lexical validity
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        logger.info("Downloading NLTK words corpus...")
        nltk.download('words')
    from nltk.corpus import words
    common_words_set = set(w.lower() for w in words.words())

    # Load existing results to ensure we append new steps and overwrite existing ones
    results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load existing results from {output_file}. Check output file.")
            return

    step = re.search(r'step(\d+)', checkpoint_path).group(1)
    logger.info(f"Evaluating Step {step}...")
    
    checkpoint_data = {"prompts": [], "metrics": {}}
    
    for i, prompt in enumerate(PROMPTS):
        logger.info(f"Evaluating prompt {i}")
        # Using defaults for temp, topP/K, etc.
        generated_text = text_generator.base_generate(prompt=prompt)
        diversity = calculate_ngram_diversity(generated_text)
        validity = calculate_lexical_validity(generated_text, common_words_set)

        checkpoint_data["prompts"].append({
            "input": prompt,
            "output": generated_text,
            "diversity": diversity,
            "lexical_validity": validity
        })

    # Self-BLEU Logic: Generate multiple samples for the SAME prompt to measure variance
    logger.info(f"Generating {NUM_SAMPLES_FOR_SELF_BLEU} samples for Self-BLEU calculation...")
    diversity_samples = []
    for _ in range(NUM_SAMPLES_FOR_SELF_BLEU):
        sample = text_generator.base_generate(prompt=DIVERSITY_PROMPT)
        diversity_samples.append(sample)

    # Global metrics for this checkpoint
    logger.info("Calculating global metrics...")
    checkpoint_data["metrics"]["avg_diversity"] = sum(p["diversity"] for p in checkpoint_data["prompts"]) / len(PROMPTS)
    checkpoint_data["metrics"]["avg_lexical_validity"] = sum(p["lexical_validity"] for p in checkpoint_data["prompts"]) / len(PROMPTS)
    checkpoint_data["metrics"]["self_bleu"] = calculate_self_bleu(diversity_samples)
    
    results[step] = checkpoint_data

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chefformer Checkpoint Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Checkpoint directory")
    parser.add_argument('-s', '--steps', nargs='+', type=int, required=True,
                        help='List of checkpoint steps to evaluate')
    parser.add_argument("-o", "--output", type=str, default="./logs/pretrain/evaluation/eval_results.json",
                        help="Path to the output JSON results file")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"], help="Compute target")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_skeleton = Chefformer(model_settings=app_settings.model)

    # Instantiate the service shell once
    generation_service = TextGenerationService(
        model=model_skeleton, 
        tokenizer=tokenizer, 
        device=device
    )

    for step in args.steps:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch0_step{step}.pth")
        try:
            if os.path.exists(checkpoint_path):
                generation_service.update_model_weights(checkpoint_path)
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            continue
        run_checkpoint_evaluation(generation_service, args.output)
