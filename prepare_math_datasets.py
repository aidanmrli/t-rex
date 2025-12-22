import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm

def extract_gsm8k_answer(answer_str):
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()

def extract_math_answer(solution_str):
    # Search for \boxed{...} or \boxed{...}$
    match = re.search(r"\\boxed\{(.*)\}", solution_str)
    if match:
        return match.group(1).strip()
    return solution_str.strip()

def format_example(question, solution):
    # We provide multiple keys to be compatible with different OpenRLHF configs
    # input/prompt: for the question
    # output: for the full solution (SFT)
    # answer/label: for the ground truth (RL)
    
    # Identify if it's GSM8K or MATH to extract answer
    if "####" in solution:
        answer = extract_gsm8k_answer(solution)
    else:
        answer = extract_math_answer(solution)
        
    return {
        "input": question,
        "prompt": question,
        "output": solution,
        "answer": answer,
        "label": answer
    }

def process_and_save(dataset_name, subset_name, split, output_path, question_key, solution_key):
    print(f"Processing {dataset_name} ({subset_name if subset_name else ''}) {split}...")
    try:
        if subset_name:
            ds = load_dataset(dataset_name, subset_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
            
        with open(output_path, "w") as f:
            for ex in tqdm(ds):
                formatted = format_example(ex[question_key], ex[solution_key])
                f.write(json.dumps(formatted) + "\n")
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

def main():
    os.makedirs("data", exist_ok=True)
    
    # 1. GSM8K (openai/gsm8k)
    process_and_save("openai/gsm8k", "main", "train", "data/gsm8k_train.jsonl", "question", "answer")
    process_and_save("openai/gsm8k", "main", "test", "data/gsm8k_test.jsonl", "question", "answer")
    
    # 2. GSM8K Platinum (madrylab/gsm8k-platinum) - Test only
    process_and_save("madrylab/gsm8k-platinum", None, "test", "data/gsm8k_platinum_test.jsonl", "question", "answer")
    
    # 3. MATH (EleutherAI/hendrycks_math)
    process_and_save("EleutherAI/hendrycks_math", None, "train", "data/math_train.jsonl", "problem", "solution")
    process_and_save("EleutherAI/hendrycks_math", None, "test", "data/math_test.jsonl", "problem", "solution")

    print("\nDone! All datasets processed and saved to ./data/")
    print("Files created:")
    for f in os.listdir("data"):
        if f.endswith(".jsonl"):
            print(f" - data/{f}")

if __name__ == "__main__":
    main()