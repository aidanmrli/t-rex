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

def format_math500_example(example):
    """Format MATH-500 example which already has answer extracted."""
    return {
        "input": example["problem"],
        "prompt": example["problem"],
        "output": example["solution"],
        "answer": example["answer"],
        "label": example["answer"],
        # MATH-500 has additional metadata
        "subject": example.get("subject", ""),
        "level": example.get("level", ""),
        "unique_id": example.get("unique_id", "")
    }

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

def process_and_save(dataset_name, subset_names, split, output_path, question_key, solution_key):
    if subset_names is None:
        subset_names = [None]
    elif isinstance(subset_names, str):
        subset_names = [subset_names]
        
    print(f"Processing {dataset_name} {split}...")
    try:
        with open(output_path, "w") as f:
            for subset in subset_names:
                if subset:
                    print(f"  - Loading subset: {subset}")
                    ds = load_dataset(dataset_name, subset, split=split)
                else:
                    ds = load_dataset(dataset_name, split=split)
                    
                for ex in tqdm(ds, desc=f"Subset {subset}" if subset else "Dataset"):
                    formatted = format_example(ex[question_key], ex[solution_key])
                    f.write(json.dumps(formatted) + "\n")
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

def process_math500(output_path):
    """Process MATH-500 dataset which has pre-extracted answers."""
    print("Processing HuggingFaceH4/MATH-500 test...")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        with open(output_path, "w") as f:
            for ex in tqdm(ds, desc="MATH-500"):
                formatted = format_math500_example(ex)
                f.write(json.dumps(formatted) + "\n")
        print(f"  - Saved {len(ds)} problems to {output_path}")
    except Exception as e:
        print(f"Error processing MATH-500: {e}")

def main():
    data_dir = "trex/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. GSM8K (openai/gsm8k)
    process_and_save("openai/gsm8k", "main", "train", os.path.join(data_dir, "gsm8k_train.jsonl"), "question", "answer")
    process_and_save("openai/gsm8k", "main", "test", os.path.join(data_dir, "gsm8k_test.jsonl"), "question", "answer")
    
    # 2. GSM8K Platinum (madrylab/gsm8k-platinum) - Test only
    process_and_save("madrylab/gsm8k-platinum", None, "test", os.path.join(data_dir, "gsm8k_platinum_test.jsonl"), "question", "answer")
    
    # 3. MATH (EleutherAI/hendrycks_math)
    math_subsets = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    process_and_save("EleutherAI/hendrycks_math", math_subsets, "train", os.path.join(data_dir, "math_train.jsonl"), "problem", "solution")
    process_and_save("EleutherAI/hendrycks_math", math_subsets, "test", os.path.join(data_dir, "math_test.jsonl"), "problem", "solution")
    
    # 4. MATH-500 (HuggingFaceH4/MATH-500) - Curated 500-problem subset from OpenAI's PRM800K
    # See: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    process_math500(os.path.join(data_dir, "math500_test.jsonl"))

    print(f"\nDone! All datasets processed and saved to {data_dir}/")
    print("Files created:")
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".jsonl"):
            print(f" - {os.path.join(data_dir, f)}")

if __name__ == "__main__":
    main()