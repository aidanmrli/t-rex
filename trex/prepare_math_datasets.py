import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

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
    parser = argparse.ArgumentParser(description="Prepare math datasets for SFT/RL.")
    parser.add_argument(
        "--include_prm800k",
        action="store_true",
        help="Include PRM800K processing (explore/format).",
    )
    parser.add_argument(
        "--only_prm800k",
        action="store_true",
        help="Skip other dataset preparation and run only PRM800K stages.",
    )
    parser.add_argument(
        "--prm800k_stage",
        type=str,
        default="explore",
        choices=["explore", "format", "both"],
        help="PRM800K stage: explore schema, format SFT JSONL, or both.",
    )
    parser.add_argument(
        "--prm800k_source",
        type=str,
        default="hf",
        choices=["hf", "git"],
        help="PRM800K source: HuggingFace dataset or local git repo checkout.",
    )
    parser.add_argument(
        "--prm800k_dataset",
        type=str,
        default="tasksource/PRM800K",
        help="HuggingFace dataset name for PRM800K.",
    )
    parser.add_argument(
        "--prm800k_config",
        type=str,
        default=None,
        help="Optional HF config name for PRM800K (e.g., phase2).",
    )
    parser.add_argument(
        "--prm800k_split",
        type=str,
        default="train",
        help="Dataset split for PRM800K when using HF source.",
    )
    parser.add_argument(
        "--prm800k_eval_split",
        type=str,
        default=None,
        help="Optional eval split for PRM800K when using HF source.",
    )
    parser.add_argument(
        "--prm800k_repo_path",
        type=str,
        default=None,
        help="Path to local openai/prm800k repo (required for source=git).",
    )
    parser.add_argument(
        "--prm800k_output",
        type=str,
        default=None,
        help="Output path for PRM800K SFT train JSONL.",
    )
    parser.add_argument(
        "--prm800k_eval_output",
        type=str,
        default=None,
        help="Output path for PRM800K SFT eval JSONL.",
    )
    parser.add_argument(
        "--prm800k_schema_output",
        type=str,
        default=None,
        help="Output path for PRM800K schema JSON.",
    )
    parser.add_argument(
        "--prm800k_samples_output",
        type=str,
        default=None,
        help="Output path for PRM800K sample rows JSONL.",
    )
    parser.add_argument(
        "--prm800k_max_samples",
        type=int,
        default=None,
        help="Limit number of PRM800K records processed (for debugging).",
    )
    parser.add_argument(
        "--prm800k_explore_samples",
        type=int,
        default=5,
        help="Number of PRM800K samples to save during explore stage.",
    )
    parser.add_argument(
        "--prm800k_filter_correct",
        action="store_true",
        help="Keep only samples where pre_generated_answer matches ground_truth_answer.",
    )
    parser.add_argument(
        "--prm800k_use_pre_generated_steps",
        action="store_true",
        help="Prefer question.pre_generated_steps when available.",
    )
    parser.add_argument(
        "--prm800k_streaming",
        action="store_true",
        help="Use HF streaming for PRM800K (HF source only).",
    )
    args = parser.parse_args()

    data_dir = "trex/data"
    os.makedirs(data_dir, exist_ok=True)

    if not args.only_prm800k:
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

    if args.include_prm800k:
        process_prm800k(
            data_dir=data_dir,
            stage=args.prm800k_stage,
            source=args.prm800k_source,
            dataset_name=args.prm800k_dataset,
            dataset_config=args.prm800k_config,
            split=args.prm800k_split,
            eval_split=args.prm800k_eval_split,
            repo_path=args.prm800k_repo_path,
            output_path=args.prm800k_output,
            eval_output_path=args.prm800k_eval_output,
            schema_output=args.prm800k_schema_output,
            samples_output=args.prm800k_samples_output,
            max_samples=args.prm800k_max_samples,
            explore_samples=args.prm800k_explore_samples,
            filter_correct=args.prm800k_filter_correct,
            use_pre_generated_steps=args.prm800k_use_pre_generated_steps,
            streaming=args.prm800k_streaming,
        )

    print(f"\nDone! All datasets processed and saved to {data_dir}/")
    print("Files created:")
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".jsonl"):
            print(f" - {os.path.join(data_dir, f)}")

PRM800K_STEP_HEADER_RE = re.compile(r"^\s*##\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


def _resolve_prm800k_data_dir(repo_path: str) -> Path:
    root = Path(repo_path)
    candidates = [root / "data", root / "prm800k" / "data"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find PRM800K data directory under {repo_path}."
    )


def _iter_prm800k_hf(
    dataset_name: str,
    split: str,
    streaming: bool,
    dataset_config: Optional[str] = None,
) -> Iterable[Dict]:
    try:
        if dataset_config:
            dataset = load_dataset(
                dataset_name, dataset_config, split=split, streaming=streaming
            )
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        return dataset
    except Exception as exc:
        if streaming:
            raise
        # Some PRM800K HF versions fail to cast Arrow types during download.
        # Retry with streaming to avoid dataset materialization.
        print(
            f"[PRM800K] load_dataset failed in non-streaming mode ({exc}). "
            "Retrying with streaming=True."
        )
        if dataset_config:
            dataset = load_dataset(
                dataset_name, dataset_config, split=split, streaming=True
            )
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        return dataset


def _iter_prm800k_git(repo_path: str) -> Iterator[Dict]:
    data_dir = _resolve_prm800k_data_dir(repo_path)
    jsonl_files = sorted(p for p in data_dir.glob("*.jsonl") if p.is_file())
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}.")
    for path in jsonl_files:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _normalize_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    value = str(text)
    value = value.replace("\\boxed{", "").replace("}", "")
    value = value.replace("$", "").replace(",", "").replace("\\!", "")
    value = value.replace("Answer", "").replace("answer", "")
    value = value.strip()
    return value


def _extract_answer_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    answer_match = re.search(r"#\s*Answer\s*[:\n]+\s*([^\n]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    answer_match = re.search(r"\bAnswer\b\s*[:\n]+\s*([^\n]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Fallback: last numeric token.
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
    if numbers:
        return numbers[-1].strip()
    return None


def _strip_step_header(text: str) -> str:
    return PRM800K_STEP_HEADER_RE.sub("", text).strip()


def _join_steps_with_delimiter(steps: List[str]) -> str:
    cleaned = [_strip_step_header(s) for s in steps if s and s.strip()]
    return "\n\n".join(cleaned)


def _normalize_solution_to_delimiter(solution: str) -> str:
    if not solution:
        return ""
    if PRM800K_STEP_HEADER_RE.search(solution):
        parts = PRM800K_STEP_HEADER_RE.split(solution)
        steps = [p.strip() for p in parts if p and p.strip()]
        return "\n\n".join(steps)
    parts = [p.strip() for p in re.split(r"\n{2,}", solution) if p.strip()]
    if not parts:
        return solution.strip()
    return "\n\n".join(parts)


def _extract_prm800k_fields(example: Dict) -> Dict[str, Optional[object]]:
    question = example.get("question") if isinstance(example.get("question"), dict) else {}
    label = example.get("label") if isinstance(example.get("label"), dict) else {}

    def pick(*keys):
        for key in keys:
            if key in example and example[key] not in (None, ""):
                return example[key]
            if key in question and question[key] not in (None, ""):
                return question[key]
            if key in label and label[key] not in (None, ""):
                return label[key]
        return None

    def select_step_text(step: Dict) -> Optional[str]:
        completions = step.get("completions") if isinstance(step.get("completions"), list) else []
        chosen_idx = step.get("chosen_completion")
        chosen_text = None
        if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
            chosen_text = completions[chosen_idx].get("text")
        if not chosen_text:
            human_completion = step.get("human_completion")
            if isinstance(human_completion, dict):
                chosen_text = human_completion.get("text")
            elif isinstance(human_completion, str):
                chosen_text = human_completion
        if not chosen_text and completions:
            completions_sorted = sorted(
                completions,
                key=lambda c: (c.get("rating", 0), bool(c.get("text"))),
                reverse=True,
            )
            chosen_text = completions_sorted[0].get("text") if completions_sorted else None
        if not chosen_text:
            if isinstance(step.get("text"), str):
                chosen_text = step["text"]
        return chosen_text.strip() if isinstance(chosen_text, str) and chosen_text.strip() else None

    pre_generated_steps = None
    label_steps = label.get("steps") if isinstance(label, dict) else None
    if isinstance(label_steps, list) and label_steps:
        extracted_steps: List[str] = []
        for step in label_steps:
            if not isinstance(step, dict):
                continue
            chosen_text = select_step_text(step)
            if chosen_text:
                extracted_steps.append(chosen_text)

        if extracted_steps:
            pre_generated_steps = extracted_steps

    return {
        "problem": pick("problem", "prompt", "question", "instruction", "input"),
        "ground_truth_solution": pick(
            "ground_truth_solution", "solution", "reference_solution"
        ),
        "ground_truth_answer": pick(
            "ground_truth_answer", "answer", "reference_answer"
        ),
        "pre_generated_steps": pre_generated_steps or pick("pre_generated_steps", "steps"),
        "pre_generated_answer": pick("pre_generated_answer"),
    }


def _format_prm800k_example(
    example: Dict,
    filter_correct: bool,
    use_pre_generated_steps: bool,
) -> Optional[Dict[str, str]]:
    fields = _extract_prm800k_fields(example)
    problem = fields["problem"]
    if not problem:
        return None

    steps = fields["pre_generated_steps"] if use_pre_generated_steps else None

    output = None
    if isinstance(steps, list) and steps:
        normalized_steps: List[str] = []
        if any(isinstance(s, dict) for s in steps):
            # Recover chosen completion text if the list still contains step dicts.
            for step in steps:
                if isinstance(step, dict):
                    step_text = None
                    if isinstance(step.get("completions"), list):
                        completions = step["completions"]
                        chosen_idx = step.get("chosen_completion")
                        if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
                            step_text = completions[chosen_idx].get("text")
                    if not step_text and isinstance(step.get("human_completion"), dict):
                        step_text = step["human_completion"].get("text")
                    if not step_text and isinstance(step.get("text"), str):
                        step_text = step["text"]
                    if isinstance(step_text, str) and step_text.strip():
                        normalized_steps.append(step_text.strip())
                elif isinstance(step, str) and step.strip():
                    normalized_steps.append(step.strip())
        else:
            normalized_steps = [str(s).strip() for s in steps if str(s).strip()]

        if normalized_steps:
            output = _join_steps_with_delimiter(normalized_steps)
    elif isinstance(steps, str) and steps.strip():
        output = _normalize_solution_to_delimiter(steps)
    else:
        solution = fields["ground_truth_solution"]
        if isinstance(solution, str) and solution.strip():
            output = _normalize_solution_to_delimiter(solution)

    if not output:
        return None

    if filter_correct:
        pred = _normalize_answer(fields["pre_generated_answer"])
        gold = _normalize_answer(fields["ground_truth_answer"])
        if pred is None:
            if isinstance(steps, list) and steps:
                last_step = steps[-1]
                if isinstance(last_step, dict):
                    last_text = None
                    if isinstance(last_step.get("completions"), list):
                        completions = last_step["completions"]
                        chosen_idx = last_step.get("chosen_completion")
                        if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
                            last_text = completions[chosen_idx].get("text")
                    if not last_text and isinstance(last_step.get("human_completion"), dict):
                        last_text = last_step["human_completion"].get("text")
                    if not last_text and isinstance(last_step.get("text"), str):
                        last_text = last_step["text"]
                    pred = _normalize_answer(_extract_answer_from_text(last_text or ""))
                else:
                    pred = _normalize_answer(_extract_answer_from_text(str(last_step)))
            elif isinstance(output, str) and output:
                pred = _normalize_answer(_extract_answer_from_text(output))
        if pred is not None and gold is not None and pred != gold:
            return None

    answer = fields["ground_truth_answer"] or ""
    return {
        "input": problem,
        "prompt": problem,
        "output": output,
        "answer": answer,
        "label": answer,
    }


def explore_prm800k(
    records: Iterable[Dict],
    schema_output: str,
    samples_output: str,
    max_samples: int,
) -> None:
    columns = set()
    question_columns = set()
    label_columns = set()
    samples: List[Dict] = []
    for idx, row in enumerate(records):
        columns.update(row.keys())
        if isinstance(row.get("question"), dict):
            question_columns.update(row["question"].keys())
        if isinstance(row.get("label"), dict):
            label_columns.update(row["label"].keys())
        if idx < max_samples:
            samples.append(row)
        if idx + 1 >= max_samples:
            break

    schema = {
        "num_samples_seen": len(samples),
        "columns": sorted(columns),
        "question_columns": sorted(question_columns),
        "label_columns": sorted(label_columns),
    }
    with open(schema_output, "w") as f:
        json.dump(schema, f, indent=2)
    with open(samples_output, "w") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")


def _get_prm800k_iter(
    source: str,
    dataset_name: str,
    split: str,
    repo_path: Optional[str],
    streaming: bool,
    dataset_config: Optional[str],
) -> Iterable[Dict]:
    if source == "hf":
        return _iter_prm800k_hf(dataset_name, split, streaming, dataset_config=dataset_config)
    if source == "git":
        if not repo_path:
            raise ValueError("prm800k_repo_path is required when source=git")
        return _iter_prm800k_git(repo_path)
    raise ValueError(f"Unknown PRM800K source: {source}")


def process_prm800k(
    data_dir: str,
    stage: str,
    source: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    eval_split: Optional[str],
    repo_path: Optional[str],
    output_path: Optional[str],
    eval_output_path: Optional[str],
    schema_output: Optional[str],
    samples_output: Optional[str],
    max_samples: Optional[int],
    explore_samples: int,
    filter_correct: bool,
    use_pre_generated_steps: bool,
    streaming: bool,
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    schema_output = schema_output or os.path.join(data_dir, "prm800k_schema.json")
    samples_output = samples_output or os.path.join(data_dir, "prm800k_samples.jsonl")
    output_path = output_path or os.path.join(data_dir, "prm800k_sft_train.jsonl")
    eval_output_path = eval_output_path or os.path.join(data_dir, "prm800k_sft_eval.jsonl")

    if stage in ("explore", "both"):
        records = _get_prm800k_iter(
            source=source,
            dataset_name=dataset_name,
            split=split,
            repo_path=repo_path,
            streaming=streaming,
            dataset_config=dataset_config,
        )
        explore_prm800k(
            records=records,
            schema_output=schema_output,
            samples_output=samples_output,
            max_samples=explore_samples,
        )

    if stage in ("format", "both"):
        records = _get_prm800k_iter(
            source=source,
            dataset_name=dataset_name,
            split=split,
            repo_path=repo_path,
            streaming=streaming,
            dataset_config=dataset_config,
        )
        count = 0
        written = 0
        with open(output_path, "w") as f:
            for row in tqdm(records, desc="PRM800K train"):
                if max_samples is not None and count >= max_samples:
                    break
                count += 1
                formatted = _format_prm800k_example(
                    row, filter_correct=filter_correct, use_pre_generated_steps=use_pre_generated_steps
                )
                if formatted is None:
                    continue
                f.write(json.dumps(formatted) + "\n")
                written += 1
        print(f"PRM800K train written: {written}")

        if eval_split:
            eval_records = _get_prm800k_iter(
                source=source,
                dataset_name=dataset_name,
                split=eval_split,
                repo_path=repo_path,
                streaming=streaming,
                dataset_config=dataset_config,
            )
            count = 0
            written = 0
            with open(eval_output_path, "w") as f:
                for row in tqdm(eval_records, desc="PRM800K eval"):
                    if max_samples is not None and count >= max_samples:
                        break
                    count += 1
                    formatted = _format_prm800k_example(
                        row, filter_correct=filter_correct, use_pre_generated_steps=use_pre_generated_steps
                    )
                    if formatted is None:
                        continue
                    f.write(json.dumps(formatted) + "\n")
                    written += 1
            print(f"PRM800K eval written: {written}")


if __name__ == "__main__":
    main()
