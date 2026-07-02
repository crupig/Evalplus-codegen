# .venv/lib/python3.12/site-packages/evalplus/codegen.py
import json
import os
from typing import Dict, List, Optional

from evalplus.data import get_evalperf_data, get_human_eval_plus, get_mbpp_plus
from evalplus.provider import DecoderBase, make_model
from evalplus.sanitize import sanitize
from evalplus.utils import progress


def codegen(
    target_path: str,
    model: DecoderBase,
    model_name: str,
    dataset: Dict,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
):
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        os.makedirs(target_path, exist_ok=True)

    print(f"Sanitized code outputs will be saved to {target_path}")
    print(f"Raw outputs will be saved to {raw_target_path}")

    backend_type: str = type(model).__name__
    with progress(backend_type) as p:
        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                outputs, probs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for sample_idx, impl, prob in zip(range(len(outputs)), outputs, probs):
                    solution = prompt + impl if model.is_direct_completion() else impl
                    sanitized_solution = sanitize(
                        solution, entrypoint=task["entry_point"]
                    )
                    
                    if 'gpt-' not in model_name:
                        # post processing log probabilities
                        impl_wo_empty_lines = "\n".join([line for line in impl.splitlines() if line.strip() != ""])
                        sanitized_solution_wo_empty_lines = "\n".join([line for line in sanitized_solution.splitlines() if line.strip() != ""])
                        
                        if sanitized_solution_wo_empty_lines in impl_wo_empty_lines:
                            start_idx = impl.index(sanitized_solution_wo_empty_lines.splitlines()[0])
                            num_of_tokens_before_sanitized_solution = len(model.tokenizer.encode(impl[:start_idx]))
                            num_of_tokens_of_sanitized_solution = len(model.tokenizer.encode(sanitized_solution))
                            prob = prob[num_of_tokens_before_sanitized_solution:
                                        num_of_tokens_before_sanitized_solution + num_of_tokens_of_sanitized_solution]
                        
                    solution_idx = "Python--MBPP--TaskID::{0}--GeneratedBy::{1}--SampleID::{2:02d}"\
                                            .format(task_id, model_name.split("/")[-1], sample_idx)
                    if target_path.endswith(".jsonl"):
                        # Writing the sanitized version
                        with open(target_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "task_id": task_id,
                                        "sample_idx": sample_idx,
                                        "generated_by" : model_name.split("/")[-1],
                                        "prompt": prompt,
                                        "solution": sanitized_solution,
                                        "raw_output": impl,
                                        "log_probabilities": prob,
                                        "solution_idx": solution_idx,
                                    }
                                )
                                + "\n"
                            )

                        # Writing the raw version
                        with open(raw_target_path, "a") as f:
                            f.write(
                                json.dumps({"task_id": task_id, "solution": solution})
                                + "\n"
                            )
                    else:
                        # Writing the sanitized version
                        with open(
                            os.path.join(target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(sanitized_solution)

                        # Writing the raw version
                        with open(
                            os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(solution)
                    sidx += 1


def run_codegen(
    model: str,
    dataset: str,
    root: str = "evalplus_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    ids_subset: List[str] = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    tp: int = 1,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
    attn_implementation: str = "eager",
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    dtype: str = "bfloat16",
    gptqmodel_backend: str = "auto",  # For GPTQModel
    gguf_file: Optional[str] = None,
):
    assert dataset in ["humaneval", "mbpp", "evalperf"], f"Invalid dataset {dataset}"
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]

    # Make dir for codes generated by each model
    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"

    # target_path = os.path.join(root, dataset, identifier)
    target_path = os.path.join(root, identifier)
    if jsonl_fmt:
        # target_path += ".jsonl"
        target_path += f"-{os.getpid()}.jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)

    if dataset == "humaneval":
        dataset_dict = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        dataset_dict = get_mbpp_plus(version=version)
    elif dataset == "evalperf":
        original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        dataset_dict = {k: original_dataset[k] for k in get_evalperf_data()}
        assert id_range is None, "id_range not supported for evalperf"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    all_tasks_complete = False
    if jsonl_fmt and os.path.isfile(target_path):
        task_counts = {}
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

            all_tasks_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )

    if all_tasks_complete:
        print("All samples are already cached. Skipping codegen.")
        return target_path

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    # Make project dir
    os.makedirs(root, exist_ok=True)

    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    if evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    
    if 'gpt-' not in model:
        assert bs == 1, "Batch size must be 1 for correct log probabilities calculation. (The developer of the script is either lazy or not skilled enough to handle batch size > 1 logprobs extraction!)"
    # Model creation
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        dtype=dtype,
        gptqmodel_backend=gptqmodel_backend,
        gguf_file=gguf_file,
    )

    if ids_subset is not None:
        print(f"\nRunning code generation on a subset of {len(ids_subset)} tasks.\n")
        dataset_dict = {k: dataset_dict[k] for k in ids_subset if k in dataset_dict}

    codegen(
        target_path=target_path,
        dataset=dataset_dict,
        greedy=greedy,
        model=model_runner,
        model_name=model,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
    )

    # force shutdown the model runner
    del model_runner
    import gc

    gc.collect()

    return target_path


def main():
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
