import json
import random

from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry


def _normalize_kwargs(raw) -> list[dict]:
    """Convert HuggingFace {key,value} pair sequences or plain dicts to list[dict]."""
    out = []
    for entry in (raw or []):
        if not entry:
            out.append({})
            continue
        if isinstance(entry, dict) and not (set(entry) <= {"key", "value"}):
            out.append(dict(entry))
            continue
        # HuggingFace may store each instruction's kwargs as a sequence of {key,value} pairs
        items = [entry] if isinstance(entry, dict) else list(entry)
        d = {}
        for pair in items:
            k = pair.get("key", "") if isinstance(pair, dict) else getattr(pair, "key", "")
            v = pair.get("value", "") if isinstance(pair, dict) else getattr(pair, "value", "")
            try:
                d[k] = json.loads(v)
            except Exception:
                d[k] = v
        out.append(d)
    return out


def load_lighteval_tasks(
    task_names: list[str],
    num_prompts_per_task: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load evaluation prompts from lighteval tasks.

    task_names: lighteval format, e.g. ["extended|ifeval|0|0", "lighteval|gsm8k|5|0"]
    Returns list of dicts with: prompt, target, choices, category, task_name,
    instruction_ids, instruction_kwargs (populated for IFEval, empty otherwise).
    """
    registry = Registry(tasks=",".join(task_names), custom_tasks=None)
    task_dict = registry.load_tasks()
    LightevalTask.load_datasets(tasks=task_dict, dataset_loading_processes=1)

    rng = random.Random(seed)
    out: list[dict] = []

    for task_name, task in task_dict.items():
        category = task_name.split("|")[1].split(":")[0]
        docs = list(task.eval_docs())
        if num_prompts_per_task and num_prompts_per_task < len(docs):
            docs = rng.sample(docs, num_prompts_per_task)

        for doc in docs:
            prefix = (doc.instruction + "\n") if doc.instruction else ""
            prompt = prefix + doc.query
            golds = doc.get_golds()
            target = golds[0] if golds else ""

            # IFEval stores instruction metadata in doc.specific
            specific = getattr(doc, "specific", None) or {}
            instruction_ids = specific.get("instruction_id_list", [])
            instruction_kwargs = _normalize_kwargs(specific.get("kwargs", []))

            out.append({
                "prompt": prompt,
                "target": str(target),
                "choices": doc.choices or [],
                "category": category,
                "task_name": task_name,
                "instruction_ids": instruction_ids,
                "instruction_kwargs": instruction_kwargs,
            })

    return out
