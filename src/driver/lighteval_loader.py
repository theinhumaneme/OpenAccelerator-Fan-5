from __future__ import annotations

import json
import random


def load_lighteval_tasks(
    task_names: list[str],
    num_prompts_per_task: int | None = None,
    seed: int = 42,
) -> list[dict]:
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry

    registry = Registry(tasks=",".join(task_names), custom_tasks=None)
    task_dict = registry.load_tasks()
    LightevalTask.load_datasets(tasks=task_dict, dataset_loading_processes=1)

    rng = random.Random(seed)
    records: list[dict] = []
    for task_name, task in task_dict.items():
        category = task_name.split("|")[1].split(":")[0]
        docs = list(task.eval_docs())
        if num_prompts_per_task and num_prompts_per_task < len(docs):
            docs = rng.sample(docs, num_prompts_per_task)

        for doc in docs:
            prefix = (doc.instruction + "\n") if doc.instruction else ""
            golds = doc.get_golds()
            specific = getattr(doc, "specific", None) or {}
            records.append(
                {
                    "prompt": prefix + doc.query,
                    "target": str(golds[0]) if golds else "",
                    "choices": doc.choices or [],
                    "category": category,
                    "task_name": task_name,
                    "instruction_ids": specific.get("instruction_id_list", []),
                    "instruction_kwargs": _normalize_kwargs(specific.get("kwargs", [])),
                }
            )

    return records


def _normalize_kwargs(raw) -> list[dict]:
    out = []
    for entry in raw or []:
        if not entry:
            out.append({})
            continue
        if isinstance(entry, dict) and not (set(entry) <= {"key", "value"}):
            out.append(dict(entry))
            continue

        items = [entry] if isinstance(entry, dict) else list(entry)
        kwargs = {}
        for pair in items:
            key = pair.get("key", "") if isinstance(pair, dict) else getattr(pair, "key", "")
            value = pair.get("value", "") if isinstance(pair, dict) else getattr(pair, "value", "")
            try:
                kwargs[key] = json.loads(value)
            except Exception:
                kwargs[key] = value
        out.append(kwargs)
    return out
