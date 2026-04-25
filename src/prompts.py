import random

from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry


def load_lighteval_tasks(
    task_names: list[str],
    num_prompts_per_task: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load evaluation prompts from lighteval tasks.

    task_names: lighteval format, e.g. ["leaderboard|mmlu:abstract_algebra|5|0"]
    Returns list of dicts with: prompt, target, choices, category, task_name
    """
    registry = Registry(tasks=",".join(task_names), custom_tasks=None)
    task_dict = registry.load_tasks()
    LightevalTask.load_datasets(tasks=task_dict, dataset_loading_processes=1)

    rng = random.Random(seed)
    out: list[dict] = []

    for task_name, task in task_dict.items():
        # Derive category from "suite|task:subset|fewshot|version"
        category = task_name.split("|")[1].split(":")[0]
        docs = list(task.eval_docs())
        if num_prompts_per_task and num_prompts_per_task < len(docs):
            docs = rng.sample(docs, num_prompts_per_task)

        for doc in docs:
            prefix = (doc.instruction + "\n") if doc.instruction else ""
            prompt = prefix + doc.query
            golds = doc.get_golds()
            target = golds[0] if golds else ""
            out.append({
                "prompt": prompt,
                "target": str(target),
                "choices": doc.choices or [],
                "category": category,
                "task_name": task_name,
            })

    return out
