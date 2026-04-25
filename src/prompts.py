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


def _get_task_dict(task_names: list[str]) -> dict:
    """
    Build a task dict that works across lighteval versions.

    lighteval >= 0.10: Registry(tasks=...).load_tasks()
    lighteval 0.6-0.9: Registry(lighteval_tasks=None).get_task_dict(task_names)
    """
    task_str = ",".join(task_names)

    # Try new API first (0.10+)
    try:
        registry = Registry(tasks=task_str, custom_tasks=None)
        return registry.load_tasks()
    except TypeError:
        pass

    # Fallback: older API
    try:
        registry = Registry(lighteval_tasks=None, custom_tasks=None)
        return registry.get_task_dict(task_names)
    except Exception as e:
        # Surface a helpful message listing what is available
        try:
            registry = Registry(lighteval_tasks=None, custom_tasks=None)
            available = sorted(registry.TASK_TABLE.keys())[:40]
            raise RuntimeError(
                f"Could not load tasks {task_names}.\n"
                f"First 40 available tasks in this lighteval install:\n"
                + "\n".join(f"  {t}" for t in available)
            ) from e
        except AttributeError:
            raise RuntimeError(
                f"Could not load tasks {task_names} and could not list available tasks. "
                f"Original error: {e}"
            ) from e


def _task_category(task_name: str) -> str:
    """
    Extract a human-readable category from a lighteval task name.

    Handles both formats:
      "suite|task:subset|fewshot|version"  (4-part) -> "task"
      "task:subset|fewshot"                (2-part) -> "task"
    """
    parts = task_name.split("|")
    segment = parts[1] if len(parts) >= 3 else parts[0]
    return segment.split(":")[0]


def load_lighteval_tasks(
    task_names: list[str],
    num_prompts_per_task: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load evaluation prompts from lighteval tasks.

    task_names: lighteval format, e.g. ["ifeval|0", "gsm8k|5"] or
                4-part format ["leaderboard|ifeval|0|0", "leaderboard|gsm8k|5|0"]
    Returns list of dicts with: prompt, target, choices, category, task_name,
    instruction_ids, instruction_kwargs (populated for IFEval, empty otherwise).
    """
    task_dict = _get_task_dict(task_names)
    LightevalTask.load_datasets(tasks=task_dict, dataset_loading_processes=1)

    rng = random.Random(seed)
    out: list[dict] = []

    for task_name, task in task_dict.items():
        category = _task_category(task_name)
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
