import re
from collections.abc import Mapping
from typing import Any

import httpx
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read either result.name or result['name']; useful across dataclasses/dicts."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cmp(actual: int | float, target: int | float, relation: str) -> bool:
    relation = str(relation or "at least").strip().lower()
    return {
        "at least": actual >= target,
        "at most": actual <= target,
        "less than": actual < target,
        "more than": actual > target,
        "exactly": actual == target,
    }.get(relation, actual >= target)


# ---------------------------------------------------------------------------
# Prompt validation
# ---------------------------------------------------------------------------

def validate_prompt(prompt: str) -> list[str]:
    """Return warnings for benchmark prompt issues that can corrupt scoring.

    This deliberately returns warnings instead of raising so callers can aggregate
    issues across a full dataset. Treat any returned warning as a failed validation.
    """
    warnings: list[str] = []
    prompt = prompt or ""

    mcq_marker = "Answer the following multiple choice question."
    if prompt.count(mcq_marker) > 1:
        warnings.append("prompt appears duplicated: MCQ instruction marker occurs more than once")

    if re.search(r"where\s+LETTER\s+is\s+one\s+of\s+ABCD", prompt, re.I) and re.search(
        r"(?m)^\s*[E-Z]\s*:", prompt
    ):
        warnings.append("prompt says LETTER is one of ABCD, but choices include letters beyond D")

    answer_markers = len(re.findall(r"(?im)^\s*Answer\s*:\s*$", prompt))
    if answer_markers > 1:
        warnings.append("prompt contains more than one empty 'Answer:' marker")

    return warnings


def validate_results(results: list) -> dict[str, Any]:
    """Validate prompts in a result list and return a compact report."""
    bad: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        issues = validate_prompt(str(_get(r, "prompt", "")))
        if issues:
            bad.append({"index": i, "issues": issues})
    return {"n_results": len(results), "n_bad_prompts": len(bad), "bad_prompts": bad}


# ---------------------------------------------------------------------------
# IFEval instruction checker
# ---------------------------------------------------------------------------

def _check_instruction(prompt: str, output: str, instruction_id: str, kwargs: dict) -> bool:
    """Return True if output satisfies an IFEval instruction.

    Unknown instruction types fail closed instead of passing by default. That is
    important for benchmark integrity: unsupported instructions should not inflate
    scores silently.
    """
    ns, _, name = str(instruction_id or "").partition(":")
    kwargs = kwargs or {}
    output = output or ""
    prompt = prompt or ""

    if ns == "length_constraints":
        if name == "number_words":
            return _cmp(len(output.split()), int(kwargs.get("num_words", 0)), kwargs.get("relation", "at least"))
        if name == "number_sentences":
            count = len([s for s in re.split(r"(?<=[.!?])\s+", output.strip()) if s])
            return _cmp(count, int(kwargs.get("num_sentences", 0)), kwargs.get("relation", "at least"))
        if name == "number_paragraphs":
            count = len([p for p in output.split("\n\n") if p.strip()])
            return _cmp(count, int(kwargs.get("num_paragraphs", 0)), kwargs.get("relation", "at least"))
        if name == "nth_paragraph_first_word":
            paras = [p.strip() for p in output.split("\n\n") if p.strip()]
            nth = int(kwargs.get("nth_paragraph", 1)) - 1
            first_word = str(kwargs.get("first_word", "")).lower()
            return nth < len(paras) and bool(paras[nth].split()) and paras[nth].split()[0].lower().rstrip(".,!?;:") == first_word

    elif ns == "keywords":
        if name == "existence":
            return all(str(kw).lower() in output.lower() for kw in (kwargs.get("keywords") or []))
        if name == "forbidden_words":
            return all(str(fw).lower() not in output.lower() for fw in (kwargs.get("forbidden_words") or []))
        if name == "frequency":
            kw = str(kwargs.get("keyword", "")).lower()
            return _cmp(output.lower().count(kw), int(kwargs.get("frequency", 0)), kwargs.get("relation", "at least"))
        if name == "letter_frequency":
            letter = str(kwargs.get("letter", "")).lower()
            return _cmp(output.lower().count(letter), int(kwargs.get("let_frequency", 0)), kwargs.get("let_relation", "at least"))

    elif ns == "detectable_format":
        if name == "number_bullet_lists":
            # Count contiguous bullet-list blocks, not individual bullet items.
            lines = output.splitlines()
            blocks = 0
            in_block = False
            for line in lines:
                is_bullet = bool(re.match(r"^\s*[-*•]\s+", line))
                if is_bullet and not in_block:
                    blocks += 1
                in_block = is_bullet
            return _cmp(blocks, int(kwargs.get("num_bullet_lists", kwargs.get("num_bullets", 0))), kwargs.get("relation", "at least"))
        if name in ("number_highlighted_sections", "constrain_highlighted_sections"):
            highlights = len(re.findall(r"\*[^*\n]+\*", output))
            n = int(kwargs.get("num_highlights", 0))
            return highlights == n if name == "constrain_highlighted_sections" else highlights >= n
        if name == "multiple_sections":
            splitter = kwargs.get("section_spliter", kwargs.get("section_splitter", "Section"))
            return output.count(str(splitter)) >= int(kwargs.get("num_sections", 2))

    elif ns == "detectable_content":
        if name == "number_placeholders":
            return len(re.findall(r"\[[^\]]+\]", output)) >= int(kwargs.get("num_placeholders", 1))
        if name == "postscript":
            return str(kwargs.get("postscript_marker", "P.S.")) in output

    elif ns == "startend":
        if name == "end_checker":
            return output.strip().lower().endswith(str(kwargs.get("end_phrase", "")).lower().strip())
        if name == "quotation":
            s = output.strip()
            return s.startswith('"') and s.endswith('"')

    elif ns == "change_case":
        words = re.findall(r"\b[A-Za-z]+\b", output)
        if name == "english_capital":
            return bool(words) and all(w.isupper() for w in words)
        if name == "english_lowercase":
            return bool(words) and all(w.islower() for w in words)
        if name == "capital_word_frequency":
            caps = sum(1 for w in words if w.isupper())
            return _cmp(caps, int(kwargs.get("capital_frequency", 0)), kwargs.get("capital_relation", "at least"))

    elif ns == "punctuation":
        if name == "no_comma":
            return "," not in output

    elif ns == "combination":
        if name == "two_responses":
            return "******" in output
        if name == "repeat_prompt":
            return prompt.strip().lower() in output.lower()

    print(f"  [warn] Unsupported IFEval instruction: {instruction_id}")
    return False


def score_ifeval(results: list) -> dict:
    """IFEval prompt-level and instruction-level accuracy."""
    inst_pass = inst_total = prompt_pass = prompt_total = 0
    unsupported_or_failed = []

    for idx, r in enumerate(results):
        instruction_ids = _get(r, "instruction_ids", None)
        if not instruction_ids:
            continue

        prompt_total += 1
        all_pass = True
        kw_list = _get(r, "instruction_kwargs", None) or [{}] * len(instruction_ids)
        for iid, kw in zip(instruction_ids, kw_list):
            passed = _check_instruction(str(_get(r, "prompt", "")), str(_get(r, "output", "")), iid, kw or {})
            inst_pass += int(passed)
            inst_total += 1
            if not passed:
                all_pass = False
                unsupported_or_failed.append({"index": idx, "instruction_id": iid})
        prompt_pass += int(all_pass)

    if prompt_total == 0 and results:
        print("  [warn] score_ifeval: no instruction_ids found — doc.specific may not contain IFEval metadata")

    return {
        "prompt_accuracy": prompt_pass / prompt_total if prompt_total else 0.0,
        "instruction_accuracy": inst_pass / inst_total if inst_total else 0.0,
        "n_prompts": prompt_total,
        "n_instructions": inst_total,
        "failed_instructions": unsupported_or_failed,
    }


# ---------------------------------------------------------------------------
# MCQ / generation accuracy
# ---------------------------------------------------------------------------

def extract_answer_letter(output: str, valid_letters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str | None:
    """Extract a multiple-choice answer from model output.

    Preference order:
    1. Last explicit `Answer: X` line/span.
    2. Last standalone `answer is X` span.
    3. First standalone letter at the very beginning of output.
    """
    output = output or ""
    letters = re.escape(valid_letters)

    matches = re.findall(rf"(?i)\banswer\s*:\s*([{letters}])\b", output)
    if matches:
        return matches[-1].upper()

    matches = re.findall(rf"(?i)\banswer\s+(?:is|would be)\s*([{letters}])\b", output)
    if matches:
        return matches[-1].upper()

    m = re.search(rf"^\s*([{letters}])\b", output, re.I)
    if m:
        return m.group(1).upper()

    return None


def _norm_text(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _prefix_accuracy(results: list) -> dict:
    """Accuracy for MC and generation tasks.

    Single-char targets are parsed as multiple-choice answers using `Answer: X`
    when available. Multi-char targets are normalized and searched in full output.
    """
    correct = total = 0
    parse_failures: list[int] = []

    for idx, r in enumerate(results):
        target = _get(r, "target", None)
        if not target:
            continue
        nt = _norm_text(target)
        if not nt:
            continue

        total += 1
        output = str(_get(r, "output", ""))
        if len(nt) == 1 and nt.isalpha():
            pred = extract_answer_letter(output)
            if pred is None:
                parse_failures.append(idx)
            correct += int(pred == str(target).strip().upper())
        else:
            correct += int(nt in _norm_text(output))

    return {
        "accuracy": correct / total if total else 0.0,
        "n_scored": total,
        "n_parse_failures": len(parse_failures),
        "parse_failure_indices": parse_failures[:25],
    }


def compute_task_accuracy(results: list) -> dict:
    """Dispatch to IFEval scorer for IFEval results; answer-parse for everything else."""
    ifeval = [r for r in results if "ifeval" in str(_get(r, "category", "") or "").lower()]
    other = [r for r in results if "ifeval" not in str(_get(r, "category", "") or "").lower()]

    out: dict = {}
    validation = validate_results(results)
    if validation["n_bad_prompts"]:
        out["prompt_validation"] = validation

    if ifeval:
        out["ifeval"] = score_ifeval(ifeval)
        out["accuracy"] = out["ifeval"]["prompt_accuracy"]
    if other:
        pa = _prefix_accuracy(other)
        out["other_accuracy"] = pa["accuracy"]
        out["n_other_scored"] = pa["n_scored"]
        out["n_answer_parse_failures"] = pa["n_parse_failures"]
        if pa["parse_failure_indices"]:
            out["answer_parse_failure_indices"] = pa["parse_failure_indices"]
        if "accuracy" not in out:
            out["accuracy"] = pa["accuracy"]
    return out


# ---------------------------------------------------------------------------
# vLLM metrics helpers
# ---------------------------------------------------------------------------

def _parse_prometheus_samples(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$", line)
        if not m:
            continue
        name, value = m.group(1), float(m.group(2))
        samples[name] = samples.get(name, 0.0) + value
    return samples


def fetch_acceptance_rate(metrics_url: str, debug: bool = False) -> float | None:
    """Fetch speculative acceptance rate from a Prometheus metrics endpoint.

    Supports either a direct gauge containing acceptance_rate, or counter-style
    metrics where acceptance can be estimated as accepted/proposed draft tokens.
    Set debug=True to print speculative/draft/accept metric lines.
    """
    try:
        resp = httpx.get(metrics_url, timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Could not reach metrics endpoint: {e}")
        return None

    text = resp.text
    if debug:
        for line in text.splitlines():
            lower = line.lower()
            if "spec" in lower or "draft" in lower or "accept" in lower:
                print(line)

    # First prefer any direct acceptance-rate/efficiency gauge.
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        lower = line.lower()
        if ("acceptance_rate" in lower or "acceptance rate" in lower or "spec_decode_efficiency" in lower) and not lower.startswith("#"):
            m = re.search(r"\s([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", line)
            if m:
                return float(m.group(1))

    samples = _parse_prometheus_samples(text)
    accepted = 0.0
    proposed = 0.0
    emitted = 0.0
    draft = 0.0

    for name, value in samples.items():
        lname = name.lower()
        if "spec" not in lname and "draft" not in lname:
            continue
        if "accepted" in lname:
            accepted += value
        if any(k in lname for k in ("proposed", "draft", "scored")):
            draft += value
        if any(k in lname for k in ("proposed", "draft")):
            proposed += value
        if "emitted" in lname or "output" in lname:
            emitted += value

    denom = proposed or draft or emitted
    if accepted > 0 and denom > 0:
        return accepted / denom
    return None


# ---------------------------------------------------------------------------
# Token confidence buckets and aggregate stats
# ---------------------------------------------------------------------------

def bucket_by_token_confidence(token_logprobs: list[float], n_buckets: int = 4) -> dict[str, float]:
    """Bucket tokens by log-probability.

    Low logprob = low confidence. This is token confidence, not task difficulty.
    Returns mean logprob per bucket with non-overlapping bucket boundaries.
    """
    if not token_logprobs:
        return {}
    arr = np.array([x for x in token_logprobs if x is not None and np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {}

    thresholds = np.percentile(arr, np.linspace(0, 100, n_buckets + 1))
    labels = ["very_low_confidence", "low_confidence", "high_confidence", "very_high_confidence"]
    buckets: dict[str, float] = {}

    for i in range(n_buckets):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == n_buckets - 1:
            mask = (arr >= lo) & (arr <= hi)
        else:
            mask = (arr >= lo) & (arr < hi)
        label = labels[i] if i < len(labels) else f"bucket_{i}"
        buckets[label] = float(np.mean(arr[mask])) if mask.any() else 0.0
    return buckets


# Backward-compatible name. Prefer bucket_by_token_confidence in new code.
def bucket_by_difficulty(token_logprobs: list[float], n_buckets: int = 4) -> dict[str, float]:
    return bucket_by_token_confidence(token_logprobs, n_buckets=n_buckets)


def _finite_array(values: list[Any]) -> np.ndarray:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return np.array(out, dtype=float)


def aggregate(results: list) -> dict[str, float]:
    if not results:
        return {}

    ttfts = _finite_array([_get(r, "ttft_ms", None) for r in results])
    totals = _finite_array([_get(r, "total_ms", None) for r in results])
    tps = _finite_array([_get(r, "throughput_tps", None) for r in results])

    def stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {
                f"mean_{prefix}": float("nan"),
                f"p50_{prefix}": float("nan"),
                f"p95_{prefix}": float("nan"),
            }
        return {
            f"mean_{prefix}": float(np.mean(arr)),
            f"p50_{prefix}": float(np.percentile(arr, 50)),
            f"p95_{prefix}": float(np.percentile(arr, 95)),
        }

    out: dict[str, float] = {"n": len(results)}
    out.update(stats("ttft_ms", ttfts))
    out.update(stats("total_ms", totals))
    out.update(stats("throughput_tps", tps))
    out["n_valid_ttft"] = int(ttfts.size)
    out["n_valid_total"] = int(totals.size)
    out["n_valid_throughput"] = int(tps.size)
    return out
