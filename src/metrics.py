import re
import httpx
import numpy as np


# ---------------------------------------------------------------------------
# IFEval instruction checker
# ---------------------------------------------------------------------------

def _cmp(actual: int, target: int, relation: str) -> bool:
    return {
        "at least": actual >= target,
        "at most": actual <= target,
        "less than": actual < target,
        "more than": actual > target,
        "exactly": actual == target,
    }.get(relation, actual >= target)


def _check_instruction(prompt: str, output: str, instruction_id: str, kwargs: dict) -> bool:
    """Return True if the model output satisfies the given IFEval instruction."""
    ns, _, name = instruction_id.partition(":")

    if ns == "length_constraints":
        if name == "number_words":
            return _cmp(len(output.split()), int(kwargs.get("num_words", 0)), kwargs.get("relation", "at least"))
        if name == "number_sentences":
            count = len([s for s in re.split(r"(?<=[.!?])\s+", output.strip()) if s])
            return _cmp(count, int(kwargs.get("num_sentences", 0)), kwargs.get("relation", "at least"))
        if name == "number_paragraphs":
            count = len([p for p in output.split("\n\n") if p.strip()])
            return count >= int(kwargs.get("num_paragraphs", 0))
        if name == "nth_paragraph_first_word":
            paras = [p.strip() for p in output.split("\n\n") if p.strip()]
            nth = int(kwargs.get("nth_paragraph", 1)) - 1
            first_word = str(kwargs.get("first_word", "")).lower()
            return nth < len(paras) and paras[nth].split()[0].lower().rstrip(".,!?;:") == first_word

    elif ns == "keywords":
        if name == "existence":
            return all(kw.lower() in output.lower() for kw in (kwargs.get("keywords") or []))
        if name == "forbidden_words":
            return all(fw.lower() not in output.lower() for fw in (kwargs.get("forbidden_words") or []))
        if name == "frequency":
            kw = str(kwargs.get("keyword", "")).lower()
            return _cmp(output.lower().count(kw), int(kwargs.get("frequency", 0)), kwargs.get("relation", "at least"))
        if name == "letter_frequency":
            letter = str(kwargs.get("letter", "")).lower()
            return _cmp(output.lower().count(letter), int(kwargs.get("let_frequency", 0)), kwargs.get("let_relation", "at least"))

    elif ns == "detectable_format":
        if name == "number_bullet_lists":
            bullets = len(re.findall(r"^\s*[-*•]\s", output, re.MULTILINE))
            return _cmp(bullets, int(kwargs.get("num_bullets", 0)), "at least")
        if name in ("number_highlighted_sections", "constrain_highlighted_sections"):
            highlights = len(re.findall(r"\*[^*\n]+\*", output))
            n = int(kwargs.get("num_highlights", 0))
            return highlights == n if name == "constrain_highlighted_sections" else highlights >= n
        if name == "multiple_sections":
            splitter = kwargs.get("section_spliter", "Section")
            return output.count(splitter) >= int(kwargs.get("num_sections", 2))

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

    return True  # unknown instruction type: pass by default


def score_ifeval(results: list) -> dict:
    """IFEval prompt-level and instruction-level accuracy."""
    inst_pass = inst_total = prompt_pass = prompt_total = 0
    for r in results:
        if not r.instruction_ids:
            continue
        prompt_total += 1
        all_pass = True
        kw_list = r.instruction_kwargs or [{}] * len(r.instruction_ids)
        for iid, kw in zip(r.instruction_ids, kw_list):
            passed = _check_instruction(r.prompt, r.output, iid, kw or {})
            inst_pass += int(passed)
            inst_total += 1
            if not passed:
                all_pass = False
        prompt_pass += int(all_pass)
    if prompt_total == 0 and results:
        print("  [warn] score_ifeval: no instruction_ids found — doc.specific may not contain IFEval metadata")
    return {
        "prompt_accuracy": prompt_pass / prompt_total if prompt_total else 0.0,
        "instruction_accuracy": inst_pass / inst_total if inst_total else 0.0,
        "n_prompts": prompt_total,
        "n_instructions": inst_total,
    }


def _prefix_accuracy(results: list) -> dict:
    """Accuracy for MC and generation tasks.

    Single-char targets (e.g. MMLU-Pro A/B/C/D): match at the start of output.
    Multi-char targets (e.g. GSM8K numbers): search the full output, since
    reasoning chains put the answer at the end.
    """
    correct = total = 0
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    for r in results:
        if not r.target:
            continue
        total += 1
        nt = norm(r.target)
        if not nt:
            continue
        no = norm(r.output)
        if len(nt) == 1:
            correct += int(no.startswith(nt))
        else:
            correct += int(nt in no)
    return {"accuracy": correct / total if total else 0.0, "n_scored": total}


def compute_task_accuracy(results: list) -> dict:
    """Dispatch to IFEval scorer for IFEval results; prefix-match for everything else."""
    ifeval = [r for r in results if "ifeval" in (r.category or "")]
    other = [r for r in results if "ifeval" not in (r.category or "")]

    out: dict = {}
    if ifeval:
        out["ifeval"] = score_ifeval(ifeval)
        out["accuracy"] = out["ifeval"]["prompt_accuracy"]
    if other:
        pa = _prefix_accuracy(other)
        out["other_accuracy"] = pa["accuracy"]
        out["n_other_scored"] = pa["n_scored"]
        if "accuracy" not in out:
            out["accuracy"] = pa["accuracy"]
    return out


def fetch_acceptance_rate(metrics_url: str) -> float | None:
    try:
        resp = httpx.get(metrics_url, timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] Could not reach metrics endpoint: {e}")
        return None

    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        if "acceptance_rate" in line.lower():
            m = re.search(r"[\s}]([\d.eE+\-]+)\s*$", line)
            if m:
                return float(m.group(1))
    return None


def bucket_by_difficulty(token_logprobs: list[float], n_buckets: int = 4) -> dict[str, float]:
    """
    Bucket tokens by their log-probability (proxy for difficulty).
    High logprob = model is confident = easy token.
    Returns the mean logprob per bucket.
    """
    if not token_logprobs:
        return {}
    arr = np.array(token_logprobs, dtype=float)
    # Sort ascending so bucket 0 = hardest (most negative logprob)
    thresholds = np.percentile(arr, np.linspace(0, 100, n_buckets + 1))
    labels = ["very_hard", "hard", "easy", "very_easy"]
    buckets: dict[str, float] = {}
    for i in range(n_buckets):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (arr >= lo) & (arr <= hi)
        label = labels[i] if i < len(labels) else f"bucket_{i}"
        buckets[label] = float(np.mean(arr[mask])) if mask.any() else 0.0
    return buckets


def aggregate(results: list) -> dict[str, float]:
    if not results:
        return {}
    ttfts = np.array([r.ttft_ms for r in results])
    totals = np.array([r.total_ms for r in results])
    tps = np.array([r.throughput_tps for r in results])
    return {
        "n": len(results),
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p50_ttft_ms": float(np.percentile(ttfts, 50)),
        "p95_ttft_ms": float(np.percentile(ttfts, 95)),
        "mean_total_ms": float(np.mean(totals)),
        "mean_throughput_tps": float(np.mean(tps)),
        "p50_throughput_tps": float(np.percentile(tps, 50)),
    }
