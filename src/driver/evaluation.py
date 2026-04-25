from __future__ import annotations

import re


def compute_task_accuracy(results: list) -> dict:
    ifeval = [result for result in results if "ifeval" in (result.category or "")]
    other = [result for result in results if "ifeval" not in (result.category or "")]

    output: dict = {}
    if ifeval:
        output["ifeval"] = score_ifeval(ifeval)
        output["accuracy"] = output["ifeval"]["prompt_accuracy"]
    if other:
        prefix = _prefix_accuracy(other)
        output["other_accuracy"] = prefix["accuracy"]
        output["n_other_scored"] = prefix["n_scored"]
        if "accuracy" not in output:
            output["accuracy"] = prefix["accuracy"]
    return output


def score_ifeval(results: list) -> dict:
    inst_pass = inst_total = prompt_pass = prompt_total = 0
    for result in results:
        if not result.instruction_ids:
            continue
        prompt_total += 1
        all_pass = True
        kwargs_list = result.instruction_kwargs or [{}] * len(result.instruction_ids)
        for instruction_id, kwargs in zip(result.instruction_ids, kwargs_list):
            passed = _check_instruction(result.prompt, result.output, instruction_id, kwargs or {})
            inst_pass += int(passed)
            inst_total += 1
            if not passed:
                all_pass = False
        prompt_pass += int(all_pass)

    return {
        "prompt_accuracy": prompt_pass / prompt_total if prompt_total else 0.0,
        "instruction_accuracy": inst_pass / inst_total if inst_total else 0.0,
        "n_prompts": prompt_total,
        "n_instructions": inst_total,
    }


def _prefix_accuracy(results: list) -> dict:
    correct = total = 0
    for result in results:
        if not result.target:
            continue
        total += 1
        if _normalize(result.target) and _normalize(result.target) in _normalize(
            result.output[: len(result.target) + 30]
        ):
            correct += 1
    return {"accuracy": correct / total if total else 0.0, "n_scored": total}


def _check_instruction(prompt: str, output: str, instruction_id: str, kwargs: dict) -> bool:
    namespace, _, name = instruction_id.partition(":")

    if namespace == "length_constraints":
        if name == "number_words":
            return _compare(len(output.split()), int(kwargs.get("num_words", 0)), kwargs.get("relation", "at least"))
        if name == "number_sentences":
            count = len([s for s in re.split(r"(?<=[.!?])\s+", output.strip()) if s])
            return _compare(count, int(kwargs.get("num_sentences", 0)), kwargs.get("relation", "at least"))
        if name == "number_paragraphs":
            count = len([p for p in output.split("\n\n") if p.strip()])
            return count >= int(kwargs.get("num_paragraphs", 0))
        if name == "nth_paragraph_first_word":
            paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
            nth = int(kwargs.get("nth_paragraph", 1)) - 1
            first_word = str(kwargs.get("first_word", "")).lower()
            return (
                nth < len(paragraphs)
                and paragraphs[nth].split()[0].lower().rstrip(".,!?;:") == first_word
            )

    if namespace == "keywords":
        if name == "existence":
            return all(keyword.lower() in output.lower() for keyword in kwargs.get("keywords") or [])
        if name == "forbidden_words":
            return all(word.lower() not in output.lower() for word in kwargs.get("forbidden_words") or [])
        if name == "frequency":
            keyword = str(kwargs.get("keyword", "")).lower()
            return _compare(
                output.lower().count(keyword),
                int(kwargs.get("frequency", 0)),
                kwargs.get("relation", "at least"),
            )
        if name == "letter_frequency":
            letter = str(kwargs.get("letter", "")).lower()
            return _compare(
                output.lower().count(letter),
                int(kwargs.get("let_frequency", 0)),
                kwargs.get("let_relation", "at least"),
            )

    if namespace == "detectable_format":
        if name == "number_bullet_lists":
            bullets = len(re.findall(r"^\s*[-*]\s", output, re.MULTILINE))
            return _compare(bullets, int(kwargs.get("num_bullets", 0)), "at least")
        if name in ("number_highlighted_sections", "constrain_highlighted_sections"):
            highlights = len(re.findall(r"\*[^*\n]+\*", output))
            expected = int(kwargs.get("num_highlights", 0))
            return highlights == expected if name == "constrain_highlighted_sections" else highlights >= expected
        if name == "multiple_sections":
            splitter = kwargs.get("section_spliter", "Section")
            return output.count(splitter) >= int(kwargs.get("num_sections", 2))

    if namespace == "detectable_content":
        if name == "number_placeholders":
            return len(re.findall(r"\[[^\]]+\]", output)) >= int(kwargs.get("num_placeholders", 1))
        if name == "postscript":
            return str(kwargs.get("postscript_marker", "P.S.")) in output

    if namespace == "startend":
        if name == "end_checker":
            return output.strip().lower().endswith(str(kwargs.get("end_phrase", "")).lower().strip())
        if name == "quotation":
            stripped = output.strip()
            return stripped.startswith('"') and stripped.endswith('"')

    if namespace == "change_case":
        words = re.findall(r"\b[A-Za-z]+\b", output)
        if name == "english_capital":
            return bool(words) and all(word.isupper() for word in words)
        if name == "english_lowercase":
            return bool(words) and all(word.islower() for word in words)
        if name == "capital_word_frequency":
            caps = sum(1 for word in words if word.isupper())
            return _compare(caps, int(kwargs.get("capital_frequency", 0)), kwargs.get("capital_relation", "at least"))

    if namespace == "punctuation" and name == "no_comma":
        return "," not in output

    if namespace == "combination":
        if name == "two_responses":
            return "******" in output
        if name == "repeat_prompt":
            return prompt.strip().lower() in output.lower()

    return True


def _compare(actual: int, target: int, relation: str) -> bool:
    return {
        "at least": actual >= target,
        "at most": actual <= target,
        "less than": actual < target,
        "more than": actual > target,
        "exactly": actual == target,
    }.get(relation, actual >= target)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
