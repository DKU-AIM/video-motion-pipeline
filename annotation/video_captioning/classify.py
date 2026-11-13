"""Replay or classify caption JSONL while retaining one row per input player."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

RULES = {
    "슛": r"슛(?:을|를)?\s*(?:하|쏘|던|시도)|점프슛|레이업|덩크|(?:점프(?:를)?\s*(?:하고|해)|(?:골대|림)(?:를|을)?\s*향해)\s*(?:농구)?공을\s*(?:던|쏘|쏜)",
    "드리블": r"드리블|\bdribbl(?:e|es|ed|ing)(?![a-z])|(?:농구)?공을\s*(?:바닥(?:에|으로)\s*)?튀[기긴길겨]|(?:농구)?공을\s*바닥에\s*(?:바운드|튕[기긴겨])",
    "패스": r"패스|(?:팀원|동료|같은 팀 선수)(?:에게|한테)\s*(?:농구)?공을\s*(?:건네|건넨|넘기|넘긴|전달|던)|(?:농구)?공을\s*(?:팀원|동료)에게\s*(?:건네|건넨|넘기|넘긴|전달|던)",
    "리바운드": r"리바운드|(?:슛|투구).*?(?:림|골대|백보드).*?(?:맞고|튕겨|튀어).*?(?:공|볼).*?(?:잡|걷어|확보)",
    "수비": r"수비(?:를\s*)?(?:하|중)|상대(?:팀)?(?:\s*선수)?(?:의)?\s*(?:진로|동선)을\s*막|상대(?:팀)?\s*선수(?:를|에게)\s*(?:따라붙|밀착)",
    "이동": r"걷[고는습]|걸어|달리[고며는]|달립|달린다|뛰[고며는]|뜁|이동(?:하며|하고|하는|한다|합니다)|코트를\s*가로지르|코트를\s*가로지른|움직이(?:고|며|는|다)|움직입니다",
    "공 소지": r"(?:농구)?공을\s*(?:들고|들며|들었다|쥐고|쥔\s*채|잡고\s*(?:있|서|이동))",
    "경합": r"몸싸움(?:을)?\s*(?:하|하고|벌이|벌인|중)|(?:농구)?공을\s*두고\s*(?:다투|경합)|경합(?:을)?\s*(?:하|중)",
}
UNCERTAIN = re.compile(r"거나|보이지\s*않|있지\s*않|(?:하지|치지|튀기지|던지지|쏘지|잡지|막지|걷지|달리지|들지|쥐지|이동하지)\s*않|안\s*(?:하|치|튀|던|쏘|잡|막)|할 수 없|같습니다|추정|하려|하기 위해|준비")
CLAUSE_BOUNDARY = re.compile(r"[.!?;\n]|,(?!\s*(?:아니|않|못))|\s+(?:하지만|그러나)\s+|(?<=고)\s+있으며\s*|(?<=후)\s+")
CATEGORIES = (*RULES, "검수 필요", "미분류")


def classify(caption, force_review=False):
    evidence = {}
    ambiguous = False
    for clause in CLAUSE_BOUNDARY.split(caption):
        matches = {label: match.group() for label, pattern in RULES.items()
                   if (match := re.search(pattern, clause, re.I))}
        if UNCERTAIN.search(clause):
            ambiguous = ambiguous or bool(matches)
            continue
        for label, phrase in matches.items():
            evidence.setdefault(label, phrase)
    if force_review:
        return ["검수 필요"], evidence
    if evidence:
        return list(evidence), evidence
    if ambiguous:
        return ["검수 필요"], evidence
    return ["미분류"], evidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--review-id", action="append", default=[])
    args = parser.parse_args()
    source = args.input.read_text()
    if args.input.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    else:
        rows = json.loads(source)["models"]["qwen"]
    results = []
    for row in rows:
        identifier = row["clip_id"]
        caption = row.get("raw_output") or ""
        labels, evidence = classify(caption, identifier in args.review_id or not caption)
        results.append(dict(id=identifier, caption=caption, labels=labels, evidence=evidence))
    if len({row["id"] for row in results}) != len(results):
        raise ValueError("Expected a unique ID for each input row.")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter(label for row in results for label in row["labels"])
    (args.output / "matches.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    with (args.output / "counts.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "count"])
        writer.writerows((category, counts[category]) for category in CATEGORIES)
    print(dict(counts))

if __name__ == "__main__":
    main()
