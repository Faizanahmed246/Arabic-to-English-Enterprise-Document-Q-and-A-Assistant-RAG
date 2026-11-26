import argparse
import json
import math
import os
import secrets
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from rag.api import app
from rag.embeddings import INDEX_VERSION, MODEL, REVISION

ROOT = Path(__file__).resolve().parent
TOP_K = 5


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def first_evidence_rank(passages: list[dict], evidence: list[dict]) -> int | None:
    for rank, passage in enumerate(passages, 1):
        for target in evidence:
            if target["source"] == passage["source"] and normalize(target["quote"]) in normalize(passage["excerpt"]):
                return rank
    return None


def rate(count: int, total: int) -> float | None:
    return round(100 * count / total, 2) if total else None


def retrieval_summary(rows: list[dict]) -> dict:
    ranks = [row["first_evidence_rank"] for row in rows if row["evidence"]]
    hits = sum(rank is not None for rank in ranks)
    return {
        "answerable_questions": len(ranks),
        "hits_at_5": hits,
        "hit_rate_at_5_pct": rate(hits, len(ranks)),
        "mrr_at_5": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 4) if ranks else None,
    }


def latency_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"p50_ms": round(median(ordered), 2), "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def answer_summary(rows: list[dict]) -> dict:
    generated = [row for row in rows if "generated" in row]
    answered = [row for row in generated if row["generated"]["status"] == "answered"]
    reviewed = [row for row in answered if isinstance(row.get("faithful"), bool)]
    faithful = sum(row["faithful"] for row in reviewed)
    unsupported = [row for row in generated if not row["evidence"]]
    abstained = sum(row["generated"]["status"] == "insufficient_evidence" for row in unsupported)
    answerable = [row for row in generated if row["evidence"]]
    answered_answerable = sum(row["generated"]["status"] == "answered" for row in answerable)
    metrics = [row["generated"]["metrics"] for row in generated]
    return {
        "generated_responses": len(generated),
        "answered_responses": len(answered),
        "reviewed_answers": len(reviewed),
        "faithful_answers": faithful if reviewed else None,
        "faithfulness_pct": rate(faithful, len(answered)) if len(reviewed) == len(answered) else None,
        "answerable_coverage_pct": rate(answered_answerable, len(answerable)),
        "unanswerable_abstention_pct": rate(abstained, len(unsupported)),
        "total_input_tokens": sum(item["input_tokens"] or 0 for item in metrics) if generated else None,
        "total_output_tokens": sum(item["output_tokens"] or 0 for item in metrics) if generated else None,
        "end_to_end_latency": latency_summary([item["total_ms"] for item in metrics]),
    }


def run(with_answers: bool) -> dict:
    config = {"RAG_API_KEY": secrets.token_hex(24)}
    if not with_answers:
        config["ANTHROPIC_API_KEY"] = ""
    cache = Path(os.environ.get("RAG_MODEL_CACHE", "data/models")).resolve()
    policies = sorted((ROOT / "policies").glob("*.md"))
    questions = [json.loads(line) for line in (ROOT / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    chunk_count = 0
    with TemporaryDirectory(prefix="bilingual-rag-") as data:
        config.update(RAG_DATA_DIR=data, RAG_MODEL_CACHE=str(cache))
        with patch.dict(os.environ, config), TestClient(app) as client:
            headers = {"X-API-Key": config["RAG_API_KEY"]}
            for policy in policies:
                response = client.put(f"/v1/documents/{quote(policy.name)}", content=policy.read_bytes(), headers=headers)
                chunk_count += response.raise_for_status().json()["chunks"]
            for item in questions:
                query = {"question": item["question"], "top_k": TOP_K}
                result = client.post("/v1/search", json=query, headers=headers).raise_for_status().json()
                row = {
                    **item,
                    "first_evidence_rank": first_evidence_rank(result["passages"], item["evidence"]),
                    "retrieval_ms": result["retrieval_ms"],
                    "retrieved": [
                        {key: passage[key] for key in ("source", "location", "similarity")}
                        for passage in result["passages"]
                    ],
                }
                if with_answers:
                    response = client.post("/v1/ask", json={**query, "language": item["language"]}, headers=headers)
                    if response.is_success:
                        row["generated"] = response.json()
                        if row["generated"]["status"] == "answered":
                            row["faithful"] = None
                    else:
                        row["generation_error"] = {"status_code": response.status_code, **response.json()}
                rows.append(row)
    cross = [row for row in rows if row["language"] != row["source_language"]]
    same = [row for row in rows if row["language"] == row["source_language"]]
    return {
        "corpus": "synthetic bilingual policies",
        "embedding_model": MODEL,
        "embedding_revision": REVISION,
        "index_version": INDEX_VERSION,
        "policy_documents": len(policies),
        "indexed_chunks": chunk_count,
        "questions": len(rows),
        "unanswerable_questions": sum(not row["evidence"] for row in rows),
        "retrieval": {
            **retrieval_summary(rows),
            "cross_language": retrieval_summary(cross),
            "same_language": retrieval_summary(same),
            "latency": latency_summary([row["retrieval_ms"] for row in rows]),
        },
        "answers": {"requested": with_answers, "errors": sum("generation_error" in row for row in rows), **answer_summary(rows)},
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fresh index on the bundled bilingual policies.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--with-answers", action="store_true", help="Call Claude for each question.")
    mode.add_argument("--review", type=Path, help="Recompute faithfulness after manual review of an answer report.")
    parser.add_argument("--output", type=Path, default=Path("evaluation-report.json"))
    args = parser.parse_args()
    load_dotenv()
    if args.with_answers and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        parser.error("Set ANTHROPIC_API_KEY before running --with-answers.")
    if args.review:
        report = json.loads(args.review.read_text(encoding="utf-8"))
        if not report["answers"]["requested"]:
            parser.error("Choose a report produced using --with-answers.")
        report["answers"].update(answer_summary(report["items"]))
    else:
        report = run(args.with_answers)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
