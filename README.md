# Arabic–English Enterprise Document Q&A

A FastAPI service that answers policy questions in Arabic or English and cites the exact excerpts it used. It indexes PDF, DOCX, Markdown, and UTF-8 text in Chroma using multilingual E5 embeddings. Claude generates answers from the retrieved passages. Each citation includes the filename, location, chunk ID, and excerpt.

## Run locally

From the project directory in PowerShell with Python 3.12 installed, create an environment and generate an API key.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
```

Save the generated value as `RAG_API_KEY` in `.env`. Add `ANTHROPIC_API_KEY` there to enable `/v1/ask`. The default Claude model is `claude-haiku-4-5-20251001`. Start the API with the following command. The first start downloads the pinned ONNX model and tokenizer.

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag.api:app --env-file .env --no-access-log
```

Open the interactive API docs at `http://127.0.0.1:8000/docs` and use **Authorize** with your `RAG_API_KEY`. You can also upload and query from a second PowerShell session in the project directory.

```powershell
$api = 'http://127.0.0.1:8000'
$headers = @{ 'X-API-Key' = (Read-Host 'Enter RAG_API_KEY') }
Invoke-RestMethod -Method Put -Uri "$api/v1/documents/travel-expenses.en.md" `
  -Headers $headers -ContentType 'application/octet-stream' `
  -InFile 'evaluation/policies/travel-expenses.en.md'

$body = @{ question = 'ما الحد الأقصى لتعويض تكلفة الفندق في الليلة قبل الضرائب؟'; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/v1/search" -Headers $headers `
  -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Use `/v1/ask` with the same JSON for a cited answer. You can set `language` to `ar` or `en`. With `auto`, the service chooses a language from the question.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Reports service readiness after model and index startup. |
| `PUT /v1/documents/{filename}` | Indexes a file of up to 10 MiB. Repeating an identical upload is idempotent. Updated contents under the same filename replace the active version. |
| `POST /v1/search` | Returns up to ten ranked excerpts and retrieval latency. Works without a Claude key. |
| `POST /v1/ask` | Returns an answer with validated citation references, or an insufficient-evidence response, plus timing and token usage. |

## How it works

- Ingestion records PDF pages, DOCX body blocks, or text line ranges. The chunker uses up to 256 model tokens with a 48-token overlap. Scanned or image-only PDFs require OCR before upload.
- The pinned multilingual E5 model embeds chunks and questions locally using `passage:` and `query:` prefixes. A persistent Chroma collection performs cosine search. A SQLite manifest makes new versions visible after their chunks are indexed.
- Claude receives the question and the top-ranked excerpts. The service maps each validated passage ID to the source and excerpt. Faithfulness connects the meaning of a claim to the cited excerpt and is scored during answer evaluation.
- The service logs request IDs, answer status, retrieval and generation latency, and Claude input and output token counts as JSON. It omits questions and document text from these logs.

The included deployment targets a single shared corpus with an API key that grants read and write access. Claude receives the retrieved policy excerpts on answer requests.

## Reproduce the evaluation

```powershell
.\.venv\Scripts\python.exe -m evaluation.run --output evaluation-report.json
.\.venv\Scripts\python.exe -m evaluation.run --with-answers --output evaluation-report-answers.json
```

The first command evaluates retrieval over eight fictional policies and 30 questions. The 24 answerable questions have an annotated source and exact evidence quote. The six unanswerable questions measure abstention when you run with Claude. Hit rate@5 counts an answerable question as a hit when one of the five retrieved excerpts contains its annotated evidence.

The included local run reached **83.33% hit rate@5 (20/24)**, with **75%** on cross-language questions (12/16) and **100%** on same-language questions (8/8). See `retrieval-results.json` for per-question evidence ranks and latency. Answer faithfulness is pending a live Claude run and review.

With `--with-answers`, the report records actual Claude responses, citations, timings, and tokens. Review each answered row with this rubric:

- Set `faithful` to `true` when every factual claim follows from its cited excerpt, including amounts, conditions, and exceptions across languages.
- Set `faithful` to `false` when any factual claim lacks support, misstates a cited detail, or cites an unrelated excerpt.

Score the reviewed file with `python -m evaluation.run --review evaluation-report-answers.json --output evaluation-report-scored.json`. The report computes faithfulness once all substantive responses have been reviewed. It also reports answer coverage, abstention on unanswerable prompts, and any API errors.

## Docker and Render

```powershell
docker build -t bilingual-policy-rag .
docker run --rm -p 127.0.0.1:8000:10000 --env-file .env `
  --mount source=policy-rag-data,target=/var/data bilingual-policy-rag
```

The Docker build caches the embedding assets. The named volume stores the Chroma collection and manifest across runs.

For a hosted portfolio demo, connect this project repository as a Render Blueprint and provide `ANTHROPIC_API_KEY`. The supplied `render.yaml` defines a paid single-instance Docker web service with a persistent disk, a generated `RAG_API_KEY`, and `/health` as its health check. Get `RAG_API_KEY` from Render's Environment panel and use your service URL in the commands above. [Render Blueprints](https://render.com/docs/blueprint-spec) · [Render persistent disks](https://render.com/docs/disks)

[Multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small) · [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure) · [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
