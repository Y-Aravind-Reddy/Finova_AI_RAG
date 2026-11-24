# FINOVA Role-Based Banking Policy RAG Backend

A FastAPI backend that answers banking-policy questions using only the documents a user's **backend-verified role** (and tenant) is permitted to see. It uses LangChain, Gemini embeddings, and either Gemini or Groq as the LLM. **Supabase** does everything else: Auth verifies users, the `profiles` table stores each user's role and tenant, and the `policy_chunks` / `policy_indexes` tables (pgvector) store the document chunks and their embeddings. There is no local index folder.

> **The ten sample PDFs in `data/` are fictional training examples.** They describe an invented "Finova Demo Bank". They are **not** official bank, RBI or regulatory policy, and every figure in them is made up.

> **Status: development / pilot quality, not production-ready.** See [Security limitations](#security-limitations).

---

## Project layout

```
FINOVA_ROLE_BASED_RAG/
├── data/<role>/                 PDFs per role (sub-folder <tenant_id>/ = tenant-specific docs)
├── supabase/migrations/         SQL for the policy_chunks / policy_indexes tables + search functions
├── src/
│   ├── config.py                settings + validation (pydantic-settings)
│   ├── prompts.py               secure role-aware prompt, injection defences
│   ├── embeddings.py            Gemini embeddings
│   ├── chains.py                Gemini / Groq LLM selection, chain construction
│   ├── rag_pipeline.py          role/tenant-filtered retrieval + grounded answers
│   ├── vector_store.py          Supabase (pgvector) client for policy chunks
│   ├── role_manager.py          Supabase token validation, role authorization
│   └── document_loader.py       PDF extraction + chunking + access metadata
├── scripts/generate_sample_pdfs.py   regenerates the 10 fictional sample PDFs
├── tests/                       offline unit/integration tests (no API keys needed)
├── ingest.py                    embeds data/ PDFs into Supabase, per role
└── main.py                      FastAPI app
```

Roles: `SUPER_ADMIN`, `ADMIN`, `FINANCE_ANALYST`, `SUPPORT_EXECUTIVE`, `CUSTOMER`. Each role searches **only its own chunks**. SUPER_ADMIN cannot read other roles' documents either. If you want role inheritance, change `ROLE_INDEX_ACCESS` in `src/role_manager.py` deliberately, as a reviewed policy decision.

---

## 1. Python setup

Python **3.11+** is required. The project was tested on Python 3.13.

```bash
cd FINOVA_ROLE_BASED_RAG
python -m venv .venv            # Windows with several Pythons: py -3.13 -m venv .venv
```

Activate the virtual environment:

| Shell | Command |
|---|---|
| PowerShell | `.venv\Scripts\Activate.ps1` |
| cmd | `.venv\Scripts\activate.bat` |
| Git Bash | `source .venv/Scripts/activate` |
| macOS / Linux | `source .venv/bin/activate` |

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + pytest (production uses requirements.txt)
```

## 3. Environment variables

Copy `.env.example` to `.env` and fill it in. `.env` is git-ignored, so never commit it.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | **Required** for embeddings (ingestion and every query), and for the LLM when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL` | Gemini chat and embedding models |
| `GROQ_API_KEY` / `GROQ_MODEL` | Needed only when `LLM_PROVIDER=groq` |
| `LLM_PROVIDER` | `gemini` or `groq` |
| `EMBEDDING_PROVIDER` | `gemini` (the only supported value) |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` | Supabase project settings. The service-role key must stay **server-side only** |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K` | Chunking and retrieval |
| `AUTH_MODE` | `supabase` (default) or `demo` |
| `ALLOW_DEMO_AUTH` | Must be `true` **and** `APP_ENV` must not be `production` before `AUTH_MODE=demo` is accepted |
| `APP_ENV` | `production` (default), `development` or `test` |
| `EMBEDDING_DIMENSIONS` | Embedding size (default `3072`, Gemini's default). Must match `vector(N)` in the Supabase migration |
| `CORS_ORIGINS` | Comma-separated browser origins allowed to call the API |
| `CORS_ORIGIN_REGEX` | Optional anchored regex for changing origins (e.g. AI Studio / Cloud Run previews) |
| `RETRIEVAL_SCORE_THRESHOLD` | Minimum cosine relevance (0–1) for a chunk to be used; 0 disables |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | Bound each LLM call (defaults 45 s, 2 retries). Without them a provider outage hangs `/chat` for minutes |
| `LLM_FALLBACK_PROVIDER` | Optional (`groq` or `gemini`, must differ from `LLM_PROVIDER`). Tried when the primary LLM fails, e.g. Gemini `503 high demand` |

Invalid combinations stop the app at startup. For example, overlap ≥ chunk size, demo auth in production, or a non-https Supabase URL.

**Changing `GEMINI_EMBEDDING_MODEL` requires `python ingest.py --rebuild`.** Each role's `policy_indexes` row records the model it was built with, and the API (and the SQL search function) refuse to query a mismatched index.

## 4. Supabase setup

1. Create a Supabase project. Copy the project URL, the anon key and the service-role key into `.env`.
1. **Run the migration once:** open Supabase → **SQL Editor**, paste the contents of `supabase/migrations/20261124000000_policy_vector_store.sql` and click **Run**. It enables `pgvector` and creates `policy_indexes`, `policy_chunks`, `match_policy_chunks()` and `activate_policy_build()`. It changes nothing that already exists and is safe to re-run. Both tables have RLS on with no policies, and anon/authenticated access is revoked, so only the backend's service-role key can read policy text.
2. The backend reads three columns of `public.profiles` (verified against the live project schema): `role` (enum `user_role_type`: `SUPER_ADMIN`, `ADMIN`, `FINANCE_ANALYST`, `SUPPORT_EXECUTIVE`, `CUSTOMER`), `status` (enum `user_status_type`: `Active`, `Suspended`, `Pending Review`, `KYC Pending`) and `tenant_id` (NOT NULL, FK to `tenant_workspaces.id`). Only `status = 'Active'` may use the API; every other value, a missing value or a missing profile is rejected with 403. Keep RLS enabled and never give users write access to their own `role`, `status` or `tenant_id`.

**How the backend authenticates a request:**

1. It sends the bearer token to `GET /auth/v1/user`. Supabase rejects expired, forged and revoked tokens.
2. It reads `role`, `status` and `tenant_id` from `profiles` with the service-role key, using the **verified** user id.
3. It rejects the request (403) when the user has no profile, is inactive, or has an unknown role.

The backend never reads a role from the request body or headers. `/chat` rejects bodies that contain extra fields such as `role`.

### Demo mode (local development only)

```env
APP_ENV=development
AUTH_MODE=demo
ALLOW_DEMO_AUTH=true
```

In demo mode, tokens take the form `demo:<ROLE>[:<tenant_id>]`, for example `Authorization: Bearer demo:CUSTOMER:bank-1`. The app refuses to start if demo mode is combined with `APP_ENV=production`, which is the default.

## 5. Adding PDFs

* `data/<role>/*.pdf`: shared by all tenants (`tenant_id = GLOBAL`). Use this only for generic, non-confidential policy.
* `data/<role>/<tenant_id>/*.pdf`: visible only to users whose profile `tenant_id` matches. Tenant ids must match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`.

Scanned or image-only PDFs need OCR first. Encrypted PDFs are rejected. Invalid PDFs are skipped and reported, or fail the role when you pass `--strict`.

To regenerate the fictional samples: `python scripts/generate_sample_pdfs.py --force`.

## 6. Run ingestion

```bash
python ingest.py                      # build missing / changed indexes for all roles
python ingest.py --rebuild            # force rebuild
python ingest.py --role CUSTOMER      # one role (repeatable)
python ingest.py --dry-run            # load + chunk only, no API calls
```

Ingestion needs `GEMINI_API_KEY`, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (except `--dry-run`). Each role's chunks are inserted under a new build id; `activate_policy_build()` then switches the role to that build and deletes the old one in a single transaction. A failed run keeps the previous build live and removes its own partial rows. A role whose source PDFs, chunk settings and embedding model are unchanged is skipped, with no embedding calls. Each role's outcome is saved in `policy_indexes.last_run_*`. Exit codes: `0` success, `1` at least one role failed, `2` configuration or Supabase error (for example a missing `GEMINI_API_KEY`, or the migration not applied).

You can run ingestion from your machine: it writes to the same Supabase project the deployed API reads, so new documents are live without a redeploy.

## 7. Start FastAPI

```bash
uvicorn main:app --reload --port 8000
```

## 8. Test through Swagger

Open <http://127.0.0.1:8000/docs>.

1. `GET /health` and `GET /roles` need no auth.
2. Click **Authorize** and paste a Supabase access token (the token only, without `Bearer `). Without this step every protected call returns `401 Missing bearer token`, which is expected. In demo mode, paste `demo:CUSTOMER:bank-1` instead.
3. Call `POST /chat` with `{"question": "How do I close my account?"}`. The response looks like this:

```json
{
  "answer": "… [account_terms_and_fees.pdf, p. 2]",
  "role": "CUSTOMER",
  "found": true,
  "sources": [{"document": "account_terms_and_fees.pdf", "pages": [2]}],
  "source_documents": ["account_terms_and_fees.pdf"],
  "source_pages": [2],
  "context_warning": false
}
```

4. `POST /ingest/status` is limited to SUPER_ADMIN and ADMIN. It is read-only, because ingestion runs offline via the CLI. SUPER_ADMIN sees filenames, tenant ids, total chunk counts and `last_ingest` (the outcome and error of the last `ingest.py` run). ADMIN sees build state and chunk counts limited to GLOBAL + their own tenant.

To get a real token for testing:

```bash
curl -X POST "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"..."}'
# use the "access_token" field
```

Error codes: `401` (missing, invalid or expired token), `403` (inactive user, no profile, or role not allowed), `422` (invalid body), `502` (LLM or embedding provider failure), `503` (index missing/incompatible/corrupted, or Supabase/keys not configured), `500` (unexpected; no stack trace is returned). Every response carries an `X-Request-ID` header that also appears in the server logs.

### Automated tests

```bash
python -m pytest
```

The tests run offline. They use a deterministic keyword embedding and a fake LLM. They cover configuration validation, PDF loading, ingestion (all 5 indexes, skip/rebuild, failure rollback), role and tenant isolation, prompt-injection sanitisation, Supabase auth logic (mocked HTTP), and the API endpoints. They do **not** call Gemini, Groq or Supabase.

## 9. Connecting the React frontend

Add the frontend origin to `CORS_ORIGINS`, then send the Supabase session token. Never send a role.

```ts
import { createClient } from "@supabase/supabase-js";
const supabase = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY);

export async function askPolicy(question: string) {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session) throw new Error("Not signed in");
  const res = await fetch(`${import.meta.env.VITE_API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${session.access_token}` },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json(); // { answer, role, found, sources, source_documents, source_pages }
}
```

The frontend only ever needs the **anon** key. Never ship the service-role key to the browser.

## 10. Deploy to Railway

The repo is Railway-ready: `railway.json` (start command + `/health` health check), `start.sh`, `.python-version` (3.13) and `requirements.txt`.

**What happens on each deploy:** `start.sh` runs `python ingest.py`, which compares `data/` with what is stored in Supabase and embeds only roles whose PDFs changed (normally none), then starts uvicorn on Railway's `$PORT`. The indexes live in Supabase, so Railway's temporary disk does not matter and no volume is needed.

1. Push the project to GitHub (`.env` and `.venv` are git-ignored — never commit them). If the repo root is the parent folder, set **Settings → Root Directory** to `FINOVA_ROLE_BASED_RAG`.
2. Railway → **New Project → Deploy from GitHub repo**.
3. **Variables** tab — add (values from your local `.env`):

   ```
   GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL
   GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER
   SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
   APP_ENV=production   AUTH_MODE=supabase   ALLOW_DEMO_AUTH=false
   CORS_ORIGINS=https://<your-frontend-domain>
   CORS_ORIGIN_REGEX=^https://[a-z0-9-]+\.(run\.app|usercontent\.goog)$   # optional, for AI Studio / Cloud Run URLs
   ```

   Never use demo auth on Railway.
4. **Settings → Networking → Generate Domain**. You get e.g. `https://finova-rag-production.up.railway.app`.
5. Check `https://<railway-domain>/health` → `"status": "ok", "indexes_built": 5`, and `https://<railway-domain>/docs`.

### Troubleshooting `"status": "degraded"` / `indexes_built` below 5

`/health` stays HTTP 200 so the service keeps running, but `/chat` returns 503 for every role whose index is missing. To find the cause:

1. Railway → Deployments → **Deploy Logs**: look for the `Ingestion summary` table and the `Config: … gemini_key_set=…` / `Index X NOT ready` lines printed at startup.
2. Or call `POST /ingest/status` as SUPER_ADMIN and read `last_ingest` (per-role `status`/`detail` from the last run; secrets are redacted). A configuration error that stopped ingestion before it reached Supabase appears only in the deploy logs.
3. Typical causes: the Supabase migration has not been run (logs say `apply supabase/migrations/…`); `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` missing in Railway Variables; `GEMINI_API_KEY` missing or invalid; `GEMINI_EMBEDDING_MODEL` differs from the one that built the index; Gemini quota/rate limits during ingestion. Running `python ingest.py` locally against the same Supabase project fixes missing indexes without a redeploy.

### Connecting a Google AI Studio frontend

The frontend must log users in with Supabase (same project as `SUPABASE_URL`) and send the **access token** to the Railway URL:

```ts
const API_URL = "https://<your-railway-domain>";   // no trailing slash

const { data: { session } } = await supabase.auth.getSession();
const res = await fetch(`${API_URL}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${session!.access_token}` },
  body: JSON.stringify({ question }),
});
```

Find your frontend's exact origin (browser devtools → Network → `Origin` request header) and add it to `CORS_ORIGINS`, or cover changing preview URLs with an anchored `CORS_ORIGIN_REGEX`. A CORS error in the browser console means the origin is not allowed. Only the Supabase **anon** key goes in the frontend; the service-role key lives only in Railway variables.

---

## Security limitations

The project is **not production-ready** until the items below are addressed.

1. **Tenant isolation is enforced in SQL, but all tenants share one table.** `match_policy_chunks()` only returns rows whose `role_scope`/`allowed_roles` match the verified role and whose `tenant_id` is `GLOBAL` or the user's own tenant; the API then re-checks every row. A user with no `tenant_id` sees only `GLOBAL` documents. Remaining caveats:
   * anyone holding the service-role key (or database access) can read every tenant's chunks;
   * removing one tenant's data means re-running ingestion for that role;
   * search is an exact scan of one role's rows — fine for thousands of chunks; beyond ~100k per role add an HNSW index (see the migration comment).

   Also add audit logging.
2. **Do not ingest individual customers' confidential data** into these indexes. The `CUSTOMER` role is one shared scope per tenant, and there is no per-end-customer document isolation.
3. **Keep RLS on the policy tables.** They hold the full text of every policy document. Do not add RLS policies that expose them to `anon` or `authenticated`; the frontend never needs them.
4. **Prompt injection is mitigated, not eliminated.** The mitigations are: delimited and sanitised context, a system prompt that treats documents as untrusted data, heuristic flagging (`context_warning`), and citations taken from metadata rather than model output. A malicious document can still influence an answer, so control who can add PDFs.
5. **No rate limiting, request quotas or response caching.** Add these at an API gateway or with middleware. Every `/chat` request makes three or four Supabase calls (auth, profile, search, plus the per-role index state every 30 s), one embedding call and one LLM call.
6. **Answers can still be wrong.** LLM output is not legal or regulatory advice. Grounding reduces hallucination but does not eliminate it.
7. **The service-role key bypasses RLS.** The backend uses it for the filtered `profiles` lookup and the policy tables. Store it in a secret manager in production.

## What has and has not been verified

* ✅ Offline: the test suite passes against an in-memory stand-in for the Supabase policy tables (same filters as the SQL), ingestion stores all five roles, failed rebuilds keep the live build, and API auth fails closed.
* ❌ Not verified here: real Gemini embeddings and chat, Groq chat, and real Supabase token and profile lookups. These need your credentials. Run `python ingest.py`, then test `/chat` with a real token.
