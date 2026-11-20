# FINOVA Role-Based Banking Policy RAG Backend

A FastAPI backend that answers banking-policy questions using only the documents a user's **backend-verified role** (and tenant) is permitted to see. It uses LangChain, Gemini embeddings, FAISS, and either Gemini or Groq as the LLM. Supabase Auth handles authentication and a `profiles` table stores each user's role.

> **The ten sample PDFs in `data/` are fictional training examples.** They describe an invented "Finova Demo Bank". They are **not** official bank, RBI or regulatory policy, and every figure in them is made up.

> **Status: development / pilot quality, not production-ready.** See [Security limitations](#security-limitations).

---

## Project layout

```
FINOVA_ROLE_BASED_RAG/
├── data/<role>/                 PDFs per role (sub-folder <tenant_id>/ = tenant-specific docs)
├── vectorstore/<role>/          FAISS index + manifest.json per role (built by ingest.py)
├── src/
│   ├── config.py                settings + validation (pydantic-settings)
│   ├── prompts.py               secure role-aware prompt, injection defences
│   ├── embeddings.py            Gemini embeddings
│   ├── chains.py                Gemini / Groq LLM selection, chain construction
│   ├── rag_pipeline.py          role/tenant-filtered retrieval + grounded answers
│   ├── role_manager.py          Supabase token validation, role authorization
│   └── document_loader.py       PDF extraction + chunking + access metadata
├── scripts/generate_sample_pdfs.py   regenerates the 10 fictional sample PDFs
├── tests/                       offline unit/integration tests (no API keys needed)
├── ingest.py                    builds one FAISS index per role
└── main.py                      FastAPI app
```

Roles: `SUPER_ADMIN`, `ADMIN`, `FINANCE_ANALYST`, `SUPPORT_EXECUTIVE`, `CUSTOMER`. Each role searches **only its own index**. SUPER_ADMIN cannot read other roles' documents either. If you want role inheritance, change `ROLE_INDEX_ACCESS` in `src/role_manager.py` deliberately, as a reviewed policy decision.

---

## 1. Python setup

Python **3.11+** is required. The project was tested on Python 3.13. Very new interpreters (for example 3.15) may not have `faiss-cpu` wheels yet.

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
| `VECTORSTORE_PATH` | Index root folder (default `vectorstore`) |
| `CORS_ORIGINS` | Comma-separated browser origins allowed to call the API |
| `CORS_ORIGIN_REGEX` | Optional anchored regex for changing origins (e.g. AI Studio / Cloud Run previews) |
| `RETRIEVAL_FETCH_K`, `RETRIEVAL_SCORE_THRESHOLD` | Number of candidates fetched before permission filtering, and the minimum relevance score |

Invalid combinations stop the app at startup. For example, overlap ≥ chunk size, demo auth in production, or a non-https Supabase URL.

**Changing `GEMINI_EMBEDDING_MODEL` requires `python ingest.py --rebuild`.** Each index records the model it was built with, and the API refuses to query a mismatched index.

## 4. Supabase setup

1. Create a Supabase project. Copy the project URL, the anon key and the service-role key into `.env`.
2. In the SQL editor, create the profiles table:

```sql
create table public.profiles (
  id          uuid primary key references auth.users (id) on delete cascade,
  role        text not null check (role in
                ('SUPER_ADMIN','ADMIN','FINANCE_ANALYST','SUPPORT_EXECUTIVE','CUSTOMER')),
  is_active   boolean not null default false,
  tenant_id   text,               -- the bank / organisation the user belongs to
  created_at  timestamptz not null default now()
);

alter table public.profiles enable row level security;

-- Users may read (not write) their own profile; the frontend may display it.
create policy "read own profile" on public.profiles
  for select to authenticated using (auth.uid() = id);
-- Deliberately NO insert/update/delete policies: users can never change their
-- own role. Roles are assigned by admins via the dashboard or service-role tooling.
```

3. Create users with Supabase Auth, then insert a profile row for each one:

```sql
insert into public.profiles (id, role, is_active, tenant_id)
values ('<auth user uuid>', 'CUSTOMER', true, 'bank-1');
```

**How the backend authenticates a request:**

1. It sends the bearer token to `GET /auth/v1/user`. Supabase rejects expired, forged and revoked tokens.
2. It reads `role`, `is_active` and `tenant_id` from `profiles` with the service-role key, using the **verified** user id.
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

Each role builds into a temporary folder that is swapped in only after it saves successfully, so a failed run keeps the previous index. A role whose source PDFs, chunk settings and embedding model are unchanged is skipped. Exit codes: `0` success, `1` at least one role failed, `2` configuration error (for example a missing `GEMINI_API_KEY`).

## 7. Start FastAPI

```bash
uvicorn main:app --reload --port 8000
```

## 8. Test through Swagger

Open <http://127.0.0.1:8000/docs>.

1. `GET /health` and `GET /roles` need no auth.
2. Click **Authorize** and paste a Supabase access token. In demo mode, paste `demo:CUSTOMER:bank-1` instead.
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

4. `POST /ingest/status` is limited to SUPER_ADMIN and ADMIN. It is read-only, because ingestion runs offline via the CLI. Filenames and tenant ids are shown only to SUPER_ADMIN.

To get a real token for testing:

```bash
curl -X POST "$SUPABASE_URL/auth/v1/token?grant_type=password" \
  -H "apikey: $SUPABASE_ANON_KEY" -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"..."}'
# use the "access_token" field
```

Error codes: `401` (missing, invalid or expired token), `403` (inactive user, no profile, or role not allowed), `422` (invalid body), `502` (LLM failure), `503` (index not built, or Supabase/keys not configured).

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

**What happens on each deploy:** `start.sh` runs `python ingest.py` (embeds `data/` with Gemini), then starts uvicorn on Railway's `$PORT`. Container disks are ephemeral, so indexes are rebuilt on every deploy/restart — cheap for the samples. For larger document sets, add a **Railway Volume** mounted at `/data` and set `VECTORSTORE_PATH=/data/vectorstore`; unchanged indexes are then skipped.

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
5. Check `https://<railway-domain>/health` → `"indexes_built": 5`, and `https://<railway-domain>/docs`.

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

1. **Tenant isolation is logical, not physical.** All tenants' documents for a role share one FAISS index. Isolation comes from `tenant_id` and `allowed_roles` metadata, which is checked inside the FAISS search filter and again after retrieval. A user with no `tenant_id` sees only `GLOBAL` documents. This prevents cross-tenant results from reaching the API, but:
   * an operator or anyone with file access can read every tenant's chunks in `vectorstore/`;
   * filtering happens after the nearest-neighbour search (`RETRIEVAL_FETCH_K` candidates), so a tenant with few documents in a large shared index may get fewer results;
   * removing one tenant's data means rebuilding the role index.

   For production, use a separate index per (role, tenant) or a vector database with native tenant partitioning and row-level security (for example pgvector with RLS). Also add audit logging.
2. **Do not ingest individual customers' confidential data** into these indexes. The `CUSTOMER` role is one shared scope per tenant, and there is no per-end-customer document isolation.
3. **FAISS docstores are pickle files.** `load_local` runs with `allow_dangerous_deserialization=True`. Only load indexes produced by `ingest.py`, and keep `vectorstore/` writable only by the deployment user.
4. **Prompt injection is mitigated, not eliminated.** The mitigations are: delimited and sanitised context, a system prompt that treats documents as untrusted data, heuristic flagging (`context_warning`), and citations taken from metadata rather than model output. A malicious document can still influence an answer, so control who can add PDFs.
5. **No rate limiting, request quotas or response caching.** Add these at an API gateway or with middleware. Every `/chat` request makes two Supabase calls, one embedding call and one LLM call.
6. **Answers can still be wrong.** LLM output is not legal or regulatory advice. Grounding reduces hallucination but does not eliminate it.
7. **The service-role key bypasses RLS.** The backend uses it only for a single filtered `profiles` lookup. Store it in a secret manager in production.
8. `langchain-community` (which provides the FAISS wrapper) prints a deprecation notice. Plan a move to a maintained vector-store integration.

## What has and has not been verified

* ✅ Offline: the test suite passes, ingestion builds all five indexes with a stand-in embedding model, and API auth fails closed.
* ❌ Not verified here: real Gemini embeddings and chat, Groq chat, and real Supabase token and profile lookups. These need your credentials. Run `python ingest.py`, then test `/chat` with a real token.
