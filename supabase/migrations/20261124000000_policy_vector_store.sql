-- FINOVA policy vector store (replaces the local FAISS vectorstore/ folder).
--
-- Non-destructive: creates two new tables and two functions. It does not touch
-- profiles, tenant_workspaces or any other existing object. Safe to re-run.
--
-- Access model: only the backend (service_role key) may read or write these
-- objects. RLS is enabled with NO policies, and anon/authenticated privileges
-- are revoked, so a browser holding the anon key or a user JWT cannot read any
-- policy text or embeddings.

create extension if not exists vector with schema extensions;

set search_path = public, extensions;

-- One row per role: which build is live, what model built it, and the outcome
-- of the last ingestion run.
create table if not exists public.policy_indexes (
    role                 text primary key
                         check (role in ('SUPER_ADMIN', 'ADMIN', 'FINANCE_ANALYST', 'SUPPORT_EXECUTIVE', 'CUSTOMER')),
    active_build_id      uuid,
    built_at             timestamptz,
    embedding_model      text,
    fingerprint          text,
    chunk_count          integer not null default 0,
    documents            text[] not null default '{}',
    tenant_chunk_counts  jsonb not null default '{}'::jsonb,
    manifest             jsonb not null default '{}'::jsonb,
    last_run_at          timestamptz,
    last_run_status      text,
    last_run_detail      text,
    updated_at           timestamptz not null default now()
);

-- One row per chunk. 3072 = gemini-embedding-001 default output size.
-- No ANN index: retrieval is an exact scan of one role's rows, which is fast
-- for policy-document volumes (thousands of chunks) and never misses results
-- because of filtering. For >100k chunks per role, add an HNSW index on a
-- halfvec(3072) expression.
create table if not exists public.policy_chunks (
    id             uuid primary key default gen_random_uuid(),
    build_id       uuid not null,
    role_scope     text not null
                   check (role_scope in ('SUPER_ADMIN', 'ADMIN', 'FINANCE_ANALYST', 'SUPPORT_EXECUTIVE', 'CUSTOMER')),
    allowed_roles  text[] not null,
    tenant_id      text not null,
    chunk_id       text not null,
    doc_id         text not null,
    source         text not null,
    source_path    text not null,
    page           integer not null,
    content        text not null,
    embedding      vector(3072) not null,
    created_at     timestamptz not null default now()
);

create index if not exists policy_chunks_role_build_tenant_idx
    on public.policy_chunks (role_scope, build_id, tenant_id);

alter table public.policy_indexes enable row level security;
alter table public.policy_chunks  enable row level security;

revoke all on table public.policy_indexes from anon, authenticated;
revoke all on table public.policy_chunks  from anon, authenticated;
grant select, insert, update, delete on table public.policy_indexes to service_role;
grant select, insert, update, delete on table public.policy_chunks  to service_role;

-- Similarity search. Role, tenant, active build and embedding model are all
-- enforced here, in the database; the API re-checks role/tenant afterwards.
create or replace function public.match_policy_chunks(
    p_role             text,
    p_tenant_id        text,
    p_embedding_model  text,
    p_query_embedding  vector(3072),
    p_match_count      integer
)
returns table (
    chunk_id       text,
    doc_id         text,
    source         text,
    source_path    text,
    page           integer,
    role_scope     text,
    allowed_roles  text[],
    tenant_id      text,
    content        text,
    similarity     double precision
)
language sql
stable
security invoker
set search_path = public, extensions, pg_temp
as $$
    select c.chunk_id, c.doc_id, c.source, c.source_path, c.page,
           c.role_scope, c.allowed_roles, c.tenant_id, c.content,
           1 - (c.embedding <=> p_query_embedding) as similarity
    from public.policy_chunks c
    join public.policy_indexes i
      on i.role = c.role_scope
     and i.active_build_id = c.build_id
     and i.embedding_model = p_embedding_model
    where c.role_scope = p_role
      and p_role = any (c.allowed_roles)
      and (c.tenant_id = 'GLOBAL' or (p_tenant_id is not null and c.tenant_id = p_tenant_id))
    order by c.embedding <=> p_query_embedding
    limit least(greatest(coalesce(p_match_count, 5), 1), 50);
$$;

-- Atomically make p_build_id the live build for p_role and delete older builds.
-- Fails (leaving the previous build live) if the build is empty or incomplete.
create or replace function public.activate_policy_build(
    p_role      text,
    p_build_id  uuid,
    p_manifest  jsonb
)
returns void
language plpgsql
security invoker
set search_path = public, extensions, pg_temp
as $$
declare
    v_count integer;
begin
    perform pg_advisory_xact_lock(hashtext('finova_policy_build:' || p_role));

    select count(*) into v_count
    from public.policy_chunks
    where role_scope = p_role and build_id = p_build_id;

    if v_count = 0 then
        raise exception 'Build % has no chunks for role %', p_build_id, p_role;
    end if;
    if v_count <> (p_manifest ->> 'chunk_count')::integer then
        raise exception 'Build % has % chunks but the manifest expects %',
            p_build_id, v_count, p_manifest ->> 'chunk_count';
    end if;

    insert into public.policy_indexes as i (
        role, active_build_id, built_at, embedding_model, fingerprint, chunk_count,
        documents, tenant_chunk_counts, manifest, updated_at
    )
    values (
        p_role, p_build_id, now(), p_manifest ->> 'embedding_model', p_manifest ->> 'fingerprint', v_count,
        coalesce(array(select jsonb_array_elements_text(p_manifest -> 'documents')), '{}'),
        coalesce(p_manifest -> 'tenant_chunk_counts', '{}'::jsonb), p_manifest, now()
    )
    on conflict (role) do update set
        active_build_id     = excluded.active_build_id,
        built_at            = excluded.built_at,
        embedding_model     = excluded.embedding_model,
        fingerprint         = excluded.fingerprint,
        chunk_count         = excluded.chunk_count,
        documents           = excluded.documents,
        tenant_chunk_counts = excluded.tenant_chunk_counts,
        manifest            = excluded.manifest,
        updated_at          = excluded.updated_at;

    delete from public.policy_chunks
    where role_scope = p_role and build_id <> p_build_id;
end;
$$;

revoke all on function public.match_policy_chunks(text, text, text, vector, integer) from public, anon, authenticated;
revoke all on function public.activate_policy_build(text, uuid, jsonb) from public, anon, authenticated;
grant execute on function public.match_policy_chunks(text, text, text, vector, integer) to service_role;
grant execute on function public.activate_policy_build(text, uuid, jsonb) to service_role;

-- Make the new tables/functions visible to PostgREST immediately.
notify pgrst, 'reload schema';
