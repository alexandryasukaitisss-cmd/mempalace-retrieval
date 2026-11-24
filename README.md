# MemPalace Retrieval

[![CI](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval/actions/workflows/ci.yml)

> **Works standalone with any Python memory or RAG store** (zero MemPalace dependency required) — permission-first hybrid retrieval with Reciprocal Rank Fusion, stale-fact rejection, and evidence-based abstention.

[Русский](README.ru.md) · [MIT](LICENSE)

At its simplest, this searches your notes so the answer comes from the current fact, not an old copy. If a fact was revoked, changed, or belongs to another project, it is not served.

A small Python reference implementation for searching agent memory without stale or inaccessible facts. It is an extraction from a local retrieval layer built around [MemPalace](https://github.com/MemPalace/mempalace) and a canonical fact store. [Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger) is the companion project for recording how facts change.

**Search from the current facts.** Reject out-of-scope, revoked, or outdated snapshots before they take up ranking slots. Combine phrase, word, optional embedding, and fresh-record results. Skip weak matches and check the canonical record again before returning it.

## Try it

Python 3.11 or newer. The demo and tests need no database, account, model, or network.

~~~sh
python3 -B demo.py
python3 -B -m unittest discover -s tests -v
~~~

The demo shows a dispatch window changing from Tuesday to Thursday: the old index stops answering, and the fresh record can answer before the index is rebuilt.

## Companion: vector identity

The retrieval layer does not produce embeddings. Its logical continuation is **[Embedding Contract](https://github.com/alexandryasukaitisss-cmd/embedding-contract)**: provider-neutral identity, safe backfill, local Gemma via Ollama, and an explicit Gemini adapter. Use it to ensure that only vectors from the same model, dimensions, and prompt contract share an index.

## Use it in your own store

~~~python
from retrieval import IndexedMemory, Memory, search

fact = Memory("dispatch", "team", "2", "Dispatch window: Thursday morning.", 1700000000)
store = {fact.memory_id: fact}
hits = search(
    "dispatch window",
    [IndexedMemory(fact)],
    read_current=store.get,
    allowed_scopes={"team"},
)
print(hits[0].memory.text)
~~~

The host supplies the caller's authorized scopes and an authoritative record reader; the query never selects permissions. The reader returns the full current record or None. Keep revision, scope, status, timestamp, and content in the indexed snapshot. Supply recent canonical records via **fresh** to bridge index lag. Optional embeddings must use the same space and dimensions. Rank fusion combines the ordered lanes; its score is not a probability.

This version scans small in-memory snapshots. It does not open a MemPalace database or run an embedding model. Use an authorization-aware, coherent store and enforce access when serving results. The second read catches changes observed before that check; it cannot prevent later changes. Calibrate evidence thresholds on your own retrieval tasks. See [design and attribution](NOTICE.md).

## Local-First Agent Memory Stack

This repository is part of a modular, local-first stack for reliable AI agent memory (each component works standalone):

| Layer | Repository | Responsibility |
| --- | --- | --- |
| **1. Temporal Fact Ledger** | [memory-ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger) | Append-only SQLite event journal, provenance, `A → B → A` supersession, conflict rollback vs retraction |
| **2. Permission-First Retrieval** | [mempalace-retrieval](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval) | Scope filtering before ranking, Reciprocal Rank Fusion, stale-fact rejection, evidence-based abstention |
| **3. Vector Space Integrity** | [embedding-contract](https://github.com/alexandryasukaitisss-cmd/embedding-contract) | Fails fast on model/dimension/prefix drift across Ollama Gemma and optional Gemini embeddings |
