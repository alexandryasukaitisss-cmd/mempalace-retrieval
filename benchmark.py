"""Reproducible offline before/after benchmark for permission-first hybrid retrieval.

Compares a naive index-only retriever (no pre-ranking scope gate, no canonical
revalidation, no fresh-write lane, no evidence-based abstention) against
`retrieval.search` on a deterministic 24-query fixture covering five failure modes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

from retrieval import Embedding, IndexedMemory, Memory, _terms, _tokens, search


def _naive_search(query: str, index: list[IndexedMemory], *, limit: int = 5) -> list[Memory]:
    """Baseline index-only search: ranks index snapshots by term overlap without guards."""
    q_terms = _terms(_tokens(query))
    scored = []
    for item in index:
        overlap = len(q_terms & _terms(_tokens(item.snapshot.text)))
        if overlap > 0:
            scored.append((overlap, item.snapshot.updated_at, item.snapshot))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].memory_id))
    return [row[2] for row in scored[:limit]]


def build_suite():
    """Build deterministic store, stale index snapshots, fresh writes, and 24 test queries."""
    store: dict[str, Memory] = {}
    index: list[IndexedMemory] = []
    fresh: list[Memory] = []
    cases: list[dict] = []

    # 1. Active in-scope facts (6 queries)
    active_facts = [
        ("deploy-window", "Deployment freeze starts Friday at 18:00 UTC.", "deployment freeze friday"),
        ("backup-policy", "Database snapshots retain daily backups for 30 days.", "database snapshots daily"),
        ("oncall-rotation", "Primary on-call rotation hands off every Monday morning.", "on-call rotation monday"),
        ("staging-url", "Staging health check runs against internal smoke endpoint.", "staging health smoke"),
        ("release-branch", "Hotfix releases branch directly from latest production tag.", "hotfix releases production"),
        ("audit-retention", "Security audit logs archive monthly to cold storage.", "security audit archive"),
    ]
    for idx, (mid, text, query) in enumerate(active_facts, 1):
        rec = Memory(mid, "team", "1", text, float(100 + idx))
        store[mid] = rec
        index.append(IndexedMemory(rec, Embedding("demo-v1", (1.0, 0.0))))
        cases.append({"category": "active_recall", "query": query, "expected_id": mid})

    # 2. Cross-scope private facts (6 queries — caller only has scope "team")
    private_facts = [
        ("hr-salary", "Executive salary review schedule finalized for Q4.", "executive salary review"),
        ("legal-nda", "Confidential merger counterparty NDA signed Tuesday.", "confidential merger nda"),
        ("sec-root", "Production break-glass root token stored in hardware vault.", "break-glass root vault"),
        ("fin-payroll", "Payroll wire batch approval threshold raised for December.", "payroll wire approval"),
        ("med-leave", "Private medical leave accommodation recorded for staff.", "medical leave accommodation"),
        ("board-deck", "Board compensation committee draft minutes attached.", "board compensation minutes"),
    ]
    for idx, (mid, text, query) in enumerate(private_facts, 1):
        rec = Memory(mid, "private", "1", text, float(200 + idx))
        store[mid] = rec
        index.append(IndexedMemory(rec, Embedding("demo-v1", (1.0, 0.0))))
        cases.append({"category": "cross_scope_leak", "query": query, "forbidden_id": mid})

    # 3. Stale index snapshots superseded by fresh canonical writes (4 queries)
    stale_pairs = [
        (
            "dispatch-day",
            "Parcel dispatch window scheduled for Tuesday morning.",
            "Parcel dispatch window moved to Thursday afternoon.",
            "parcel dispatch window",
        ),
        (
            "api-limit",
            "Public webhook rate limit set to 60 requests per minute.",
            "Public webhook rate limit increased to 240 requests per minute.",
            "webhook rate limit",
        ),
        (
            "tls-min",
            "Gateway minimum protocol allows TLS 1.2 legacy ciphers.",
            "Gateway minimum protocol requires TLS 1.3 strict ciphers.",
            "gateway minimum protocol",
        ),
        (
            "cache-ttl",
            "Session lookup cache expires after 900 seconds.",
            "Session lookup cache expires after 120 seconds.",
            "session lookup cache",
        ),
    ]
    for idx, (mid, old_text, new_text, query) in enumerate(stale_pairs, 1):
        old_rec = Memory(mid, "team", "1", old_text, float(300 + idx))
        new_rec = replace(old_rec, revision="2", text=new_text, updated_at=float(400 + idx))
        store[mid] = new_rec
        index.append(IndexedMemory(old_rec))
        fresh.append(new_rec)
        cases.append({
            "category": "stale_update",
            "query": query,
            "stale_revision": "1",
            "expected_revision": "2",
            "expected_id": mid,
        })

    # 4. Revoked facts still present in the search index (4 queries)
    revoked_facts = [
        ("old-vpn", "Legacy office VPN gateway uses shared group password.", "legacy office vpn"),
        ("old-promo", "Partner discount voucher SPRING50 applies sitewide.", "partner discount voucher"),
        ("old-endpoint", "Deprecated billing webhook posts to v1 callback path.", "deprecated billing webhook"),
        ("old-approver", "Temporary vendor invoice signoff delegated to contractor.", "vendor invoice signoff"),
    ]
    for idx, (mid, text, query) in enumerate(revoked_facts, 1):
        old_rec = Memory(mid, "team", "1", text, float(500 + idx), status="active")
        revoked_rec = replace(old_rec, revision="2", status="revoked", updated_at=float(600 + idx))
        store[mid] = revoked_rec
        index.append(IndexedMemory(old_rec))
        cases.append({"category": "revoked_leak", "query": query, "forbidden_id": mid})

    # 5. Unsupported weak-overlap queries (4 queries — only 1 of 3 content terms appears)
    weak_queries = [
        "friday glacier observatory",
        "monthly telescope telemetry",
        "monday submarine sonar",
        "storage vineyard harvest",
    ]
    for query in weak_queries:
        cases.append({"category": "unsupported_abstain", "query": query})

    return store, index, fresh, cases


def run_benchmark() -> dict:
    store, index, fresh, cases = build_suite()
    allowed = {"team"}

    metrics = {
        "naive": {
            "active_recall": [0, 0],
            "fresh_update_recall": [0, 0],
            "cross_scope_leaks": [0, 0],
            "stale_fact_leaks": [0, 0],
            "revoked_fact_leaks": [0, 0],
            "unsupported_false_positives": [0, 0],
        },
        "mempalace_retrieval": {
            "active_recall": [0, 0],
            "fresh_update_recall": [0, 0],
            "cross_scope_leaks": [0, 0],
            "stale_fact_leaks": [0, 0],
            "revoked_fact_leaks": [0, 0],
            "unsupported_false_positives": [0, 0],
        },
    }

    for case in cases:
        cat = case["category"]
        query = case["query"]

        naive_hits = _naive_search(query, index)
        guarded_hits = [
            hit.memory
            for hit in search(
                query,
                index,
                allowed_scopes=allowed,
                read_current=store.get,
                fresh=fresh,
            )
        ]

        for label, hits in (("naive", naive_hits), ("mempalace_retrieval", guarded_hits)):
            bucket = metrics[label]
            if cat == "active_recall":
                bucket["active_recall"][1] += 1
                if hits and hits[0].memory_id == case["expected_id"]:
                    bucket["active_recall"][0] += 1
            elif cat == "cross_scope_leak":
                bucket["cross_scope_leaks"][1] += 1
                if any(h.memory_id == case["forbidden_id"] for h in hits):
                    bucket["cross_scope_leaks"][0] += 1
            elif cat == "stale_update":
                bucket["stale_fact_leaks"][1] += 1
                bucket["fresh_update_recall"][1] += 1
                if any(h.memory_id == case["expected_id"] and h.revision == case["stale_revision"] for h in hits):
                    bucket["stale_fact_leaks"][0] += 1
                if hits and hits[0].memory_id == case["expected_id"] and hits[0].revision == case["expected_revision"]:
                    bucket["fresh_update_recall"][0] += 1
            elif cat == "revoked_leak":
                bucket["revoked_fact_leaks"][1] += 1
                if any(h.memory_id == case["forbidden_id"] for h in hits):
                    bucket["revoked_fact_leaks"][0] += 1
            elif cat == "unsupported_abstain":
                bucket["unsupported_false_positives"][1] += 1
                if hits:
                    bucket["unsupported_false_positives"][0] += 1

    return metrics


def format_markdown(metrics: dict) -> str:
    rows = [
        ("Active in-scope recall (higher is better)", "active_recall"),
        ("Updated-fact recall before reindex (higher is better)", "fresh_update_recall"),
        ("Cross-scope private leaks (lower is better)", "cross_scope_leaks"),
        ("Stale-revision answers served (lower is better)", "stale_fact_leaks"),
        ("Revoked-fact answers served (lower is better)", "revoked_fact_leaks"),
        ("Unsupported weak-match false positives (lower is better)", "unsupported_false_positives"),
    ]
    lines = [
        "| Metric (24-query offline suite) | Naive index search | MemPalace Retrieval |",
        "| --- | --- | --- |",
    ]
    for title, key in rows:
        n_ok, n_tot = metrics["naive"][key]
        g_ok, g_tot = metrics["mempalace_retrieval"][key]
        lines.append(f"| {title} | {n_ok}/{n_tot} ({round(100 * n_ok / n_tot)}%) | **{g_ok}/{g_tot} ({round(100 * g_ok / g_tot)}%)** |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print raw JSON metrics")
    args = parser.parse_args()
    metrics = run_benchmark()
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_markdown(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
