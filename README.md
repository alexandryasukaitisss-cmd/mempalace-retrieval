# MemPalace Retrieval

[Русский](README.ru.md) · [MIT](LICENSE)

A small Python reference implementation for searching agent memory without serving stale or inaccessible facts. It is an extraction from a local retrieval layer built around [MemPalace](https://github.com/MemPalace/mempalace) and a canonical fact store. [Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger) is the companion project for recording how facts change.

**Search from the current facts.** Reject out-of-scope, revoked, or outdated snapshots before they take up ranking slots. Combine phrase, word, optional embedding, and fresh-record results. Skip weak matches and check the canonical record again before returning it.

## Try it

Python 3.11 or newer. The demo and tests need no database, account, model, or network.

~~~sh
python3 -B demo.py
python3 -B -m unittest discover -s tests -v
~~~

The demo shows a dispatch window changing from Tuesday to Thursday: the old index stops answering, and the fresh record can answer before the index is rebuilt.

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
