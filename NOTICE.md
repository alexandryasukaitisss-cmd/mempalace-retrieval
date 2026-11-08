# Design and attribution

This is an independent adaptation of a local retrieval layer built alongside [MemPalace](https://github.com/MemPalace/mempalace) (MIT) and [Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger) (MIT). It does not bundle either project or their storage formats.

The extraction preserves eligibility before lane depth, weighted reciprocal rank fusion, evidence thresholds, and a final canonical read. It uses token and phrase matching instead of the original SQLite and stemming lanes. The host supplies fresh records and optional embeddings. Lexical coverage can qualify a candidate when no embeddings are supplied. The module returns no hit unless at least one candidate passes its own evidence check.

The MIT license covers this implementation and its fictional fixtures.
