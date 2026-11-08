from dataclasses import replace
import math
import unittest

from retrieval import Embedding, IndexedMemory, Memory, search

CURRENT = Memory("pickup", "team", "2", "Train pickup window is Friday.", 2.0)


def retrieve(query, records=(IndexedMemory(CURRENT),), **kwargs):
    params = {"allowed_scopes": {"team"}, "read_current": {CURRENT.memory_id: CURRENT}.get}
    return search(query, records, **(params | kwargs))


class RetrievalTest(unittest.TestCase):
    def test_denied_and_stale_records_cannot_fill_lane_depth(self):
        denied = Memory("a-private", "private", "1", "Train pickup", 3.0)
        revoked = Memory("a-revoked", "team", "1", "Train pickup", 3.0)
        stale = Memory("a-stale", "team", "1", "Train pickup", 3.0)
        store = {
            "a-revoked": replace(revoked, status="revoked"),
            "a-stale": replace(stale, revision="2"),
            "pickup": CURRENT,
        }
        looked_up = []

        def current(mid):
            looked_up.append(mid)
            return store.get(mid)

        hits = retrieve(
            "train pickup",
            [IndexedMemory(record) for record in (denied, revoked, stale, CURRENT)],
            read_current=current, limit=1, depth=1,
        )
        self.assertEqual([hit.memory for hit in hits], [CURRENT])
        self.assertNotIn("a-private", looked_up)
        self.assertEqual(hits[0].lanes, ("phrase", "lexical"))

    def test_canonical_revalidation_drops_changed_results(self):
        for change in (
            None, replace(CURRENT, status="revoked"), replace(CURRENT, scope="private"),
            replace(CURRENT, text="Train pickup window is Saturday."), replace(CURRENT, revision="3"),
            {"text": "Unexpected adapter result"},
        ):
            with self.subTest(change=change):
                sequence = iter((CURRENT, change))
                self.assertEqual(retrieve("train pickup", read_current=lambda _: next(sequence)), [])

    def test_fresh_lane_can_serve_replacement_missing_from_index(self):
        stale = replace(CURRENT, revision="1", text="Train pickup is Thursday.")
        records = [IndexedMemory(stale)]
        self.assertEqual(retrieve("train pickup", records), [])
        hits = retrieve("train pickup", records, fresh=[CURRENT, CURRENT])
        self.assertEqual([hit.memory for hit in hits], [CURRENT])
        self.assertIn("fresh", hits[0].lanes)

    def test_abstains_with_only_one_of_three_terms(self):
        self.assertEqual(retrieve("train orchid apple"), [])
        self.assertEqual(retrieve("train pickup"), retrieve("train pickup"))

    def test_valid_dense_evidence_and_embedding_space_compatibility(self):
        query = Embedding("toy-v1", (1.0, 0.0))
        records = [IndexedMemory(CURRENT, Embedding("toy-v1", (1.0, 0.0)))]
        self.assertEqual(retrieve("railway schedule", records, query_embedding=query)[0].lanes, ("dense",))
        for invalid in (
            Embedding("different", (1.0, 0.0)), Embedding("toy-v1", (1.0,)),
            Embedding("toy-v1", (math.nan, 0.0)), Embedding("toy-v1", (math.inf, 0.0)),
            Embedding("toy-v1", (0.0, 0.0)), Embedding("toy-v1", (True, 0.0)),
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(retrieve(
                    "railway schedule", [IndexedMemory(CURRENT, invalid)], query_embedding=query,
                ), [])
                self.assertEqual(len(retrieve(
                    "train pickup", [IndexedMemory(CURRENT, invalid)], query_embedding=query,
                )), 1)
        with self.assertRaisesRegex(ValueError, "invalid embedding"):
            retrieve("train pickup", query_embedding=Embedding("toy-v1", (math.nan, 0.0)))

    def test_dense_and_lexical_agreement_needs_evidence_from_both(self):
        records = [IndexedMemory(CURRENT, Embedding("toy-v1", (0.5, math.sqrt(0.75))))]
        query = Embedding("toy-v1", (1.0, 0.0))
        self.assertEqual(retrieve("train orchid", records), [])
        self.assertEqual(retrieve("orchid apple", records, query_embedding=query), [])
        self.assertEqual(retrieve("train orchid", records, query_embedding=query)[0].lanes, ("lexical", "dense"))

    def test_unicode_matching_preserves_the_stored_text(self):
        original = Memory("note", "team", "1", "Café рядом с парком 👩‍🔬", 1.0)
        hits = retrieve(" \ufeffCAFE\u0301\u200b\u00a0рядом ", [IndexedMemory(original)], read_current=lambda _: original)
        self.assertEqual(hits[0].memory.text, original.text)
        for empty in ("", "   ", "\x00\u200b\ufeff", "как это", "👩‍🔬"):
            with self.subTest(empty=empty):
                self.assertEqual(retrieve(empty), [])
        for bad in (None, "train\ud800", "x" * 4097):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                retrieve(bad)

    def test_adapter_failure_propagates_without_returning_index_text(self):
        def unavailable(_):
            raise RuntimeError("store unavailable")

        with self.assertRaisesRegex(RuntimeError, "store unavailable"):
            retrieve("train pickup", read_current=unavailable)
        with self.assertRaises(ValueError):
            retrieve("train pickup", allowed_scopes="team")
        self.assertEqual(retrieve("train pickup", allowed_scopes=set()), [])

    def test_ties_are_deterministic_and_duplicate_rows_do_not_boost_scores(self):
        other = replace(CURRENT, memory_id="earlier")
        records = [IndexedMemory(CURRENT), IndexedMemory(other)]
        params = {"read_current": {x.memory_id: x for x in (CURRENT, other)}.get}
        hits = retrieve("train pickup", records, **params)
        self.assertEqual([hit.memory.memory_id for hit in hits], ["earlier", "pickup"])
        self.assertEqual(retrieve("train pickup", records[::-1] + records, **params), hits)


if __name__ == "__main__":
    unittest.main()
