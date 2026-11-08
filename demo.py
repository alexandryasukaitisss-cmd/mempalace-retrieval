"""Run an offline example with fictional records."""
from dataclasses import replace
from retrieval import IndexedMemory, Memory, search

main = Memory("parcel", "team", "1", "Parcel dispatch window is Tuesday morning.", 1.0)
private = Memory("notes", "private", "1", "Parcel dispatch notes are private.", 2.0)
store = {record.memory_id: record for record in (main, private)}
index = [IndexedMemory(main), IndexedMemory(private)]


def retrieve(query, **kwargs):
    return search(query, index, allowed_scopes={"team"}, read_current=store.get, **kwargs)


if __name__ == "__main__":
    for hit in retrieve("parcel dispatch"):
        print("Before change:", hit.memory.text)
    updated = replace(main, revision="2", text="Parcel dispatch window is Thursday morning.", updated_at=3.0)
    store["parcel"] = updated
    print("With a stale index:", retrieve("parcel dispatch"))
    for hit in retrieve("parcel dispatch", fresh=[updated]):
        print("With the fresh record:", hit.memory.text)
    print("Weak evidence:", retrieve("orchard budget forecast"))
