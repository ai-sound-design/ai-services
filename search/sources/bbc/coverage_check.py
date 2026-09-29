"""Check how much of the BBC archive the harvest plan actually reaches.

Runs the same slicing as harvest.py but writes nothing. Reports the number of
distinct sounds reached, how many the API total claims, and where the
difference goes: records without an id or description (unusable for text
retrieval, skipped by to_row) and slices that exceed the 1000-result window
and cannot be split further.

    docker compose run --rm --no-deps indexer python sources/bbc/coverage_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harvest  # noqa: E402

plan = harvest.slices((0, harvest.MAX_DURATION))
print("slices:", len(plan), flush=True)
ids, skipped, capped = set(), 0, 0
for duration, categories in plan:
    total, results = harvest.query(0, harvest.MAX_WINDOW, duration, categories)
    if total > harvest.MAX_WINDOW:
        capped += total - harvest.MAX_WINDOW
    for record in results:
        row = harvest.to_row(record)
        if not row:
            skipped += 1
            continue
        ids.add(row["external_id"])
print("distinct ids reached:", len(ids))
print("skipped by to_row:", skipped)
print("lost to capped slices (>1000, unsplittable):", capped)
