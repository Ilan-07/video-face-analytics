"""Apply human-confirmed cross-scene merges from reports/merge_candidates.csv.

Reads the review sheet, unions face_ids whose `verdict` is "same", relabels
identities.csv accordingly, and recomputes analytics -> a human-certified
featured-cast count. Pairs left blank or "different" are not merged. The review
sheet only contains non-co-occurring pairs, so a confirmed merge can't violate
the co-occurrence constraint.
"""
import pandas as pd

import config
import util

log = util.get_logger()
SAME = {"same", "y", "yes", "1", "true"}


def run() -> int:
    cf = config.REPORT_DIR / "merge_candidates.csv"
    if not cf.exists():
        raise RuntimeError("no merge_candidates.csv; run review_merges.py first")
    cands = pd.read_csv(cf).fillna("")
    confirmed = [(r.face_a, r.face_b) for r in cands.itertuples()
                 if str(r.verdict).strip().lower() in SAME]
    if not confirmed:
        log.info("no confirmed merges (fill verdict=same in %s)", cf.name)
        return 0

    ident = pd.read_csv(config.IDENTITIES_CSV)
    named = ident["face_id"] != "unknown"
    before = ident[named]["face_id"].nunique()

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in confirmed:
        parent[find(a)] = find(b)

    ident["face_id"] = ident["face_id"].map(
        lambda f: find(f) if f in parent else f)

    # back up the pre-merge mapping, then persist
    (config.DATA / "identities.premerge.csv").write_text(
        config.IDENTITIES_CSV.read_text())
    ident.to_csv(config.IDENTITIES_CSV, index=False)
    after = ident[ident["face_id"] != "unknown"]["face_id"].nunique()
    log.info("applied %d confirmed merges: %d -> %d identities",
             len(confirmed), before, after)

    import analytics
    analytics.run()
    return len(confirmed)


if __name__ == "__main__":
    run()
