"""
demo_full.py — Trace one loan through all three phases of MOLOA.

This is the presentation demo. It reads the prediction files the notebook
produced (phase1/2/3_predictions.parquet) and follows a single loan across
its whole lifecycle:

    Phase 1 (origination)  -> Phase 2 (month 6)  -> Phase 3 (distress)

Because it reads finished predictions rather than re-running the models, it
is fast and has no heavy dependencies — ideal for a live demo.

Usage:
    # Auto-pick an interesting loan that appears in all three phases
    python demo_full.py

    # Trace a specific loan
    python demo_full.py --loan F22Q1000123

    # List a few good demo candidates and exit
    python demo_full.py --list

Requires (download from your Google Drive MOLOA_data folder):
    phase1_predictions.parquet
    phase2_predictions.parquet
    phase3_predictions.parquet

Set the folder with MOLOA_DATA_DIR (defaults to ./demo, where the parquets are committed).
Dependencies: pandas, pyarrow  (see requirements.txt)
"""

import argparse
import os
import sys

import pandas as pd

DATA_DIR = os.environ.get("MOLOA_DATA_DIR", "./demo")

P1 = os.path.join(DATA_DIR, "phase1_predictions.parquet")
P2 = os.path.join(DATA_DIR, "phase2_predictions.parquet")
P3 = os.path.join(DATA_DIR, "phase3_predictions.parquet")

KEY = "LOAN_SEQ_NO"

# ---- pretty-printing helpers ------------------------------------------------

BAR = "=" * 66
SUB = "-" * 66


def _need(path, label):
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: {label} not found at '{path}'.\n"
            f"Download the phase*_predictions.parquet files from your Google "
            f"Drive MOLOA_data folder into '{DATA_DIR}', or set MOLOA_DATA_DIR."
        )


def load():
    _need(P1, "Phase 1 predictions")
    p1 = pd.read_parquet(P1)
    p2 = pd.read_parquet(P2) if os.path.exists(P2) else None
    p3 = pd.read_parquet(P3) if os.path.exists(P3) else None
    return p1, p2, p3


def pick_demo_loan(p1, p2, p3):
    """Choose a loan that appears in all three phases and tells a good story:
    approved at origination, but later distressed — the lifecycle payoff."""
    ids = set(p1[KEY])
    if p2 is not None:
        ids &= set(p2[KEY])
    if p3 is not None:
        ids &= set(p3[KEY])
    if not ids:
        # fall back to any loan present in Phase 1
        return p1[KEY].iloc[0]

    candidates = p3[p3[KEY].isin(ids)] if p3 is not None else p1[p1[KEY].isin(ids)]
    # Prefer a distressed loan whose modification had a large positive CATE
    if p3 is not None and "phase3_cate_point" in candidates.columns:
        candidates = candidates.sort_values("phase3_cate_point", ascending=False)
    return candidates[KEY].iloc[0]


def row_for(df, loan):
    if df is None:
        return None
    hit = df[df[KEY] == loan]
    return hit.iloc[0] if len(hit) else None


def fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None and pd.notna(x) else "—"


def show(loan, p1, p2, p3):
    r1 = row_for(p1, loan)
    r2 = row_for(p2, loan)
    r3 = row_for(p3, loan)

    print(BAR)
    print(f"  MOLOA — lifecycle trace for loan  {loan}")
    print(BAR)

    # ---------- PHASE 1 ----------
    print("\n[ PHASE 1 — ORIGINATION (month 0) ]")
    if r1 is not None:
        dec = "APPROVE" if int(r1["phase1_lar_decision"]) == 0 else "DENY / ESCALATE"
        print(f"  Routed default probability : {fmt_pct(r1['phase1_lar_proba'])}")
        print(f"  Decision                   : {dec}")
        # Which agents the router leaned on
        gates = {c.replace("phase1_lar_gate_", ""): r1[c]
                 for c in r1.index if c.startswith("phase1_lar_gate_")}
        if gates:
            top = max(gates, key=gates.get)
            print(f"  Router leaned most on      : {top.capitalize()}Agent "
                  f"({fmt_pct(gates[top])} of the weight)")
    else:
        print("  (loan not found in Phase 1)")

    # ---------- PHASE 2 ----------
    print("\n[ PHASE 2 — EARLY MONITORING (month 6) ]")
    if r2 is not None:
        watch = "WATCHLIST" if int(r2["phase2_lar_decision"]) == 1 else "CONTINUE"
        print(f"  Updated default probability: {fmt_pct(r2['phase2_lar_proba'])}")
        print(f"  Cox partial hazard         : {r2['phase2_cox_partial_hazard']:.3f}")
        print(f"  Monitoring decision        : {watch}")
        if r1 is not None:
            delta = r2["phase2_lar_proba"] - r1["phase1_lar_proba"]
            arrow = "risk ROSE" if delta > 0 else "risk fell"
            print(f"  Change since origination   : {arrow} by {abs(delta)*100:.1f} pts")
    else:
        print("  (loan was not observed at month 6)")

    # ---------- PHASE 3 ----------
    print("\n[ PHASE 3 — DISTRESS INTERVENTION (first 60+DPD) ]")
    if r3 is not None:
        rec = r3.get("phase3_intervention_recommendation", "—")
        p_mod = r3.get("phase3_cure_proba_if_modified")
        p_not = r3.get("phase3_cure_proba_if_not_modified")
        cate = r3.get("phase3_cate_point")
        print(f"  Propensity to be modified  : {fmt_pct(r3.get('phase3_propensity_score'))}")
        print(f"  Cure chance IF modified    : {fmt_pct(p_mod)}")
        print(f"  Cure chance IF NOT modified: {fmt_pct(p_not)}")
        if cate is not None and pd.notna(cate):
            print(f"  Estimated treatment effect : {cate*100:+.1f} pts "
                  f"(CI [{r3['phase3_cate_lo']*100:+.1f}, {r3['phase3_cate_hi']*100:+.1f}])")
        print(f"  Recommended action         : {rec}")
    else:
        print("  (loan never became distressed — no intervention needed)")

    print("\n" + BAR)
    print("  End of lifecycle trace.")
    print(BAR)


def list_candidates(p1, p2, p3, n=8):
    ids = set(p1[KEY])
    if p2 is not None:
        ids &= set(p2[KEY])
    if p3 is not None:
        ids &= set(p3[KEY])
    print(f"Loans present in all three phases (showing up to {n}):\n")
    if p3 is not None and "phase3_cate_point" in p3.columns:
        pool = p3[p3[KEY].isin(ids)].sort_values("phase3_cate_point", ascending=False)
    else:
        pool = p1[p1[KEY].isin(ids)]
    for loan in pool[KEY].head(n):
        print(f"  python demo_full.py --loan {loan}")


def main():
    ap = argparse.ArgumentParser(description="Trace one loan through all 3 MOLOA phases.")
    ap.add_argument("--loan", type=str, help="Specific LOAN_SEQ_NO to trace.")
    ap.add_argument("--list", action="store_true", help="List good demo loans and exit.")
    args = ap.parse_args()

    p1, p2, p3 = load()

    if args.list:
        list_candidates(p1, p2, p3)
        return

    loan = args.loan or pick_demo_loan(p1, p2, p3)
    if args.loan and loan not in set(p1[KEY]):
        sys.exit(f"Loan '{loan}' not found in Phase 1 predictions.")
    show(loan, p1, p2, p3)


if __name__ == "__main__":
    main()
