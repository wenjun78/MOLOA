"""
predict.py — Score a mortgage across all three MOLOA phases from simple inputs.

This mirrors the interactive "What-if" scorer in moloa_demo.html. It uses the
same approximation of the trained-model behaviour, so it needs no model files,
no PyTorch, and no version-sensitive pickles — it just runs.

    Phase 1 (origination) -> Phase 2 (month 6) -> Phase 3 (distress)

Usage:
    # Score the built-in strong and weak example applicants
    python predict.py --demo

    # Score from a JSON file of applicant values
    python predict.py --input applicant_strong.json

    # Score directly from the command line
    python predict.py --fico 780 --dti 25 --ltv 70 --rate 6.2 --upb 250000 --months-late 0

JSON / CLI fields (all optional, sensible defaults applied):
    fico, dti, ltv, rate, upb, months_late,
    property_type (SF/PU/CO/MH), occupancy (P/S/I),
    purpose (P/C/N), first_time_homebuyer (Y/N)

Dependencies: none (standard library only).
"""

import argparse
import json
import math
import sys


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Phase approximations — ported directly from moloa_demo.html
# (approxPhase1 / approxPhase2 / approxPhase3). Geographic uses a fixed prior
# instead of the HTML's small random term, so CLI output is reproducible.
# ---------------------------------------------------------------------------

def approx_phase1(fico, dti, ltv, rate, upb, prop_type, fthb, occupancy, purpose):
    credit = clamp(0.4 + (700 - fico) * 0.005 + (dti - 35) * 0.012, 0.05, 0.95)
    collateral_base = 0.3 + max(0, ltv - 80) * 0.012
    prop_risk = {"SF": 0, "PU": 0.02, "CO": 0.04, "MH": 0.10}.get(prop_type, 0)
    occ_risk = {"P": 0, "S": 0.03, "I": 0.06}.get(occupancy, 0)
    collateral = clamp(collateral_base + prop_risk + occ_risk, 0.05, 0.95)
    geographic = 0.32  # fixed prior (HTML uses 0.30 + small noise)
    product_base = 0.35 + (rate - 5) * 0.04 + (0.05 if upb > 500000 else 0)
    purpose_risk = {"P": 0, "C": 0.03, "N": 0.01}.get(purpose, 0)
    product = clamp(product_base + purpose_risk, 0.05, 0.95)
    borrower = clamp(0.3 + (0.05 if fthb == "Y" else 0), 0.1, 0.9)

    gates = {"credit": 0.42, "collateral": 0.15, "geographic": 0.14,
             "product": 0.18, "borrower": 0.11}
    lar = (gates["credit"] * credit + gates["collateral"] * collateral
           + gates["geographic"] * geographic + gates["product"] * product
           + gates["borrower"] * borrower)
    specialist = {"credit": credit, "collateral": collateral,
                  "geographic": geographic, "product": product, "borrower": borrower}
    return {"lar": lar, "gates": gates, "specialist": specialist}


def approx_phase2(p1_lar, m6, ltv):
    trajectory = clamp(0.25 + m6 * 0.18, 0.05, 0.99)
    equity = clamp(0.3 + max(0, ltv - 80) * 0.012, 0.05, 0.99)
    prior = p1_lar
    if m6 >= 2:
        gates = {"trajectory": 0.70, "equity": 0.10, "priordecision": 0.20}
    elif m6 >= 1:
        gates = {"trajectory": 0.45, "equity": 0.10, "priordecision": 0.45}
    else:
        gates = {"trajectory": 0.07, "equity": 0.05, "priordecision": 0.88}
    lar = (gates["trajectory"] * trajectory + gates["equity"] * equity
           + gates["priordecision"] * prior)
    cox = math.exp(lar * 5 + m6 * 0.6) * 0.5
    return {"lar": lar, "gates": gates,
            "specialist": {"trajectory": trajectory, "equity": equity, "priordecision": prior},
            "cox": cox}


def approx_phase3(distressed, fico, ltv):
    if not distressed:
        return {"triggered": False}
    cate = 25 + (700 - fico) * 0.05 - (ltv - 80) * 0.15
    cate = clamp(cate, -30, 80)
    cate_lo, cate_hi = cate - 30, cate + 25
    rec = "SPECIALIST"
    if cate > 5 and cate_lo > 0:
        rec = "MODIFY"
    elif cate > -2 and (cate_hi - cate_lo) < 30:
        rec = "FORBEAR"
    return {"triggered": True, "cate": cate, "cate_lo": cate_lo,
            "cate_hi": cate_hi, "recommendation": rec}


# ---------------------------------------------------------------------------
# Scoring + presentation
# ---------------------------------------------------------------------------

DEFAULTS = {
    "fico": 740, "dti": 30, "ltv": 80, "rate": 6.5, "upb": 250000,
    "months_late": 0, "property_type": "SF", "occupancy": "P",
    "purpose": "P", "first_time_homebuyer": "N",
}


def normalise(app: dict) -> dict:
    a = dict(DEFAULTS)
    for k, v in app.items():
        key = k.lower().replace("-", "_")
        if key in a:
            a[key] = v
    return a


def score(app: dict) -> dict:
    a = normalise(app)
    fico, dti, ltv = int(a["fico"]), int(a["dti"]), int(a["ltv"])
    rate, upb = float(a["rate"]), int(a["upb"])
    m6 = int(a["months_late"])
    fthb = str(a["first_time_homebuyer"]).upper()[:1]

    p1 = approx_phase1(fico, dti, ltv, rate, upb, a["property_type"], fthb,
                       a["occupancy"], a["purpose"])
    p1_decision = "DENY / ESCALATE" if p1["lar"] > 0.50 else "APPROVE"
    dom1 = max(p1["gates"], key=p1["gates"].get)

    p2 = approx_phase2(p1["lar"], m6, ltv)
    p2_decision = "WATCHLIST" if p2["lar"] > 0.55 else "CONTINUE"
    dom2 = max(p2["gates"], key=p2["gates"].get)

    distressed = m6 >= 2
    p3 = approx_phase3(distressed, fico, ltv)

    return {"inputs": a, "p1": p1, "p1_decision": p1_decision, "dom1": dom1,
            "p2": p2, "p2_decision": p2_decision, "dom2": dom2, "p3": p3}


BAR = "=" * 66


def render(app_label, r):
    a = r["inputs"]
    print(BAR)
    print(f"  MOLOA lifecycle score — {app_label}")
    print(f"  FICO {a['fico']} · DTI {a['dti']}% · LTV {a['ltv']}% · "
          f"rate {a['rate']} · UPB ${a['upb']:,} · {a['months_late']} mo late")
    print(BAR)

    print("\n[ PHASE 1 — ORIGINATION (month 0) ]")
    print(f"  Default probability : {r['p1']['lar']*100:.1f}%")
    print(f"  Decision            : {r['p1_decision']}")
    print(f"  Router leaned on    : {r['dom1'].capitalize()}Agent "
          f"({r['p1']['gates'][r['dom1']]*100:.0f}% of the weight)")

    print("\n[ PHASE 2 — EARLY MONITORING (month 6) ]")
    print(f"  Updated probability : {r['p2']['lar']*100:.1f}%")
    print(f"  Cox partial hazard  : {r['p2']['cox']:.3f}")
    print(f"  Monitoring decision : {r['p2_decision']}")
    print(f"  Router leaned on    : {r['dom2'].capitalize()}")

    print("\n[ PHASE 3 — DISTRESS INTERVENTION (first 60+DPD) ]")
    if r["p3"]["triggered"]:
        p3 = r["p3"]
        print(f"  Estimated treatment effect : {p3['cate']:+.1f} pts "
              f"(CI [{p3['cate_lo']:+.1f}, {p3['cate_hi']:+.1f}])")
        print(f"  Recommended action         : {p3['recommendation']}")
    else:
        print("  Loan is not distressed — no intervention needed.")
    print(BAR + "\n")


DEMO_STRONG = {"fico": 790, "dti": 22, "ltv": 65, "rate": 6.1, "upb": 210000, "months_late": 0}
DEMO_WEAK = {"fico": 585, "dti": 52, "ltv": 97, "rate": 8.9, "upb": 420000, "months_late": 3}


def main():
    ap = argparse.ArgumentParser(description="Score a mortgage across all 3 MOLOA phases.")
    ap.add_argument("--demo", action="store_true", help="Score built-in strong & weak applicants.")
    ap.add_argument("--input", type=str, help="Path to a JSON file of applicant values.")
    for f in ["fico", "dti", "ltv", "upb", "months-late"]:
        ap.add_argument(f"--{f}", type=int)
    ap.add_argument("--rate", type=float)
    ap.add_argument("--property-type", type=str)
    ap.add_argument("--occupancy", type=str)
    ap.add_argument("--purpose", type=str)
    ap.add_argument("--first-time-homebuyer", type=str)
    args = ap.parse_args()

    if args.demo:
        render("STRONG applicant", score(DEMO_STRONG))
        render("WEAK applicant", score(DEMO_WEAK))
        return

    if args.input:
        with open(args.input) as f:
            render(args.input, score(json.load(f)))
        return

    # CLI-flag mode: gather any provided flags
    cli = {k: v for k, v in vars(args).items()
           if v is not None and k not in ("demo", "input")}
    if cli:
        render("command-line applicant", score(cli))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
