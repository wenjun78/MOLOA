"""
predict.py — Score a single mortgage application with the MOLOA Phase 1 pipeline.

A standalone command-line wrapper around the trained Phase 1 artifacts
(five specialist agents + the Learned Agent Router). It demonstrates that the
research pipeline runs as a tool, separate from the notebook.

Usage:
    # Score the built-in example applicant
    python predict.py --demo

    # Score from a JSON file mapping feature name -> value
    python predict.py --input applicant.json

Requires the trained artifact produced by the notebook:
    phase1_models.pkl

Set its location with the MOLOA_DATA_DIR environment variable
(defaults to ./MOLOA_data).

Dependencies: numpy, torch, xgboost, scikit-learn  (see requirements.txt)
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

# torch and xgboost are needed at unpickle time because the saved objects
# (LearnedAgentRouter state_dict, SpecialistAgent.model) reference them.
import torch
import torch.nn as nn


DATA_DIR = os.environ.get("MOLOA_DATA_DIR", "./MOLOA_data")
MODELS_PATH = os.path.join(DATA_DIR, "phase1_models.pkl")


# ---------------------------------------------------------------------------
# Model class definitions.
#
# These MUST match the classes the notebook used when it pickled the artifacts,
# so that pickle.load() can reconstruct the SpecialistAgent objects, and so we
# can rebuild the LearnedAgentRouter and load its saved state_dict.
# ---------------------------------------------------------------------------

class SpecialistAgent:
    """XGBoost classifier on a non-overlapping feature partition.

    Reconstructed at unpickle time. Only the attributes used for inference
    (`col_indices`, `model`) are relied upon here.
    """

    def __init__(self, name=None, feature_names=None, spw_value=None,
                 early_stopping_rounds=40):
        self.name = name
        self.feature_names = feature_names
        self.col_indices = None  # populated from the pickle
        self.model = None        # populated from the pickle

    def predict(self, X):
        """Return (recommendation, confidence, reasoning, probability)."""
        Xd = X[:, self.col_indices]
        proba = self.model.predict_proba(Xd)[:, 1]
        rec = (proba >= 0.5).astype(int)
        conf = np.abs(proba - 0.5) * 2
        return rec, conf, None, proba


class LearnedAgentRouter(nn.Module):
    """Per-loan attention gate over the specialist agents (Contribution 1)."""

    def __init__(self, n_features, n_agents=5, hidden=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(n_agents * 2 + n_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_features, agent_probas, agent_confs):
        gate_input = torch.cat([agent_probas, agent_confs, x_features], dim=1)
        gates = torch.softmax(self.gate_net(gate_input), dim=1)
        weighted = (gates * agent_probas).sum(dim=1)
        return torch.sigmoid(weighted + self.bias), gates


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_artifacts(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: trained models not found at '{path}'.\n"
            f"Run the notebook (notebooks/MOLOA_full_pipeline.ipynb) first to "
            f"produce phase1_models.pkl, or set MOLOA_DATA_DIR to its folder."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def build_feature_vector(applicant: dict, all_features: list) -> np.ndarray:
    """Lay the applicant's values out in the exact ALL_FEATURES order.

    Missing features are filled with NaN — XGBoost handles missing natively,
    which matches the notebook's sentinel handling.
    """
    row = np.full(len(all_features), np.nan, dtype=np.float32)
    unknown = []
    for k, v in applicant.items():
        if k in all_features:
            try:
                row[all_features.index(k)] = float(v)
            except (TypeError, ValueError):
                unknown.append(k)  # non-numeric (e.g. categorical string)
        else:
            unknown.append(k)
    if unknown:
        print(f"  (note: ignored {len(unknown)} unrecognised/non-numeric "
              f"feature(s): {', '.join(unknown)})", file=sys.stderr)
    return row.reshape(1, -1)


def score(applicant: dict, art: dict) -> dict:
    all_features = art["ALL_FEATURES"]
    agents = art["agents"]
    scaler = art["scaler_full"]
    threshold = art.get("best_t_lar_refit", art.get("best_t_lar_train", 0.5))

    # 1. Build the feature vector in the trained order
    X = build_feature_vector(applicant, all_features)

    # 2. Each specialist agent emits a probability + confidence
    agent_names = list(agents.keys())
    probas = np.column_stack([agents[n].predict(X)[3] for n in agent_names])
    confs = np.column_stack([agents[n].predict(X)[1] for n in agent_names])

    # 3. Rebuild the LAR and load its trained weights
    lar = LearnedAgentRouter(len(all_features), n_agents=len(agent_names))
    lar.load_state_dict(art["lar_state_dict"])
    lar.eval()

    # 4. Forward pass: scaled features + agent signals -> routed probability
    X_scaled = torch.FloatTensor(scaler.transform(np.nan_to_num(X)))
    p_t = torch.FloatTensor(probas)
    c_t = torch.FloatTensor(confs)
    with torch.no_grad():
        routed_proba, gates = lar(X_scaled, p_t, c_t)

    default_prob = float(routed_proba.item())
    decision = "DENY / ESCALATE" if default_prob >= threshold else "APPROVE"

    # Per-agent contribution = how much the router leaned on each agent
    gate_weights = {n: round(float(g), 4)
                    for n, g in zip(agent_names, gates.squeeze().tolist())}
    agent_probs = {n: round(float(p), 4)
                   for n, p in zip(agent_names, probas.ravel().tolist())}

    return {
        "default_probability": round(default_prob, 4),
        "decision_threshold": round(float(threshold), 4),
        "decision": decision,
        "agent_probabilities": agent_probs,
        "router_gate_weights": gate_weights,
    }


# A representative low-risk applicant for --demo.
# Feature names must match those in the trained ALL_FEATURES; unknown keys are
# ignored with a note, so this works as an illustration regardless of schema.
DEMO_APPLICANT = {
    "CREDIT_SCORE": 760,
    "ORIG_DTI": 28,
    "ORIG_LTV": 75,
    "ORIG_UPB": 240000,
    "ORIG_INTEREST_RATE": 6.5,
    "ORIG_LOAN_TERM": 360,
    "NUM_UNITS": 1,
}


def main():
    parser = argparse.ArgumentParser(
        description="Score a mortgage application with MOLOA Phase 1.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true",
                       help="Score the built-in example applicant.")
    group.add_argument("--input", type=str,
                       help="Path to a JSON file mapping feature name -> value.")
    args = parser.parse_args()

    if args.demo:
        applicant = DEMO_APPLICANT
        print("Scoring built-in demo applicant...\n", file=sys.stderr)
    else:
        with open(args.input) as f:
            applicant = json.load(f)

    art = load_artifacts(MODELS_PATH)
    result = score(applicant, art)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
