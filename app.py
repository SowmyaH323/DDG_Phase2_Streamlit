import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import torch
import xgboost as xgb
from Bio.PDB import PDBParser

# -----------------------------
# Streamlit-safe output folder
# -----------------------------
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="ΔΔG Phase-2 Mutation Prioritization", layout="wide")

# -----------------------------
# DEBUG (check files in repo)
# -----------------------------
st.sidebar.write("CWD:", os.getcwd())
st.sidebar.write("Files:", os.listdir("."))

# -----------------------------
# Paths (repo root)
# -----------------------------
XGB_PATH  = Path("xgb_phase2_v4w_huber_weighted.json")
BIAS_PATH = Path("xgb_v4w_bias.txt")
CNN_PATH  = Path("cnn_phase2_v2_best.pt")
GNN_PATH  = Path("gnn_phase2_best.pt")  # <-- state_dict (NOT TorchScript)

DEVICE = "cpu"
parser = PDBParser(QUIET=True)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a:i for i,a in enumerate(AA_LIST)}

HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
         'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
VOLUME = {'A':88.6,'C':108.5,'D':111.1,'E':138.4,'F':189.9,'G':60.1,'H':153.2,'I':166.7,'K':168.6,'L':166.7,
          'M':162.9,'N':114.1,'P':112.7,'Q':143.8,'R':173.4,'S':89.0,'T':116.1,'V':140.0,'W':227.8,'Y':193.6}
CHARGE = {'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0}

def parse_mutation(mut: str):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut.strip().upper())
    if not m:
        raise ValueError(f"Bad mutation format: {mut} (expected like P66A)")
    return m.group(1), int(m.group(2)), m.group(3)

# ---------------------------------------------------------
# Minimal CNN & GNN architectures MUST match training
# If your saved files are FULL models, these are not used.
# If they are state_dict, these must match EXACTLY.
# ---------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class MutationAwareCNNv2(nn.Module):
    # IMPORTANT: set channels to match your checkpoint
    # From your earlier key shapes, first conv was [16,2,3,3]
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # mutation vector length=4 (pos, dh, dv, dq)
        self.mut_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, mut4=None):
        z = self.cnn(x).view(x.size(0), -1)
        if mut4 is None:
            mut4 = torch.zeros((x.size(0),4), device=x.device, dtype=x.dtype)
        m = self.mut_mlp(mut4)
        out = self.head(torch.cat([z, m], dim=1))
        return out.squeeze(-1)

class MutationAwareGNN(nn.Module):
    # If you trained with PyTorch Geometric, Streamlit Cloud may not have it.
    # So this GNN must match YOUR deployed inference version.
    # If your gnn_predict_one builds its own graph and does NOT rely on torch_geometric in Streamlit,
    # keep that same implementation here.
    def __init__(self, in_dim=25, hidden_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # placeholder “MLP over pooled node features”
        h = F.relu(self.lin1(x))
        h = F.relu(self.lin2(h))
        h = h.mean(dim=0, keepdim=True)
        return self.head(h).squeeze(-1)

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    # XGB
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))
    bias = float(BIAS_PATH.read_text().strip())

    # CNN: handle both full model OR state_dict
    cnn_obj = torch.load(CNN_PATH, map_location=DEVICE)
    if isinstance(cnn_obj, dict):
        cnn_model = MutationAwareCNNv2().to(DEVICE)
        cnn_model.load_state_dict(cnn_obj, strict=False)
    else:
        cnn_model = cnn_obj
    cnn_model.eval()

    # GNN: state_dict only (NO torch.jit.load)
    gnn_state = torch.load(GNN_PATH, map_location=DEVICE)
    if isinstance(gnn_state, dict):
        gnn_model = MutationAwareGNN(in_dim=25, hidden_dim=64).to(DEVICE)
        gnn_model.load_state_dict(gnn_state, strict=False)
    else:
        # if someone uploaded full model accidentally
        gnn_model = gnn_state
    gnn_model.eval()

    return booster, bias, cnn_model, gnn_model

booster, xgb_bias, cnn_model, gnn_model = load_models()
st.success("✅ Models loaded (XGB + CNN + GNN)")

# -----------------------------
# Predictors
# -----------------------------
def xgb_predict_one(mutation: str):
    wt, pos, mt = parse_mutation(mutation)
    wt_vec = np.zeros(20); mt_vec = np.zeros(20)
    wt_vec[AA_INDEX[wt]] = 1
    mt_vec[AA_INDEX[mt]] = 1
    dh = HYDRO[mt] - HYDRO[wt]
    dv = VOLUME[mt] - VOLUME[wt]
    dq = CHARGE[mt] - CHARGE[wt]
    feat = np.concatenate([wt_vec, mt_vec, np.array([pos, dh, dv, dq], dtype=np.float32)])
    pred = booster.predict(xgb.DMatrix(feat.reshape(1, -1)))[0] + xgb_bias
    return float(pred), True

def _get_residues_with_ca(pdb_path, chain_id):
    structure = parser.get_structure("p", pdb_path)
    residues = []
    for model in structure:
        for ch in model:
            if ch.id == chain_id:
                for res in ch:
                    if "CA" in res:
                        residues.append(res)
    return residues

def cnn_predict_one(pdb_path, chain_id, mutation):
    wt, pos, mt = parse_mutation(mutation)
    residues = _get_residues_with_ca(pdb_path, chain_id)
    if len(residues) == 0:
        return 0.0, False

    n = len(residues)
    coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)

    # contact map
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    cmap = (dist <= 8.0).astype(np.float32)

    # mutation mask
    idx = next((i for i, r in enumerate(residues) if r.id[1] == pos), None)
    if idx is None:
        return 0.0, False
    mask = np.zeros_like(cmap, dtype=np.float32)
    mask[idx, :] = 1.0
    mask[:, idx] = 1.0

    H = 128
    X = np.zeros((2, H, H), dtype=np.float32)
    X[0, :min(n,H), :min(n,H)] = cmap[:H, :H]
    X[1, :min(n,H), :min(n,H)] = mask[:H, :H]

    dh = HYDRO[mt] - HYDRO[wt]
    dv = VOLUME[mt] - VOLUME[wt]
    dq = CHARGE[mt] - CHARGE[wt]
    mut4 = np.array([pos, dh, dv, dq], dtype=np.float32)

    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        mt4 = torch.tensor(mut4, dtype=torch.float32).unsqueeze(0)
        y = cnn_model(xt, mt4).item()
    return float(y), True

def gnn_predict_one(pdb_path, chain_id, mutation):
    # NOTE: this uses a very simple pooled-node-feature MLP unless you implement your real GNN graph inference here.
    wt, pos, mt = parse_mutation(mutation)
    residues = _get_residues_with_ca(pdb_path, chain_id)
    if len(residues) == 0:
        return 0.0, False

    # node features (25): 20 one-hot + 3 props + 2 flags (mut idx, etc) if you want
    x = np.zeros((len(residues), 25), dtype=np.float32)
    for i, r in enumerate(residues):
        aa1 = r.get_resname()[0]
        if aa1 in AA_INDEX:
            x[i, AA_INDEX[aa1]] = 1.0

    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        y = gnn_model(xt).item()
    return float(y), True

def predict_ensemble_one(pdb_id, chain_id, mutation, weights=(1/3, 1/3, 1/3)):
    pdb_path = f"{pdb_id.upper()}.pdb"
    if not Path(pdb_path).exists():
        return {"error": f"PDB file not found in repo: {pdb_path}"}

    xgb_p, _ = xgb_predict_one(mutation)
    cnn_p, cnn_ok = cnn_predict_one(pdb_path, chain_id, mutation)
    gnn_p, gnn_ok = gnn_predict_one(pdb_path, chain_id, mutation)

    wx, wc, wg = weights
    ens = (wx * xgb_p) + (wc * cnn_p) + (wg * gnn_p)

    return {
        "PDB_ID": pdb_id.upper(),
        "chain": chain_id,
        "mutation": mutation,
        "xgb": xgb_p,
        "cnn": cnn_p,
        "gnn": gnn_p,
        "ens": float(ens),
        "cnn_ok": bool(cnn_ok),
        "gnn_ok": bool(gnn_ok),
    }

# -----------------------------
# UI
# -----------------------------
st.title("ΔΔG Phase-2 — Mutation Prioritization (XGB + CNN + GNN)")

col1, col2 = st.columns(2)
with col1:
    pdb_id = st.text_input("PDB ID (must be uploaded as PDB_ID.pdb in repo)", value="1RX4").strip().upper()
    chain_id = st.text_input("Chain", value="A").strip()
with col2:
    mutation = st.text_input("Mutation (e.g., P66A)", value="P66A").strip().upper()
    wx = st.slider("Weight XGB", 0.0, 1.0, 0.3333)
    wc = st.slider("Weight CNN", 0.0, 1.0, 0.3333)
    wg = st.slider("Weight GNN", 0.0, 1.0, 0.3333)

if st.button("Predict"):
    s = wx + wc + wg
    if s == 0:
        st.error("Weights sum to 0")
    else:
        weights = (wx/s, wc/s, wg/s)
        out = predict_ensemble_one(pdb_id, chain_id, mutation, weights=weights)
        if "error" in out:
            st.error(out["error"])
        else:
            st.json(out)







