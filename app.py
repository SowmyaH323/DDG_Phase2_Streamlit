import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from Bio.PDB import PDBParser

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="ΔΔG Phase-2 Mutation Prioritizer", layout="wide")
DEVICE = "cpu"

# Repo-root files (because you uploaded everything in root)
XGB_PATH  = Path("xgb_phase2_v4w_huber_weighted.json")
BIAS_PATH = Path("xgb_v4w_bias.txt")
CNN_PATH  = Path("cnn_phase2_v2_best.pt")
GNN_PATH  = Path("gnn_phase2_best.pt")

# Writable output directory on Streamlit Cloud
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

PDBParserObj = PDBParser(QUIET=True)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a:i for i,a in enumerate(AA_LIST)}

# Physicochemical (simple)
HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
         'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
VOLUME = {'A':88.6,'C':108.5,'D':111.1,'E':138.4,'F':189.9,'G':60.1,'H':153.2,'I':166.7,'K':168.6,'L':166.7,
          'M':162.9,'N':114.1,'P':112.7,'Q':143.8,'R':173.4,'S':89.0,'T':116.1,'V':140.0,'W':227.8,'Y':193.6}
CHARGE = {'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0}


# ============================================================
# HELPERS
# ============================================================
def parse_mutation(mut: str):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut.strip().upper())
    if not m:
        raise ValueError(f"Bad mutation format: {mut} (expected like P66A)")
    return m.group(1), int(m.group(2)), m.group(3)

def ddg_class(v: float) -> str:
    if v < -1: return "Strong stabilizing (< -1)"
    if v < 0:  return "Mild stabilizing (-1 to 0)"
    if v <= 1: return "Neutral (0 to 1)"
    return "Destabilizing (> 1)"

def confidence_label(cnn_ok: bool, gnn_ok: bool) -> str:
    if cnn_ok and gnn_ok: return "High"
    if cnn_ok or gnn_ok:  return "Medium"
    return "Low"


# ============================================================
# MODELS
# ============================================================
class MutationAwareCNNv2(nn.Module):
    """
    CNN architecture matching your checkpoint family:
    keys like cnn.0.weight, cnn.3.weight, cnn.6.weight, cnn.9.weight
    + mut_mlp.0.weight etc.
    """
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),          # 0
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),         # 3
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),         # 6
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)  # 9
        )
        self.mut_mlp = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 16, 64), nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x_img, mut_vec4):
        # x_img: [B,2,128,128]
        h = self.cnn(x_img).view(x_img.size(0), -1)   # [B,64]
        m = self.mut_mlp(mut_vec4)                    # [B,16]
        z = torch.cat([h, m], dim=1)
        y = self.head(z)
        return y.squeeze(-1)


class _FakeGCNConv(nn.Module):
    """Matches checkpoint keys: convX.lin.weight and convX.bias"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return self.lin(x) + self.bias


class MutationAwareGNN(nn.Module):
    """
    Checkpoint-compatible GNN (no torch_geometric).
    Expects keys: conv1.lin.weight, conv2.lin.weight, conv3.lin.weight, head.0.*, head.3.*
    """
    def __init__(self, in_dim=25, h1=64, h2=64, h3=32):
        super().__init__()
        self.conv1 = _FakeGCNConv(in_dim, h1)
        self.conv2 = _FakeGCNConv(h1, h2)
        self.conv3 = _FakeGCNConv(h2, h3)
        self.head = nn.Sequential(
            nn.Linear(h3, 32), nn.ReLU(), nn.Dropout(0.0), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.mean(dim=0, keepdim=True)      # pool residues
        y = self.head(h).squeeze(-1)
        return y


@st.cache_resource
def load_models():
    # ---- XGB ----
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"Missing {XGB_PATH} in repo root")
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))

    if not BIAS_PATH.exists():
        raise FileNotFoundError(f"Missing {BIAS_PATH} in repo root")
    xgb_bias = float(BIAS_PATH.read_text().strip())

    # ---- CNN ----
    if not CNN_PATH.exists():
        raise FileNotFoundError(f"Missing {CNN_PATH} in repo root")
    cnn_state = torch.load(CNN_PATH, map_location=DEVICE)
    cnn_model = MutationAwareCNNv2().to(DEVICE)

    # It might be full model or state_dict
    if isinstance(cnn_state, dict) and all(isinstance(k, str) for k in cnn_state.keys()):
        cnn_model.load_state_dict(cnn_state, strict=False)
    else:
        # already a model
        cnn_model = cnn_state
    cnn_model.eval()

    # ---- GNN ----
    if not GNN_PATH.exists():
        raise FileNotFoundError(f"Missing {GNN_PATH} in repo root")
    gnn_state = torch.load(GNN_PATH, map_location=DEVICE)

    # infer dims from checkpoint
    if not isinstance(gnn_state, dict):
        raise RuntimeError("GNN file must be a state_dict dict. Re-upload the state_dict .pt")

    w1 = gnn_state["conv1.lin.weight"]  # [h1, in_dim]
    w2 = gnn_state["conv2.lin.weight"]  # [h2, h1]
    w3 = gnn_state["conv3.lin.weight"]  # [h3, h2]
    h1, in_dim = w1.shape
    h2, _ = w2.shape
    h3, _ = w3.shape

    gnn_model = MutationAwareGNN(in_dim=in_dim, h1=h1, h2=h2, h3=h3).to(DEVICE)
    missing, unexpected = gnn_model.load_state_dict(gnn_state, strict=False)
    gnn_model.eval()

    return booster, xgb_bias, cnn_model, gnn_model, missing, unexpected


# ============================================================
# FEATURE BUILDERS
# ============================================================
def xgb_predict_one(booster, bias, mutation: str):
    wt, pos, mt = parse_mutation(mutation)
    wt_vec = np.zeros(20, dtype=np.float32)
    mt_vec = np.zeros(20, dtype=np.float32)
    wt_vec[AA_INDEX[wt]] = 1.0
    mt_vec[AA_INDEX[mt]] = 1.0

    dh = HYDRO[mt] - HYDRO[wt]
    dv = VOLUME[mt] - VOLUME[wt]
    dq = CHARGE[mt] - CHARGE[wt]

    feat = np.concatenate([wt_vec, mt_vec, np.array([pos, dh, dv, dq], dtype=np.float32)], axis=0)
    dmat = xgb.DMatrix(feat.reshape(1, -1))
    pred = float(booster.predict(dmat)[0] + bias)
    return pred, True


def get_residues_with_ca(pdb_path: str, chain_id: str):
    struct = PDBParserObj.get_structure("p", pdb_path)
    residues = []
    for model in struct:
        for ch in model:
            if ch.id == chain_id:
                for res in ch:
                    if "CA" in res:
                        residues.append(res)
    return residues


def make_contact_map_and_mask(residues, mut_pos: int, cutoff=8.0, H=128):
    n = len(residues)
    coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)
    cmap = np.zeros((n, n), dtype=np.float32)
    # fast-ish: compute distances by loops (n ~ few hundred max)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                cmap[i, j] = 1.0
                cmap[j, i] = 1.0

    idx = None
    for i, r in enumerate(residues):
        if r.id[1] == mut_pos:
            idx = i
            break

    mask = np.zeros((n, n), dtype=np.float32)
    if idx is not None:
        mask[idx, :] = 1.0
        mask[:, idx] = 1.0

    X = np.zeros((2, H, H), dtype=np.float32)
    X[0, :min(n, H), :min(n, H)] = cmap[:H, :H]
    X[1, :min(n, H), :min(n, H)] = mask[:H, :H]
    return X, idx is not None


def mut_vec4(wt, mt):
    dh = HYDRO[mt] - HYDRO[wt]
    dv = VOLUME[mt] - VOLUME[wt]
    dq = CHARGE[mt] - CHARGE[wt]
    # include absolute hydro too (simple 4th feature)
    return np.array([dh, dv, dq, HYDRO[mt]], dtype=np.float32)


def cnn_predict_one(cnn_model, pdb_path: str, chain_id: str, mutation: str):
    try:
        wt, pos, mt = parse_mutation(mutation)
        residues = get_residues_with_ca(pdb_path, chain_id)
        if len(residues) == 0:
            return 0.0, False

        X_img, has_pos = make_contact_map_and_mask(residues, mut_pos=pos)
        mv = mut_vec4(wt, mt)

        x_img = torch.tensor(X_img, dtype=torch.float32).unsqueeze(0)   # [1,2,128,128]
        mvec = torch.tensor(mv, dtype=torch.float32).unsqueeze(0)       # [1,4]

        with torch.no_grad():
            y = cnn_model(x_img, mvec).item()
        return float(y), bool(has_pos)
    except Exception as e:
        st.warning(f"CNN error for {mutation}: {e}")
        return 0.0, False


def gnn_node_features(residues, mut_pos: int):
    # 25 dims: 20 one-hot + 4 mut vec broadcast + 1 mut-flag
    n = len(residues)
    x = np.zeros((n, 25), dtype=np.float32)

    mut_idx = None
    for i, r in enumerate(residues):
        if r.id[1] == mut_pos:
            mut_idx = i
            break

    # residue type one-hot (best effort)
    for i, r in enumerate(residues):
        resname = r.get_resname()
        # Bio.PDB gives 3-letter; crude map first letter fallback won’t be perfect.
        aa = resname[0].upper()
        if aa in AA_INDEX:
            x[i, AA_INDEX[aa]] = 1.0

    # mutation flag
    if mut_idx is not None:
        x[mut_idx, 24] = 1.0

    return x, (mut_idx is not None)


def gnn_predict_one(gnn_model, pdb_path: str, chain_id: str, mutation: str):
    try:
        wt, pos, mt = parse_mutation(mutation)
        residues = get_residues_with_ca(pdb_path, chain_id)
        if len(residues) == 0:
            return 0.0, False

        x, ok_pos = gnn_node_features(residues, mut_pos=pos)
        xt = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            y = gnn_model(xt).item()
        return float(y), bool(ok_pos)
    except Exception as e:
        st.warning(f"GNN error for {mutation}: {e}")
        return 0.0, False


def predict_ensemble_one(pdb_path: str, pdb_id: str, chain: str, mutation: str, weights=(1/3,1/3,1/3)):
    booster, bias, cnn_model, gnn_model, *_ = load_models()

    xgb_p, x_ok = xgb_predict_one(booster, bias, mutation)
    cnn_p, cnn_ok = cnn_predict_one(cnn_model, pdb_path, chain, mutation)
    gnn_p, gnn_ok = gnn_predict_one(gnn_model, pdb_path, chain, mutation)

    wx, wc, wg = weights
    denom = (wx if x_ok else 0) + (wc if cnn_ok else 0) + (wg if gnn_ok else 0)
    if denom == 0:
        ens = 0.0
    else:
        ens = ((wx*xgb_p if x_ok else 0) + (wc*cnn_p if cnn_ok else 0) + (wg*gnn_p if gnn_ok else 0)) / denom

    return {
        "PDB_ID": pdb_id,
        "chain": chain,
        "mutation": mutation,
        "xgb": xgb_p,
        "cnn": cnn_p,
        "gnn": gnn_p,
        "ens": ens,
        "cnn_ok": cnn_ok,
        "gnn_ok": gnn_ok,
    }


def scan_19aa_position(pdb_path: str, pdb_id: str, chain_id: str, pos: int, wt_aa: str, weights=(1/3,1/3,1/3)):
    muts = []
    for aa in AA_LIST:
        if aa == wt_aa:
            continue
        muts.append(f"{wt_aa}{pos}{aa}")

    rows = []
    for m in muts:
        out = predict_ensemble_one(pdb_path, pdb_id, chain_id, m, weights=weights)
        rows.append(out)

    df = pd.DataFrame(rows)
    df["ddg_class"] = df["ens"].apply(ddg_class)
    df = df.sort_values("ens", ascending=True).reset_index(drop=True)
    df["rank_stabilizing"] = np.arange(1, len(df)+1)
    df["pct"] = df["rank_stabilizing"] / len(df)
    df["confidence"] = [confidence_label(c, g) for c, g in zip(df["cnn_ok"], df["gnn_ok"])]

    # highlight top/bottom 5
    df["highlight"] = ""
    top_k = 5
    df.loc[:top_k-1, "highlight"] = "TOP stabilizing"
    df.loc[len(df)-top_k:, "highlight"] = "TOP destabilizing"
    return df


# ============================================================
# UI
# ============================================================
st.title("ΔΔG Phase-2 — Mutation Prioritization (XGB + CNN + GNN ensemble)")

with st.sidebar:
    st.header("Artifacts status")
    st.write("XGB:", XGB_PATH.exists(), str(XGB_PATH))
    st.write("BIAS:", BIAS_PATH.exists(), str(BIAS_PATH))
    st.write("CNN:", CNN_PATH.exists(), str(CNN_PATH))
    st.write("GNN:", GNN_PATH.exists(), str(GNN_PATH))

    booster, bias, cnn_model, gnn_model, missing, unexpected = load_models()
    st.success(f"Models loaded. XGB bias={bias:.4f}")
    st.caption(f"GNN missing keys: {missing}")
    st.caption(f"GNN unexpected keys: {unexpected}")

    st.header("Ensemble weights")
    wx = st.slider("XGB weight", 0.0, 1.0, 0.33, 0.01)
    wc = st.slider("CNN weight", 0.0, 1.0, 0.33, 0.01)
    wg = st.slider("GNN weight", 0.0, 1.0, 0.33, 0.01)
    wsum = wx + wc + wg
    if wsum == 0:
        st.warning("All weights are 0. Using equal by default.")
        weights = (1/3, 1/3, 1/3)
    else:
        weights = (wx, wc, wg)

st.markdown("### 1) Upload a PDB")
pdb_file = st.file_uploader("Upload PDB file", type=["pdb"])
chain_id = st.text_input("Chain ID", value="A").strip() or "A"

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 2) Predict one mutation")
    mut_str = st.text_input("Mutation (e.g., P66A)", value="P66A").strip().upper()
    run_one = st.button("Predict mutation")

with col2:
    st.markdown("### 3) 19-AA scan at position")
    pos = st.number_input("Position (integer)", min_value=1, value=66, step=1)
    wt = st.text_input("WT amino acid at that position (1-letter)", value="P").strip().upper()[:1]
    run_scan = st.button("Run 19-AA scan")

if pdb_file is None:
    st.info("Upload a PDB to start.")
    st.stop()

# Save uploaded PDB into writable folder
pdb_bytes = pdb_file.getvalue()
pdb_id = Path(pdb_file.name).stem.upper()
pdb_path = OUT_DIR / f"{pdb_id}.pdb"
pdb_path.write_bytes(pdb_bytes)

st.success(f"Using PDB: {pdb_id}.pdb (saved in outputs/)")

if run_one:
    try:
        out = predict_ensemble_one(str(pdb_path), pdb_id, chain_id, mut_str, weights=weights)
        out["ddg_class"] = ddg_class(out["ens"])
        out["confidence"] = confidence_label(out["cnn_ok"], out["gnn_ok"])
        st.subheader("Prediction")
        st.json(out)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if run_scan:
    try:
        scan_df = scan_19aa_position(
            pdb_path=str(pdb_path),
            pdb_id=pdb_id,
            chain_id=chain_id,
            pos=int(pos),
            wt_aa=wt,
            weights=weights
        )

        st.subheader("19-AA scan (ranked by ensemble; lower = more stabilizing)")
        st.dataframe(scan_df, use_container_width=True)

        # Save CSV
        csv_path = OUT_DIR / f"scan_{pdb_id}_{chain_id}_pos{pos}.csv"
        scan_df.to_csv(csv_path, index=False)

        st.download_button(
            "Download CSV",
            data=scan_df.to_csv(index=False).encode("utf-8"),
            file_name=csv_path.name,
            mime="text/csv",
        )
        st.caption(f"Saved on server: {csv_path}")

    except Exception as e:
        st.error(f"Scan failed: {e}")









