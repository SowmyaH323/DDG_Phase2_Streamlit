import io
import re
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from Bio.PDB import PDBParser

# -----------------------------
# PyG (GNN)
# -----------------------------
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYG_OK = True
except Exception:
    PYG_OK = False

st.set_page_config(page_title="ΔΔG Phase-2 Mutation Scanner", layout="wide")

DEVICE = "cpu"
PARSER = PDBParser(QUIET=True)

# -----------------------------
# Repo paths (files in repo root)
# -----------------------------
REPO = Path(".")
XGB_PATH  = REPO / "xgb_phase2_v4w_huber_weighted.json"
BIAS_PATH = REPO / "xgb_v4w_bias.txt"
CNN_PATH  = REPO / "cnn_phase2_v2_best.pt"
GNN_PATH  = REPO / "gnn_phase2_best.pt"

# Temp storage (Streamlit Cloud allows /tmp)
TMP_DIR = Path("/tmp/ddg_phase2")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# AA utilities
# -----------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA_LIST)}

AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
         'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
VOLUME = {'A':88.6,'C':108.5,'D':111.1,'E':138.4,'F':189.9,'G':60.1,'H':153.2,'I':166.7,'K':168.6,
          'L':166.7,'M':162.9,'N':114.1,'P':112.7,'Q':143.8,'R':173.4,'S':89.0,'T':116.1,'V':140.0,
          'W':227.8,'Y':193.6}
CHARGE = {'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,'M':0,'N':0,'P':0,
          'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0}

def parse_mutation(mut: str):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut.strip().upper())
    if not m:
        raise ValueError(f"Bad mutation: {mut}")
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

# -----------------------------
# Phase-2 CNN architecture
# (matches your checkpoint key patterns)
# -----------------------------
class MutationAwareCNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.mut_mlp = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU()
        )

        # head.3 exists because of Dropout at index 2
        self.head = nn.Sequential(
            nn.Linear(80, 64), nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x, mut_vec):
        z = self.cnn(x).view(x.size(0), -1)     # (B,64)
        m = self.mut_mlp(mut_vec)               # (B,16)
        h = torch.cat([z, m], dim=1)            # (B,80)
        y = self.head(h)                        # (B,1)
        return y.squeeze(-1)

# -----------------------------
# Phase-2 GNN architecture
# -----------------------------
class MutationAwareGNN(nn.Module):
    def __init__(self, in_dim=25, hidden_dim=64):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError("torch_geometric not available")

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)

        self.head = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(32, 1)
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        g = global_mean_pool(x, batch)
        y = self.head(g)
        return y.squeeze(-1)

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))
    xgb_bias = float(BIAS_PATH.read_text().strip()) if BIAS_PATH.exists() else 0.0

    # CNN
    cnn_obj = torch.load(CNN_PATH, map_location=DEVICE)
    if isinstance(cnn_obj, dict):
        cnn_model = MutationAwareCNNv2().to(DEVICE)
        inc = cnn_model.load_state_dict(cnn_obj, strict=False)
        cnn_missing = list(getattr(inc, "missing_keys", []))
        cnn_unexpected = list(getattr(inc, "unexpected_keys", []))
    else:
        cnn_model = cnn_obj
        cnn_missing, cnn_unexpected = [], []
    cnn_model.eval()

    # GNN
    gnn_obj = torch.load(GNN_PATH, map_location=DEVICE)
    if isinstance(gnn_obj, dict):
        if not PYG_OK:
            raise RuntimeError("GNN is a state_dict but torch_geometric is not available.")
        gnn_model = MutationAwareGNN(in_dim=25, hidden_dim=64).to(DEVICE)
        inc2 = gnn_model.load_state_dict(gnn_obj, strict=False)
        gnn_missing = list(getattr(inc2, "missing_keys", []))
        gnn_unexpected = list(getattr(inc2, "unexpected_keys", []))
    else:
        gnn_model = gnn_obj
        gnn_missing, gnn_unexpected = [], []
    gnn_model.eval()

    meta = {
        "cnn_missing": cnn_missing,
        "cnn_unexpected": cnn_unexpected,
        "gnn_missing": gnn_missing,
        "gnn_unexpected": gnn_unexpected,
    }
    return booster, xgb_bias, cnn_model, gnn_model, meta

# -----------------------------
# Structure helpers
# -----------------------------
def get_chain_residues_with_ca(structure, chain_id: str):
    residues = []
    for model in structure:
        for ch in model:
            if ch.id == chain_id:
                for res in ch:
                    if "CA" in res and res.get_resname().upper() in AA3_TO_1:
                        residues.append(res)
    return residues

def wt_aa_from_pdb(pdb_path: str, chain_id: str, pos: int):
    structure = PARSER.get_structure("p", pdb_path)
    residues = get_chain_residues_with_ca(structure, chain_id)
    for r in residues:
        if r.id[1] == pos:
            return AA3_TO_1.get(r.get_resname().upper(), None)
    return None

# -----------------------------
# Predictions
# -----------------------------
def xgb_predict_one(booster, bias: float, mutation: str):
    wt, pos, mt = parse_mutation(mutation)

    wt_vec = np.zeros(20, dtype=np.float32)
    mt_vec = np.zeros(20, dtype=np.float32)
    wt_vec[AA_INDEX[wt]] = 1.0
    mt_vec[AA_INDEX[mt]] = 1.0

    feat = np.concatenate([
        wt_vec, mt_vec,
        np.array([pos,
                  HYDRO[mt] - HYDRO[wt],
                  VOLUME[mt] - VOLUME[wt],
                  CHARGE[mt] - CHARGE[wt]], dtype=np.float32)
    ])

    dmat = xgb.DMatrix(feat.reshape(1, -1))
    pred = float(booster.predict(dmat)[0] + bias)
    return pred, True

def cnn_predict_one(cnn_model, pdb_path: str, chain_id: str, mutation: str, cutoff=8.0, H=128):
    try:
        wt, pos, mt = parse_mutation(mutation)

        structure = PARSER.get_structure("p", pdb_path)
        residues = get_chain_residues_with_ca(structure, chain_id)
        if len(residues) == 0:
            return 0.0, False

        coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)
        n = len(coords)

        cmap = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                    cmap[i, j] = cmap[j, i] = 1.0

        idx = None
        for i, r in enumerate(residues):
            if r.id[1] == pos:
                idx = i
                break
        if idx is None:
            return 0.0, False

        mask = np.zeros((n, n), dtype=np.float32)
        mask[idx, :] = 1.0
        mask[:, idx] = 1.0

        X = np.zeros((2, H, H), dtype=np.float32)
        m = min(n, H)
        X[0, :m, :m] = cmap[:m, :m]
        X[1, :m, :m] = mask[:m, :m]

        dh = HYDRO[mt] - HYDRO[wt]
        dv = VOLUME[mt] - VOLUME[wt]
        dq = CHARGE[mt] - CHARGE[wt]
        mut_vec = np.array([[dh, dv, dq, 1.0]], dtype=np.float32)

        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            mv = torch.tensor(mut_vec, dtype=torch.float32)
            y = cnn_model(t, mv).item()

        return float(y), True
    except Exception:
        return 0.0, False

def gnn_predict_one(gnn_model, pdb_path: str, chain_id: str, mutation: str, cutoff=8.0):
    if not PYG_OK:
        return 0.0, False

    try:
        wt, pos, mt = parse_mutation(mutation)

        structure = PARSER.get_structure("p", pdb_path)
        residues = get_chain_residues_with_ca(structure, chain_id)
        if len(residues) == 0:
            return 0.0, False

        coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)

        edges = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                    edges.append([i, j])
                    edges.append([j, i])
        if len(edges) == 0:
            return 0.0, False

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        x = torch.zeros((len(residues), 25), dtype=torch.float32)
        mut_idx = None
        for i, r in enumerate(residues):
            aa1 = AA3_TO_1[r.get_resname().upper()]
            x[i, AA_INDEX[aa1]] = 1.0
            if r.id[1] == pos:
                mut_idx = i
        if mut_idx is None:
            return 0.0, False

        dh = HYDRO[mt] - HYDRO[wt]
        dv = VOLUME[mt] - VOLUME[wt]
        dq = CHARGE[mt] - CHARGE[wt]

        x[:, 20] = 0.0
        x[mut_idx, 20] = 1.0
        x[:, 21] = dh
        x[:, 22] = dv
        x[:, 23] = dq
        x[:, 24] = 1.0

        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(x.size(0), dtype=torch.long)

        with torch.no_grad():
            y = gnn_model(data).item()

        return float(y), True
    except Exception:
        return 0.0, False

def predict_ensemble_one(booster, bias, cnn_model, gnn_model, pdb_path: str, chain_id: str, mutation: str, weights):
    w_xgb, w_cnn, w_gnn = weights
    xgb_p, _ = xgb_predict_one(booster, bias, mutation)
    cnn_p, cnn_ok = cnn_predict_one(cnn_model, pdb_path, chain_id, mutation)
    gnn_p, gnn_ok = gnn_predict_one(gnn_model, pdb_path, chain_id, mutation)
    ens = (w_xgb * xgb_p) + (w_cnn * cnn_p) + (w_gnn * gnn_p)
    return {
        "mutation": mutation,
        "xgb": float(xgb_p),
        "cnn": float(cnn_p),
        "gnn": float(gnn_p),
        "ens": float(ens),
        "cnn_ok": bool(cnn_ok),
        "gnn_ok": bool(gnn_ok),
    }

def scan_19aa(pdb_path: str, chain_id: str, pos: int, wt_aa: str, booster, bias, cnn_model, gnn_model, weights):
    rows = []
    for aa in AA_LIST:
        if aa == wt_aa:
            continue
        mut = f"{wt_aa}{pos}{aa}"
        rows.append(predict_ensemble_one(booster, bias, cnn_model, gnn_model, pdb_path, chain_id, mut, weights))
    return pd.DataFrame(rows)

def prioritize_scan(scan_df: pd.DataFrame, pdb_id: str, chain_id: str, top_k=5):
    dfp = scan_df.copy()
    dfp["PDB_ID"] = pdb_id
    dfp["chain"] = chain_id
    dfp = dfp.sort_values("ens", ascending=True).reset_index(drop=True)
    dfp["rank_stabilizing"] = np.arange(1, len(dfp) + 1)
    dfp["pct"] = dfp["rank_stabilizing"] / len(dfp)
    dfp["ddg_class"] = dfp["ens"].apply(ddg_class)
    dfp["confidence"] = [confidence_label(c, g) for c, g in zip(dfp["cnn_ok"], dfp["gnn_ok"])]
    dfp["highlight"] = ""
    dfp.loc[:top_k - 1, "highlight"] = "TOP stabilizing"
    dfp.loc[len(dfp) - top_k:, "highlight"] = "TOP destabilizing"
    return dfp

# -----------------------------
# FASTA helpers (XGB-only)
# -----------------------------
def clean_fasta_to_sequence(text: str) -> str:
    if text is None:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    if lines[0].startswith(">"):
        lines = lines[1:]
    seq = "".join(lines).upper()
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq

def wt_from_fasta(seq: str, pos_1based: int):
    if not seq:
        return None
    if pos_1based < 1 or pos_1based > len(seq):
        return None
    wt = seq[pos_1based - 1]
    return wt if wt in AA_INDEX else None

def scan_19aa_xgb_only(seq: str, pos: int, booster, bias):
    wt = wt_from_fasta(seq, pos)
    if wt is None:
        raise ValueError("WT could not be determined from FASTA at this position.")
    out_rows = []
    for aa in AA_LIST:
        if aa == wt:
            continue
        mut = f"{wt}{pos}{aa}"
        xgb_p, _ = xgb_predict_one(booster, bias, mut)
        out_rows.append({
            "mutation": mut,
            "xgb": float(xgb_p),
            "cnn": 0.0,
            "gnn": 0.0,
            "ens": float(xgb_p),
            "cnn_ok": False,
            "gnn_ok": False,
        })
    return pd.DataFrame(out_rows), wt

# -----------------------------
# UI
# -----------------------------
st.title("ΔΔG Phase-2 Mutation Scanner (XGB + CNN + GNN Ensemble)")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Artifacts status")
    st.write("XGB:", XGB_PATH.exists(), XGB_PATH.name)
    st.write("BIAS:", BIAS_PATH.exists(), BIAS_PATH.name)
    st.write("CNN:", CNN_PATH.exists(), CNN_PATH.name)
    st.write("GNN:", GNN_PATH.exists(), GNN_PATH.name)
    st.write("PyG available:", PYG_OK)

    st.subheader("Ensemble weights")
    wx = st.slider("XGB weight", 0.0, 1.0, 0.33, 0.01, key="wx")
    wc = st.slider("CNN weight", 0.0, 1.0, 0.33, 0.01, key="wc")
    wg = st.slider("GNN weight", 0.0, 1.0, 0.33, 0.01, key="wg")
    s = wx + wc + wg
    weights = (1/3, 1/3, 1/3) if s == 0 else (wx/s, wc/s, wg/s)
    st.caption(f"Normalized weights = {weights}")

with right:
    st.subheader("Run predictions")
    tab_pdb, tab_fasta = st.tabs(["PDB Mode (XGB+CNN+GNN)", "FASTA Mode (XGB only)"])

    # -------------------------
    # TAB 1: PDB MODE
    # -------------------------
    with tab_pdb:
        st.markdown("### Upload PDB and run 19-AA scan (structure-based)")
        uploaded = st.file_uploader("Upload a PDB file", type=["pdb"], key="pdb_uploader")
        chain_id = st.text_input("Chain ID", value="A", max_chars=2, key="pdb_chain")
        pos = st.number_input("Position (residue number)", min_value=1, value=66, step=1, key="pdb_pos")
        wt_override = st.text_input("WT AA override (optional, 1-letter)", value="", key="pdb_wt").strip().upper()
        run = st.button("Run 19-AA scan (PDB mode)", key="run_pdb")

        if run:
            if uploaded is None:
                st.error("Please upload a PDB file.")
                st.stop()

            pdb_bytes = uploaded.getvalue()
            pdb_name = Path(uploaded.name).stem.upper()
            pdb_path = TMP_DIR / f"{pdb_name}.pdb"
            pdb_path.write_bytes(pdb_bytes)

            try:
                booster, bias, cnn_model, gnn_model, meta = load_models()
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

            st.success(f"Models loaded. XGB bias={bias:.4f}")
            if meta["cnn_missing"] or meta["cnn_unexpected"] or meta["gnn_missing"] or meta["gnn_unexpected"]:
                with st.expander("Model load details (missing/unexpected keys)"):
                    st.write(meta)

            wt_aa = wt_override if wt_override in AA_INDEX else wt_aa_from_pdb(str(pdb_path), chain_id, int(pos))
            if wt_aa is None:
                st.error("Could not detect WT amino acid from PDB at this chain/position. Please enter WT override.")
                st.stop()

            st.info(f"Using PDB: {pdb_name}.pdb | chain: {chain_id} | position: {int(pos)} | WT: {wt_aa}")

            scan_df = scan_19aa(
                pdb_path=str(pdb_path),
                chain_id=chain_id,
                pos=int(pos),
                wt_aa=wt_aa,
                booster=booster,
                bias=bias,
                cnn_model=cnn_model,
                gnn_model=gnn_model,
                weights=weights
            )
            scan_prior = prioritize_scan(scan_df, pdb_id=pdb_name, chain_id=chain_id, top_k=5)

            # ---- show confidence clearly
            conf_mode = scan_prior["confidence"].mode()[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", conf_mode)
            c2.metric("CNN OK rate", f"{scan_prior['cnn_ok'].mean():.2f}")
            c3.metric("GNN OK rate", f"{scan_prior['gnn_ok'].mean():.2f}")

            st.markdown("### 19-AA scan (ranked by ensemble; lower = more stabilizing)")
            st.dataframe(
                scan_prior[["mutation","ens","ddg_class","confidence","cnn_ok","gnn_ok","xgb","cnn","gnn","rank_stabilizing","pct","highlight","PDB_ID","chain"]],
                use_container_width=True
            )

            st.markdown("### Bar plot (ensemble ΔΔG per mutation)")
            st.bar_chart(scan_prior.set_index("mutation")["ens"])

            csv_bytes = scan_prior.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"scan_{pdb_name}_{chain_id}_pos{int(pos)}.csv",
                mime="text/csv"
            )

    # -------------------------
    # TAB 2: FASTA MODE (XGB ONLY)
    # -------------------------
    with tab_fasta:
        st.markdown("### Paste FASTA and run 19-AA scan (sequence-based, XGB only)")
        fasta_text = st.text_area(
            "FASTA sequence (or raw sequence)",
            height=200,
            placeholder=">my_protein\nMSTNPKPQR...\n",
            key="fasta_text"
        )
        fasta_label = st.text_input("Protein label (for output filenames)", value="FASTA_PROTEIN", key="fasta_label")
        pos_fa = st.number_input("Position (1-based residue number)", min_value=1, value=66, step=1, key="fasta_pos")
        run_fa = st.button("Run 19-AA scan (FASTA mode)", key="run_fasta")

        if run_fa:
            seq = clean_fasta_to_sequence(fasta_text)
            if not seq:
                st.error("FASTA/sequence is empty. Please paste a sequence.")
                st.stop()

            try:
                booster, bias, cnn_model, gnn_model, meta = load_models()
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

            wt = wt_from_fasta(seq, int(pos_fa))
            if wt is None:
                st.error(f"Position {int(pos_fa)} is out of range (sequence length={len(seq)}) or contains invalid AA.")
                st.stop()

            st.success(f"XGB loaded. Bias={bias:.4f} (FASTA mode uses XGB only)")
            st.info(f"FASTA length: {len(seq)} | position: {int(pos_fa)} | WT from FASTA: {wt} | Mode: XGB only")

            scan_df, wt_used = scan_19aa_xgb_only(seq, int(pos_fa), booster, bias)
            scan_prior = prioritize_scan(scan_df, pdb_id=fasta_label.strip().upper(), chain_id="FASTA", top_k=5)

            st.metric("Confidence", "Low (XGB only)")

            st.markdown("### 19-AA scan (XGB-only; lower = more stabilizing)")
            st.dataframe(
                scan_prior[["mutation","ens","ddg_class","confidence","xgb","rank_stabilizing","pct","highlight","PDB_ID","chain"]],
                use_container_width=True
            )

            st.markdown("### Bar plot (XGB ΔΔG per mutation)")
            st.bar_chart(scan_prior.set_index("mutation")["ens"])

            csv_bytes = scan_prior.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"scan_{fasta_label.strip().upper()}_pos{int(pos_fa)}_XGBonly.csv",
                mime="text/csv"
            )
























