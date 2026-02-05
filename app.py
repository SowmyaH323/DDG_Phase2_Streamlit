import os
import re
import io
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import torch
import xgboost as xgb

from Bio.PDB import PDBParser

# =========================
# CONFIG
# =========================
DEVICE = "cpu"

DRIVE_BASE = Path("/content/drive/MyDrive/DDG_Phase2")
PDB_ROOT = DRIVE_BASE / "pdbs"
ARTIFACTS_DIR = DRIVE_BASE / "artifacts_phase2"
OUT_DIR = ARTIFACTS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

XGB_PATH  = ARTIFACTS_DIR / "xgb_phase2_v4w_huber_weighted.json"
BIAS_PATH = ARTIFACTS_DIR / "xgb_v4w_bias.txt"
CNN_PATH  = ARTIFACTS_DIR / "cnn_phase2_v2_best.pt"
GNN_PATH  = ARTIFACTS_DIR / "gnn_phase2_best.pt"


# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource
def load_models():
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))
    bias = float(BIAS_PATH.read_text().strip())

    cnn_state = torch.load(CNN_PATH, map_location=DEVICE)
    cnn_model = torch.load(CNN_PATH, map_location=DEVICE)
    if isinstance(cnn_model, dict):
        cnn_model = None

    # safest method
    cnn_model = torch.load(CNN_PATH, map_location=DEVICE)
    if hasattr(cnn_model, "eval"):
        cnn_model.eval()

    gnn_model = torch.load(GNN_PATH, map_location=DEVICE)
    if hasattr(gnn_model, "eval"):
        gnn_model.eval()

    return booster, bias, cnn_model, gnn_model


booster, xgb_bias, cnn_model, gnn_model = load_models()


# =========================
# PROPERTIES
# =========================
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a:i for i,a in enumerate(AA_LIST)}

HYDRO = {
 'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,
 'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}

VOLUME = {
 'A':88.6,'C':108.5,'D':111.1,'E':138.4,'F':189.9,'G':60.1,'H':153.2,'I':166.7,'K':168.6,
 'L':166.7,'M':162.9,'N':114.1,'P':112.7,'Q':143.8,'R':173.4,'S':89.0,'T':116.1,'V':140.0,
 'W':227.8,'Y':193.6
}

CHARGE = {
 'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,'M':0,'N':0,'P':0,
 'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0
}


# =========================
# HELPERS
# =========================
def parse_mutation(mut):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut.strip().upper())
    if not m:
        raise ValueError("Mutation must look like P66A")
    return m.group(1), int(m.group(2)), m.group(3)


def ddg_class(val):
    if val < -1:
        return "Strong stabilizing (< -1)"
    elif -1 <= val < 0:
        return "Mild stabilizing (-1 to 0)"
    elif 0 <= val <= 1:
        return "Neutral (0 to 1)"
    else:
        return "Destabilizing (> 1)"


def confidence_label(cnn_ok, gnn_ok):
    if cnn_ok and gnn_ok:
        return "High"
    elif cnn_ok or gnn_ok:
        return "Medium"
    else:
        return "Low"


# =========================
# XGB PREDICT
# =========================
def xgb_predict_one(mutation):
    wt,pos,mt = parse_mutation(mutation)

    wt_vec = np.zeros(20); mt_vec = np.zeros(20)
    wt_vec[AA_INDEX[wt]] = 1
    mt_vec[AA_INDEX[mt]] = 1

    dh = HYDRO[mt] - HYDRO[wt]
    dv = VOLUME[mt] - VOLUME[wt]
    dq = CHARGE[mt] - CHARGE[wt]

    feat = np.concatenate([
        wt_vec, mt_vec,
        np.array([pos, dh, dv, dq], dtype=np.float32)
    ])

    dmat = xgb.DMatrix(feat.reshape(1,-1))
    pred = booster.predict(dmat)[0] + xgb_bias
    return float(pred), True


# =========================
# CNN PREDICT
# =========================
parser = PDBParser(QUIET=True)

def cnn_predict_one(pdb_path, chain_id, mutation):
    try:
        wt,pos,mt = parse_mutation(mutation)
        structure = parser.get_structure("p", pdb_path)

        residues = []
        for model in structure:
            for ch in model:
                if ch.id == chain_id:
                    for res in ch:
                        if "CA" in res:
                            residues.append(res)

        if len(residues) == 0:
            return 0.0, False

        n = len(residues)
        cmap = np.zeros((n,n), dtype=np.float32)

        for i in range(n):
            for j in range(i+1,n):
                if np.linalg.norm(residues[i]["CA"].coord - residues[j]["CA"].coord) <= 8.0:
                    cmap[i,j] = cmap[j,i] = 1.0

        idx = next((i for i,r in enumerate(residues) if r.id[1]==pos), None)
        mask = np.zeros_like(cmap)
        if idx is not None:
            mask[idx,:] = mask[:,idx] = 1.0

        H = 128
        X = np.zeros((2,H,H), dtype=np.float32)
        X[0,:n,:n] = cmap[:H,:H]
        X[1,:n,:n] = mask[:H,:H]

        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            y = cnn_model(xt).item()

        return float(y), True
    except Exception as e:
        return 0.0, False


# =========================
# GNN PREDICT (Simple mode)
# =========================
def gnn_predict_one(pdb_path, chain_id, mutation):
    try:
        wt,pos,mt = parse_mutation(mutation)
        structure = parser.get_structure("p", pdb_path)

        residues = []
        for model in structure:
            for ch in model:
                if ch.id == chain_id:
                    for res in ch:
                        if "CA" in res:
                            residues.append(res)

        if len(residues) == 0:
            return 0.0, False

        coords = np.array([r["CA"].coord for r in residues])
        mut_idx = next((i for i,r in enumerate(residues) if r.id[1]==pos), None)
        if mut_idx is None:
            return 0.0, False

        edges = []
        for i in range(len(coords)):
            for j in range(i+1,len(coords)):
                if np.linalg.norm(coords[i]-coords[j]) <= 8.0:
                    edges.append((i,j))
                    edges.append((j,i))

        # if gnn_model is scripted and expects tensors
        x = torch.zeros((len(residues), 25), dtype=torch.float32)
        for i,r in enumerate(residues):
            aa = r.get_resname()[0]
            if aa in AA_INDEX:
                x[i, AA_INDEX[aa]] = 1.0

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        with torch.no_grad():
            y = gnn_model(x, edge_index, mut_idx).item()

        return float(y), True

    except Exception as e:
        return 0.0, False


# =========================
# ENSEMBLE
# =========================
def predict_ensemble_one(pdb_path, chain_id, mutation, weights=(1/3,1/3,1/3)):
    wx, wc, wg = weights

    xgb_p, _ = xgb_predict_one(mutation)
    cnn_p, cnn_ok = cnn_predict_one(pdb_path, chain_id, mutation)
    gnn_p, gnn_ok = gnn_predict_one(pdb_path, chain_id, mutation)

    ens = (wx*xgb_p) + (wc*cnn_p) + (wg*gnn_p)

    return {
        "mutation": mutation,
        "xgb": xgb_p,
        "cnn": cnn_p,
        "gnn": gnn_p,
        "ens": ens,
        "cnn_ok": cnn_ok,
        "gnn_ok": gnn_ok,
        "confidence": confidence_label(cnn_ok, gnn_ok),
        "ddg_class": ddg_class(ens)
    }


# =========================
# SCAN 19AA
# =========================
def scan_19aa_position(pdb_path, pdb_id, chain_id, pos, wt_aa):
    muts = []
    for aa in AA_LIST:
        if aa == wt_aa:
            continue
        muts.append(f"{wt_aa}{pos}{aa}")

    rows = []
    for m in muts:
        out = predict_ensemble_one(pdb_path, chain_id, m)
        out["PDB_ID"] = pdb_id
        out["chain"] = chain_id
        rows.append(out)

    df = pd.DataFrame(rows)
    return df


def prioritize_scan(scan_df, top_k=5):
    dfp = scan_df.copy()
    dfp = dfp.sort_values("ens", ascending=True).reset_index(drop=True)

    dfp["rank_stabilizing"] = dfp.index + 1
    dfp["pct"] = dfp["rank_stabilizing"] / len(dfp)

    dfp["highlight"] = ""
    dfp.loc[dfp.index < top_k, "highlight"] = "TOP stabilizing"
    dfp.loc[dfp.index >= len(dfp)-top_k, "highlight"] = "TOP destabilizing"

    return dfp


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="ΔΔG Phase-2 Mutation Scanner", layout="wide")
st.title("🧬 Phase-2 ΔΔG Mutation Scanner (XGB + CNN + GNN Ensemble)")


tab1, tab2 = st.tabs(["🔹 Predict One Mutation", "🔹 Scan 19AA Position"])


# -------------------------
# TAB 1
# -------------------------
with tab1:
    st.subheader("Predict a Single Mutation")

    uploaded_pdb = st.file_uploader("Upload PDB file", type=["pdb"])
    chain_id = st.text_input("Chain ID", value="A")
    mutation = st.text_input("Mutation (e.g., P66A)", value="P66A")

    weights = st.slider("Ensemble Weight (XGB / CNN / GNN)", 0.0, 1.0, 0.33)
    weights_tuple = (weights, weights, 1 - (2*weights))
    if weights_tuple[2] < 0:
        weights_tuple = (1/3,1/3,1/3)

    if uploaded_pdb is not None:
        pdb_id = Path(uploaded_pdb.name).stem.upper()
        tmp_path = OUT_DIR / uploaded_pdb.name
        tmp_path.write_bytes(uploaded_pdb.read())

        if st.button("Run Prediction"):
            out = predict_ensemble_one(str(tmp_path), chain_id, mutation, weights=(1/3,1/3,1/3))
            st.json(out)

            st.success(f"Confidence: {out['confidence']} | Class: {out['ddg_class']}")


# -------------------------
# TAB 2
# -------------------------
with tab2:
    st.subheader("Scan All 19 Mutations at a Position")

    uploaded_pdb2 = st.file_uploader("Upload PDB file for scanning", type=["pdb"], key="scan_pdb")
    chain2 = st.text_input("Chain ID (scan)", value="A", key="scan_chain")
    pos = st.number_input("Position", min_value=1, max_value=2000, value=66)
    wt_aa = st.text_input("Wildtype AA (optional)", value="P")

    if uploaded_pdb2 is not None:
        pdb_id2 = Path(uploaded_pdb2.name).stem.upper()
        tmp_path2 = OUT_DIR / uploaded_pdb2.name
        tmp_path2.write_bytes(uploaded_pdb2.read())

        if st.button("Run 19AA Scan"):
            scan_df = scan_19aa_position(str(tmp_path2), pdb_id2, chain2, pos, wt_aa)
            scan_prior = prioritize_scan(scan_df, top_k=5)

            st.dataframe(scan_prior)

            csv_bytes = scan_prior.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"scan_{pdb_id2}_{chain2}_pos{pos}.csv",
                mime="text/csv"
            )

            save_path = OUT_DIR / f"scan_{pdb_id2}_{chain2}_pos{pos}.csv"
            scan_prior.to_csv(save_path, index=False)
            st.success(f"Saved automatically to: {save_path}")
