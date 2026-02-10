import io
import re
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st

# -----------------------------
# Core ML deps
# -----------------------------
import torch
import xgboost as xgb

# Biopython
from Bio.PDB import PDBParser

# PyG (needed for GNN)
try:
    from torch_geometric.data import Data
    PYG_OK = True
except Exception:
    PYG_OK = False

st.set_page_config(page_title="ΔΔG Phase-2 Mutation Scanner", layout="wide")

DEVICE = "cpu"
PARSER = PDBParser(QUIET=True)

# -----------------------------
# Paths (repo root files)
# -----------------------------
REPO = Path(".")
XGB_PATH  = REPO / "xgb_phase2_v4w_huber_weighted.json"
BIAS_PATH = REPO / "xgb_v4w_bias.txt"
CNN_PATH  = REPO / "cnn_phase2_v2_best.pt"
GNN_PATH  = REPO / "gnn_phase2_best.pt"

# Where to store uploaded PDB temporarily (allowed)
TMP_DIR = Path("/tmp/ddg_phase2")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# AA utils
# -----------------------------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a:i for i,a in enumerate(AA_LIST)}

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

def confidence_label(cnn_ok: bool, gnn_ok: bool, ens: float) -> str:
    # High: both structural models ok
    # Medium: one ok
    # Low: neither ok
    if cnn_ok and gnn_ok: return "High"
    if cnn_ok or gnn_ok: return "Medium"
    return "Low"

# -----------------------------
# Model loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # XGB
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))
    xgb_bias = float(BIAS_PATH.read_text().strip()) if BIAS_PATH.exists() else 0.0

    # CNN (your file can be full model or state_dict)
    cnn_obj = torch.load(CNN_PATH, map_location=DEVICE)
    if isinstance(cnn_obj, dict):
        # State dict: we need a compatible architecture.
        # EASIEST practical path for Streamlit: save the CNN as full model in Colab next time.
        # For now: try to load as scripted module if dict fails; else raise.
        raise RuntimeError("CNN file is a state_dict. Please upload the FULL model .pt (torch.save(model)).")
    cnn_model = cnn_obj
    cnn_model.eval()

    # GNN (same story)
    gnn_obj = torch.load(GNN_PATH, map_location=DEVICE)
    if isinstance(gnn_obj, dict):
        raise RuntimeError("GNN file is a state_dict. Please upload the FULL model .pt (torch.save(model)).")
    gnn_model = gnn_obj
    gnn_model.eval()

    return booster, xgb_bias, cnn_model, gnn_model

# -----------------------------
# Feature + prediction
# -----------------------------
def xgb_predict_one(booster, bias: float, mutation: str):
    wt, pos, mt = parse_mutation(mutation)

    wt_vec = np.zeros(20, dtype=np.float32); mt_vec = np.zeros(20, dtype=np.float32)
    wt_vec[AA_INDEX[wt]] = 1.0
    mt_vec[AA_INDEX[mt]] = 1.0

    feat = np.concatenate([
        wt_vec, mt_vec,
        np.array([pos,
                  HYDRO[mt]-HYDRO[wt],
                  VOLUME[mt]-VOLUME[wt],
                  CHARGE[mt]-CHARGE[wt]], dtype=np.float32)
    ])
    dmat = xgb.DMatrix(feat.reshape(1,-1))
    pred = float(booster.predict(dmat)[0] + bias)
    return pred, True

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

def cnn_predict_one(cnn_model, pdb_path: str, chain_id: str, mutation: str, cutoff=8.0, H=128):
    try:
        wt, pos, mt = parse_mutation(mutation)
        structure = PARSER.get_structure("p", pdb_path)
        residues = get_chain_residues_with_ca(structure, chain_id)
        if len(residues) == 0:
            return 0.0, False

        n = len(residues)
        coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)

        # contact map
        cmap = np.zeros((n,n), dtype=np.float32)
        for i in range(n):
            for j in range(i+1,n):
                if np.linalg.norm(coords[i]-coords[j]) <= cutoff:
                    cmap[i,j] = cmap[j,i] = 1.0

        # mutation mask row/col
        idx = None
        for i,r in enumerate(residues):
            if r.id[1] == pos:
                idx = i
                break
        if idx is None:
            return 0.0, False

        mask = np.zeros((n,n), dtype=np.float32)
        mask[idx,:] = 1.0
        mask[:,idx] = 1.0

        X = np.zeros((2,H,H), dtype=np.float32)
        X[0,:min(n,H),:min(n,H)] = cmap[:H,:H]
        X[1,:min(n,H),:min(n,H)] = mask[:H,:H]

        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            y = cnn_model(t).item()
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
            for j in range(i+1, len(coords)):
                if np.linalg.norm(coords[i]-coords[j]) <= cutoff:
                    edges.append([i,j]); edges.append([j,i])
        if len(edges) == 0:
            return 0.0, False

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # node feats: 20 one-hot + 5 simple props (optional); keep 25 to match your Phase-2
        x = torch.zeros((len(residues), 25), dtype=torch.float32)
        mut_idx = None
        for i,r in enumerate(residues):
            aa1 = AA3_TO_1[r.get_resname().upper()]
            x[i, AA_INDEX[aa1]] = 1.0
            if r.id[1] == pos:
                mut_idx = i
        if mut_idx is None:
            return 0.0, False

        # extra dims (5): [is_mut, dh, dv, dq, 1]
        # keep it simple: same for all nodes except is_mut=1 at mutation site
        dh = HYDRO[mt] - HYDRO[wt]
        dv = VOLUME[mt] - VOLUME[wt]
        dq = CHARGE[mt] - CHARGE[wt]
        x[:,20] = 0.0
        x[mut_idx,20] = 1.0
        x[:,21] = dh
        x[:,22] = dv
        x[:,23] = dq
        x[:,24] = 1.0

        data = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            y = gnn_model(data).item()
        return float(y), True
    except Exception:
        return 0.0, False

def predict_ensemble_one(booster, bias, cnn_model, gnn_model, pdb_path: str, chain_id: str, mutation: str, weights):
    w_xgb, w_cnn, w_gnn = weights

    xgb_p, xgb_ok = xgb_predict_one(booster, bias, mutation)
    cnn_p, cnn_ok = cnn_predict_one(cnn_model, pdb_path, chain_id, mutation)
    gnn_p, gnn_ok = gnn_predict_one(gnn_model, pdb_path, chain_id, mutation)

    # If CNN/GNN fail, we still keep output but confidence will drop
    ens = (w_xgb * xgb_p) + (w_cnn * cnn_p) + (w_gnn * gnn_p)

    return {
        "mutation": mutation,
        "xgb": xgb_p,
        "cnn": cnn_p,
        "gnn": gnn_p,
        "ens": float(ens),
        "cnn_ok": cnn_ok,
        "gnn_ok": gnn_ok,
    }

def scan_19aa(pdb_path: str, chain_id: str, pos: int, wt_aa: str, booster, bias, cnn_model, gnn_model, weights):
    out_rows = []
    for aa in AA_LIST:
        if aa == wt_aa:
            continue
        mut = f"{wt_aa}{pos}{aa}"
        out = predict_ensemble_one(booster, bias, cnn_model, gnn_model, pdb_path, chain_id, mut, weights)
        out_rows.append(out)
    df_scan = pd.DataFrame(out_rows)
    return df_scan

def prioritize_scan(scan_df: pd.DataFrame, pdb_id: str, chain_id: str):
    dfp = scan_df.copy()
    dfp["PDB_ID"] = pdb_id
    dfp["chain"] = chain_id

    dfp = dfp.sort_values("ens", ascending=True).reset_index(drop=True)
    dfp["rank_stabilizing"] = np.arange(1, len(dfp)+1)
    dfp["pct"] = dfp["rank_stabilizing"] / len(dfp)

    dfp["ddg_class"] = dfp["ens"].apply(ddg_class)
    dfp["confidence"] = [confidence_label(c,o,e) for c,o,e in zip(dfp["cnn_ok"], dfp["gnn_ok"], dfp["ens"])]

    # Highlight top/bottom 5
    top_k = 5
    dfp["highlight"] = ""
    dfp.loc[:top_k-1, "highlight"] = "TOP stabilizing"
    dfp.loc[len(dfp)-top_k:, "highlight"] = "TOP destabilizing"
    return dfp

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
    wx = st.slider("XGB weight", 0.0, 1.0, 0.33, 0.01)
    wc = st.slider("CNN weight", 0.0, 1.0, 0.33, 0.01)
    wg = st.slider("GNN weight", 0.0, 1.0, 0.33, 0.01)
    s = wx + wc + wg
    if s == 0:
        weights = (1/3, 1/3, 1/3)
    else:
        weights = (wx/s, wc/s, wg/s)
    st.caption(f"Normalized weights = {weights}")

with right:
    st.subheader("Upload PDB and run 19-AA scan")

    uploaded = st.file_uploader("Upload a PDB file", type=["pdb"])
    chain_id = st.text_input("Chain ID", value="A", max_chars=2)
    pos = st.number_input("Position (residue number)", min_value=1, value=66, step=1)

    wt_override = st.text_input("WT AA override (optional, 1-letter)", value="").strip().upper()
    run = st.button("Run 19-AA scan")

    if run:
        if uploaded is None:
            st.error("Please upload a PDB file.")
            st.stop()

        pdb_bytes = uploaded.getvalue()
        pdb_name = Path(uploaded.name).stem.upper()
        pdb_path = TMP_DIR / f"{pdb_name}.pdb"
        pdb_path.write_bytes(pdb_bytes)

        # Load models
        try:
            booster, bias, cnn_model, gnn_model = load_models()
            st.success(f"Models loaded. XGB bias={bias:.4f}")
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()

        # Determine WT AA
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
        scan_prior = prioritize_scan(scan_df, pdb_id=pdb_name, chain_id=chain_id)

        st.markdown("### 19-AA scan (ranked by ensemble; lower = more stabilizing)")
        st.dataframe(scan_prior, use_container_width=True)

        # Bar plot: top 10 stabilizing and top 10 destabilizing
        st.markdown("### Bar plot: top stabilizing vs destabilizing")
        topN = 10
        a = scan_prior.head(topN)[["mutation","ens"]].copy()
        b = scan_prior.tail(topN)[["mutation","ens"]].copy()
        a["group"] = "Top stabilizing"
        b["group"] = "Top destabilizing"
        plot_df = pd.concat([a,b], ignore_index=True)
        st.bar_chart(plot_df.set_index("mutation")["ens"])

        # Download CSV
        csv_bytes = scan_prior.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"scan_{pdb_name}_{chain_id}_pos{int(pos)}.csv",
            mime="text/csv"
        )











