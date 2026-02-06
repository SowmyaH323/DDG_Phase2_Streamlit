import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import xgboost as xgb
import torch

from Bio.PDB import PDBParser

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="ΔΔG Phase-2 Mutation Scanner", layout="wide")
st.title("ΔΔG Phase-2 Mutation Scanner (XGB + CNN + GNN Ensemble)")

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = Path("/tmp/ddg_outputs")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Files in repo root (your case)
# -----------------------------
XGB_PATH  = BASE_DIR / "xgb_phase2_v4w_huber_weighted.json"   # you said latest
BIAS_PATH = BASE_DIR / "xgb_v4w_bias.txt"
CNN_PATH  = BASE_DIR / "cnn_phase2_v2_best.pt"
GNN_PATH = "gnn_phase2_best_scripted.pt"
gnn_model = torch.jit.load(GNN_PATH, map_location="cpu")
gnn_model.eval()


DEVICE = "cpu"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA_LIST)}

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

PARSER = PDBParser(QUIET=True)

def parse_mutation(mut: str):
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut.strip().upper())
    if not m:
        raise ValueError(f"Bad mutation format: {mut} (expected like P66A)")
    return m.group(1), int(m.group(2)), m.group(3)

def ddg_class(val: float) -> str:
    if val < -1.0:
        return "Strong stabilizing (< -1)"
    if val < 0.0:
        return "Mild stabilizing (-1 to 0)"
    if val <= 1.0:
        return "Neutral (0 to 1)"
    return "Destabilizing (> 1)"

def confidence_label(cnn_ok: bool, gnn_ok: bool) -> str:
    # simple and honest: both structure models working => High
    if cnn_ok and gnn_ok:
        return "High"
    if cnn_ok or gnn_ok:
        return "Medium"
    return "Low"

def try_load_torch_model(path: Path, name: str):
    """
    Best path for Streamlit: TorchScript (.pt via torch.jit.save).
    Fallback: torch.load full model object.
    If it is state_dict/OrderedDict -> raise with instruction.
    """
    # 1) TorchScript
    try:
        m = torch.jit.load(str(path), map_location=DEVICE)
        m.eval()
        st.success(f"✅ {name} loaded (TorchScript): {path.name}")
        return m
    except Exception:
        pass

    # 2) Full model object
    obj = torch.load(str(path), map_location=DEVICE)
    if hasattr(obj, "eval"):
        obj.eval()
        st.success(f"✅ {name} loaded (torch.load model): {path.name}")
        return obj

    # 3) state_dict/OrderedDict (cannot eval without class code)
    if isinstance(obj, dict):
        raise RuntimeError(
            f"{name} file looks like a state_dict (OrderedDict). "
            f"Streamlit cannot reconstruct the model class automatically.\n\n"
            f"✅ Fix (NO retraining): in Colab, load your working {name} model and run:\n"
            f"  scripted = torch.jit.script(model)\n"
            f"  torch.jit.save(scripted, '{path.name}')\n"
            f"Then upload that new .pt to GitHub and redeploy."
        )
    raise RuntimeError(f"Unknown {name} file format: {path.name}")

@st.cache_resource
def load_models():
    # XGB
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"Missing: {XGB_PATH}")
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))

    # bias
    if not BIAS_PATH.exists():
        raise FileNotFoundError(f"Missing: {BIAS_PATH}")
    bias = float(BIAS_PATH.read_text().strip())

    # CNN/GNN
    cnn_model = try_load_torch_model(CNN_PATH, "CNN")
    gnn_model = try_load_torch_model(GNN_PATH, "GNN")

    return booster, bias, cnn_model, gnn_model

def xgb_predict_one(booster, bias, mutation: str):
    wt, pos, mt = parse_mutation(mutation)

    wt_vec = np.zeros(20, dtype=np.float32)
    mt_vec = np.zeros(20, dtype=np.float32)
    if wt in AA_INDEX: wt_vec[AA_INDEX[wt]] = 1.0
    if mt in AA_INDEX: mt_vec[AA_INDEX[mt]] = 1.0

    dh = float(HYDRO[mt] - HYDRO[wt])
    dv = float(VOLUME[mt] - VOLUME[wt])
    dq = float(CHARGE[mt] - CHARGE[wt])

    feat = np.concatenate([
        wt_vec, mt_vec,
        np.array([pos, dh, dv, dq], dtype=np.float32)
    ], axis=0)

    dmat = xgb.DMatrix(feat.reshape(1, -1))
    pred = float(booster.predict(dmat)[0] + bias)
    return pred, True

def get_residues_with_ca(structure, chain_id: str):
    residues = []
    for model in structure:
        for ch in model:
            if ch.id == chain_id:
                for res in ch:
                    if "CA" in res:
                        residues.append(res)
    return residues

def build_contact_map_and_mask(pdb_bytes: bytes, chain_id: str, pos: int, cutoff=8.0, H=128):
    structure = PARSER.get_structure("p", io_bytes(pdb_bytes))
    residues = get_residues_with_ca(structure, chain_id)
    if len(residues) == 0:
        return None, None, False

    n = min(len(residues), H)
    coords = np.array([residues[i]["CA"].coord for i in range(n)], dtype=np.float32)

    # contact map
    cmap = np.zeros((H, H), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                cmap[i, j] = 1.0
                cmap[j, i] = 1.0

    # mask row/col for mutation position
    idx = None
    for i in range(n):
        if residues[i].id[1] == pos:
            idx = i
            break

    mask = np.zeros((H, H), dtype=np.float32)
    if idx is not None:
        mask[idx, :n] = 1.0
        mask[:n, idx] = 1.0

    X = np.stack([cmap, mask], axis=0)  # (2,H,H)
    return X, residues, True

def mut_vec_4(wt: str, pos: int, mt: str):
    dh = float(HYDRO[mt] - HYDRO[wt])
    dv = float(VOLUME[mt] - VOLUME[wt])
    dq = float(CHARGE[mt] - CHARGE[wt])
    return np.array([pos, dh, dv, dq], dtype=np.float32)

def cnn_predict_one(cnn_model, pdb_bytes: bytes, chain_id: str, mutation: str):
    try:
        wt, pos, mt = parse_mutation(mutation)
        X, residues, ok = build_contact_map_and_mask(pdb_bytes, chain_id, pos)
        if not ok:
            return 0.0, False

        # If your CNN expects only (B,2,128,128):
        x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1,2,128,128)

        with torch.no_grad():
            y = cnn_model(x_t).squeeze().item()
        return float(y), True
    except Exception as e:
        st.warning(f"CNN error for {mutation}: {e}")
        return 0.0, False

def gnn_predict_one(gnn_model, pdb_bytes: bytes, chain_id: str, mutation: str):
    """
    This assumes your saved GNN is TorchScript (recommended) and accepts:
      (x, edge_index, mut_idx) or something similar.

    If your scripted model expects a single tensor input, adjust after we see its signature.
    """
    try:
        wt, pos, mt = parse_mutation(mutation)

        structure = PARSER.get_structure("p", io_bytes(pdb_bytes))
        residues = get_residues_with_ca(structure, chain_id)
        if len(residues) == 0:
            return 0.0, False

        coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)
        n = coords.shape[0]

        # edges within 8Å
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= 8.0:
                    edges.append([i, j])
                    edges.append([j, i])

        if len(edges) == 0:
            return 0.0, False

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2,E)

        # node features: 20 one-hot + 3 props + 2 flags = 25
        x = torch.zeros((n, 25), dtype=torch.float32)
        for i, r in enumerate(residues):
            # Bio.PDB residue name is 3-letter; we approximate to first letter isn’t valid.
            # We will mark AA one-hot as unknown if not mapped.
            resname = r.get_resname().upper()
            # crude mapping: 3-letter -> 1-letter (minimal)
            map3to1 = {
                "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K",
                "LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V",
                "TRP":"W","TYR":"Y"
            }
            aa = map3to1.get(resname, None)
            if aa in AA_INDEX:
                x[i, AA_INDEX[aa]] = 1.0
                x[i, 20] = float(HYDRO[aa])
                x[i, 21] = float(VOLUME[aa])
                x[i, 22] = float(CHARGE[aa])
            # mutation flag later

        # mut_idx (0-based index in residues list)
        mut_idx = None
        for i, r in enumerate(residues):
            if r.id[1] == pos:
                mut_idx = i
                break
        if mut_idx is None:
            return 0.0, False

        x[mut_idx, 23] = 1.0  # mutated position flag
        # optional: encode target mutant physchem deltas at the mutated node
        x[mut_idx, 24] = float(HYDRO[mt] - HYDRO[wt])

        with torch.no_grad():
            # Most robust attempt: try common signatures
            try:
                y = gnn_model(x, edge_index, torch.tensor([mut_idx], dtype=torch.long)).squeeze().item()
            except Exception:
                y = gnn_model(x, edge_index).squeeze().item()

        return float(y), True
    except Exception as e:
        st.warning(f"GNN error for {mutation}: {e}")
        return 0.0, False

def ensemble_predict_one(booster, bias, cnn_model, gnn_model, pdb_bytes, chain_id, mutation, weights):
    wx, wc, wg = weights
    preds = {}

    xgb_p, x_ok = xgb_predict_one(booster, bias, mutation)
    cnn_p, cnn_ok = cnn_predict_one(cnn_model, pdb_bytes, chain_id, mutation)
    gnn_p, gnn_ok = gnn_predict_one(gnn_model, pdb_bytes, chain_id, mutation)

    # If any structural model fails, keep ens but mark confidence lower
    ens = wx * xgb_p + wc * cnn_p + wg * gnn_p

    preds.update({
        "mutation": mutation,
        "xgb": xgb_p,
        "cnn": cnn_p,
        "gnn": gnn_p,
        "ens": float(ens),
        "cnn_ok": bool(cnn_ok),
        "gnn_ok": bool(gnn_ok),
        "ddg_class": ddg_class(float(ens)),
        "confidence": confidence_label(bool(cnn_ok), bool(gnn_ok)),
    })
    return preds

def scan_19aa_position(booster, bias, cnn_model, gnn_model, pdb_bytes, pdb_id, chain_id, pos, wt_aa=None, weights=(1/3,1/3,1/3)):
    if wt_aa is None:
        # user must supply if unknown
        raise ValueError("wt_aa is required (example: 'P')")

    rows = []
    for mt in AA_LIST:
        if mt == wt_aa:
            continue
        mut = f"{wt_aa}{pos}{mt}"
        out = ensemble_predict_one(booster, bias, cnn_model, gnn_model, pdb_bytes, chain_id, mut, weights)
        out["PDB_ID"] = pdb_id
        out["chain"] = chain_id
        rows.append(out)

    dfscan = pd.DataFrame(rows).sort_values("ens", ascending=True).reset_index(drop=True)
    dfscan["rank_stabilizing"] = np.arange(1, len(dfscan) + 1)
    dfscan["pct"] = dfscan["rank_stabilizing"] / float(len(dfscan))

    # highlight
    top_k = 5
    dfscan["highlight"] = ""
    dfscan.loc[:top_k-1, "highlight"] = "TOP stabilizing"
    dfscan.loc[len(dfscan)-top_k:, "highlight"] = "TOP destabilizing"
    return dfscan

def prioritize_scan(dfscan, top_k=5):
    # already ranked; just return as is (kept for your workflow)
    return dfscan.copy()

def io_bytes(b: bytes):
    # Bio.PDB parser needs a file-like. We'll build an in-memory handle.
    import io
    return io.BytesIO(b)

# -----------------------------
# Load models (shows errors nicely)
# -----------------------------
try:
    booster, xgb_bias, cnn_model, gnn_model = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Inputs")

pdb_id = st.sidebar.text_input("Protein ID label (any name)", value="NEW_PROT")
chain_id = st.sidebar.text_input("Chain ID", value="A").strip()

pdb_file = st.sidebar.file_uploader("Upload PDB (.pdb)", type=["pdb"])
if pdb_file is None:
    st.info("Upload a PDB file to enable CNN/GNN and ensemble scanning.")
    st.stop()

pdb_bytes = pdb_file.read()

mutation = st.sidebar.text_input("Single mutation (e.g., P66A)", value="P66A").strip().upper()

st.sidebar.subheader("Ensemble weights")
wx = st.sidebar.slider("XGB weight", 0.0, 1.0, 1/3)
wc = st.sidebar.slider("CNN weight", 0.0, 1.0, 1/3)
wg = st.sidebar.slider("GNN weight", 0.0, 1.0, 1/3)
s = wx + wc + wg
if s == 0:
    st.sidebar.error("Weights sum to 0. Set at least one > 0.")
    st.stop()
weights = (wx/s, wc/s, wg/s)
st.sidebar.caption(f"Normalized weights = {weights}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Single mutation prediction")
    if st.button("Predict mutation"):
        out = ensemble_predict_one(
            booster, xgb_bias, cnn_model, gnn_model,
            pdb_bytes, chain_id, mutation, weights
        )
        st.json(out)

with col2:
    st.subheader("19-aa scan at a position")
    pos = st.number_input("Position (integer)", value=66, step=1)
    wt_aa = st.text_input("WT residue at position (one-letter, e.g., P)", value="P").strip().upper()

    if st.button("Run 19-aa scan"):
        scan_df = scan_19aa_position(
            booster, xgb_bias, cnn_model, gnn_model,
            pdb_bytes,
            pdb_id=pdb_id,
            chain_id=chain_id,
            pos=int(pos),
            wt_aa=wt_aa,
            weights=weights
        )
        scan_prior = prioritize_scan(scan_df, top_k=5)

        st.dataframe(scan_prior, use_container_width=True)

        # Download CSV (no write-permission issues)
        csv_bytes = scan_prior.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scan CSV",
            data=csv_bytes,
            file_name=f"scan_{pdb_id}_{chain_id}_pos{int(pos)}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Note: This app writes outputs only to memory/downloads (Streamlit Cloud file system is read-only).")




