"""patch_main_infer.py — inject IF7 + orthogonal into infer() return block"""
from pathlib import Path

p = Path("stage3/inference_engine/main.py")
src = p.read_text(encoding="utf-8")

OLD = '''        return {
            "turn": self._turn,
            "V": round(v, 6),
            "next_frame_prediction": nfp,
            "language_token_output": lto,
            "gate_status": gate_status,
            "M1": round(m_vals["M1"], 4),
            "M2": round(m_vals["M2"], 4),
            "M3": round(m_vals["M3"], 4),
            "binary_scalar": round(l2["binary_scalar"], 4),
            "geometry_scalar": round(l2["geometry_scalar"], 4),
            "language_scalar": round(l2["language_scalar"], 4),
        }'''

NEW = '''        # IF7: relational arbitration over scalar deltas
        scalars = [m_vals["M1"], m_vals["M2"], m_vals["M3"],
                   l2["binary_scalar"], l2["geometry_scalar"], l2["language_scalar"]]
        if7_fusion = self._if7.step(scalars)
        orth = orthogonal_reparameterize(v, scalars)

        return {
            "turn": self._turn,
            "V": round(v, 6),
            "next_frame_prediction": nfp,
            "language_token_output": lto,
            "gate_status": gate_status,
            "M1": round(m_vals["M1"], 4),
            "M2": round(m_vals["M2"], 4),
            "M3": round(m_vals["M3"], 4),
            "binary_scalar": round(l2["binary_scalar"], 4),
            "geometry_scalar": round(l2["geometry_scalar"], 4),
            "language_scalar": round(l2["language_scalar"], 4),
            "if7_policy_fusion": round(if7_fusion, 6),
            "if7_coherence": round(self._if7.coherence, 6),
            "if7_stability_loss": round(self._if7.stability_loss, 6),
            "if7_uncertainty": round(self._if7.uncertainty, 6),
            "orthogonal_vector": [round(x, 4) for x in orth],
        }'''

if OLD in src:
    src = src.replace(OLD, NEW)
    p.write_text(src, encoding="utf-8")
    print("PATCHED OK")
else:
    print("OLD block not found — showing return area:")
    idx = src.find("return {")
    print(repr(src[idx:idx+500]))
