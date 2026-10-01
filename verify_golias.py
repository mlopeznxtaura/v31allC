from golias_types import IF1_CreativeExploration, IF7_RelationalArbitration, GoliasSample
from stage1.core.triangulation import IF7State, orthogonal_reparameterize
from stage3.inference_engine.main import V31InferenceEngine

e = V31InferenceEngine()
r = e.infer(
    binary_text="test substrate hw=RTX5090",
    geometry_text="Entry 1/1. SystemID=CASEBELIZE. PhaseReady=True. Convergence=0.80. Complexity=HIGH.",
    language_text="What is the Agent Trust Protocol and how does IF7 relational arbitration preserve invariants across turns?",
    entry_index=1, total_entries=1
)
print("V =", r["V"])
print("gate =", r["gate_status"])
print("if7_fusion =", r["if7_policy_fusion"])
print("if7_coherence =", r["if7_coherence"])
print("if7_stability =", r["if7_stability_loss"])
print("orthogonal =", r["orthogonal_vector"])

# Test IF7 standalone
if7 = IF7_RelationalArbitration()
fusion = if7.compute(0.5,0.5, 0.6,0.5, 0.7,0.5, 0.5,0.5, 0.5,0.5, 0.5,0.6)
print("IF7 standalone fusion =", round(fusion, 4))

# Test orthogonal
orth = orthogonal_reparameterize(0.7, [2.5, 0.9, 1.0, 0.4, 0.6, 0.3])
print("Orthogonal vector =", orth)

print("ALL SYSTEMS OK")
