import json, sys
with open("/mnt/d/Users/archi/Downloads/screen_rec_1779777342_badas.json") as f:
    d = json.load(f)
print("type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys())[:15])
    for k in d:
        v = d[k]
        if isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
            print("  first:", json.dumps(v[0], indent=2)[:400])
        else:
            print(f"  {k}: {str(v)[:100]}")
elif isinstance(d, list):
    print(f"list[{len(d)}]")
    print("first:", json.dumps(d[0], indent=2)[:500])
