import json
with open("output/master_corpus.jsonl", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx == 3234:
            print(f"Row {idx} content:")
            print(line)
            # Find where 'nan' is (case insensitive)
            pos = line.lower().find("nan")
            print(f"Position: {pos}")
            print(f"Context: {line[max(0, pos-20):pos+30]}")
