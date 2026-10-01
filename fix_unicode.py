import re, pathlib
p = pathlib.Path("deep_train.py")
c = p.read_text(encoding="utf-8")
c = c.replace("f\"\\n{'\\u2500'*72}\"", '"-"*72')
c = c.replace("\u2500", "-")
p.write_text(c, encoding="utf-8")
print("fixed")
