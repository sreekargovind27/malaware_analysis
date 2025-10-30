import os
import shutil
from pathlib import Path

OUTPUT = "all_python_files"
shutil.rmtree(OUTPUT, ignore_errors=True)
os.makedirs(OUTPUT)

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "venv", "node_modules"]]
    for f in files:
        if f.endswith(".py"):
            src = Path(root) / f
            parent_name = src.parent.name.replace(".", "_")
            dst = Path(OUTPUT) / f"{parent_name}__{f}"
            shutil.copy2(src, dst)
            print(f"Copied: {src}")

print(f"\n✅ Done! Check '{OUTPUT}/' folder")