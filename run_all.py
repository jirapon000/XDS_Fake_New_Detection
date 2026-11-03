"""
Re-run every notebook in order and save all figures as PNG files.

Usage (from the project root, with the env activated):
    python run_all.py                 # run everything, then save figures
    python run_all.py --only SVM      # run only notebooks whose path contains "SVM"
    python run_all.py --figures-only  # don't run anything, just save figures from the current outputs
"""
import argparse
import base64
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

# Order matters: each step uses files made by the steps before it
NOTEBOOKS = [
    "combine_dataset.ipynb",
    "dataset_eda.ipynb",
    "clean_dataset.ipynb",
    "Model/Baseline Model/Logistic_regression_TF-IDF.ipynb",
    "Model/Common Models/Decision_tree.ipynb",
    "Model/Common Models/Random_forest.ipynb",
    "Model/Common Models/LightGBM.ipynb",
    "Model/Common Models/XGBoost.ipynb",
    "Model/Common Models/SVM.ipynb",
    "Model/XAI Synthesis/xai_synthesis.ipynb",
    "Model/XAI Evaluation/xai_evaluation.ipynb",
]
FIG_DIR = "figures"


def run_notebook(path):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(path) or "."}})
    client.execute()
    nbformat.write(nb, path)          # save the new outputs back into the notebook
    return nb


def save_figures(nb, path):
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(FIG_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):     # clear old figures so removed plots don't linger
        if f.endswith(".png"):
            os.remove(os.path.join(out_dir, f))
    n = 0
    for c, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        k = 0
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if png:
                k += 1
                n += 1
                fname = f"{name}_cell{c:02d}_{k}.png"
                with open(os.path.join(out_dir, fname), "wb") as fh:
                    fh.write(base64.b64decode(png))
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run only notebooks whose path contains this text")
    parser.add_argument("--figures-only", action="store_true", help="skip running, just save figures")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))     # always work from the project root
    todo = [p for p in NOTEBOOKS if not args.only or args.only.lower() in p.lower()]

    missing = [p for p in todo if not os.path.exists(p)]
    if missing:
        print("These notebooks were not found (check the names in NOTEBOOKS):")
        for p in missing:
            print("   ", p)
        sys.exit(1)

    t_all = time.time()
    for k, path in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{k}/{len(todo)}] {path} ...", flush=True)
        try:
            nb = nbformat.read(path, as_version=4) if args.figures_only else run_notebook(path)
        except Exception as e:
            print(f"\nFAILED in {path}:\n{e}")
            print("Fix the error, then continue from here with:  python run_all.py --only \"<name>\"")
            sys.exit(1)
        n = save_figures(nb, path)
        print(f"      done in {time.time() - t0:.0f} s, {n} figures saved", flush=True)

    print(f"\nAll done in {(time.time() - t_all) / 60:.1f} min. Figures are in '{FIG_DIR}/'.")


if __name__ == "__main__":
    main()