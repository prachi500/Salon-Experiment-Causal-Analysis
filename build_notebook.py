"""Convert analysis_src.py (percent format) into an executed Jupyter notebook."""
import re
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

src = Path("analysis_src.py").read_text()
chunks = re.split(r"^# %%(?: \[markdown\])?\s*$", src, flags=re.M)
heads = re.findall(r"^# %%( \[markdown\])?\s*$", src, flags=re.M)
nb = nbf.v4.new_notebook()
for head, body in zip(heads, chunks[1:]):
    body = body.strip("\n")
    if head:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbf.v4.new_code_cell(body))
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "Salon_Experiment_Causal_Analysis.ipynb")
print("cells:", len(nb.cells))
