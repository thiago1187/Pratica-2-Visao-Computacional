"""Limpa as saídas das células marcadas com a tag `dataset-images` (licença CC BY-NC-ND: não versionar imagens).

Uso: python tools/clean_notebook.py notebook_executado.ipynb notebook_limpo.ipynb
"""
import sys

import nbformat

src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
n = 0
for c in nb.cells:
    if c.cell_type == "code" and "dataset-images" in c.metadata.get("tags", []):
        c.outputs, c.execution_count = [], None
        n += 1
nbformat.write(nb, dst)
leaks = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.outputs
            if "image/png" in o.get("data", {}) and "dataset-images" in c.metadata.get("tags", []))
print(f"{n} células limpas; imagens de dataset restantes nas saídas marcadas: {leaks}")
