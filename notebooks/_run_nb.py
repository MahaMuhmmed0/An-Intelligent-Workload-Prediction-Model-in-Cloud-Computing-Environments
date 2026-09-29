"""Execute a notebook's code cells as a script (no jupyter needed) for verification."""
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
src = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
src = src.replace("plt.show()", "plt.close('all')")
g = {"__name__": "__main__"}
exec(compile(src, sys.argv[1], "exec"), g)
