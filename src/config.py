"""Central paths for the antisym-ddg project. Override the project root with PROTIEN_ROOT."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PROTIEN_ROOT", Path(__file__).resolve().parents[1]))
DATA = str(ROOT / "data")          # parsed/ (committed) + raw clones & pdbs (gitignored)
WORK = str(ROOT / "artifacts")     # intermediate npz/json caches (gitignored)
FIGURES = str(ROOT / "figures")
RESULTS = str(ROOT / "results")
MODELS = str(ROOT / "models")
for _d in (DATA, WORK, FIGURES, RESULTS, MODELS, DATA + "/parsed", DATA + "/pdbs"):
    Path(_d).mkdir(parents=True, exist_ok=True)
