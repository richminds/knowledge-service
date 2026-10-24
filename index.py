"""Vercel entrypoint (pyproject.toml [tool.vercel] entrypoint = "index:app").

The application lives in src/app and imports itself as the top-level `app`
package, the way `uvicorn --app-dir src` runs it locally. Vercel imports this
module from the repository root instead, so put src/ on the path first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.main import app  # noqa: E402

__all__ = ["app"]
