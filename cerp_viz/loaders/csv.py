from typing import Any

import pandas as pd

from cerp_viz.core.base import BaseLoader


class CSVLoader(BaseLoader):
    def accepts(self, source: Any) -> bool:
        name = source if isinstance(source, str) else getattr(source, "name", "")
        return name.lower().endswith(".csv")

    def load(self, source: Any) -> dict[str, pd.DataFrame]:
        return {"Sheet1": pd.read_csv(source)}
