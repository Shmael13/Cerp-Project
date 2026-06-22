from typing import Any

import pandas as pd

from cerp_viz.core.base import BaseLoader


class ExcelLoader(BaseLoader):
    def accepts(self, source: Any) -> bool:
        name = source if isinstance(source, str) else getattr(source, "name", "")
        return name.lower().endswith((".xlsx", ".xls", ".xlsm"))

    def load(self, source: Any) -> dict[str, pd.DataFrame]:
        xl = pd.ExcelFile(source)
        return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
