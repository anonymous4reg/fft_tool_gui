from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedData:
    path: Path
    dataframe: pd.DataFrame
    sheet_name: str | None = None


def load_csv(path: str | Path, has_header: bool | None = None) -> LoadedData:
    p = Path(path)
    if has_header is None:
        preview = pd.read_csv(p, header=None, nrows=1, dtype=str, keep_default_na=False)
        first_row_values = [str(v).strip() for v in preview.iloc[0].tolist()] if not preview.empty else []
        inferred_has_header = True
        if first_row_values:
            if any(any(ch.isalpha() for ch in v) for v in first_row_values):
                inferred_has_header = True
            else:
                numeric_like = 0
                for v in first_row_values:
                    try:
                        float(v)
                        numeric_like += 1
                    except Exception:
                        pass
                inferred_has_header = (numeric_like / max(1, len(first_row_values))) < 0.8
        has_header = inferred_has_header

    if has_header is False:
        df = pd.read_csv(p, header=None)
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(p, header=0)
    return LoadedData(path=p, dataframe=df, sheet_name=None)


def get_xlsx_sheet_names(path: str | Path) -> list[str]:
    p = Path(path)
    excel = pd.ExcelFile(p)
    return list(excel.sheet_names)


def load_xlsx(path: str | Path, sheet_name: str | int | None = None) -> LoadedData:
    p = Path(path)
    df = pd.read_excel(p, sheet_name=sheet_name)
    if sheet_name is None:
        resolved_sheet = None
    else:
        resolved_sheet = str(sheet_name)
    return LoadedData(path=p, dataframe=df, sheet_name=resolved_sheet)
