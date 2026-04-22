from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedData:
    path: Path
    dataframe: pd.DataFrame
    sheet_name: str | None = None


def load_csv(path: str | Path) -> LoadedData:
    p = Path(path)
    df = pd.read_csv(p)
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

