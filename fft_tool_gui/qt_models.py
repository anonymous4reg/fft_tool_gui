from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6 import QtCore


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return int(self._df.shape[0])

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return int(self._df.shape[1])

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role not in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if value is None:
            return ""
        return str(value)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return str(section)
        return str(self._df.index[section])

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

