from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from fft_tool_gui.data_io import LoadedData, get_xlsx_sheet_names, load_csv, load_xlsx
from fft_tool_gui.fft_compute import compute_fft_amplitude
from fft_tool_gui.qt_models import DataFrameModel

PREVIEW_ROWS = 2000
SNIFF_ROWS = 1000
PLOT_MAX_POINTS = 20000


class LoadWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(str)

    def __init__(self, path: Path, sheet_name: str | None):
        super().__init__()
        self._path = path
        self._sheet_name = sheet_name

    @QtCore.Slot()
    def run(self) -> None:
        try:
            p = self._path
            suffix = p.suffix.lower()
            if suffix == ".csv":
                loaded = load_csv(p)
                sheet_names = None
            elif suffix == ".xlsx":
                sheet_names = get_xlsx_sheet_names(p)
                if self._sheet_name is None:
                    sheet = sheet_names[0] if sheet_names else None
                else:
                    sheet = self._sheet_name
                loaded = load_xlsx(p, sheet_name=sheet)
            else:
                raise ValueError("仅支持 .csv / .xlsx")

            self.finished.emit(loaded, sheet_names)
        except Exception as e:
            self.failed.emit(str(e))


@dataclass
class PlotState:
    frequency_hz: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    time_x: np.ndarray | None = None
    time_y: np.ndarray | None = None
    time_is_seconds: bool = False


class RightPanViewBox(pg.ViewBox):
    def mouseDragEvent(self, ev: Any, axis: Any = None) -> None:
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            if ev.isFinish():
                return
            x_range, y_range = self.viewRange()
            bounds = self.sceneBoundingRect()
            w = float(bounds.width()) if bounds.width() else 1.0
            h = float(bounds.height()) if bounds.height() else 1.0
            dx = ev.pos().x() - ev.lastPos().x()
            dy = ev.pos().y() - ev.lastPos().y()
            x_scale = (x_range[1] - x_range[0]) / w
            y_scale = (y_range[1] - y_range[0]) / h
            self.translateBy(x=-dx * x_scale, y=dy * y_scale)
            return
        super().mouseDragEvent(ev, axis=axis)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FFT Tool GUI")

        self._loaded: LoadedData | None = None
        self._plot_state = PlotState()
        self._load_thread: QtCore.QThread | None = None
        self._load_worker: LoadWorker | None = None
        self._progress: QtWidgets.QProgressDialog | None = None

        self._build_ui()
        self._wire_events()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        file_group = QtWidgets.QGroupBox("数据导入")
        file_layout = QtWidgets.QGridLayout(file_group)
        left_layout.addWidget(file_group)

        self.btn_open = QtWidgets.QPushButton("导入文件 (.csv/.xlsx)")
        self.lbl_path = QtWidgets.QLabel("未选择文件")
        self.lbl_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cbo_sheet = QtWidgets.QComboBox()
        self.cbo_sheet.setEnabled(False)

        file_layout.addWidget(self.btn_open, 0, 0, 1, 2)
        file_layout.addWidget(QtWidgets.QLabel("路径:"), 1, 0)
        file_layout.addWidget(self.lbl_path, 1, 1)
        file_layout.addWidget(QtWidgets.QLabel("Sheet:"), 2, 0)
        file_layout.addWidget(self.cbo_sheet, 2, 1)

        self.table = QtWidgets.QTableView()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        left_layout.addWidget(self.table, stretch=1)

        select_group = QtWidgets.QGroupBox("选择信号")
        select_layout = QtWidgets.QGridLayout(select_group)
        left_layout.addWidget(select_group)

        self.rdo_column = QtWidgets.QRadioButton("按列")
        self.rdo_row = QtWidgets.QRadioButton("按行")
        self.rdo_column.setChecked(True)

        self.cbo_column = QtWidgets.QComboBox()
        self.cbo_time = QtWidgets.QComboBox()
        self.cbo_time.setEnabled(True)
        self.spn_row = QtWidgets.QSpinBox()
        self.spn_row.setMinimum(0)
        self.spn_row.setMaximum(0)
        self.spn_row.setEnabled(False)

        select_layout.addWidget(self.rdo_column, 0, 0)
        select_layout.addWidget(self.rdo_row, 0, 1)
        select_layout.addWidget(QtWidgets.QLabel("列名:"), 1, 0)
        select_layout.addWidget(self.cbo_column, 1, 1)
        select_layout.addWidget(QtWidgets.QLabel("时间列:"), 2, 0)
        select_layout.addWidget(self.cbo_time, 2, 1)
        select_layout.addWidget(QtWidgets.QLabel("行号:"), 3, 0)
        select_layout.addWidget(self.spn_row, 3, 1)

        self.spn_range_start = QtWidgets.QSpinBox()
        self.spn_range_start.setMinimum(0)
        self.spn_range_start.setMaximum(0)
        self.spn_range_end = QtWidgets.QSpinBox()
        self.spn_range_end.setMinimum(0)
        self.spn_range_end.setMaximum(0)
        self.btn_range_all = QtWidgets.QPushButton("全选范围")

        select_layout.addWidget(QtWidgets.QLabel("范围起点:"), 4, 0)
        select_layout.addWidget(self.spn_range_start, 4, 1)
        select_layout.addWidget(QtWidgets.QLabel("范围终点:"), 5, 0)
        select_layout.addWidget(self.spn_range_end, 5, 1)
        select_layout.addWidget(self.btn_range_all, 6, 0, 1, 2)

        param_group = QtWidgets.QGroupBox("FFT 参数")
        param_layout = QtWidgets.QGridLayout(param_group)
        left_layout.addWidget(param_group)

        self.spn_fs = QtWidgets.QDoubleSpinBox()
        self.spn_fs.setMinimum(0.000001)
        self.spn_fs.setMaximum(1e12)
        self.spn_fs.setDecimals(6)
        self.spn_fs.setValue(1000.0)
        self.spn_fs.setSuffix(" Hz")

        self.chk_auto_nfft = QtWidgets.QCheckBox("自动 NFFT(下一次幂)")
        self.chk_auto_nfft.setChecked(True)
        self.spn_nfft = QtWidgets.QSpinBox()
        self.spn_nfft.setMinimum(2)
        self.spn_nfft.setMaximum(1_000_000_000)
        self.spn_nfft.setValue(1024)
        self.spn_nfft.setEnabled(False)

        self.chk_window = QtWidgets.QCheckBox("加窗")
        self.cbo_window = QtWidgets.QComboBox()
        self.cbo_window.addItems(["hann", "hamming", "blackman", "flattop", "kaiser"])
        self.cbo_window.setEnabled(False)
        self.spn_window_param = QtWidgets.QDoubleSpinBox()
        self.spn_window_param.setMinimum(0.0)
        self.spn_window_param.setMaximum(1e6)
        self.spn_window_param.setDecimals(6)
        self.spn_window_param.setValue(14.0)
        self.spn_window_param.setEnabled(False)

        self.btn_fft = QtWidgets.QPushButton("计算 FFT")

        param_layout.addWidget(QtWidgets.QLabel("采样率 fs:"), 0, 0)
        param_layout.addWidget(self.spn_fs, 0, 1)
        param_layout.addWidget(self.chk_auto_nfft, 1, 0, 1, 2)
        param_layout.addWidget(QtWidgets.QLabel("NFFT:"), 2, 0)
        param_layout.addWidget(self.spn_nfft, 2, 1)
        param_layout.addWidget(self.chk_window, 3, 0)
        param_layout.addWidget(self.cbo_window, 3, 1)
        param_layout.addWidget(QtWidgets.QLabel("窗参数(仅 kaiser beta):"), 4, 0)
        param_layout.addWidget(self.spn_window_param, 4, 1)
        param_layout.addWidget(self.btn_fft, 5, 0, 1, 2)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)

        tools_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(tools_row)

        self.btn_reset_view = QtWidgets.QPushButton("重置视图")
        tools_row.addWidget(self.btn_reset_view)
        tools_row.addStretch(1)

        plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_layout.addWidget(plots_splitter, stretch=1)

        self.time_plot = pg.PlotWidget(viewBox=RightPanViewBox())
        self.time_plot.showGrid(x=True, y=True, alpha=0.25)
        self.time_plot.setLabel("left", "Amplitude")
        plots_splitter.addWidget(self.time_plot)

        self.fft_plot = pg.PlotWidget(viewBox=RightPanViewBox())
        self.fft_plot.showGrid(x=True, y=True, alpha=0.25)
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Amplitude")
        plots_splitter.addWidget(self.fft_plot)

        plots_splitter.setStretchFactor(0, 1)
        plots_splitter.setStretchFactor(1, 3)

        self.lbl_pick = QtWidgets.QLabel("点选: -")
        self.lbl_pick.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self.lbl_pick)

        self._time_curve = self.time_plot.plot([], [], pen=pg.mkPen(160, 160, 160, width=1))
        self._time_sel_curve = self.time_plot.plot([], [], pen=pg.mkPen(220, 0, 0, width=2))

        self._curve = self.fft_plot.plot([], [], pen=pg.mkPen(width=2))
        self._marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(width=2), brush=pg.mkBrush(255, 255, 0, 180))
        self.fft_plot.addItem(self._marker)

        self.time_plot.setMenuEnabled(False)
        self.fft_plot.setMenuEnabled(False)
        self.time_plot.plotItem.vb.setMouseMode(pg.ViewBox.RectMode)
        self.fft_plot.plotItem.vb.setMouseMode(pg.ViewBox.RectMode)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    def _wire_events(self) -> None:
        self.btn_open.clicked.connect(self._on_open)
        self.cbo_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        self.chk_auto_nfft.toggled.connect(self._on_auto_nfft_toggled)
        self.chk_window.toggled.connect(self._on_window_toggled)
        self.cbo_window.currentIndexChanged.connect(self._on_window_changed)
        self.rdo_column.toggled.connect(self._on_signal_mode_changed)
        self.btn_fft.clicked.connect(self._on_fft)
        self.btn_reset_view.clicked.connect(self._on_reset_view)

        self.cbo_column.currentIndexChanged.connect(self._update_time_plot)
        self.cbo_time.currentIndexChanged.connect(self._update_time_plot)
        self.spn_row.valueChanged.connect(self._update_time_plot)
        self.spn_range_start.valueChanged.connect(self._update_time_plot)
        self.spn_range_end.valueChanged.connect(self._update_time_plot)
        self.btn_range_all.clicked.connect(self._on_range_all)

        self.fft_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

    def _set_loading(self, loading: bool, message: str = "") -> None:
        widgets = [
            self.btn_open,
            self.cbo_sheet,
            self.table,
            self.rdo_column,
            self.rdo_row,
            self.cbo_column,
            self.cbo_time,
            self.spn_row,
            self.spn_range_start,
            self.spn_range_end,
            self.btn_range_all,
            self.spn_fs,
            self.chk_auto_nfft,
            self.spn_nfft,
            self.chk_window,
            self.cbo_window,
            self.spn_window_param,
            self.btn_fft,
            self.btn_reset_view,
        ]
        for w in widgets:
            w.setEnabled(not loading)

        if loading:
            if self._progress is None:
                self._progress = QtWidgets.QProgressDialog(self)
                self._progress.setWindowTitle("加载中")
                self._progress.setRange(0, 0)
                self._progress.setCancelButton(None)
                self._progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            self._progress.setLabelText(message or "正在读取文件…")
            self._progress.show()
        else:
            if self._progress is not None:
                self._progress.hide()

    def _set_dataframe(self, df: pd.DataFrame) -> None:
        preview_df = df.head(PREVIEW_ROWS)
        model = DataFrameModel(preview_df)
        self.table.setModel(model)
        self.table.resizeColumnsToContents()

        self.cbo_column.clear()
        numeric_cols = []
        for col in df.columns:
            series = pd.to_numeric(df[col].head(SNIFF_ROWS), errors="coerce")
            if np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)).any():
                numeric_cols.append(str(col))
        self.cbo_column.addItems(numeric_cols)

        self.cbo_time.blockSignals(True)
        self.cbo_time.clear()
        self.cbo_time.addItem("无")
        for col in df.columns:
            self.cbo_time.addItem(str(col))
        inferred = self._infer_time_column(df)
        if inferred:
            idx = self.cbo_time.findText(inferred)
            if idx >= 0:
                self.cbo_time.setCurrentIndex(idx)
        else:
            self.cbo_time.setCurrentIndex(0)
        self.cbo_time.blockSignals(False)

        if df.shape[0] > 0:
            self.spn_row.setMaximum(max(0, int(df.shape[0] - 1)))
        else:
            self.spn_row.setMaximum(0)

        self._sync_range_controls()
        self._update_time_plot()

    def _on_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            str(Path.cwd()),
            "Data Files (*.csv *.xlsx);;CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return

        p = Path(path)
        self.lbl_path.setText(str(p))
        self._start_load(path=p, sheet_name=None)

    def _on_sheet_changed(self) -> None:
        if not self._loaded:
            return
        if self._loaded.path.suffix.lower() != ".xlsx":
            return
        sheet = self.cbo_sheet.currentText()
        if not sheet:
            return
        self._start_load(path=self._loaded.path, sheet_name=sheet)

    def _start_load(self, path: Path, sheet_name: str | None) -> None:
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait(200)
            self._load_thread = None
            self._load_worker = None

        self._set_loading(True, f"正在读取: {path.name}")

        thread = QtCore.QThread(self)
        worker = LoadWorker(path=path, sheet_name=sheet_name)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_load_finished)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @QtCore.Slot(object, object)
    def _on_load_finished(self, loaded: object, sheet_names: object) -> None:
        try:
            self._loaded = loaded  # type: ignore[assignment]
            if sheet_names is None:
                self.cbo_sheet.blockSignals(True)
                self.cbo_sheet.clear()
                self.cbo_sheet.setEnabled(False)
                self.cbo_sheet.blockSignals(False)
            else:
                names = list(sheet_names)
                self.cbo_sheet.blockSignals(True)
                self.cbo_sheet.clear()
                self.cbo_sheet.addItems(names)
                self.cbo_sheet.setEnabled(True)
                if self._loaded.sheet_name:
                    idx = self.cbo_sheet.findText(self._loaded.sheet_name)
                    if idx >= 0:
                        self.cbo_sheet.setCurrentIndex(idx)
                self.cbo_sheet.blockSignals(False)

            self._set_dataframe(self._loaded.dataframe)
        finally:
            self._set_loading(False)

    @QtCore.Slot(str)
    def _on_load_failed(self, message: str) -> None:
        self._loaded = None
        self._set_loading(False)
        QtWidgets.QMessageBox.critical(self, "导入失败", message)

    def _on_auto_nfft_toggled(self, checked: bool) -> None:
        self.spn_nfft.setEnabled(not checked)

    def _on_window_toggled(self, checked: bool) -> None:
        self.cbo_window.setEnabled(checked)
        self._update_window_param_enable()

    def _on_window_changed(self) -> None:
        self._update_window_param_enable()

    def _update_window_param_enable(self) -> None:
        enabled = self.chk_window.isChecked() and (self.cbo_window.currentText().lower() == "kaiser")
        self.spn_window_param.setEnabled(enabled)

    def _on_signal_mode_changed(self) -> None:
        by_col = self.rdo_column.isChecked()
        self.cbo_column.setEnabled(by_col)
        self.cbo_time.setEnabled(by_col)
        self.spn_row.setEnabled(not by_col)
        self._sync_range_controls()
        self._update_time_plot()

    def _infer_time_column(self, df: pd.DataFrame) -> str | None:
        keywords = ("time", "timestamp", "t", "sec", "ms", "us", "ns")
        for col in df.columns:
            name = str(col).strip().lower()
            if any(k in name for k in keywords):
                return str(col)
        for col in df.columns:
            s = df[col].head(SNIFF_ROWS)
            if pd.api.types.is_datetime64_any_dtype(s):
                return str(col)
            parsed = pd.to_datetime(s, errors="coerce")
            ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
            if ratio >= 0.9:
                return str(col)
        return None

    def _datetime_like_series_to_seconds(self, s: pd.Series) -> np.ndarray:
        t = pd.to_datetime(s, errors="coerce")
        t_np = t.to_numpy(dtype="datetime64[ns]")
        t_ns = t_np.astype("int64").astype(float)
        t_ns[t.isna().to_numpy()] = np.nan
        if not np.isfinite(t_ns).any():
            raise ValueError("时间列无法解析为有效时间")
        return (t_ns - np.nanmin(t_ns)) / 1e9

    def _get_time_axis_and_signal(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if not self._loaded:
            raise ValueError("请先导入数据")
        df = self._loaded.dataframe
        if df.empty:
            raise ValueError("数据为空")

        if self.rdo_column.isChecked():
            col = self.cbo_column.currentText()
            if not col:
                raise ValueError("未选择列")
            y_series = pd.to_numeric(df[col], errors="coerce")
            y = y_series.to_numpy(dtype=float, na_value=np.nan)

            time_name = self.cbo_time.currentText().strip()
            if time_name and time_name != "无" and time_name in df.columns:
                t_raw = df[time_name]
                if pd.api.types.is_datetime64_any_dtype(t_raw):
                    x = self._datetime_like_series_to_seconds(t_raw)
                    is_seconds = True
                else:
                    parsed = pd.to_datetime(t_raw, errors="coerce")
                    if float(parsed.notna().mean()) >= 0.9:
                        x = self._datetime_like_series_to_seconds(t_raw)
                        is_seconds = True
                    else:
                        x = pd.to_numeric(t_raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                        is_seconds = False
            else:
                x = np.arange(y.size, dtype=float)
                is_seconds = False

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if y.size < 2:
                raise ValueError("所选列的有效数值不足")
            return x, y, is_seconds

        row = int(self.spn_row.value())
        row_values = df.iloc[row, :].to_numpy()
        y_series = pd.to_numeric(pd.Series(row_values), errors="coerce").dropna()
        y = y_series.to_numpy(dtype=float)
        x = np.arange(y.size, dtype=float)
        if y.size < 2:
            raise ValueError("所选行的有效数值不足")
        return x, y, False

    def _sync_range_controls(self) -> None:
        if not self._loaded:
            n = 0
        else:
            df = self._loaded.dataframe
            if self.rdo_column.isChecked():
                n = int(df.shape[0])
            else:
                n = int(df.shape[1])

        max_idx = max(0, n - 1)
        for spn in (self.spn_range_start, self.spn_range_end):
            spn.blockSignals(True)
            spn.setMaximum(max_idx)
            if spn.value() > max_idx:
                spn.setValue(max_idx)
            spn.blockSignals(False)

        if n > 0 and self.spn_range_end.value() < self.spn_range_start.value():
            self.spn_range_end.setValue(self.spn_range_start.value())

    def _get_selected_segment(self) -> tuple[np.ndarray, np.ndarray, bool]:
        x, y, is_seconds = self._get_time_axis_and_signal()
        if y.size == 0:
            raise ValueError("数据为空")
        self._sync_range_controls()
        start = int(self.spn_range_start.value())
        end = int(self.spn_range_end.value())
        if end < start:
            start, end = end, start
        end = min(end, int(y.size - 1))
        start = max(0, min(start, end))
        return x[start : end + 1], y[start : end + 1], is_seconds

    def _on_range_all(self) -> None:
        try:
            _, y, _ = self._get_time_axis_and_signal()
            if y.size == 0:
                return
            self.spn_range_start.setValue(0)
            self.spn_range_end.setValue(int(y.size - 1))
        finally:
            self._update_time_plot()

    def _update_time_plot(self) -> None:
        if not self._loaded:
            self._time_curve.setData([], [])
            self._time_sel_curve.setData([], [])
            self._plot_state.time_x = None
            self._plot_state.time_y = None
            return

        try:
            x, y, is_seconds = self._get_time_plot_series(max_points=PLOT_MAX_POINTS)
            x_sel, y_sel, _ = self._get_time_plot_series(max_points=PLOT_MAX_POINTS, selected_only=True)
        except Exception:
            self._time_curve.setData([], [])
            self._time_sel_curve.setData([], [])
            self._plot_state.time_x = None
            self._plot_state.time_y = None
            return

        self._plot_state.time_x = x
        self._plot_state.time_y = y
        self._plot_state.time_is_seconds = is_seconds

        self._time_curve.setData(x, y, connect="finite", autoDownsample=True, clipToView=True)

        self._sync_range_controls()
        self._time_sel_curve.setData(x_sel, y_sel, connect="finite", autoDownsample=True, clipToView=True)

        if self.rdo_column.isChecked() and self.cbo_time.currentText().strip() != "无":
            if is_seconds:
                self.time_plot.setLabel("bottom", "Time", units="s")
            else:
                self.time_plot.setLabel("bottom", "Time")
        else:
            self.time_plot.setLabel("bottom", "Sample")

    def _get_time_plot_series(
        self,
        max_points: int,
        selected_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if not self._loaded:
            raise ValueError("请先导入数据")
        df = self._loaded.dataframe
        if df.empty:
            raise ValueError("数据为空")

        if self.rdo_column.isChecked():
            col = self.cbo_column.currentText()
            if not col:
                raise ValueError("未选择列")

            if selected_only:
                start = int(self.spn_range_start.value())
                end = int(self.spn_range_end.value())
                if end < start:
                    start, end = end, start
                start = max(0, min(start, int(df.shape[0] - 1)))
                end = max(0, min(end, int(df.shape[0] - 1)))
                data_slice = slice(start, end + 1)
            else:
                data_slice = slice(None)

            y_raw = df[col].iloc[data_slice]
            n = int(y_raw.shape[0])
            step = max(1, int(np.ceil(n / max_points))) if max_points > 0 else 1
            y_raw = y_raw.iloc[::step]
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan)

            time_name = self.cbo_time.currentText().strip()
            if time_name and time_name != "无" and time_name in df.columns:
                t_raw = df[time_name].iloc[data_slice].iloc[::step]
                if pd.api.types.is_datetime64_any_dtype(t_raw):
                    x = self._datetime_like_series_to_seconds(t_raw)
                    is_seconds = True
                else:
                    parsed = pd.to_datetime(t_raw, errors="coerce")
                    if float(parsed.notna().mean()) >= 0.9:
                        x = self._datetime_like_series_to_seconds(t_raw)
                        is_seconds = True
                    else:
                        x = pd.to_numeric(t_raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                        is_seconds = False
            else:
                if selected_only:
                    x = np.arange(int(self.spn_range_start.value()), int(self.spn_range_end.value()) + 1, step, dtype=float)
                else:
                    x = np.arange(0, n, step, dtype=float)
                is_seconds = False

            mask = np.isfinite(x) & np.isfinite(y)
            return x[mask], y[mask], is_seconds

        row = int(self.spn_row.value())
        row_values = df.iloc[row, :]
        n = int(row_values.shape[0])
        if selected_only:
            start = int(self.spn_range_start.value())
            end = int(self.spn_range_end.value())
            if end < start:
                start, end = end, start
            start = max(0, min(start, n - 1))
            end = max(0, min(end, n - 1))
            idx = np.arange(start, end + 1, dtype=int)
        else:
            idx = np.arange(0, n, dtype=int)

        step = max(1, int(np.ceil(idx.size / max_points))) if max_points > 0 else 1
        idx = idx[::step]
        y = pd.to_numeric(row_values.iloc[idx], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        x = idx.astype(float)
        mask = np.isfinite(y)
        return x[mask], y[mask], False

    def _on_fft(self) -> None:
        try:
            y = self._get_selected_y_for_fft()
            fs = float(self.spn_fs.value())
            if self.chk_auto_nfft.isChecked():
                nfft = None
            else:
                nfft = int(self.spn_nfft.value())

            window_enabled = self.chk_window.isChecked()
            window_name = self.cbo_window.currentText()
            window_param = float(self.spn_window_param.value()) if window_name.lower() == "kaiser" else None

            result = compute_fft_amplitude(
                x=y,
                fs_hz=fs,
                nfft=nfft,
                window_enabled=window_enabled,
                window_name=window_name,
                window_param=window_param,
            )

            self._plot_state.frequency_hz = result.frequency_hz
            self._plot_state.amplitude = result.amplitude

            self._curve.setData(result.frequency_hz, result.amplitude)
            self._marker.setData([])
            self.fft_plot.enableAutoRange()
            self.statusBar().showMessage(f"N={y.size}, NFFT={result.nfft}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "FFT 失败", str(e))

    def _get_selected_y_for_fft(self) -> np.ndarray:
        if not self._loaded:
            raise ValueError("请先导入数据")
        df = self._loaded.dataframe
        if df.empty:
            raise ValueError("数据为空")

        self._sync_range_controls()
        start = int(self.spn_range_start.value())
        end = int(self.spn_range_end.value())
        if end < start:
            start, end = end, start

        if self.rdo_column.isChecked():
            col = self.cbo_column.currentText()
            if not col:
                raise ValueError("未选择列")
            start = max(0, min(start, int(df.shape[0] - 1)))
            end = max(0, min(end, int(df.shape[0] - 1)))
            y_raw = df[col].iloc[start : end + 1]
            y = pd.to_numeric(y_raw, errors="coerce").dropna().to_numpy(dtype=float)
        else:
            row = int(self.spn_row.value())
            row_values = df.iloc[row, :]
            n = int(row_values.shape[0])
            start = max(0, min(start, n - 1))
            end = max(0, min(end, n - 1))
            y = pd.to_numeric(row_values.iloc[start : end + 1], errors="coerce").dropna().to_numpy(dtype=float)

        if y.size < 2:
            raise ValueError("所选范围的有效数值不足")
        return y

    def _on_plot_clicked(self, mouse_event: Any) -> None:
        if self._plot_state.frequency_hz is None or self._plot_state.amplitude is None:
            return
        if mouse_event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        view_pos = self.fft_plot.plotItem.vb.mapSceneToView(mouse_event.scenePos())
        x_click = float(view_pos.x())

        f = self._plot_state.frequency_hz
        a = self._plot_state.amplitude
        if f.size == 0:
            return

        idx = int(np.clip(np.searchsorted(f, x_click), 0, f.size - 1))
        if idx > 0 and idx < f.size:
            left = idx - 1
            right = idx
            if abs(f[left] - x_click) < abs(f[right] - x_click):
                idx = left
            else:
                idx = right

        fx = float(f[idx])
        ay = float(a[idx])
        self._marker.setData([fx], [ay])
        self.lbl_pick.setText(f"点选: f={fx:.6g} Hz, y={ay:.6g}")

    def _on_reset_view(self) -> None:
        self.time_plot.enableAutoRange()
        self.fft_plot.enableAutoRange()


def run() -> None:
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(1200, 700)
    window.show()
    raise SystemExit(app.exec())
