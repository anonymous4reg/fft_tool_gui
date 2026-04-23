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

    def __init__(self, path: Path, sheet_name: str | None, csv_has_header: bool | None):
        super().__init__()
        self._path = path
        self._sheet_name = sheet_name
        self._csv_has_header = csv_has_header

    @QtCore.Slot()
    def run(self) -> None:
        try:
            p = self._path
            suffix = p.suffix.lower()
            if suffix == ".csv":
                loaded = load_csv(p, has_header=self._csv_has_header)
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._drag_start_range: tuple[list[float], list[float]] | None = None
        self._drag_start_pos: QtCore.QPointF | None = None
        self._drag_item: QtWidgets.QGraphicsItem | None = None
        self._drag_mode: str | None = None

    def _apply_left_drag_zoom(self, ev: Any) -> None:
        if self._drag_start_range is None or self._drag_start_pos is None:
            return

        start_pos = self._drag_start_pos
        end_pos = QtCore.QPointF(ev.pos())
        dx = float(end_pos.x() - start_pos.x())
        dy = float(end_pos.y() - start_pos.y())
        if dx == 0.0 and dy == 0.0:
            return

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        if abs_dx > abs_dy * 2.0:
            axis_mode = "x"
        elif abs_dy > abs_dx * 2.0:
            axis_mode = "y"
        else:
            axis_mode = "xy"

        zoom_out = dx < 0.0 and dy < 0.0

        start_scene = self.mapToScene(start_pos)
        end_scene = self.mapToScene(end_pos)
        p1 = self.mapSceneToView(start_scene)
        p2 = self.mapSceneToView(end_scene)
        x1 = float(min(p1.x(), p2.x()))
        x2 = float(max(p1.x(), p2.x()))
        y1 = float(min(p1.y(), p2.y()))
        y2 = float(max(p1.y(), p2.y()))

        x0_range, y0_range = self._drag_start_range
        x0_min, x0_max = float(x0_range[0]), float(x0_range[1])
        y0_min, y0_max = float(y0_range[0]), float(y0_range[1])

        center_pos = QtCore.QPointF((start_pos.x() + end_pos.x()) / 2.0, (start_pos.y() + end_pos.y()) / 2.0)
        center_view = self.mapSceneToView(self.mapToScene(center_pos))
        cx = float(center_view.x())
        cy = float(center_view.y())

        if zoom_out:
            bounds = self.boundingRect()
            view_w = float(bounds.width()) if bounds.width() else 1.0
            view_h = float(bounds.height()) if bounds.height() else 1.0
            box_w = max(1.0, abs_dx)
            box_h = max(1.0, abs_dy)

            if axis_mode in ("x", "xy"):
                fx = max(1e-6, box_w / view_w)
                new_w = (x0_max - x0_min) / fx
                self.setXRange(cx - new_w / 2.0, cx + new_w / 2.0, padding=0.0)
            else:
                self.setXRange(x0_min, x0_max, padding=0.0)

            if axis_mode in ("y", "xy"):
                fy = max(1e-6, box_h / view_h)
                new_h = (y0_max - y0_min) / fy
                self.setYRange(cy - new_h / 2.0, cy + new_h / 2.0, padding=0.0)
            else:
                self.setYRange(y0_min, y0_max, padding=0.0)
            return

        if axis_mode in ("x", "xy"):
            if x2 > x1:
                self.setXRange(x1, x2, padding=0.0)
        else:
            self.setXRange(x0_min, x0_max, padding=0.0)

        if axis_mode in ("y", "xy"):
            if y2 > y1:
                self.setYRange(y1, y2, padding=0.0)
        else:
            self.setYRange(y0_min, y0_max, padding=0.0)

    def _clear_drag_feedback(self) -> None:
        if self._drag_item is not None:
            try:
                if self._drag_item.scene() is not None:
                    self._drag_item.scene().removeItem(self._drag_item)
            except Exception:
                pass
        self._drag_item = None
        self._drag_mode = None

    def _ensure_drag_item(self, mode: str) -> None:
        if self._drag_mode == mode and self._drag_item is not None:
            return
        self._clear_drag_feedback()
        pen = pg.mkPen("#ffcc00", width=2)
        if mode == "xy":
            item = QtWidgets.QGraphicsRectItem()
            item.setPen(pen)
            item.setBrush(pg.mkBrush(255, 204, 0, 40))
        else:
            item = QtWidgets.QGraphicsPathItem()
            item.setPen(pen)
        item.setZValue(10_000)
        item.setParentItem(self)
        self._drag_item = item
        self._drag_mode = mode

    def _update_drag_feedback(self, ev: Any) -> None:
        if self._drag_start_pos is None:
            return
        start_pos = self._drag_start_pos
        end_pos = QtCore.QPointF(ev.pos())
        dx = float(end_pos.x() - start_pos.x())
        dy = float(end_pos.y() - start_pos.y())
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx < 2.0 and abs_dy < 2.0:
            self._clear_drag_feedback()
            return

        if abs_dx > abs_dy * 2.0 and abs_dy <= 10.0:
            mode = "x"
        elif abs_dy > abs_dx * 2.0 and abs_dx <= 10.0:
            mode = "y"
        else:
            mode = "xy"

        self._ensure_drag_item(mode)
        if self._drag_item is None:
            return

        x1 = float(min(start_pos.x(), end_pos.x()))
        x2 = float(max(start_pos.x(), end_pos.x()))
        y1 = float(min(start_pos.y(), end_pos.y()))
        y2 = float(max(start_pos.y(), end_pos.y()))

        if mode == "xy":
            rect_item = self._drag_item  # type: ignore[assignment]
            if isinstance(rect_item, QtWidgets.QGraphicsRectItem):
                rect_item.setRect(QtCore.QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)))
            return

        path_item = self._drag_item
        if not isinstance(path_item, QtWidgets.QGraphicsPathItem):
            return

        path = QtGui.QPainterPath()
        bar_len = 22.0
        half = bar_len / 2.0
        if mode == "x":
            ymid = float((start_pos.y() + end_pos.y()) / 2.0)
            path.moveTo(x1, ymid - half)
            path.lineTo(x1, ymid + half)
            path.moveTo(x2, ymid - half)
            path.lineTo(x2, ymid + half)
            path.moveTo(x1, ymid)
            path.lineTo(x2, ymid)
        else:
            xmid = float((start_pos.x() + end_pos.x()) / 2.0)
            path.moveTo(xmid - half, y1)
            path.lineTo(xmid + half, y1)
            path.moveTo(xmid - half, y2)
            path.lineTo(xmid + half, y2)
            path.moveTo(xmid, y1)
            path.lineTo(xmid, y2)
        path_item.setPath(path)

    def mouseDragEvent(self, ev: Any, axis: Any = None) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            if ev.isStart():
                self._drag_start_range = self.viewRange()
                self._drag_start_pos = QtCore.QPointF(ev.buttonDownPos())
            self._update_drag_feedback(ev)
            if ev.isFinish():
                self._apply_left_drag_zoom(ev)
                self._clear_drag_feedback()
            return

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

    def wheelEvent(self, ev: Any, axis: Any = None) -> None:
        delta: float | None = None
        ev_delta = getattr(ev, "delta", None)
        if ev_delta is not None:
            try:
                delta = float(ev_delta() if callable(ev_delta) else ev_delta)
            except Exception:
                delta = None
        if delta is None:
            try:
                delta = float(ev.angleDelta().y())
            except Exception:
                delta = None
        if delta is None:
            super().wheelEvent(ev, axis=axis)
            return

        if delta == 0.0:
            return

        mods: Any = None
        ev_mods = getattr(ev, "modifiers", None)
        if ev_mods is not None and callable(ev_mods):
            try:
                mods = ev_mods()
            except Exception:
                mods = None
        if mods is None:
            mods = QtWidgets.QApplication.keyboardModifiers()
        zoom_y_only = bool(mods & QtCore.Qt.ControlModifier)
        zoom_x_only = bool(mods & (QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier))

        steps = delta / 120.0
        base = 0.98
        scale = base**steps
        if zoom_x_only or zoom_y_only:
            sx = scale if zoom_x_only else 1.0
            sy = scale if zoom_y_only else 1.0
        else:
            sx = scale
            sy = scale

        center = self.mapSceneToView(ev.scenePos())
        self.scaleBy((sx, sy), center=center)
        ev.accept()

    def mouseClickEvent(self, ev: Any) -> None:
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            ev.accept()
            self.enableAutoRange()
            return
        super().mouseClickEvent(ev)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FFT Tool GUI")

        self._loaded: LoadedData | None = None
        self._raw_df: pd.DataFrame | None = None
        self._df: pd.DataFrame | None = None
        self._use_transposed = False
        self._plot_state = PlotState()
        self._load_thread: QtCore.QThread | None = None
        self._load_worker: LoadWorker | None = None
        self._progress: QtWidgets.QProgressDialog | None = None
        self._pending_time_autorange = False
        self._updating_range_inputs = False

        self._build_ui()
        self._wire_events()

    def _sync_param_enabled_states(self) -> None:
        self.spn_nfft.setEnabled(not self.chk_auto_nfft.isChecked())
        self.cbo_window.setEnabled(self.chk_window.isChecked())
        self._update_window_param_enable()

    def _cleanup_loader(self) -> None:
        if self._load_thread is not None:
            try:
                self._load_thread.quit()
                self._load_thread.wait(200)
            except RuntimeError:
                pass
        self._load_thread = None
        self._load_worker = None

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
        self.cbo_csv_header = QtWidgets.QComboBox()
        self.cbo_csv_header.addItems(["自动识别", "有表头", "无表头"])

        file_layout.addWidget(self.btn_open, 0, 0, 1, 2)
        file_layout.addWidget(QtWidgets.QLabel("路径:"), 1, 0)
        file_layout.addWidget(self.lbl_path, 1, 1)
        file_layout.addWidget(QtWidgets.QLabel("Sheet:"), 2, 0)
        file_layout.addWidget(self.cbo_sheet, 2, 1)
        file_layout.addWidget(QtWidgets.QLabel("CSV 表头:"), 3, 0)
        file_layout.addWidget(self.cbo_csv_header, 3, 1)

        self.table = QtWidgets.QTableView()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        left_layout.addWidget(self.table, stretch=1)

        select_group = QtWidgets.QGroupBox("选择信号")
        select_layout = QtWidgets.QGridLayout(select_group)
        left_layout.addWidget(select_group)

        self.btn_transpose = QtWidgets.QPushButton("转置数据: 关")
        self.btn_transpose.setCheckable(True)
        self.btn_transpose.setEnabled(False)
        self.cbo_column = QtWidgets.QComboBox()
        self.cbo_time = QtWidgets.QComboBox()
        self.cbo_time.setEnabled(True)

        select_layout.addWidget(self.btn_transpose, 0, 0, 1, 2)
        select_layout.addWidget(QtWidgets.QLabel("时间列:"), 1, 0)
        select_layout.addWidget(self.cbo_time, 1, 1)
        select_layout.addWidget(QtWidgets.QLabel("信号列:"), 2, 0)
        select_layout.addWidget(self.cbo_column, 2, 1)

        self.spn_range_start = QtWidgets.QSpinBox()
        self.spn_range_start.setMinimum(0)
        self.spn_range_start.setMaximum(0)
        self.spn_range_end = QtWidgets.QSpinBox()
        self.spn_range_end.setMinimum(0)
        self.spn_range_end.setMaximum(0)
        self.spn_range_len = QtWidgets.QSpinBox()
        self.spn_range_len.setMinimum(1)
        self.spn_range_len.setMaximum(1)
        self.spn_range_len.setValue(1)
        self.spn_range_len.setEnabled(False)
        self.btn_range_all = QtWidgets.QPushButton("全选范围")

        select_layout.addWidget(QtWidgets.QLabel("信号范围起点:"), 3, 0)
        select_layout.addWidget(self.spn_range_start, 3, 1)
        select_layout.addWidget(QtWidgets.QLabel("信号范围终点:"), 4, 0)
        select_layout.addWidget(self.spn_range_end, 4, 1)
        select_layout.addWidget(QtWidgets.QLabel("信号长度L:"), 5, 0)
        select_layout.addWidget(self.spn_range_len, 5, 1)
        select_layout.addWidget(self.btn_range_all, 6, 0, 1, 2)

        param_group = QtWidgets.QGroupBox("FFT 参数")
        param_layout = QtWidgets.QGridLayout(param_group)
        left_layout.addWidget(param_group)

        self.spn_fs = QtWidgets.QDoubleSpinBox()
        self.spn_fs.setMinimum(0.000001)
        self.spn_fs.setMaximum(1e12)
        self.spn_fs.setDecimals(2)
        self.spn_fs.setValue(10000.0)
        self.spn_fs.setSuffix(" Hz")

        self.chk_auto_nfft = QtWidgets.QCheckBox("自动 NFFT(下一次幂)")
        self.chk_auto_nfft.setChecked(False)
        self.spn_nfft = QtWidgets.QSpinBox()
        self.spn_nfft.setMinimum(2)
        self.spn_nfft.setMaximum(1_000_000_000)
        self.spn_nfft.setValue(10000)
        self.spn_nfft.setEnabled(True)

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
        param_layout.addWidget(QtWidgets.QLabel("NFFT(FFT变换点数):"), 2, 0)
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

        self._time_curve = self.time_plot.plot([], [], pen=pg.mkPen("#1f77b4", width=1))
        self._time_sel_curve = self.time_plot.plot([], [], pen=pg.mkPen("#d62728", width=2))

        self._curve = self.fft_plot.plot([], [], pen=pg.mkPen("#9467bd", width=2))
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
        self.cbo_csv_header.currentIndexChanged.connect(self._on_csv_header_changed)
        self.chk_auto_nfft.toggled.connect(self._on_auto_nfft_toggled)
        self.chk_window.toggled.connect(self._on_window_toggled)
        self.cbo_window.currentIndexChanged.connect(self._on_window_changed)
        self.btn_transpose.toggled.connect(self._on_transpose_toggled)
        self.btn_fft.clicked.connect(self._on_fft)
        self.btn_reset_view.clicked.connect(self._on_reset_view)

        self.cbo_column.currentIndexChanged.connect(self._on_signal_column_changed)
        self.cbo_time.currentIndexChanged.connect(self._on_time_column_changed)
        self.spn_range_start.valueChanged.connect(self._on_range_start_changed)
        self.spn_range_end.valueChanged.connect(self._on_range_end_changed)
        self.spn_range_len.valueChanged.connect(self._on_range_len_changed)
        self.btn_range_all.clicked.connect(self._on_range_all)

        self.fft_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

    def _set_loading(self, loading: bool, message: str = "") -> None:
        widgets = [
            self.btn_open,
            self.cbo_sheet,
            self.cbo_csv_header,
            self.table,
            self.btn_transpose,
            self.cbo_column,
            self.cbo_time,
            self.spn_range_start,
            self.spn_range_end,
            self.spn_range_len,
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
            self._sync_param_enabled_states()

    def _set_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df
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

        self._sync_range_controls()
        self._pending_time_autorange = True
        self._update_time_plot()

    def _apply_current_dataframe(self) -> None:
        if self._raw_df is None:
            self._df = None
            self.btn_transpose.setEnabled(False)
            return
        df = self._raw_df.transpose() if self._use_transposed else self._raw_df
        self.btn_transpose.setEnabled(True)
        self.btn_transpose.blockSignals(True)
        self.btn_transpose.setChecked(self._use_transposed)
        self.btn_transpose.setText("转置数据: 开" if self._use_transposed else "转置数据: 关")
        self.btn_transpose.blockSignals(False)
        self._set_dataframe(df)

    def _on_signal_column_changed(self, _index: int) -> None:
        self._pending_time_autorange = True
        self._update_time_plot()

    def _on_time_column_changed(self, _index: int) -> None:
        self._pending_time_autorange = True
        self._update_time_plot()

    def _update_range_len_from_range(self) -> None:
        if self._df is None:
            return
        if self._updating_range_inputs:
            return
        self._updating_range_inputs = True
        try:
            start = int(self.spn_range_start.value())
            end = int(self.spn_range_end.value())
            if end < start:
                end = start
                self.spn_range_end.blockSignals(True)
                self.spn_range_end.setValue(end)
                self.spn_range_end.blockSignals(False)

            length = max(1, end - start + 1)
            self.spn_range_len.blockSignals(True)
            self.spn_range_len.setValue(length)
            self.spn_range_len.blockSignals(False)
        finally:
            self._updating_range_inputs = False

    def _on_range_start_changed(self, _value: int) -> None:
        self._update_range_len_from_range()
        self._update_time_plot()

    def _on_range_end_changed(self, _value: int) -> None:
        self._update_range_len_from_range()
        self._update_time_plot()

    def _on_range_len_changed(self, _value: int) -> None:
        if self._df is None:
            return
        if self._updating_range_inputs:
            return
        self._updating_range_inputs = True
        try:
            start = int(self.spn_range_start.value())
            length = int(self.spn_range_len.value())
            max_end = int(self.spn_range_end.maximum())
            end = min(max_end, start + max(1, length) - 1)
            self.spn_range_end.blockSignals(True)
            self.spn_range_end.setValue(end)
            self.spn_range_end.blockSignals(False)
            self._update_range_len_from_range()
        finally:
            self._updating_range_inputs = False
        self._update_time_plot()

    def _on_transpose_toggled(self, checked: bool) -> None:
        self._use_transposed = checked
        self._pending_time_autorange = True
        self._apply_current_dataframe()

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
        self._start_load(path=p, sheet_name=None, csv_has_header=self._get_csv_has_header_setting())

    def _on_sheet_changed(self) -> None:
        if not self._loaded:
            return
        if self._loaded.path.suffix.lower() != ".xlsx":
            return
        sheet = self.cbo_sheet.currentText()
        if not sheet:
            return
        self._start_load(path=self._loaded.path, sheet_name=sheet, csv_has_header=self._get_csv_has_header_setting())

    def _on_csv_header_changed(self) -> None:
        if not self._loaded:
            return
        if self._loaded.path.suffix.lower() != ".csv":
            return
        self._start_load(path=self._loaded.path, sheet_name=None, csv_has_header=self._get_csv_has_header_setting())

    def _get_csv_has_header_setting(self) -> bool | None:
        idx = int(self.cbo_csv_header.currentIndex())
        if idx == 1:
            return True
        if idx == 2:
            return False
        return None

    def _start_load(self, path: Path, sheet_name: str | None, csv_has_header: bool | None) -> None:
        self._cleanup_loader()

        self._set_loading(True, f"正在读取: {path.name}")

        thread = QtCore.QThread(self)
        worker = LoadWorker(path=path, sheet_name=sheet_name, csv_has_header=csv_has_header)
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
            self._raw_df = self._loaded.dataframe
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

            self._apply_current_dataframe()
        finally:
            self._cleanup_loader()
            self._set_loading(False)

    @QtCore.Slot(str)
    def _on_load_failed(self, message: str) -> None:
        self._cleanup_loader()
        self._loaded = None
        self._raw_df = None
        self._df = None
        self.btn_transpose.setEnabled(False)
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

    def _timedelta_like_series_to_seconds(self, s: pd.Series) -> np.ndarray:
        td = pd.to_timedelta(s, errors="coerce")
        td_np = td.to_numpy(dtype="timedelta64[ns]")
        td_ns = td_np.astype("int64").astype(float)
        td_ns[td.isna().to_numpy()] = np.nan
        if not np.isfinite(td_ns).any():
            raise ValueError("时间列无法解析为有效时间")
        return (td_ns - np.nanmin(td_ns)) / 1e9

    def _time_series_to_axis(self, s: pd.Series) -> tuple[np.ndarray, bool]:
        if pd.api.types.is_datetime64_any_dtype(s):
            return self._datetime_like_series_to_seconds(s), True

        x_num = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = x_num[np.isfinite(x_num)]
        if finite.size and float(finite.size) / float(x_num.size) >= 0.9:
            med = float(np.nanmedian(finite))
            base = float(np.nanmin(finite))
            if abs(med) >= 1e15:
                return (x_num - base) / 1e9, True
            if abs(med) >= 1e12:
                return (x_num - base) / 1e3, True
            if abs(med) >= 1e9:
                return x_num - base, True
            return x_num, False

        parsed_td = pd.to_timedelta(s, errors="coerce")
        if float(parsed_td.notna().mean()) >= 0.9:
            return self._timedelta_like_series_to_seconds(s), True

        parsed_dt = pd.to_datetime(s, errors="coerce")
        if float(parsed_dt.notna().mean()) >= 0.9:
            return self._datetime_like_series_to_seconds(s), True

        raise ValueError("时间列无法解析为数值/时间戳/时长")

    def _get_time_axis_and_signal(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._df is None:
            raise ValueError("请先导入数据")
        df = self._df
        if df.empty:
            raise ValueError("数据为空")

        col = self.cbo_column.currentText()
        if not col:
            raise ValueError("未选择列")
        y_series = pd.to_numeric(df[col], errors="coerce")
        y = y_series.to_numpy(dtype=float, na_value=np.nan)

        time_name = self.cbo_time.currentText().strip()
        if time_name and time_name != "无" and time_name in df.columns:
            t_raw = df[time_name]
            try:
                x, is_seconds = self._time_series_to_axis(t_raw)
            except Exception:
                x = np.arange(y.size, dtype=float)
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

    def _sync_range_controls(self) -> None:
        if self._df is None:
            n = 0
        else:
            n = int(self._df.shape[0])

        max_idx = max(0, n - 1)
        for spn in (self.spn_range_start, self.spn_range_end):
            spn.blockSignals(True)
            spn.setMaximum(max_idx)
            if spn.value() > max_idx:
                spn.setValue(max_idx)
            spn.blockSignals(False)

        if n > 0 and self.spn_range_end.value() < self.spn_range_start.value():
            self.spn_range_end.setValue(self.spn_range_start.value())
        self.spn_range_len.blockSignals(True)
        if n <= 0:
            self.spn_range_len.setEnabled(False)
            self.spn_range_len.setMaximum(1)
            self.spn_range_len.setValue(1)
        else:
            self.spn_range_len.setEnabled(True)
            self.spn_range_len.setMaximum(n)
            start = int(self.spn_range_start.value())
            end = int(self.spn_range_end.value())
            length = max(1, end - start + 1)
            self.spn_range_len.setValue(min(n, length))
        self.spn_range_len.blockSignals(False)

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
            self._update_range_len_from_range()
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
            x, y, is_seconds, used_time_col = self._get_time_plot_series(max_points=PLOT_MAX_POINTS)
            x_sel, y_sel, _, _ = self._get_time_plot_series(max_points=PLOT_MAX_POINTS, selected_only=True)
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

        if used_time_col:
            if is_seconds:
                self.time_plot.setLabel("bottom", "Time", units="s")
            else:
                self.time_plot.setLabel("bottom", "Time")
        else:
            self.time_plot.setLabel("bottom", "Sample")

        if self._pending_time_autorange:
            self._pending_time_autorange = False
            self.time_plot.enableAutoRange()

    def _get_time_plot_series(
        self,
        max_points: int,
        selected_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        if self._df is None:
            raise ValueError("请先导入数据")
        df = self._df
        if df.empty:
            raise ValueError("数据为空")

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

        col = self.cbo_column.currentText()
        if not col:
            raise ValueError("未选择列")
        y_raw = df[col].iloc[data_slice]
        n = int(y_raw.shape[0])
        step = max(1, int(np.ceil(n / max_points))) if max_points > 0 else 1
        y_raw = y_raw.iloc[::step]
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan)

        if selected_only:
            sample_x = np.arange(int(self.spn_range_start.value()), int(self.spn_range_end.value()) + 1, step, dtype=float)
        else:
            sample_x = np.arange(0, n, step, dtype=float)

        time_name = self.cbo_time.currentText().strip()
        if time_name and time_name != "无" and time_name in df.columns:
            t_raw = df[time_name].iloc[data_slice].iloc[::step]
            used_time_col = True
            try:
                x, is_seconds = self._time_series_to_axis(t_raw)
            except Exception:
                x = sample_x
                is_seconds = False
                used_time_col = False
                self.statusBar().showMessage("时间列解析失败，已回退到采样点作为横轴", 5000)
        else:
            x = sample_x
            is_seconds = False
            used_time_col = False

        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 2 and used_time_col:
            y_mask = np.isfinite(y)
            return sample_x[y_mask], y[y_mask], False, False
        return x[mask], y[mask], is_seconds, used_time_col

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
        if self._df is None:
            raise ValueError("请先导入数据")
        df = self._df
        if df.empty:
            raise ValueError("数据为空")

        self._sync_range_controls()
        start = int(self.spn_range_start.value())
        end = int(self.spn_range_end.value())
        if end < start:
            start, end = end, start

        col = self.cbo_column.currentText()
        if not col:
            raise ValueError("未选择列")
        start = max(0, min(start, int(df.shape[0] - 1)))
        end = max(0, min(end, int(df.shape[0] - 1)))
        y_raw = df[col].iloc[start : end + 1]
        y = pd.to_numeric(y_raw, errors="coerce").dropna().to_numpy(dtype=float)

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
