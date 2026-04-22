# FFT Tool GUI

一个用于从 `.csv` / `.xlsx` 导入数据、选择信号并计算 FFT 幅值谱的桌面 GUI 工具。

## 功能

- 导入数据：支持 CSV / Excel（可选择 Sheet）
- 表格预览：在界面中查看数据
- 信号选择：按列或按行选择信号，支持选择时间列与范围截取
- FFT 计算：支持自动 NFFT（下一次幂）、可选加窗（hann/hamming/blackman/flattop/kaiser）
- 绘图展示：时域与频域图，支持点选频点读取数值

## 环境准备（Conda 优先）

本项目提供 conda 环境文件 [environment.yml](environment.yml)。

创建新环境：

```powershell
conda env create -f environment.yml
conda activate fft_tool_gui
```

更新已有环境：

```powershell
conda env update -n <你的env名> -f environment.yml
conda activate <你的env名>
```

## 运行

在项目根目录执行：

```powershell
python -m fft_tool_gui
```

或在代码层面调用入口函数：

```powershell
python -c "from fft_tool_gui.app import run; run()"
```

## 数据格式建议

- CSV：第一行作为表头更易用（对应列名选择）
- Excel：支持多 Sheet，界面中选择要导入的 Sheet
- 时间列：可为数值（秒/采样点等）或时间戳（程序会尝试识别并换算为秒）

## 工程结构

- `fft_tool_gui/app.py`：主界面与交互逻辑
- `fft_tool_gui/data_io.py`：CSV/Excel 加载
- `fft_tool_gui/fft_compute.py`：FFT 计算逻辑
- `fft_tool_gui/qt_models.py`：DataFrame 的 Qt 表格模型
