from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FftResult:
    frequency_hz: np.ndarray
    amplitude: np.ndarray
    nfft: int


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_fft_amplitude(
    x: np.ndarray,
    fs_hz: float,
    nfft: int | None = None,
    window_enabled: bool = False,
    window_name: str = "hann",
    window_param: float | None = None,
) -> FftResult:
    if fs_hz <= 0:
        raise ValueError("采样率 fs 必须 > 0")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("FFT 输入必须是一维序列")
    if x.size < 2:
        raise ValueError("数据点数量不足，无法进行 FFT")

    x = x - np.nanmean(x)
    if np.any(~np.isfinite(x)):
        raise ValueError("数据包含 NaN/Inf，无法计算 FFT")

    n = int(x.size)
    if nfft is None:
        nfft = next_pow2(n)
    nfft = int(nfft)
    if nfft < 2:
        raise ValueError("NFFT 必须 >= 2")

    if window_enabled:
        if window_name.lower() == "kaiser":
            if window_param is None:
                raise ValueError("Kaiser 窗需要 beta 参数")
            w = signal.get_window(("kaiser", float(window_param)), n, fftbins=True)
        else:
            w = signal.get_window(window_name, n, fftbins=True)
        x = x * w

    y = np.fft.rfft(x, n=nfft)
    amp = np.abs(y) / n
    if amp.size > 2:
        amp[1:-1] *= 2.0

    f = np.fft.rfftfreq(nfft, d=1.0 / fs_hz)
    return FftResult(frequency_hz=f, amplitude=amp, nfft=nfft)

