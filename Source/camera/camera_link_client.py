"""Manage the hidden .NET Camera Link bridge and receive its latest frame."""

from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
from pathlib import Path
import struct
import sys
import uuid

import numpy as np
from PySide6.QtCore import QObject, QProcess, QTimer, Signal


HEADER = struct.Struct("<IIqqiiiiii")
HEADER_SIZE = HEADER.size
MAXIMUM_FRAME_BYTES = 4 * 1024 * 1024
MAP_SIZE = HEADER_SIZE + MAXIMUM_FRAME_BYTES
MAGIC = 0x53535054


@dataclass(slots=True, frozen=True)
class CameraConnectionSettings:
    cam_file: Path
    connector: str = "B"
    pixel_format: str = "Mono16"
    inspection_interval_ms: int = 500

    def validate(self) -> None:
        if not self.cam_file.is_file():
            raise ValueError(f"CAM 파일을 찾을 수 없습니다: {self.cam_file}")
        if self.connector.upper() not in {"A", "B"}:
            raise ValueError("보드 커넥터는 A 또는 B여야 합니다.")
        if self.pixel_format not in {"Mono8", "Mono16"}:
            raise ValueError("픽셀 형식은 Mono8 또는 Mono16이어야 합니다.")
        if self.inspection_interval_ms not in {500, 1000}:
            raise ValueError("검사 주기는 500ms 또는 1000ms여야 합니다.")


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def default_cam_file() -> Path:
    root = application_root()
    packaged = root / "cam File" / "LK_FLIR 640.cam"
    if packaged.is_file():
        return packaged
    return root / "reference_dll" / "SDK_Package" / "cam File" / "LK_FLIR 640.cam"


def bridge_executable() -> Path:
    root = application_root()
    packaged = root / "CameraBridge" / "CameraBridge.exe"
    if packaged.is_file():
        return packaged
    return root / "CameraBridge" / "bin" / "Release" / "net8.0-windows" / "CameraBridge.exe"


class CameraLinkClient(QObject):
    connected = Signal(str)
    connectionFailed = Signal(str)
    disconnected = Signal()
    frameReady = Signal(object, int, float, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_status)
        self._process.readyReadStandardError.connect(self._read_error)
        self._process.finished.connect(self._process_finished)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(16)
        self._poll_timer.timeout.connect(self._poll_frame)
        self._mapping: mmap.mmap | None = None
        self._map_name = ""
        self._last_frame_number = -1
        self._settings: CameraConnectionSettings | None = None
        self._disconnect_requested = False
        self._frame_format_error_reported = False
        self._process.errorOccurred.connect(self._process_error)

    @property
    def is_connected(self) -> bool:
        return self._mapping is not None and self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def settings(self) -> CameraConnectionSettings | None:
        return self._settings

    def connect_camera(self, settings: CameraConnectionSettings) -> None:
        settings.validate()
        if self._process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("카메라 연결 프로세스가 이미 실행 중입니다.")
        executable = bridge_executable()
        if not executable.is_file():
            raise FileNotFoundError(f"CameraBridge 실행파일을 찾을 수 없습니다: {executable}")
        self._settings = settings
        self._disconnect_requested = False
        self._last_frame_number = -1
        self._frame_format_error_reported = False
        self._map_name = f"Local\\SSPTool_{uuid.uuid4().hex}"
        self._mapping = mmap.mmap(-1, MAP_SIZE, tagname=self._map_name)
        arguments = [
            "--map-name", self._map_name,
            "--cam-file", str(settings.cam_file),
            "--connector", settings.connector.upper(),
            "--pixel-format", settings.pixel_format,
        ]
        self._process.setProgram(str(executable))
        self._process.setArguments(arguments)
        self._process.setWorkingDirectory(str(executable.parent))
        self._process.start()

    def disconnect_camera(self) -> None:
        self._disconnect_requested = True
        self._poll_timer.stop()
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.write(b"quit\n")
            self._process.closeWriteChannel()
            if not self._process.waitForFinished(2000):
                self._process.kill()
                self._process.waitForFinished(1000)
        self._close_mapping()

    def _read_status(self) -> None:
        while self._process.canReadLine():
            raw = bytes(self._process.readLine()).decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                status = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if status.get("status") == "connected":
                self._poll_timer.start()
                self.connected.emit(status.get("message", ""))
            elif status.get("status") in {"error", "frame_error"}:
                self.connectionFailed.emit(status.get("message", "알 수 없는 카메라 오류"))

    def _read_error(self) -> None:
        message = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if message:
            self.connectionFailed.emit(message)

    def _process_finished(self, _exit_code: int, _exit_status) -> None:
        self._poll_timer.stop()
        self._close_mapping()
        if not self._disconnect_requested:
            self.connectionFailed.emit("카메라 연결 프로세스가 종료되었습니다.")
        self.disconnected.emit()

    def _process_error(self, error) -> None:
        if not self._disconnect_requested:
            self.connectionFailed.emit(
                f"CameraBridge 실행 오류: {self._process.errorString()} ({error})"
            )

    def _close_mapping(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def _poll_frame(self) -> None:
        mapping = self._mapping
        if mapping is None:
            return
        mapping.seek(0)
        first = HEADER.unpack(mapping.read(HEADER_SIZE))
        magic, version, sequence, frame_number, width, height, pixel_bytes, length, fps_milli, state = first
        if magic != MAGIC or version != 1 or sequence % 2 or frame_number == self._last_frame_number:
            return
        if width <= 0 or height <= 0 or pixel_bytes not in {1, 2}:
            return
        expected_pixel_bytes = (
            2 if self._settings is None or self._settings.pixel_format == "Mono16" else 1
        )
        if width != 640 or height != 512 or pixel_bytes != expected_pixel_bytes:
            if not self._frame_format_error_reported:
                self._frame_format_error_reported = True
                actual = f"{width}×{height} Mono{pixel_bytes * 8}"
                expected = f"640×512 {self._settings.pixel_format if self._settings else 'Mono16'}"
                self.connectionFailed.emit(
                    f"카메라 프레임 형식이 설정과 다릅니다. 설정: {expected}, 수신: {actual}"
                )
            return
        if length != width * height * pixel_bytes or length > MAXIMUM_FRAME_BYTES:
            return
        data = mapping.read(length)
        mapping.seek(8)
        final_sequence = struct.unpack("<q", mapping.read(8))[0]
        if final_sequence != sequence or final_sequence % 2:
            return
        dtype = np.uint16 if pixel_bytes == 2 else np.uint8
        image = np.frombuffer(data, dtype=dtype).reshape(height, width).copy()
        self._last_frame_number = frame_number
        channel_state = {0: "IDLE", 1: "READY", 2: "ACTIVE"}.get(state, "UNKNOWN")
        self.frameReady.emit(image, frame_number, fps_milli / 1000.0, channel_state)

    def close(self) -> None:
        self.disconnect_camera()
