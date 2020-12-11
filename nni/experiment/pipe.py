from io import BufferedIOBase
import os
import sys

if sys.platform == 'win32':
    import _win32
    import msvcrt

    class WindowsPipe:
        def __init__(self, experiment_id: str):
            self.path: str = r'\\.\pipe\nni-' + experiment_id
            self.file = None

            self._handle = _win32.CreateNamedPipe(
                self.path,
                _win32.PIPE_ACCESS_DUPLEX,
                _win32.PIPE_TYPE_MESSAGE | _win32.PIPE_READMODE_MESSAGE | _win32.PIPE_WAIT,
                1,
                8192,
                8192,
                0,
                _win32.NULL
            )

        def connect(self) -> BufferedIOBase:
            _win32.ConnectNamedPipe(self._handle, _win32.NULL)
            fd = msvcrt.open_osfhandle(self._handle)
            self.file = os.fdopen(fd, 'rwb')
            return self.file

        def close(self) -> None:
            if self.file is not None:
                self.file.close()
            _win32.CloseHandle(self._handle)

    Pipe = WindowsPipe


else:
    import socket

    from . import management

    class UnixPipe:
        def __init__(self, experiment_id: str):
            self.path: str = str(management.create_experiment_directory(experiment_id) / 'dispatcher-pipe')
            self.file = None

            self._socket = socket.socket(socket.AF_UNIX)
            self._socket.bind(self.path)
            self._socket.listen(1)  # only accepts one connection

        def connect(self) -> BufferedIOBase:
            conn, _ = self._socket.accept()
            self.file = conn.makefile('rwb')
            return self.file

        def close(self) -> None:
            if self.file is not None:
                self.file.close()
            self._socket.close()
            os.unlink(self.path)

    Pipe = UnixPipe
