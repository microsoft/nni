from io import BufferedIOBase
import logging
import os
import sys

_logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    import _winapi
    import msvcrt

    class WindowsPipe:
        def __init__(self, experiment_id: str):
            self.path: str = r'\\.\pipe\nni-' + experiment_id
            self.file = None

            self._handle = _winapi.CreateNamedPipe(
                self.path,
                _winapi.PIPE_ACCESS_DUPLEX,
                _winapi.PIPE_TYPE_MESSAGE | _winapi.PIPE_READMODE_MESSAGE | _winapi.PIPE_WAIT,
                1,
                8192,
                8192,
                0,
                _winapi.NULL
            )

        def connect(self) -> BufferedIOBase:
            _winapi.ConnectNamedPipe(self._handle, _winapi.NULL)
            fd = msvcrt.open_osfhandle(self._handle, 0)
            self.file = os.fdopen(fd, 'w+b')
            return self.file

        def close(self) -> None:
            try:
                if self.file is not None:
                    self.file.close()
            except Exception as e:
                _logger.debug('Error on closing Windows pipe: %s', e)

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
            try:
                if self.file is not None:
                    self.file.close()
                self._socket.close()
                os.unlink(self.path)
            except Exception as e:
                _logger.debug('Error on closing POSIX pipe: %s', e)

    Pipe = UnixPipe
