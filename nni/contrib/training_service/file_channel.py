# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time
from pathlib import Path

Command = str


class _PrefixAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if isinstance(self.extra, dict) and 'prefix' in self.extra:
            return f"{self.extra['prefix']} {msg}", kwargs
        return msg, kwargs


class FileChannel:
    """
    Command channel based on access to the same directory.

    The directory can be mounted at a different place for the other side,
    as long as it's available to both of them.

    Both side must have read and write access to the directory as well as the files inside.
    If the directory doesn't exist. They must have the privilege to create it.

    :class:`FileChannel` is stateful. It (at least) has a state to mark which messages (i.e. files)
    that has been read. Recovering the channel from faults might lose that state and consume duplicated messages.
    Thus the reader side needs the state of "current reading progress" to be persistent.
    Therefore a sender can broadcast to multiple receivers via a :class:`FileChannel`,
    but it can only be listening to one channel in current implementation.

    All the files written by the peer are in the URL, starting with ``<peer_name>.``.
    Synchronization can leverage this glob pattern.

    Parameters
    ----------
    url
        A directory on the current file system.
    local_peer
        Join as which peer. IDs are used to identify self, with no other limitations.
        Possible values might be "server", "client", "1", "2", ...
        Peer name can't contain ``.``.
    remote_peer
        The peer name that is connected to.
        This only matters in :meth:`receive`.

    Warnings
    --------
    The channel is not thread-safe. Behavior is undefined when two threads / processes belonging to the same peer
    try to access the channel at the same time. Try to use the channel ONLY on rank 0 whenever possible.
    """

    def __init__(self, url: str | Path, local_peer: str, remote_peer: str):
        self._url: Path = Path(url)
        self._local_peer = local_peer
        self._remote_peer = remote_peer

        assert '.' not in self._local_peer
        assert '.' not in self._remote_peer

        self._logger = _PrefixAdapter(
            logging.getLogger(__name__),
            {'prefix': f'(file channel {local_peer} -> {remote_peer})'}
        )

        self._file_capacity: int = 100000000  # 1e8
        self._line_limit_per_file: int = 100

        # For write. Next to write is 1.
        self._write_progress: int = 1
        self._recover_write_state()

        # For read. Has already read 0.
        self._read_progress: int = 0
        self._recover_read_state()

    def __repr__(self):
        return f'{self.__class__.__name__}({self._url}, {self._local_peer}, {self._remote_peer})'

    def send(self, command: Command) -> None:
        """Send a command.

        Returns immediately without checking whether the command is received successfully.

        If the send (itself) is unsuccessful (e.g., due to the command is invalid),
        the error is logged and ignored.
        """
        if not isinstance(command, str):
            self._logger.error('Sent command must be str, found %s, ignore: %s', type(command), command)
            return

        self._url.mkdir(exist_ok=True, parents=True)

        # Find a room for this message
        if self._write_progress % self._file_capacity >= self._line_limit_per_file:
            self._logger.debug('File full. Need a new file: %d', self._write_progress)
            # 2300100 -> 2400001
            self._write_progress = (self._write_progress // self._file_capacity + 1) * self._file_capacity + 1

        filename = self._format_filename(self._local_peer, self._write_progress // self._file_capacity)

        try:
            with filename.open('a') as f:
                f.write('%016d\t' % self._write_progress + command + '\n')
                f.flush()
            self._logger.debug('Sent command: %s', command)
            self._write_progress += 1
        except:
            self._logger.exception('Write to file failed: %s', filename)

    def receive(self, non_blocking: bool = False) -> Command | None:
        """Receive a command.

        Parameters
        ----------
        non_blocking
            If ``True``, return immediately if no command is received.
            Otherwise, block until a command comes.
        """
        while True:
            # Find a new message from two places.

            # 1. Check whether there is a message from the file corresponding to current progress.
            current_filename = self._format_filename(self._remote_peer, self._read_progress // self._file_capacity)
            content = self._receive_from_file(current_filename)
            if content is not None:
                return content

            # 2. Check whether there is a message from the next file.
            next_filename = self._format_filename(self._remote_peer, self._read_progress // self._file_capacity + 1)
            content = self._receive_from_file(next_filename)
            if content is not None:
                return content

            if non_blocking:
                return None

            self._logger.debug('Nothing received. Try again later.')
            time.sleep(1.)

    def _format_filename(self, peer_name: str, file_index: int) -> Path:
        assert peer_name in [self._local_peer, self._remote_peer]
        return self._url / f'{peer_name}.{file_index:08d}'

    def _recover_write_state(self) -> None:
        while True:
            path = self._format_filename(self._local_peer, self._write_progress // self._file_capacity)
            if path.exists():
                # Regardless of whether it's full or not.
                self._write_progress += self._file_capacity
            else:
                break
        if self._write_progress > 1:
            self._logger.info('Write progress is recovered to be: %d', self._write_progress)

    def _recover_read_state(self) -> None:
        path = self._url / f'{self._local_peer}.read'
        if not path.exists():
            self._logger.debug('Reading state does not exist. Nothing to recover.')
        else:
            try:
                with path.open() as f:
                    self._read_progress = int(f.read())
                self._logger.info('Read progress is recovered to be: %d', self._read_progress)
            except:
                self._logger.exception('Reading state appears to be corrupted: %s', path)

    def _save_read_state(self) -> None:
        try:
            self._url.mkdir(exist_ok=True, parents=True)
            with (self._url / f'{self._local_peer}.read').open('w') as f:
                f.write(str(self._read_progress))
            self._logger.debug('Read progress successfully updated: %d', self._read_progress)
        except:
            self._logger.exception('Reading state fails to dump: %d', self._read_progress)

    def _receive_from_file(self, file: Path) -> str | None:
        if not file.exists():
            self._logger.debug('%s does not exist yet.', file)
            return None

        try:
            with file.open() as f:
                for line in f.readlines():
                    id, content = line.split('\t', 1)  # pylint: disable=redefined-builtin
                    if int(id) > self._read_progress:
                        content = content.rstrip('\n')
                        self._logger.debug('Received command: %s', content)
                        self._read_progress = int(id)
                        self._save_read_state()
                        return content
        except:
            self._logger.exception('File appears to be corrupted: %s', file)
            return None
