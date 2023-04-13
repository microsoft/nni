# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import atexit
import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Sequence, Callable

from .file_channel import FileChannel
from .typehint import (
    CreateCommand, KillCommand, MetricCommand, TrialStatusCommand, WakeUpCommand, ReportAwakeCommand,
    Trial, Status, typed_dict_validation
)
from .utils import graceful_kill, add_handler

_logger = logging.getLogger('nni_amlt.trial_runner')


class TrialServerHandler(BaseHTTPRequestHandler):
    """A server for trial to get parameters and report metrics to the trial runner."""

    PORT = 36378
    ADDRESS = 'http://localhost:36378'

    def __init__(self, trials: Sequence[Trial], on_metric: Callable[[MetricCommand], None], *args, **kwargs):
        self.trials = trials
        self.on_metric = on_metric
        # Must be before super.init. The handler will start to handle requests within super.init.
        super().__init__(*args, **kwargs)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _send_bad_request(self):
        self.send_response(400)
        self.end_headers()

    def _send_not_found(self):
        self.send_response(404)
        self.end_headers()

    def _send_ok(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        """GET request must be requesting parameters."""
        if not self.path.startswith('/parameter/'):
            _logger.error('Invalid path for HTTP GET: %s', self.path)
            self._send_bad_request()
            return

        trial_id = self.path.split('/')[-1]

        for trial in self.trials:
            if trial['id'] == trial_id:
                self._set_headers()
                self.wfile.write(json.dumps(trial).encode())
                return

        _logger.error('Trial ID %s not found in parameters', trial_id)
        self._send_not_found()
        return

    def do_POST(self):
        """POST request must be sending results."""
        if self.path != '/metric':
            _logger.error('Invalid path for HTTP POST: %s', self.path)
            self._send_bad_request()
            return

        content_type = self.headers.get_content_type()
        # refuse to receive non-json content
        if content_type != 'application/json':
            self._send_bad_request()
            return

        content_length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(content_length))

        if not typed_dict_validation(MetricCommand, message):
            _logger.error('Invalid message: %s', message)
            self._send_bad_request()
            return

        self.on_metric(message)
        self._send_ok()


class TrialRunner:
    """
    Runner to process incoming trial commands.

    Parameters
    ----------
    channel
        Channel to communicate with the management end.
        The runner only uses the "send" direction of the channel.
    runner_dir
        Directory for runner to save logs, save/restore checkpoints.
        Usually **unshared** between multiple ranks (nodes).
    trial_output_dir
        Directory for trials to save their output files.
        Subdirectory with trial IDs will be created inside.
        Usually **shared** between multiple ranks.
    trial_log_dir
        Path to where trial log is stored.
        Usually **unshared** between ranks.
    log_buffer_size
        Buffer size of trial stdout.
    """

    def __init__(self, channel: FileChannel, runner_dir: Path,
                 trial_output_dir: Path, trial_log_dir: Path,
                 log_buffer_size: int) -> None:
        self._channel = channel
        self._runner_dir = runner_dir
        self._trial_output_dir = trial_output_dir
        self._trial_log_dir = trial_log_dir
        self._log_buffer_size = log_buffer_size

        self._processing_trials: deque[Trial] = deque()  # including the current running one.
        self._running_process: subprocess.Popen | None = None

        if self._checkpoint_path.exists():
            self.load_checkpoint()

        self._server = self._server_start()
        atexit.register(self._server_stop)

    @property
    def _checkpoint_path(self) -> Path:
        return self._runner_dir / 'trial_runner.json'

    def _server_start(self) -> HTTPServer:
        _logger.info('Starting trial server at %s.', TrialServerHandler.ADDRESS)
        atexit.register(self._server_stop)
        server_address = ('', TrialServerHandler.PORT)
        httpd = HTTPServer(server_address, partial(TrialServerHandler, self._processing_trials, self._on_metric))

        def _start() -> None:
            httpd.serve_forever()

        threading.Thread(target=_start, daemon=True).start()
        return httpd

    def _server_stop(self) -> None:
        _logger.info('Stopping trial server.')
        atexit.unregister(self._server_stop)
        self._server.shutdown()

    def _on_metric(self, command: MetricCommand) -> None:
        self._channel.send(json.dumps(command))

    def load_checkpoint(self) -> None:
        try:
            with self._checkpoint_path.open() as f:
                checkpoint_data = json.load(f)
            self._processing_trials = deque()
            for t in checkpoint_data['queued_trials']:
                if typed_dict_validation(Trial, t):
                    self._processing_trials.append(t)
                else:
                    _logger.error('Ignored when loading checkpoint as it appears not a valid trial: %s', t)
            _logger.info('Checkpoint loaded. Processing trials: %s', self._processing_trials)
        except:
            _logger.exception('Checkpoint loading failed: %s', self._checkpoint_path)

        self._refresh_queue()

    def save_checkpoint(self) -> None:
        try:
            checkpoint_data = {
                'queued_trials': list(self._processing_trials),
            }
            self._checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            with self._checkpoint_path.open('w') as f:
                json.dump(checkpoint_data, f)
        except:
            _logger.exception('Checkpoint saving failed: %s', self._checkpoint_path)

    def check_status(self) -> list[Trial]:
        """
        Check the status of the runner and return processing trials (including running + pending).

        The caller should be responsible for :meth:`check_status` regularly.
        Otherwise the trials in the queue won't be auto-processed.
        """
        # Check the status of current running trial.
        self._refresh_queue()
        # List running and pending trials.
        return list(self._processing_trials)

    def create_trial(self, trial: Trial) -> None:
        """Submit a trial for running.

        Returns instantly.
        """
        self._processing_trials.append(trial)
        self._refresh_queue()

    def kill_trial(self, id: str) -> None:  # pylint: disable=redefined-builtin
        """Kill a trial.

        Currently must be the running trial.
        """
        if len(self._processing_trials) > 0 and self._running_process is not None:
            trial = self._processing_trials[0]
            if trial['id'] == id:
                graceful_kill(self._running_process)
                returncode = self._running_process.returncode
                _logger.info('Process %s is killed with exit code: %s', self._running_process, returncode)
                self._processing_trials.popleft()
                self._emit_status_change(trial['id'], 'interrupted')
                self._running_process = None

                # Run the next trial if any.
                self._refresh_queue()
                return

        _logger.warning('Trial %s is not running. Cannot kill it.', id)

    def send_heartbeat(self) -> float:
        """Send a heartbeat to the other side."""
        current_time = time.time()
        command = ReportAwakeCommand(
            command_type='awake',
            time=current_time,
            idle=not self._processing_trials
        )
        self._channel.send(json.dumps(command))
        return current_time

    def _refresh_queue(self) -> None:
        if not self._processing_trials:
            _logger.debug('No trials. Nothing to refresh.')
            return

        # Move the queue. See if the upfront trial is completed,
        # and whether the next trial should be run.
        if self._running_process is not None:
            if self._running_process.poll() is not None:
                returncode = self._running_process.returncode
                _logger.info('Process %s return with exit code: %s', self._running_process, returncode)
                if returncode == 0:
                    status: Status = 'succeeded'
                else:
                    status: Status = 'failed'
                trial = self._processing_trials.popleft()
                _logger.info('Trial %s ended with status: %s', trial['id'], status)
                self._emit_status_change(trial['id'], status)
                self._running_process = None

        # Run a new trial.
        if len(self._processing_trials) > 0 and self._running_process is None:
            trial = self._processing_trials[0]
            _logger.info('Running: %s', trial['command'])
            self._running_process = subprocess.Popen(
                trial['command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=self._log_buffer_size,
                shell=True,
                env=self._environ(trial)
            )
            self._start_stdout_logging(self._trial_log_dir / (trial['id'] + '.txt'))
            self._emit_status_change(trial['id'], 'running')

    def _environ(self, trial: Trial) -> dict:
        """Generate environment variables for a trial."""
        environ_base = dict(os.environ)

        output_dir = str(self._trial_output_dir / trial['id'])
        nni_environ = dict(
            NNI_PLATFORM='amlt',
            NNI_EXP_ID=trial['experiment'],
            NNI_TRIAL_JOB_ID=trial['id'],
            NNI_SYS_DIR=str(output_dir),
            NNI_OUTPUT_DIR=str(output_dir),
            NNI_TRIAL_SEQ_ID=str(trial['sequence']),
            NNI_TRIAL_COMMAND_CHANNEL='import://nni_amlt.trial_client.TrialClient'
        )

        return {
            **environ_base,
            **nni_environ
        }

    def _start_stdout_logging(self, file: Path) -> None:
        if self._running_process is None:
            _logger.error('No running process to start logging.')
            return

        def _tee(infile, log_file: Path) -> None:
            log_file.parent.mkdir(exist_ok=True, parents=True)
            with infile, open(log_file, 'ab') as f:
                for line in iter(infile.readline, b''):
                    f.write(line)
                    sys.stdout.buffer.write(line)
                    # Did not flush here.

        file.parent.mkdir(exist_ok=True, parents=True)
        t = threading.Thread(target=_tee, args=(self._running_process.stdout, file), daemon=True)
        t.start()

    def _emit_status_change(self, trial_id: str, status: Status) -> None:
        command = TrialStatusCommand(
            command_type='status',
            id=trial_id,
            status=status,
        )
        _logger.debug('Emit status change: %s', command)
        self._channel.send(json.dumps(command))


def trial_runner_loop(
    channel: str | Path,
    out: str | Path,
    rank: int,
    interval: float,
    patience: float,
    log_buffer_size: int
) -> None:
    output_dir = Path(out)
    runner_dir = output_dir / f'trial_runner_{rank}'
    trial_log_dir = output_dir / f'logs_{rank}'
    runner_dir.mkdir(exist_ok=True, parents=True)

    # Init logger if not inited.
    add_handler(_logger, runner_dir / f'trial_runner.log')

    _logger.info('Trial runner started.')

    _logger.info('Saving trial runner states to: %s', runner_dir)

    file_channel = FileChannel(channel, f'worker-{rank}', 'manager')
    _logger.info('Using channel %s to communicate with NNI manager.', file_channel)

    log_buffer_size = log_buffer_size
    _logger.info('Buffer size for trial stdout: %d', log_buffer_size)

    trial_runner = TrialRunner(file_channel, runner_dir, output_dir, trial_log_dir, log_buffer_size)

    last_good = time.time()
    last_heartbeat = time.time()
    heartbeat_interval = interval

    trial_runner.send_heartbeat()

    while True:
        if trial_runner.check_status():
            _logger.info('Trial runner has running trials. Be patient.')
            last_good = time.time()

        trial_runner.save_checkpoint()

        # Receive a control command from manager side.
        command = file_channel.receive(non_blocking=True)

        if command is not None:
            try:
                command = json.loads(command)
            except:
                _logger.exception('Command decode error. Skip this command: %s', command)
                command = None

        if command is not None:
            if not isinstance(command, dict) or 'command_type' not in command:
                _logger.error('Invalid command: %s', command)
            else:
                command_type = command['command_type']
                if command_type == 'create' and typed_dict_validation(CreateCommand, command):
                    trial_runner.create_trial(command['trial'])
                elif command_type == 'kill' and typed_dict_validation(KillCommand, command):
                    trial_runner.kill_trial(command['id'])
                elif command_type == 'wakeup' and typed_dict_validation(WakeUpCommand, command):
                    last_heartbeat = trial_runner.send_heartbeat()
                else:
                    _logger.error('Unsupported command: %s', command)

                trial_runner.save_checkpoint()

                # Reset heartbeat interval to communicate more frequently
                heartbeat_interval = interval

            # No sleep. Continue to next command.

        else:
            elapsed = time.time() - last_good
            _logger.info('No command received. Since last receiving: %f seconds (%f maximum).', elapsed, patience)

            if elapsed > patience:
                _logger.warning('No command received for too long. Quit the runner.')
                break

            if time.time() - last_heartbeat > heartbeat_interval:
                last_heartbeat = trial_runner.send_heartbeat()
                # Exponentially increase heartbeat interval
                heartbeat_interval = heartbeat_interval * 1.5

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Amulet training service trial runner')
    parser.add_argument('channel', type=str, help='The path where file channel is mounted (in cluster container)')
    parser.add_argument('out', type=str, default=None,
                        help='Checkpoint directory of the trial runner. If specified, trial runner will try to find its checkpoint.')
    parser.add_argument('--rank', type=int, default=None,
                        help='Rank of trial runner. Meaningful for distributed training. '
                             'If not set, will try to read from environment variable `RANK`.')
    parser.add_argument('--interval', type=float, default=60.,
                        help='Interval (seconds) between two polls of the channel')
    parser.add_argument('--heartbeat-max-interval', type=float, default=600.,
                        help='Max interval (seconds) between two heartbeats. '
                             'Heartbeat is used to tell the manager that the runner is still alive. '
                             'The initial heartbeat interval is `interval`. '
                             'It will be exponentially increased until it reaches this value if no message from manager is received.')
    parser.add_argument('--patience', type=float, default=1800.,
                        help='Number of seconds without any updates or running trials before the runner shutdowns')
    parser.add_argument('--log-buffer-size', type=int, default=0,
                        help='Buffer size for trial stdout. See bufsize in `subprocess.Popen`.')

    args = parser.parse_args()
    if args.rank is None:
        args.rank = int(os.environ.get('RANK', 0))
    trial_runner_loop(**vars(args))


if __name__ == '__main__':
    main()
