"""
Build docker image, start container, then set its SSH service port to VSO variable "docker_port".

Usage:
    python start_docker.py <nni-version> <container-name> <password-in-docker>
"""

import random
import socket
import sys

from _common import build_wheel, run_command, set_variable

# find idle port
port = random.randint(10000, 20000)
while True:
    sock = socket.socket()
    if sock.connect_ex(('localhost', port)) != 0:
        break  # failed to connect, so this is idle
    sock.close()
    port = random.randint(10000, 20000)

version = sys.argv[1]
container = sys.argv[2]
password = sys.argv[3]

run_command(f'docker build --build-arg NNI_RELEASE={version} -t nnidev/nni-nightly .')
run_command(f'docker run -d -t -p {port}:22 --name {container} nnidev/nni-nightly')
run_command(f'docker exec {container} useradd --create-home --password {password} nni')
run_command(['docker', 'exec', container, 'bash', '-c', f'echo "nni:{password}" | chpasswd'])
run_command(f'docker exec {container} service ssh start')
set_variable('docker_port', port)
