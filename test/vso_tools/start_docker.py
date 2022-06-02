"""
Build docker image, start container, then set its SSH service port to VSO variable "docker_port".

Usage:
    python start_docker.py <nni-version> <container-name> <password-in-docker>
"""

import argparse
import os
import random
import socket

from _common import build_wheel, run_command, set_variable

# find idle port
port = random.randint(10000, 20000)
while True:
    sock = socket.socket()
    if sock.connect_ex(('localhost', port)) != 0:
        break  # failed to connect, so this is idle
    sock.close()
    port = random.randint(10000, 20000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('version', type=str)
    parser.add_argument('container', type=str)
    parser.add_argument('password', type=str)
    parser.add_argument('--sudo', default=False, action='store_true')

    args = parser.parse_args()
    docker = 'sudo docker' if args.sudo else 'docker'
    version, container, password = args.version, args.container, args.password
    uid, gid = os.getuid(), os.getgid()

    run_command(f'{docker} build --build-arg NNI_RELEASE={version} -t nnidev/nni-nightly .')
    run_command(f'{docker} run --privileged -d -t -p {port}:22 --add-host=host.docker.internal:host-gateway --name {container} nnidev/nni-nightly')
    # The user inside docker must have the same uid and gid as outside.
    # Otherwise NFS will have permission errors.
    run_command(f'{docker} exec {container} groupadd -g {gid} nni')
    run_command(f'{docker} exec {container} useradd --create-home --password {password} -u {uid} -g {gid} nni')
    run_command(docker.split() + ['exec', container, 'bash', '-c', f'echo "nni:{password}" | chpasswd'])
    run_command(docker.split() + ['exec', container, 'bash', '-c', 'echo "nni ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers'])
    run_command(f'{docker} exec {container} service ssh start')
    set_variable('docker_port', port)

if __name__ == '__main__':
    main()
