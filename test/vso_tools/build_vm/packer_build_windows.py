"""Start and monitor a packer building process, and change WinRM ports at an appropriate time.

It first catches the name of the VM. It's a randomly generated string by packer.
It then catches the signal of packer's waiting for WinRM ready.
Actually, it will be ready soon, but packer is waiting on the wrong port.
Invoking a command here to change the port of WinRM so that packer could conect to it.

The monitor is designed to be as robust as possible, so that packer won't easily crash.
It's painful to manually clean up the resources that packer has created,
as we are reusing an existing resource group.
"""

import re
import subprocess
import sys
import time

BUILD_COMMAND = 'PACKER_LOG=1 packer build packer_windows.json'
RESOURCE_GROUP = 'nni'


def monitor_print(*args):
    print('packer build monitor:', *args, flush=True)


def main():
    process = subprocess.Popen(BUILD_COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        retcode = process.poll()

        vm_name = None
        winrm_start_waiting = None
        winrm_finished = False
        for line in process.stdout:
            try:
                line = line.decode()
                sys.stdout.write(line)
                sys.stdout.flush()
            except UnicodeDecodeError:
                monitor_print('Decode error:', str(line))

            if winrm_finished:
                continue

            # Find VM name
            vm_name_grp = re.search(r'pkrvm[a-z0-9]{10,}', line)
            if vm_name is None and vm_name_grp is not None:
                vm_name = vm_name_grp.group(0)
                monitor_print('VM name found:', vm_name)

            # Waiting for WinRM
            if winrm_start_waiting is None and 'Waiting for WinRM' in line:
                if vm_name is None:
                    monitor_print('VM name not found. This is not normal.')
                else:
                    winrm_start_waiting = time.time()
                    monitor_print('Waiting for WinRM detected. You might see some errors. No worry.')

            # After WinRM has a waiting signal, wait another minute to make sure it's ready.
            if winrm_start_waiting is not None and time.time() - winrm_start_waiting > 60:
                monitor_print('WinRM waits time has exceeded 60 seconds. Starting to invoke command to change its port.')
                result = subprocess.run(
                    'az vm run-command invoke --command-id RunPowerShellScript '
                    f'--name {vm_name} -g {RESOURCE_GROUP} '
                    '--scripts @change_winrm_port.ps1',
                    shell=True
                )
                if result.returncode != 0:
                    monitor_print('Return code of command invoking is non-zero:', result.returncode)
                else:
                    monitor_print('Command invocation successfully triggered.')

                # To make the packer resource cleanup robust,
                # WinRM is always finished regardless of subprocess status.
                winrm_finished = True

        if retcode is not None:
            if retcode != 0:
                monitor_print('packer build fails with return code:', retcode)
            else:
                monitor_print('packer build succeeds')
            return retcode

        time.sleep(1)


if __name__ == '__main__':
    sys.exit(main())
