# See change_ssh_port.sh
# Note that this script does NOT run at provision, because Azure doesn't support provision-time powershell script.
# This script is invoked with azcli (az vm run-command). See packer_build_windows.py.

winrm set winrm/config/Listener?Address=*+Transport=HTTPS '@{Port="15986"}'
New-NetFirewallRule -Name 'Custom-WinRM' -DisplayName 'Custom WinRM Port Rule' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -Program System -LocalPort 15986
Restart-Service -Name WinRM
