winrm set winrm/config/Listener?Address=*+Transport=HTTPS '@{Port="15986"}'
New-NetFirewallRule -Name 'Custom-WinRM' -DisplayName 'Custom WinRM Port Rule' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -Program System -LocalPort 15986
Restart-Service -Name WinRM
