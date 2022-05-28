Set-Item WSMan:\localhost\listener\*\Port 15986 -Force
New-NetFirewallRule -Name 'Custom-WinRM' -DisplayName 'Custom WinRM Port Rule' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -Program System -LocalPort 15986
Restart-Service -Name WinRM
