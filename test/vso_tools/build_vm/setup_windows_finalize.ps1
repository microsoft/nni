#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

# Generalize VM with sysprep
# https://docs.microsoft.com/en-us/azure/virtual-machines/windows/build-image-with-packer

# NOTE: the following *3* lines are only needed if the you have installed the Guest Agent.
while ((Get-Service RdAgent).Status -ne 'Running') { Start-Sleep -s 5 }
# Seems we don't have this.
# while ((Get-Service WindowsAzureTelemetryService).Status -ne 'Running') { Start-Sleep -s 5 }
while ((Get-Service WindowsAzureGuestAgent).Status -ne 'Running') { Start-Sleep -s 5 }

if ( Test-Path $Env:SystemRoot\windows\system32\Sysprep\unattend.xml ) {
    rm $Env:SystemRoot\windows\system32\Sysprep\unattend.xml -Force
}
& $env:SystemRoot\System32\Sysprep\Sysprep.exe /oobe /generalize /quiet /quit /mode:vm
while ($true) {
    $imageState = Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\State | Select ImageState;
    if ($imageState.ImageState -ne 'IMAGE_STATE_GENERALIZE_RESEAL_TO_OOBE') {
        Write-Output $imageState.ImageState; Start-Sleep -s 10
    } else {
        break
    }
}
