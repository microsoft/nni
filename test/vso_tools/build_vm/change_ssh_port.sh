#!/bin/bash

CONFIG_PATH=/etc/ssh/sshd_config
# CONFIG_PATH=sshd_config
sudo sed -i -e "s/#Port 22/Port 10022/g" $CONFIG_PATH
sudo service ssh restart

# {
#   "type": "shell-local",
#   "inline": [
#     "az network nsg rule create -g <resource_group> --nsg-name <network_security_group> -n {{ build `PackerRunUUID` }} \\",
#     "  --priority 100 --source-address-prefixes $(curl ifconfig.me) --source-port-ranges '*' \\",
#     "  --destination-address-prefixes '*' --destination-port-ranges '*' --access Allow --protocol '*' \\",
#     "  --description 'Disposable rule for packer build {{ build `PackerRunUUID` }}'",
#     "export NIC=$(az network nic list -g <resource_group> --query [].'name' --output tsv | grep pkrni | head -1)",
#     "echo \"NIC Found: ${NIC}\"",
#     "az network nic update -n ${NIC} -g <resource_group> --network-security-group <network_security_group>"
#   ]
# },