#!/bin/bash

# Microsoft internal subscription has a firewall (network security group),
# to deny traffic from low ports (maybe less than 10000).
# We have to change the port at VM provision, so that the VM can be connected and build scripts can run.

CONFIG_PATH=/etc/ssh/sshd_config
sudo sed -i -e "s/#Port 22/Port 10022/g" $CONFIG_PATH
sudo service ssh restart
