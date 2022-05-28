#!/bin/bash

CONFIG_PATH=/etc/ssh/sshd_config
# CONFIG_PATH=sshd_config
sudo sed -i -e "s/#Port 22/Port 10022/g" $CONFIG_PATH
sudo service ssh restart
