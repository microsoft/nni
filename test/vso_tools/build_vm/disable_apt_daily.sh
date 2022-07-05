#!/bin/bash

set -x
set -e

# References: https://gist.github.com/posilva/1cefb5bf1eeccf9382920e5d57a4b3fe

apt-get -y purge update-notifier-common ubuntu-release-upgrader-core landscape-common unattended-upgrades

systemctl kill --kill-who=all apt-daily.service
systemctl kill --kill-who=all apt-daily-upgrade.service

systemctl stop apt-daily.timer
systemctl disable apt-daily.timer
systemctl stop apt-daily.service
systemctl disable apt-daily.service

systemctl stop apt-daily-upgrade.timer
systemctl disable apt-daily-upgrade.timer
systemctl stop apt-daily-upgrade.service
systemctl disable apt-daily-upgrade.service
systemctl daemon-reload
systemctl reset-failed

rm /etc/systemd/system/timers.target.wants/apt-daily.timer
rm /etc/systemd/system/timers.target.wants/apt-daily-upgrade.timer

mv /usr/lib/apt/apt.systemd.daily /usr/lib/apt/apt.systemd.daily.DISABLED
mv /lib/systemd/system/apt-daily.service /lib/systemd/system/apt-daily.service.DISABLED
mv /lib/systemd/system/apt-daily.timer /lib/systemd/system/apt-daily.timer.DISABLED
mv /lib/systemd/system/apt-daily-upgrade.service /lib/systemd/system/apt-daily-upgrade.service.DISABLED
mv /lib/systemd/system/apt-daily-upgrade.timer /lib/systemd/system/apt-daily-upgrade.timer.DISABLED
