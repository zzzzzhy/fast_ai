#!/bin/bash
echo 'deb [arch=arm64] http://ports.ubuntu.com/ xenial main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=arm64] http://ports.ubuntu.com/ xenial-security main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=arm64] http://ports.ubuntu.com/ xenial-backports main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/armhf.list
sed -i 's#deb  http://gb.archive.ubuntu.com/ubuntu#deb  [arch=amd64]  http://gb.archive.ubuntu.com/ubuntu#g ' /etc/apt/sources.list
sed -i 's#deb   http://security.ubuntu.com/ubuntu#deb   [arch=amd64]   http://security.ubuntu.com/ubuntu#g   ' /etc/apt/sources.list

