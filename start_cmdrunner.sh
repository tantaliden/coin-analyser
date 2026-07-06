#!/bin/bash
systemctl daemon-reload
systemctl enable --now cmd-runner.service
sleep 2
systemctl is-active cmd-runner.service
echo "cmd-runner Status oben | Queue: /opt/coin/cmdq/"
