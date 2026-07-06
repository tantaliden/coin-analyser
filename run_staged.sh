#!/bin/bash
cd /opt/coin/backend
export PYTHONPATH=/opt/coin/backend
nohup /opt/coin/venv/bin/python3 /opt/training/scripts/discriminate_staged.py > /opt/coin/logs/discriminate_staged.log 2>&1 &
echo "Trichter-Test gestartet, PID $!"
