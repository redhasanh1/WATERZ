#!/bin/bash
apt-get update -qq && apt-get install -y -qq gcc g++ > /dev/null 2>&1
echo "[OK] gcc installed"
gcc --version | head -1
python /app/test/docker_test.py
