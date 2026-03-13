@echo off
python -c "import redis; r = redis.from_url('redis://default:bwQmxUCQEXUlYTWACmPbbkpnHPVpoiIa@tramway.proxy.rlwy.net:48930'); r.delete('gpu:processing'); print('GPU LOCK CLEARED!')"
pause
