@echo off
title MaxTraderAI
echo Uploading trade log...
git add trade_log.csv
git commit -m "Upload trade log" || echo No changes to commit.
git push
pause
