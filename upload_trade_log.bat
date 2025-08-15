@echo off
title MaxTraderAI
echo Uploading trade log...
git config user.name "B A"
git config user.email "b.a@idont.com"
git add trade_log.csv
git commit -m "Upload trade log" || echo No changes to commit.
git push
pause
