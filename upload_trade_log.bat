@echo off
title MaxTraderAI
echo Uploading trade log...
git config user.name "B A"
git config user.email "b.a@idont.com"
if exist logs\trade_log.csv (
    git add logs/trade_log.csv
    git commit -m "Upload trade log" || echo No changes to commit.
    git push
) else (
    echo logs\trade_log.csv not found.
)
pause
