@echo off
title MaxTraderAI
echo Updating repository...

git stash push --include-untracked -m "update" >nul 2>&1
git pull
git stash pop >nul 2>&1
pause
