@echo off
color 4f

git add .
git status
set /p a=:
git commit -m "%a%"
git push origin "master"
pause