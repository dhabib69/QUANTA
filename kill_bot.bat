@echo off
title Kill QUANTA Bot
echo Searching for background instances of bot.py...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'bot\.py' }; if ($procs) { foreach ($p in $procs) { Write-Host 'Killing bot.py (PID:' $p.ProcessId ')' -ForegroundColor Red; try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch { Write-Host 'Failed to kill PID' $p.ProcessId -ForegroundColor DarkRed } } } else { Write-Host 'No running instances of bot.py found.' -ForegroundColor Green }"

echo.
echo Operation complete. You can now run your bot peacefully.
pause
