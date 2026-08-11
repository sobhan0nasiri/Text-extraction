@echo off
REM دابل-کلیک روی همین فایل کافی است -- ppocr_env را فعال و سرویس PP-OCRv5 را روی GPU بالا می‌آورد.
call conda activate ppocr_env
cd /d "%~dp0"
python ppocr_server.py
pause