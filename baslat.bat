@echo off
title Beko AI Sistemi - Baslatici
echo [BEKO AI] Sistem hazirlaniyor...

:: Sanal ortami kontrol et ve aktif et
if not exist beko_env (
    echo [1/3] Sanal ortam olusturuluyor...
    py -3.12 -m venv beko_env
)
call beko_env\Scripts\activate

:: Gerekli kutuphaneleri yukle
echo [2/3] Kutuphaneler kontrol ediliyor...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

:: Arayüzü baslat
echo [3/3] Beko AI Paneli aciliyor...
streamlit run beko_arayuz.py
pause