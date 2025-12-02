import os

JSON_PATH = os.path.join('data', 'schedule.json')
OUTPUT_DIR = 'output'
TEMP_DIR = 'temp_raw' # Папка для сырых гигабайтных файлов перед сжатием

# Настройки поиска
TARGET_M3U8_PART = 'ru.m3u8'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# --- НАСТРОЙКИ ФАЙЛОВ ---
FILENAME_FORMAT = "{time} - {speaker} - {title}.mp4"
MAX_FILENAME_LENGTH = 120
MAX_SPEAKER_LEN = 40
MAX_TITLE_LEN = 60

# --- НАСТРОЙКИ СЖАТИЯ (FFMPEG) ---
COMPRESS_VIDEO = True 
FFMPEG_CRF = 28 
FFMPEG_PRESET = 'veryfast'
FFMPEG_SCALE = "-1:720"

# Список известных залов для фильтрации (по именам в JSON)
KNOWN_HALLS = [
    "Main Stage", 
    "Live Studio", 
    "Junior", 
    "AI Frontiers", 
    "AI in Applied Research"
]
