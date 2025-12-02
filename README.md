# AI Journey 2025 Downloader

Скрипт для автоматического скачивания, сжатия и организации лекций с конференции AI Journey.

## Требования

- Python 3.10+
- FFmpeg

## Установка

1. **Клонировать репозиторий:**

    ```bash
    git clone ...
    cd aij_2025_downloader
    ```

2. **Установить FFmpeg:**
    - Скачайте `ffmpeg.exe` (например, с [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)).
    - Положите `ffmpeg.exe` в корень этого проекта.

3. **Установить зависимости Python:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Установить браузеры для Playwright:**

    ```bash
    playwright install chromium
    ```

5. **Положить данные:**
    - Поместите JSON с расписанием в `data/schedule.json`.

## Использование

### Скачать все лекции

```bash
python main.py download --all
```

### Скачать лекции из конкретных залов

```bash
# Можно указывать несколько залов через пробел
python main.py download --halls "AI Frontiers" "Junior"
```

### Очистить временные файлы

Удаляет папку `temp_raw`.

```bash
python main.py clean

```
