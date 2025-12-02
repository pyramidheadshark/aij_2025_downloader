import json
import os
import re
import asyncio
import shutil
import argparse
import subprocess
import time
from datetime import datetime
from playwright.async_api import async_playwright
import yt_dlp
import config


def clean_name(s):
    if not s: return "Unknown"
    s = str(s).strip().replace(':', ' -').replace('/', '_').replace('\\', '_')
    s = re.sub(r'[?*<>|"]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def truncate_string(s, max_len):
    if len(s) <= max_len: return s
    return s[:max_len].rstrip() + "..."

def extract_time(iso_date_str):
    try:
        dt = datetime.fromisoformat(iso_date_str)
        return dt.strftime('%H-%M')
    except:
        return "00-00"

def is_direct_download_link(url):
    return 'vkvideo.ru' in url or 'vk.com' in url

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def compress_video(input_path, output_path):
    print(f"   --> Сжатие: {os.path.basename(input_path)} -> CRF {config.FFMPEG_CRF}")
    
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts", "-err_detect", "ignore_err",
        "-i", input_path,
        "-vf", f"scale={config.FFMPEG_SCALE}",
        "-c:v", "libx264",
        "-crf", str(config.FFMPEG_CRF),
        "-preset", config.FFMPEG_PRESET,
        "-c:a", "aac", "-b:a", "128k",
        "-map_metadata", "-1",
        output_path
    ]

    try:
        start_t = time.time()
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            print(f"   [FAIL] Ошибка FFmpeg. Лог:")
            print("="*20 + "\n" + result.stderr[-500:] + "\n" + "="*20)
            return False
            
        duration = time.time() - start_t
        old_size = os.path.getsize(input_path) / (1024*1024)
        new_size = os.path.getsize(output_path) / (1024*1024)
        
        print(f"   [OK] Сжато за {int(duration)}с. {old_size:.1f}MB -> {new_size:.1f}MB")
        return True
    except Exception as e:
        print(f"   [FAIL] Исключение при сжатии: {e}")
        return False


async def resolve_m3u8_links(unique_urls):
    results = {}
    browser_urls = [u for u in unique_urls if not is_direct_download_link(u)]
    direct_urls = [u for u in unique_urls if is_direct_download_link(u)]
    
    for url in direct_urls: results[url] = url
    if not browser_urls: return results

    CONCURRENCY = 8
    print(f"--- Запуск Chrome (Headless) для {len(browser_urls)} ссылок ---")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        async def process_url(url, idx, total):
            async with semaphore:
                context = await browser.new_context(user_agent=config.USER_AGENT)
                page = await context.new_page()
                found_m3u8 = None
                
                def handle_request(req):
                    nonlocal found_m3u8
                    if config.TARGET_M3U8_PART in req.url: found_m3u8 = req.url
                    elif '.m3u8' in req.url and not found_m3u8: found_m3u8 = req.url

                page.on("request", handle_request)
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    try: await page.wait_for_selector('video', timeout=4000)
                    except: pass 
                    for _ in range(50):
                        if found_m3u8: break
                        await page.wait_for_timeout(100)
                    
                    if found_m3u8:
                        results[url] = found_m3u8
                        print(f"[{idx}/{total}] [+] {found_m3u8.split('/')[-1]}")
                    else:
                        print(f"[{idx}/{total}] [-] FAIL: {url}")
                except Exception as e:
                    print(f"[{idx}/{total}] [!] {e}")
                finally:
                    await context.close()

        tasks = [process_url(url, i, len(browser_urls)) for i, url in enumerate(browser_urls, 1)]
        await asyncio.gather(*tasks)
        await browser.close()
    return results


def download_and_process(source_url, final_target_path, temp_dir, referer_url=None):
    """
    Скачивает RAW с притворяясь браузером (Referer), сжимает, сохраняет Final.
    """
    
    if os.path.exists(final_target_path):
        if os.path.getsize(final_target_path) < 1024:
             print(f"   [WARN] Найден пустой файл {os.path.basename(final_target_path)}. Перекачиваем.")
             os.remove(final_target_path)
        else:
             print(f"   -> Файл уже готов: {os.path.basename(final_target_path)}")
             return True

    filename = os.path.basename(final_target_path)
    raw_filename = "RAW_" + filename
    raw_path = os.path.join(temp_dir, raw_filename)
    part_path = raw_path + ".part"

    if not os.path.exists(raw_path):
        print(f"   --> Скачивание RAW: {filename}")
        
        http_headers = {
            'User-Agent': config.USER_AGENT,
        }
        if referer_url:
            http_headers['Referer'] = referer_url
            http_headers['Origin'] = "https://front.finevid.link"

        ydl_opts = {
            'outtmpl': raw_path,
            'format': 'best',
            'quiet': False, 
            'no_warnings': False,
            'concurrent_fragment_downloads': 8,
            'trim_file_name': 200,
            'http_headers': http_headers,
            'retries': 10,
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'ignoreerrors': True,
            'abort_on_unavailable_fragment': False,
            'hls_use_mpegts': True, 
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source_url])
        except Exception as e:
            print(f"   [WARN] yt-dlp завершился с ошибкой (проверяем файл...): {e}")

        if os.path.exists(raw_path): pass
        elif os.path.exists(part_path):
            print(f"   [WARN] Восстановление из .part...")
            try: shutil.move(part_path, raw_path)
            except: pass
        
        if not os.path.exists(raw_path):
             print(f"   [FAIL] Не удалось скачать файл.")
             return False
    else:
        print(f"   --> Найден загруженный RAW: {filename}")

    if config.COMPRESS_VIDEO:
        if os.path.getsize(raw_path) < 1024:
             print("   [FAIL] RAW файл пустой (возможно, бан по IP или ошибка доступа). Удаляем.")
             os.remove(raw_path)
             return False

        success = compress_video(raw_path, final_target_path)
        if success:
            try: os.remove(raw_path)
            except: pass
            return True
        else:
            return False
    else:
        try:
            shutil.move(raw_path, final_target_path)
            return True
        except Exception as e:
            print(f"   [FAIL] Ошибка перемещения: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="AIJ Downloader Pro")
    subparsers = parser.add_subparsers(dest='command', help='Commands', required=True)
    
    dl_parser = subparsers.add_parser('download', help='Скачать лекции')
    dl_parser.add_argument('--halls', nargs='+', default=[], help='Фильтр залов')
    dl_parser.add_argument('--all', action='store_true', help='Скачать ВСЕ залы')
    
    retry_parser = subparsers.add_parser('retry', help='Повторить скачивание')
    
    clean_parser = subparsers.add_parser('clean', help='Очистить кэш')
    args = parser.parse_args()

    if args.command == 'clean':
        print("--- ОЧИСТКА ---")
        if os.path.exists(config.TEMP_DIR):
            try: shutil.rmtree(config.TEMP_DIR)
            except: pass
            print(f"[OK] Удалено: {config.TEMP_DIR}")
        return

    if args.command in ['download', 'retry']:
        is_retry = (args.command == 'retry')
        
        if config.COMPRESS_VIDEO and not check_ffmpeg():
            print("\n[!!!] FFMPEG НЕ НАЙДЕН. Скачайте ffmpeg.exe.")
            return

        for d in [config.OUTPUT_DIR, config.TEMP_DIR]:
            if not os.path.exists(d): os.makedirs(d)

        with open(config.JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tasks = []
        unique_player_urls = set()
        
        target_halls = []
        if is_retry or args.all:
            target_halls = []
            print("Режим: Заполнение пропусков (Все залы)" if is_retry else "Режим: Скачивание (Все залы)")
        elif args.halls:
            target_halls = [h.lower() for h in args.halls]
            print(f"Режим: Выбранные залы {args.halls}")
        else:
            print("Используйте: download --all ИЛИ retry")
            return

        print("\n--- 1. Поиск недостающих файлов ---")
        for day in data:
            for hall in day.get('halls', []):
                hall_name = hall.get('name', 'Unknown')
                if target_halls and not any(th in hall_name.lower() for th in target_halls):
                    continue

                for topic in hall.get('topics', []):
                    if topic.get('isBreak') or not topic.get('videos'): continue
                    player_url = topic['videos'][0].get('videoUrl')
                    if not player_url: continue

                    date_folder = clean_name(day.get('concreteDate'))
                    hall_folder = clean_name(hall_name)
                    safe_title = truncate_string(clean_name(topic.get('title')), config.MAX_TITLE_LEN)
                    safe_speaker = truncate_string(clean_name(", ".join(filter(None, [s.get('fullName') for s in topic.get('speakers', [])]))) or "Speaker", config.MAX_SPEAKER_LEN)
                    time_str = extract_time(topic.get('startDate'))
                    
                    filename = config.FILENAME_FORMAT.format(time=time_str, speaker=safe_speaker, title=safe_title)
                    if len(filename) > config.MAX_FILENAME_LENGTH:
                        name_part, ext = os.path.splitext(filename)
                        filename = name_part[:config.MAX_FILENAME_LENGTH] + ext
                    
                    target_path = os.path.join(config.OUTPUT_DIR, date_folder, hall_folder, filename)
                    
                    if not os.path.exists(target_path):
                        unique_player_urls.add(player_url)
                        tasks.append({
                            'player_url': player_url,
                            'target_path': target_path,
                            'title': filename
                        })

        print(f"Необходимо скачать: {len(tasks)} файлов.")
        if len(tasks) == 0:
            print("Все файлы уже скачаны!")
            return

        m3u8_map = asyncio.run(resolve_m3u8_links(list(unique_player_urls)))
        
        print("\n--- 3. Обработка ---")
        
        stats = {'ok': 0, 'fail': 0}
        processed_files_cache = {}

        for i, task in enumerate(tasks, 1):
            p_url = task['player_url']
            if p_url not in m3u8_map:
                print(f"[{i}] SKIP: Нет видео")
                stats['fail'] += 1
                continue
            
            source_url = m3u8_map[p_url]
            target_path = task['target_path']

            if source_url in processed_files_cache and os.path.exists(processed_files_cache[source_url]):
                print(f"[{i}/{len(tasks)}] Копирование: {os.path.basename(target_path)}")
                try: 
                    shutil.copyfile(processed_files_cache[source_url], target_path)
                    stats['ok'] += 1
                except: stats['fail'] += 1
            else:
                print(f"[{i}/{len(tasks)}] Загрузка: {os.path.basename(target_path)}")
                success = download_and_process(source_url, target_path, config.TEMP_DIR, referer_url=p_url)
                if success and os.path.exists(target_path):
                    processed_files_cache[source_url] = target_path
                    stats['ok'] += 1
                else:
                    stats['fail'] += 1

        print("\n" + "="*30)
        print(f"ИТОГ: Успешно: {stats['ok']} | Провалено: {stats['fail']}")
        print("="*30)

if __name__ == "__main__":
    main()
