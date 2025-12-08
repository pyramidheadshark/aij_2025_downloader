import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import config

INPUT_DIR = config.DIR_VIDEO_RAW
OUTPUT_DIR = config.DIR_AUDIO_WAV

def check_ffmpeg():
    if os.path.exists("ffmpeg.exe"):
        os.environ["PATH"] += os.pathsep + os.getcwd()

def convert_to_wav16k(video_path, audio_path):
    # -ac 1: моно
    # -ar 16000: 16 кГц (стандарт для речевых моделей)
    # -vn: убрать видео
    cmd = [
        "ffmpeg", "-n",          # -n: не перезаписывать, если есть
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]
    subprocess.run(cmd)

def main():
    check_ffmpeg()
    
    video_files = list(INPUT_DIR.rglob("*.mp4"))
    print(f"Found {len(video_files)} video files.")
    
    pbar = tqdm(video_files, desc="Extracting Audio")
    
    for video_path in pbar:
        relative_path = video_path.relative_to(INPUT_DIR)
        audio_path = OUTPUT_DIR / relative_path.with_suffix(".wav")
        
        if audio_path.exists():
            continue
            
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        convert_to_wav16k(video_path, audio_path)

if __name__ == "__main__":
    main()
