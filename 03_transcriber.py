import os
import shutil
import torch
import warnings
import argparse
import logging
import soundfile as sf
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from transformers import AutoModel
import config

logging.getLogger("transformers").setLevel(logging.ERROR)

CHUNK_DURATION = 20.0 
OVERLAP = 2.0

def clean_huggingface_cache():
    home = Path.home()
    cache_dir = home / ".cache" / "huggingface" / "hub"
    repo_name = "models--" + config.MODEL_ID.replace("/", "--")
    target_dir = cache_dir / repo_name
    if target_dir.exists():
        try: shutil.rmtree(target_dir)
        except: pass

def get_audio_files(directory):
    return list(directory.rglob("*.wav"))

def smart_merge(text1, text2):
    if not text1: return text2
    if not text2: return text1

    look_back = min(len(text1), 100) 
    look_forward = min(len(text2), 100)
    
    tail = text1[-look_back:]
    head = text2[:look_forward]
    
    matcher = SequenceMatcher(None, tail, head)
    match = matcher.find_longest_match(0, len(tail), 0, len(head))
    
    if match.size > 5:
        return text1 + " " + text2[match.b + match.size:]
    else:
        return text1 + " " + text2

def transcribe_file_native(file_path, model, temp_root, pbar_main=None):
    data, sr = sf.read(str(file_path))
    if len(data.shape) > 1: data = data[:, 0]
    
    total_samples = len(data)
    chunk_samples = int(CHUNK_DURATION * sr)
    overlap_samples = int(OVERLAP * sr)
    step = chunk_samples - overlap_samples
    
    file_hash = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()
    file_temp_dir = temp_root / file_hash
    
    if file_temp_dir.exists(): shutil.rmtree(file_temp_dir, ignore_errors=True)
    file_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        chunks_info = []
        for i in range(0, total_samples, step):
            if total_samples - i < sr: break
            
            end = min(i + chunk_samples, total_samples)
            chunk_data = data[i:end]
            
            chunk_name = f"{i}.wav"
            chunk_path = file_temp_dir / chunk_name
            sf.write(str(chunk_path), chunk_data, sr)
            chunks_info.append(str(chunk_path))

        full_text = ""
        
        if pbar_main:
            pbar_main.set_description(f"Processing ({len(chunks_info)} chunks)")
        
        for chunk_path in chunks_info:
            try:
                text_part = model.transcribe(chunk_path)
                
                full_text = smart_merge(full_text, text_part)
                
            except Exception as e:
                pass

        return full_text

    finally:
        if file_temp_dir.exists(): 
            try: shutil.rmtree(file_temp_dir, ignore_errors=True)
            except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Process 1 file and exit")
    parser.add_argument("--clean-cache", action="store_true", help="Clean HF cache")
    args = parser.parse_args()

    if args.clean_cache:
        clean_huggingface_cache()
        print("Cache cleaned.")
        return

    print(f"--- GigaAM-v3 Native Transcriber ---")
    print(f"Device: {config.DEVICE}")
    print(f"Model Revision: {config.MODEL_REVISION}")

    try:
        print("Loading model...")
        model = AutoModel.from_pretrained(
            config.MODEL_ID, 
            revision=config.MODEL_REVISION, 
            trust_remote_code=True
        ).to(config.DEVICE)
        model.eval()
        print("Model loaded.")

    except Exception as e:
        print(f"CRITICAL Error loading components: {e}")
        return
    
    try:
        from deepmultilingualpunctuation import PunctuationModel
        punct_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")
        print("Punctuation model loaded.")
    except Exception:
        print("[WARN] deepmultilingualpunctuation failed. Skipping punctuation.")
        punct_model = None

    TEMP_CHUNKS_DIR = config.BASE_DIR / 'temp_chunks'
    if not TEMP_CHUNKS_DIR.exists(): TEMP_CHUNKS_DIR.mkdir()

    all_files = get_audio_files(config.DIR_AUDIO_WAV)
    files_to_process = []
    for wav_path in all_files:
        rel_path = wav_path.relative_to(config.DIR_AUDIO_WAV)
        txt_path = config.DIR_TEXT_RAW / rel_path.with_suffix(".txt")
        if not txt_path.exists():
            files_to_process.append(wav_path)
    
    print(f"Files to process: {len(files_to_process)}")

    pbar = tqdm(files_to_process, unit="file")
    
    for wav_path in pbar:
        short_name = wav_path.name 
        if len(short_name) > 30: short_name = short_name[:27] + "..."
        pbar.set_postfix_str(short_name)
        
        try:
            rel_path = wav_path.relative_to(config.DIR_AUDIO_WAV)
            txt_path = config.DIR_TEXT_RAW / rel_path.with_suffix(".txt")
            
            raw_text = transcribe_file_native(wav_path, model, TEMP_CHUNKS_DIR, pbar)
                        
            if punct_model and raw_text and len(raw_text) > 5:
                try:
                    final_text = punct_model.restore_punctuation(raw_text)
                except:
                    final_text = raw_text
            else:
                final_text = raw_text
            
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            if args.test:
                print(f"\n[TEST MODE] Output saved to: {txt_path}")
                print(f"Sample: {final_text[:300]}...")
                break
                
        except Exception as e:
            print(f"\n[ERR] {wav_path.name}: {e}")
            continue

    if TEMP_CHUNKS_DIR.exists():
        try: shutil.rmtree(TEMP_CHUNKS_DIR, ignore_errors=True)
        except: pass

    print("\nDone.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
