import os
import shutil
import torch
import warnings
import argparse
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel

logging.getLogger("transformers").setLevel(logging.ERROR)

INPUT_AUDIO_DIR = Path('output_audio')
OUTPUT_STT_DIR = Path('output_stt')
TEMP_CHUNKS_DIR = Path('temp_chunks')

MODEL_ID = "ai-sage/GigaAM-v3"
REVISION = "ctc" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_DURATION = 20.0 
OVERLAP = 1.0

def clean_huggingface_cache():
    home = Path.home()
    cache_dir = home / ".cache" / "huggingface" / "hub"
    repo_name = "models--" + MODEL_ID.replace("/", "--")
    target_dir = cache_dir / repo_name
    if target_dir.exists():
        try: shutil.rmtree(target_dir)
        except: pass

def get_audio_files(directory):
    return list(directory.rglob("*.wav"))

def transcribe_long_file(file_path, model, temp_dir):
    data, sr = sf.read(str(file_path))
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    total_samples = len(data)
    chunk_samples = int(CHUNK_DURATION * sr)
    overlap_samples = int(OVERLAP * sr)
    step = chunk_samples - overlap_samples
    
    full_transcription = []
    
    file_temp_dir = temp_dir / file_path.stem
    if file_temp_dir.exists():
        shutil.rmtree(file_temp_dir)
    file_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        chunk_paths = []
        for i in range(0, total_samples, step):
            if total_samples - i < sr: 
                break
                
            end = min(i + chunk_samples, total_samples)
            chunk_data = data[i:end]
            
            chunk_name = f"chunk_{i}.wav"
            chunk_path = file_temp_dir / chunk_name
            
            sf.write(str(chunk_path), chunk_data, sr)
            chunk_paths.append(str(chunk_path))

        try:
            batch_size = 4
            for i in range(0, len(chunk_paths), batch_size):
                batch = chunk_paths[i:i+batch_size]
                results = model.transcribe(batch)
                
                if isinstance(results, str):
                    full_transcription.append(results)
                elif isinstance(results, list):
                    full_transcription.extend(results)
                else:
                    full_transcription.append(str(results))
                    
        except Exception:
            for cp in chunk_paths:
                res = model.transcribe(cp)
                full_transcription.append(res)

    finally:
        if file_temp_dir.exists():
            try: shutil.rmtree(file_temp_dir)
            except: pass

    return " ".join(full_text_filter(full_transcription))

def full_text_filter(text_list):
    return [t.strip() for t in text_list if t and len(t.strip()) > 1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Process 1 file and exit")
    parser.add_argument("--clean-cache", action="store_true", help="Clean HF cache")
    args = parser.parse_args()

    if args.clean_cache:
        clean_huggingface_cache()
        print("Cache cleaned. Restart script.")
        return

    print(f"--- GigaAM-v3 Native Wrapper ---")
    print(f"Device: {DEVICE}")
    print(f"Revision: {REVISION}")

    print("Loading model via AutoModel (downloading custom code)...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID, 
            revision=REVISION, 
            trust_remote_code=True
        ).to(DEVICE)
        
        model.eval() 
        print("Model loaded successfully!")
        
        if not hasattr(model, 'transcribe'):
            print("CRITICAL: Loaded model has no .transcribe() method!")
            print(f"Model type: {type(model)}")
            return

    except Exception as e:
        print(f"CRITICAL Error loading model: {e}")
        return

    if not TEMP_CHUNKS_DIR.exists():
        TEMP_CHUNKS_DIR.mkdir()

    all_files = get_audio_files(INPUT_AUDIO_DIR)
    
    files_to_process = []
    for wav_path in all_files:
        rel_path = wav_path.relative_to(INPUT_AUDIO_DIR)
        txt_path = OUTPUT_STT_DIR / rel_path.with_suffix(".txt")
        if not txt_path.exists():
            files_to_process.append(wav_path)
    
    print(f"Files to process: {len(files_to_process)}")

    pbar = tqdm(files_to_process, desc="Transcribing")
    
    for wav_path in pbar:
        pbar.set_postfix_str(f"{wav_path.name[:20]}...")
        
        try:
            rel_path = wav_path.relative_to(INPUT_AUDIO_DIR)
            txt_path = OUTPUT_STT_DIR / rel_path.with_suffix(".txt")
            
            text = transcribe_long_file(wav_path, model, TEMP_CHUNKS_DIR)
            
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            if args.test:
                print(f"\n[TEST MODE] Output saved to: {txt_path}")
                print(f"Sample: {text[:200]}...")
                break
                
        except Exception as e:
            print(f"\n[ERR] {wav_path.name}: {e}")
            import traceback
            traceback.print_exc()

    if TEMP_CHUNKS_DIR.exists():
        try: shutil.rmtree(TEMP_CHUNKS_DIR)
        except: pass

    print("\nDone.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
