import os
import time
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types
import config

TPM_LIMIT = 125000

def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_txt_files(directory):
    return list(directory.rglob("*.txt"))

def process_batch(client, batch_data):
    request_json_str = json.dumps(batch_data, ensure_ascii=False, indent=2)
    full_prompt_text = f"{config.EDITOR_PROMPT}\n\nJSON_INPUT:\n{request_json_str}"

    try:
        response = client.models.generate_content(
            model=config.EDITOR_MODEL,
            contents=full_prompt_text,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.3,
                "top_p": 0.95
            }
        )
        return json.loads(response.text)

    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"\n[ERR] Failed to parse JSON response: {e}")
        return {}
    except Exception as e:
        print(f"\n[ERR] API call failed: {e}")
        return {}

def create_metadata_header(file_paths):
    sorted_paths = sorted(file_paths, key=lambda x: x.name)
    header = "="*65 + "\nINFO: MERGED FILE\n" + "="*65 + "\n\nINCLUDED:\n"
    for p in sorted_paths:
        header += f"📌 {p.stem.strip()}\n"
    header += "\n" + "="*65 + "\n\n"
    return header

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not config.GOOGLE_API_KEY:
        print("ERROR: NO API KEY")
        return

    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    if not config.DIR_TEXT_CLEAN.exists():
        config.DIR_TEXT_CLEAN.mkdir(parents=True)

    all_files = get_txt_files(config.DIR_TEXT_RAW)
    print(f"--- Editor: {config.EDITOR_MODEL} (JSON Batch Mode) ---")
    
    files_by_hash = defaultdict(list)
    for f in all_files:
        files_by_hash[get_file_hash(f)].append(f)
    unique_groups = list(files_by_hash.values())
    print(f"Unique groups to process: {len(unique_groups)}")
    
    batches = []
    current_batch = {}
    current_batch_len = 0

    for group in unique_groups:
        primary_file = group[0]
        text = primary_file.read_text('utf-8')
        item_id = hashlib.md5(str(group).encode()).hexdigest()
        
        if current_batch_len + len(text) > TPM_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch, current_batch_len = {}, 0
            
        current_batch[item_id] = {'text': text, 'group_meta': group}
        current_batch_len += len(text)
    
    if current_batch:
        batches.append(current_batch)

    print(f"Total batches to process: {len(batches)}")
    
    processed_count, skipped_count = 0, 0
    
    with client:
        pbar = tqdm(batches, desc="Processing Batches")
        for batch in pbar:
            first_item_meta = list(batch.values())[0]['group_meta']
            primary_file = first_item_meta[0]
            fname = f"MERGED_{primary_file.name}" if len(first_item_meta) > 1 else primary_file.name
            path = config.DIR_TEXT_CLEAN / primary_file.relative_to(config.DIR_TEXT_RAW).parent / fname
            
            if path.exists() and not args.force:
                skipped_count += len(batch)
                continue
                
            batch_to_send = {item_id: data['text'] for item_id, data in batch.items()}
            cleaned_results = process_batch(client, batch_to_send)
            
            for item_id, cleaned_text in cleaned_results.items():
                if item_id in batch:
                    group_meta = batch[item_id]['group_meta']
                    primary_file = group_meta[0]
                    
                    fname = f"MERGED_{primary_file.name}" if len(group_meta) > 1 else primary_file.name
                    relative_path = primary_file.relative_to(config.DIR_TEXT_RAW).parent
                    clean_dir = config.DIR_TEXT_CLEAN / relative_path
                    clean_path = clean_dir / fname

                    header = create_metadata_header(group_meta)
                    content = header + str(cleaned_text)

                    clean_dir.mkdir(parents=True, exist_ok=True)
                    clean_path.write_text(content, encoding='utf-8')
                    processed_count += 1
            
            if len(batches) > 1:
                time.sleep(15)

    print(f"\nDone. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
