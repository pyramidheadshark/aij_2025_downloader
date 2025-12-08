import argparse
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from readability import Readability
import torch

SIMILARITY_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
import config

def count_words(text: str) -> int:
    return len(re.findall(r'\w+', text))

def count_punctuation(text: str) -> int:
    return len(re.findall(r'[.,!?;:—"-]', text))

def count_uppercase(text: str) -> int:
    return sum(1 for char in text if char.isupper())

def get_readability_score(text: str) -> float:
    try:
        r = Readability(text)
        return r.flesch_kincaid().score
    except Exception:
        return 0.0

def find_corresponding_clean_file(stt_path: Path) -> Path | None:
    relative_path = stt_path.relative_to(config.DIR_TEXT_RAW)
    
    direct_match = config.DIR_TEXT_CLEAN / relative_path
    if direct_match.exists():
        return direct_match
        
    merged_match = config.DIR_TEXT_CLEAN / relative_path.parent / f"MERGED_{relative_path.name}"
    if merged_match.exists():
        return merged_match
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate STT vs LLM-edited text quality.")
    parser.add_argument(
        "--output", 
        type=str, 
        default="evaluation_report.csv", 
        help="Path to save the CSV report."
    )
    args = parser.parse_args()

    if not config.DIR_TEXT_RAW.exists() or not config.DIR_TEXT_CLEAN.exists():
        print("Ошибка: Директории 'output_stt' или 'output_clean' не найдены.")
        return

    print(f"🧠 Загрузка embedding-модели '{SIMILARITY_MODEL}'... (может занять время при первом запуске)")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device.upper()}")
    model = SentenceTransformer(SIMILARITY_MODEL, device=device)

    stt_files = list(config.DIR_TEXT_RAW.rglob("*.txt"))
    report_data = []

    print("📊 Сравнение файлов и вычисление метрик...")
    for stt_path in tqdm(stt_files, desc="Processing files"):
        clean_path = find_corresponding_clean_file(stt_path)
        
        if not clean_path:
            continue

        try:
            with open(stt_path, "r", encoding="utf-8") as f:
                stt_text = f.read()
            with open(clean_path, "r", encoding="utf-8") as f:
                llm_text = f.read()
                if "MERGED" in clean_path.name:
                    llm_text = re.sub(r'={10,}.*?={10,}\s*', '', llm_text, flags=re.DOTALL)

            stt_words = count_words(stt_text)
            llm_words = count_words(llm_text)
            
            embeddings = model.encode([stt_text, llm_text], convert_to_tensor=True, show_progress_bar=False)
            cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

            report_data.append({
                "file_name": stt_path.name,
                "stt_chars": len(stt_text),
                "llm_chars": len(llm_text),
                "stt_words": stt_words,
                "llm_words": llm_words,
                "word_diff_percent": round(((llm_words - stt_words) / stt_words * 100) if stt_words > 0 else 0, 2),
                "stt_punctuation": count_punctuation(stt_text),
                "llm_punctuation": count_punctuation(llm_text),
                "stt_uppercase": count_uppercase(stt_text),
                "llm_uppercase": count_uppercase(llm_text),
                "stt_readability": get_readability_score(stt_text),
                "llm_readability": get_readability_score(llm_text),
                "semantic_similarity": round(cosine_similarity, 4),
            })
        except Exception as e:
            print(f"\n[Warn] Не удалось обработать файл {stt_path.name}: {e}")

    if not report_data:
        print("Не найдено ни одной пары файлов для сравнения.")
        return

    df = pd.DataFrame(report_data)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Отчет сохранен в файл: {args.output}")

    print("\n--- Сводная статистика ---")
    avg_similarity = df['semantic_similarity'].mean() * 100
    avg_word_diff = df['word_diff_percent'].mean()
    avg_punct_increase = (df['llm_punctuation'].sum() - df['stt_punctuation'].sum()) / df['stt_punctuation'].sum() * 100 if df['stt_punctuation'].sum() > 0 else 0
    
    print(f"Среднее семантическое сходство: {avg_similarity:.2f}%")
    print(f"Среднее изменение кол-ва слов: {avg_word_diff:.2f}%")
    print(f"Общий прирост пунктуации: +{avg_punct_increase:.2f}%")
    print(f"Файлы с самым низким сходством (<95%):")
    low_similarity_files = df[df['semantic_similarity'] < 0.95].sort_values(by="semantic_similarity")
    if not low_similarity_files.empty:
        for _, row in low_similarity_files.iterrows():
            print(f"  - {row['file_name']} ({row['semantic_similarity']:.2%})")
    else:
        print("  (таких файлов не найдено, отличный результат!)")

if __name__ == "__main__":
    main()
