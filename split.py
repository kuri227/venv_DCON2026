import os
import shutil
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

# ==========================================
# 1. 設定パラメータ
# ==========================================
SR = 16000                  # サンプリングレート
DURATION = 1.5              # 1切り取りあたりの秒数
MAX_CHUNKS = 20             # 1ファイルから切り出す最大枚数（過学習防止）

# フォルダ設定
SOURCE_DIR = "to_split"     # 長い音声を最初に入れるフォルダ
RAW_DIR = "raw_data"        # 切り取り終わった元ファイルの退避先
TARGET_DIR = "input_wavs"   # 切り取ったファイルの保存先（学習用）
CSV_FILE = "multilabel_data.csv"

# ==========================================
# 2. 初期セットアップ
# ==========================================
for d in [SOURCE_DIR, RAW_DIR, TARGET_DIR]:
  os.makedirs(d, exist_ok=True)

def main():
  files_to_split = [f for f in os.listdir(
      SOURCE_DIR) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

  if not files_to_split:
    print(f"📁 '{SOURCE_DIR}' フォルダに音声ファイルがありません。")
    print(f"ネットからダウンロードした長い音声を '{SOURCE_DIR}' に入れてから再度実行してください。")
    return

  # 既存のCSVを読み込み（ファイル名被り確認用）
  if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    existing_files = set(df['filename'].tolist())
  else:
    df = pd.DataFrame(
        columns=['filename', 'label', 'environment', 'rescuer', 'victim'])
    existing_files = set()

  new_rows = []

  print(f"🔪 {len(files_to_split)} 件のファイルを分割処理します...\n")

  for file in files_to_split:
    file_path = os.path.join(SOURCE_DIR, file)
    base_name = os.path.splitext(file)[0]

    print(f"----------------------------------------")
    print(f"🎵 処理中: {file}")

    # 音声の読み込み
    try:
      y, sr = librosa.load(file_path, sr=SR)
    except Exception as e:
      print(f"  [エラー] 読み込み失敗: {e}")
      continue

    total_duration = len(y) / SR
    chunk_samples = int(DURATION * SR)

    # 分割可能な最大チャンク数を計算
    possible_chunks = int(total_duration // DURATION)

    if possible_chunks == 0:
      print(f"  [スキップ] 音声が {DURATION}秒 未満のため分割できません。")
      continue

    # 抽出するチャンク数（上限 MAX_CHUNKS）
    num_chunks = min(possible_chunks, MAX_CHUNKS)

    # 重複を避け、全体から均等にサンプリングするためのステップ幅を計算
    step = possible_chunks // num_chunks

    print(f"  -> 長さ: {total_duration:.1f}秒 | 抽出枚数: {num_chunks}枚 (等間隔抽出)")

    # ラベルの入力（このファイルの切り出しデータすべてに適用）
    print("  [ラベル設定] 該当するものに 1 を入力 (そのままEnterで 0)")
    v_in = input("    -> victim (要救助者) : ").strip()
    e_in = input("    -> environment (環境音): ").strip()
    r_in = input("    -> rescuer (救助者)  : ").strip()

    victim = 1 if v_in == '1' else 0
    environment = 1 if e_in == '1' else 0
    rescuer = 1 if r_in == '1' else 0

    labels = []
    if environment: labels.append("environment")
    if rescuer: labels.append("rescuer")
    if victim: labels.append("victim")
    label_str = ",".join(labels) if labels else "none"

    # 分割と保存
    extracted_count = 0
    for i in range(num_chunks):
      start_idx = (i * step) * chunk_samples
      end_idx = start_idx + chunk_samples
      chunk_y = y[start_idx:end_idx]

      # ファイル名の被り防止処理
      chunk_filename = f"{base_name}_part{i+1:03d}.wav"
      counter = 1
      while chunk_filename in existing_files or os.path.exists(os.path.join(TARGET_DIR, chunk_filename)):
        chunk_filename = f"{base_name}_part{i+1:03d}_{counter}.wav"
        counter += 1

      chunk_path = os.path.join(TARGET_DIR, chunk_filename)

      # 16-bit PCM WAVとして保存
      sf.write(chunk_path, chunk_y, SR, subtype='PCM_16')
      existing_files.add(chunk_filename)
      extracted_count += 1

      # CSV追記用データ
      new_rows.append({
          'filename': chunk_filename,
          'label': label_str,
          'environment': environment,
          'rescuer': rescuer,
          'victim': victim
      })

    # 元ファイルを raw_data に移動（退避）
    shutil.move(file_path, os.path.join(RAW_DIR, file))
    print(f"  ✅ {extracted_count}枚を作成し、元データを '{RAW_DIR}' に退避しました。")

  # CSVの更新
  if new_rows:
    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(CSV_FILE, index=False)
    print(f"\n🎉 完了！ 合計 {len(new_rows)} 件の新しいデータを {CSV_FILE} に追加しました。")

if __name__ == "__main__":
  main()
