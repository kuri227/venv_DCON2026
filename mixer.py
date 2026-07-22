import itertools
import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pyaudio

# ==========================================
# 1. 設定
# ==========================================
SR = 16000
DURATION = 1.5
MAX_LEN = int(SR * DURATION)

CSV_FILE = "multilabel_data.csv"
TARGET_DIR = "input_wavs"
PREFIX = "pure_mix"

# ==========================================
# 2. 音声再生関数
# ==========================================
def play_audio(audio_data, sr=SR):
  """Numpy配列をスピーカーから直接再生する"""
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
  # 音割れ防止のために少しだけ音量を下げる
  play_data = audio_data * 0.8
  stream.write(play_data.astype(np.float32).tobytes())
  stream.stop_stream()
  stream.close()
  p.terminate()

# ==========================================
# 3. スマート音声読み込み関数
# ==========================================
def load_audio_smart(file_path, is_victim=False):
  """
  要救助者の音(is_victim=True)は最大音量を中心にクロップし、
  環境音はランダムにクロップする賢い読み込み関数
  """
  try:
    y, _ = librosa.load(file_path, sr=SR)
  except Exception:
    return np.zeros(MAX_LEN)

  if len(y) <= MAX_LEN:
    y_cropped = np.pad(y, (0, MAX_LEN - len(y)), mode='constant')
  else:
    if is_victim:
      # ピーク中心クロップ (ノック音の芯を捉える)
      peak_idx = np.argmax(np.abs(y))
      half_len = MAX_LEN // 2
      start_idx = peak_idx - half_len
      end_idx = peak_idx + half_len

      if start_idx < 0:
        start_idx = 0; end_idx = MAX_LEN
      elif end_idx > len(y):
        end_idx = len(y); start_idx = len(y) - MAX_LEN
      y_cropped = y[start_idx:end_idx]
    else:
      # 環境音はランダムクロップ
      start_idx = random.randint(0, len(y) - MAX_LEN)
      y_cropped = y[start_idx:start_idx + MAX_LEN]

  # 音量の正規化 (最大値を1.0にする)
  max_amp = np.max(np.abs(y_cropped))
  if max_amp > 0:
    y_cropped = y_cropped / max_amp

  return y_cropped

# ==========================================
# 4. 次のファイル番号を取得する関数
# ==========================================
def get_next_filename(df):
  existing_files = set(df['filename'].tolist())
  counter = 1
  while True:
    filename = f"{PREFIX}_{counter:03d}.wav"
    if filename not in existing_files and not os.path.exists(os.path.join(TARGET_DIR, filename)):
      return filename
    counter += 1

# ==========================================
# 5. メインループ (Tinder式仕分け / 履歴記憶アップデート版)
# ==========================================
def main():
  if not os.path.exists(CSV_FILE):
    print(f"エラー: {CSV_FILE} が見つかりません。")
    return

  df = pd.read_csv(CSV_FILE)

  # ピュアなノック音と環境音を抽出
  df_pure_victim = df[(df['victim'] == 1) & (
      df['environment'] == 0) & (df['rescuer'] == 0)]
  df_pure_env = df[(df['victim'] == 0) & (
      df['environment'] == 1) & (df['rescuer'] == 0)]

  num_vic = len(df_pure_victim)
  num_env = len(df_pure_env)

  if num_vic == 0 or num_env == 0:
    print("合成に必要なピュアデータが不足しています。")
    return

  # --------------------------------------------------
  # ★追加：すでに採用(y)された組み合わせの履歴を取得
  # --------------------------------------------------
  if 'source_victim' in df.columns and 'source_environment' in df.columns:
    # 過去に採用した組み合わせのセットを作成（高速検索用）
    accepted_pairs = set(
        zip(df['source_victim'].dropna(), df['source_environment'].dropna()))
  else:
    accepted_pairs = set()

  # ==================================================
  # ★修正：採用済みのペアを除外して組み合わせリストを作成
  # ==================================================
  print(f"🔄 データの組み合わせを計算中... (Victim: {num_vic}件 × Environment: {num_env}件)")

  all_pairs = []
  for vic_idx in range(num_vic):
    for env_idx in range(num_env):
      vic_file = df_pure_victim.iloc[vic_idx]['filename']
      env_file = df_pure_env.iloc[env_idx]['filename']

      # 過去に採用されていない組み合わせだけをリストに追加する
      if (vic_file, env_file) not in accepted_pairs:
        all_pairs.append((vic_idx, env_idx))

  if len(all_pairs) == 0:
    print("🎉 すべての組み合わせのチェックが完了しています！もう新しいペアはありません。")
    return

  # リストを完全にシャッフルする
  random.shuffle(all_pairs)

  # 過去にいくつか採用済みの場合、残りの数を表示
  total_possible = num_vic * num_env
  print(f"✅ 全 {total_possible} 通りのうち、未判定の {len(all_pairs)} 通りをリストアップしました！")

  print("\n==================================================")
  print(" 🎧 AIデータ仕分けツール (Tinder式 MixUp)")
  print("==================================================")
  print("  [y] + Enter : 採用して保存 (Save)")
  print("  [n] + Enter : 却下して次へ (Discard)")
  print("  [r] + Enter : もう一度聞く (Replay)")
  print("  [q] + Enter : 終了 (Quit)")
  print("==================================================\n")

  accepted_count = 0
  pair_index = 0

  while pair_index < len(all_pairs):
    vic_idx, env_idx = all_pairs[pair_index]

    row_A = df_pure_victim.iloc[vic_idx]
    row_B = df_pure_env.iloc[env_idx]

    path_A = os.path.join(TARGET_DIR, row_A['filename'])
    path_B = os.path.join(TARGET_DIR, row_B['filename'])

    y_A = load_audio_smart(path_A, is_victim=True)
    y_B = load_audio_smart(path_B, is_victim=False)

    weight_A = random.uniform(0.7, 1.0)
    weight_B = random.uniform(0.1, 0.4)

    y_mix = (y_A * weight_A) + (y_B * weight_B)
    max_amp = np.max(np.abs(y_mix))
    if max_amp > 1.0:
      y_mix = y_mix / max_amp

    print(
        f"\n▶️ 再生中... [残り {len(all_pairs) - pair_index} ペア] (ノック: {row_A['filename']} / 環境音: {row_B['filename']})")
    play_audio(y_mix)

    while True:
      cmd = input(
          "  -> 採用しますか？ [y=採用 / n=却下 / r=再再生 / q=終了]: ").strip().lower()

      if cmd == 'r':
        print("  ▶️ もう一度再生します...")
        play_audio(y_mix)
      elif cmd == 'y':
        out_filename = get_next_filename(df)
        out_path = os.path.join(TARGET_DIR, out_filename)
        sf.write(out_path, y_mix, SR, subtype='PCM_16')

        # どのファイルを混ぜたか（親ファイル名）も一緒に記録する
        new_row = {
            'filename': out_filename, 'label': "environment,victim",
            'environment': 1, 'rescuer': 0, 'victim': 1,
            'source_victim': row_A['filename'],
            'source_environment': row_B['filename']
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        accepted_count += 1
        print(
            f"  ✅ 採用! {out_filename} として保存しました。(今回追加: {accepted_count}件)")
        pair_index += 1
        break

      elif cmd == 'n':
        print("  ❌ 却下しました。次の音声を生成します。")
        pair_index += 1
        break

      elif cmd == 'q':
        print(f"\n🎉 お疲れ様でした！今回新しく {accepted_count} 件のデータを追加しました！")
        return
      else:
        print("  ⚠️ 無効な入力です。y, n, r, q のいずれかを入力してください。")

  print("\n🏁 すべての組み合わせをチェックし終わりました！")

if __name__ == "__main__":
  main()
