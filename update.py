import os
import shutil
import pandas as pd

# ==========================================
# 1. 設定パラメータ
# ==========================================
CSV_FILE = "multilabel_data.csv"
RECORDING_DIR = "recording"
TARGET_DIR = "input_wavs"

def main():
  # CSVファイルの読み込み（既存のファイル名を取得して重複を防ぐ）
  if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    existing_files = set(df['filename'].tolist())
  else:
    df = pd.DataFrame(
        columns=['filename', 'label', 'environment', 'rescuer', 'victim'])
    existing_files = set()

  # 録音フォルダの確認
  if not os.path.exists(RECORDING_DIR):
    print(f"フォルダ '{RECORDING_DIR}' が見つかりません。先に録音を行ってください。")
    return

  # 追加先のフォルダを作成（念のため）
  os.makedirs(TARGET_DIR, exist_ok=True)

  # 録音フォルダ内のWAVファイルを取得（既存のファイルは除外）
  wav_files = sorted([f for f in os.listdir(
      RECORDING_DIR) if f.endswith('.wav')])
  new_files = [f for f in wav_files if f not in existing_files]

  if not new_files:
    print("✅ 新しく追加するWAVファイルはありません。（すべてCSVに登録済みです）")
    return

  print(f"📝 {len(new_files)} 件の新しいファイルが見つかりました。ラベル付けを開始します。")
  print("該当するクラスには '1' を、該当しない場合は '0'（またはそのままEnter）を入力してください。")
  print("途中で終了したい場合は 'q' を入力してEnterを押してください。\n")

  new_rows = []

  for f in new_files:
    print(f"----------------------------------------")
    print(f"🔊 ファイル: {f}")

    # 1. Victim (要救助者: ノック・タップなど)
    v_in = input(
        "  -> victim (要救助者)  [1/0, default=0, q=終了]: ").strip().lower()
    if v_in == 'q': break
    victim = 1 if v_in == '1' else 0

    # 2. Environment (環境音: 雨・ドローンなど)
    e_in = input(
        "  -> environment (環境音) [1/0, default=0, q=終了]: ").strip().lower()
    if e_in == 'q': break
    environment = 1 if e_in == '1' else 0

    # 3. Rescuer (救助者: 足音・装備品など)
    r_in = input(
        "  -> rescuer (救助者)  [1/0, default=0, q=終了]: ").strip().lower()
    if r_in == 'q': break
    rescuer = 1 if r_in == '1' else 0

    # ラベルの文字列を作成 (例: "environment,victim")
    labels = []
    if environment == 1: labels.append("environment")
    if rescuer == 1: labels.append("rescuer")
    if victim == 1: labels.append("victim")

    label_str = ",".join(labels) if labels else "none"

    # データをリストに追加
    new_rows.append({
        'filename': f,
        'label': label_str,
        'environment': environment,
        'rescuer': rescuer,
        'victim': victim
    })

    # 音声ファイルを input_wavs フォルダへコピー
    src_path = os.path.join(RECORDING_DIR, f)
    dst_path = os.path.join(TARGET_DIR, f)
    shutil.copy(src_path, dst_path)
    print(f"  [+] {TARGET_DIR} にコピーしました。")

  # CSVの更新処理
  if new_rows:
    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(CSV_FILE, index=False)
    print(f"\n🎉 完了！ {len(new_rows)} 件のデータを {CSV_FILE} に追加しました。")
    print("これで、すぐにAIの再学習 (cnn_final_model.py) を実行できます！")
  else:
    print("\n追加されたデータはありませんでした。")

if __name__ == "__main__":
  main()
