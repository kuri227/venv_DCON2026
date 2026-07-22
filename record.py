import os
import re
import wave
import threading
import pyaudio

# ==========================================
# 1. 設定パラメータ（学習モデルに完全適合）
# ==========================================
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # モノラル
RATE = 16000              # 16kHz (CNNモデルの sr=16000 に一致)
CHUNK = 1024
SAVE_DIR = "recording"
FILE_PREFIX = "タップ音"
MAX_RECORD_SECONDS = 10   # 最大録音時間 (10秒)

# ==========================================
# 2. 次のファイル番号を取得する関数
# ==========================================
def get_next_file_id():
  os.makedirs(SAVE_DIR, exist_ok=True)
  files = os.listdir(SAVE_DIR)

  # "タップ音X.wav" または "タップ音00X.wav" の数字部分を抽出
  pattern = re.compile(rf"^{FILE_PREFIX}(\d+)\.wav$")
  max_id = 0

  for f in files:
    match = pattern.match(f)
    if match:
      current_id = int(match.group(1))
      if current_id > max_id:
        max_id = current_id

  return max_id + 1

# ==========================================
# 3. マイクデバイス選択関数
# ==========================================
def select_audio_device(p):
  print("\n--- 利用可能なマイクデバイス ---")
  info = p.get_host_api_info_by_index(0)
  numdevices = info.get('deviceCount')

  valid_devices = []
  for i in range(0, numdevices):
    device_info = p.get_device_info_by_host_api_device_index(0, i)
    if device_info.get('maxInputChannels') > 0:
      print(f"[{i}] {device_info.get('name')}")
      valid_devices.append(i)
  print("--------------------------------")

  while True:
    try:
      device_idx = int(input("使用するマイクの番号を入力してください: "))
      if device_idx in valid_devices:
        return device_idx
      else:
        print("無効な番号です。リストにある番号を入力してください。")
    except ValueError:
      print("数字を入力してください。")

# ==========================================
# 4. メインルーチン
# ==========================================
def main():
  p = pyaudio.PyAudio()
  device_idx = select_audio_device(p)

  print("\n====================================================")
  print(f"設定完了: 録音デバイス [{device_idx}]")
  print(f"保存先: ./{SAVE_DIR}/")
  print("====================================================\n")

  while True:
    cmd = input("【Enterキー】を押して録音を開始します。(終了する場合は 'q' と入力してEnter): ")
    if cmd.lower() == 'q':
      print("録音プログラムを終了します。")
      break

    # ★ ここが3桁ゼロ埋めの変更点です ★
    next_id = get_next_file_id()
    filename = f"{FILE_PREFIX}{next_id:03d}.wav"
    filepath = os.path.join(SAVE_DIR, filename)

    # 録音制御用のフラグ
    stop_recording = False

    def wait_for_stop():
      nonlocal stop_recording
      input()
      stop_recording = True

    # ストリームを開く
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_idx,
                    frames_per_buffer=CHUNK)

    print(f"\n🔴 録音中... (最大 {MAX_RECORD_SECONDS} 秒)")
    print("停止するには再度【Enterキー】を押してください。")

    # 停止待ちのスレッドを開始
    stop_thread = threading.Thread(target=wait_for_stop)
    stop_thread.daemon = True
    stop_thread.start()

    frames = []
    max_chunks = int(RATE / CHUNK * MAX_RECORD_SECONDS)

    for _ in range(max_chunks):
      if stop_recording:
        break
      try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
      except IOError as e:
        print(f"録音中にエラーが発生しました: {e}")
        break

    # 録音終了処理
    stream.stop_stream()
    stream.close()

    # もし自動停止（10秒経過）した場合、スレッド側のEnter待ちを解消するための案内
    if not stop_recording:
      print(f"\n⏳ 最大録音時間({MAX_RECORD_SECONDS}秒)に達したため、自動停止しました。")
      print("次へ進むために、もう一度【Enterキー】を押してください。")
      stop_thread.join()

    # ファイルへ書き出し
    if len(frames) > 0:
      with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
      print(f"✅ 保存完了: {filepath}\n")
    else:
      print("⚠️ データが取得できませんでした。やり直してください。\n")

  p.terminate()

if __name__ == "__main__":
  main()
