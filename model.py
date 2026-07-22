import os
import time
import threading
import collections
import queue
import serial
import serial.tools.list_ports
import numpy as np
import torch
import torch.nn as nn
import librosa
import tkinter as tk
from tkinter import messagebox
import pyaudio
from scipy.signal import butter, lfilter

# Appwrite用インポート（不要な場合はコメントアウトしてください）
try:
  from appwrite.client import Client
  from appwrite.services.databases import Databases
  HAS_APPWRITE = True
except ImportError:
  HAS_APPWRITE = False

# ==========================================
# 1. 設定 (Raspberry Pi 4 最適化版)
# ==========================================
CONFIG = {
    "sr": 16000,           # AIの学習レートに合わせ、最初から16kで録音して負荷削減
    "duration": 1.5,
    "n_mels": 64,
    "hop_length": 256,
    "n_fft": 1024,
    "update_interval": 0.1,
    "baud_rate": 921600,
    "playback_chunk": 1024,
    "preferred_port": "COM5",
    "cool_down": 1.5,
    "mv_avg_window": 3     # 移動平均の窓サイズ（3回 = 300ms分）
}

# Appwrite設定 (ダミー値を実際のプロジェクトのものに書き換えてください)
APPWRITE_CONFIG = {
    "endpoint": "https://sgp.cloud.appwrite.io/v1",
    "project_id": "YOUR_APPWRITE_PROJECT_ID",
    "db_id": "YOUR_APPWRITE_DATABASE_ID",
    "col_id": "YOUR_APPWRITE_COLLECTION_ID",
    "api_key": "YOUR_API_KEY"
}

CONFIG["max_len"] = int(CONFIG["sr"] * CONFIG["duration"])
CONFIG["max_frames"] = int(np.ceil(CONFIG["max_len"] / CONFIG["hop_length"]))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_crnn_optimized.pth")
ID_TO_LABEL = {0: "Victim", 1: "Environment", 2: "Rescuer"}

# ==========================================
# 2. モデル定義 (Attention CRNN) & 音声前処理
# ==========================================

class AttentionPooling(nn.Module):
  """ノック音など瞬発的な特徴を強調する自己注目機構"""

  def __init__(self, hidden_size):
    super().__init__()
    self.attn = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.Tanh(),
        nn.Linear(hidden_size // 2, 1)
    )

  def forward(self, x):
    weights = torch.softmax(self.attn(x), dim=1)
    return torch.sum(x * weights, dim=1)

class CRNN(nn.Module):
  """学習コードと完全に一致させた最新モデル構造"""

  def __init__(self, num_classes=3):
    super(CRNN, self).__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(
            16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(
            32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(
            64), nn.ReLU(), nn.MaxPool2d(2)
    )
    rnn_input_size = 64 * 8
    hidden_size = 64
    self.rnn = nn.GRU(rnn_input_size, hidden_size,
                      batch_first=True, bidirectional=True)
    self.pool = AttentionPooling(hidden_size * 2)
    self.fc = nn.Sequential(
        nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, num_classes)
    )

  def forward(self, x):
    x = self.conv_layers(x)
    b, c, f, t = x.size()
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    out, _ = self.rnn(x)
    x = self.pool(out)
    return self.fc(x)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low, high = lowcut / nyq, highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return lfilter(b, a, data)

def extract_features(y):
  """16kHzの音声をAI用特徴量へ変換。librosa.resampleを廃止して高速化。"""
  # フィルタ適用 (100Hz-4000Hz)
  y_filtered = bandpass_filter(y, 100.0, 4000.0, CONFIG["sr"])

  # メルスペクトログラム
  mel = librosa.feature.melspectrogram(
      y=y_filtered, sr=CONFIG["sr"], n_fft=CONFIG["n_fft"],
      hop_length=CONFIG["hop_length"], n_mels=CONFIG["n_mels"]
  )
  log_mel = librosa.power_to_db(mel, ref=np.max)

  # パディング調整
  if log_mel.shape[1] > CONFIG["max_frames"]:
    log_mel = log_mel[:, :CONFIG["max_frames"]]
  else:
    log_mel = np.pad(
        log_mel, ((0, 0), (0, CONFIG["max_frames"] - log_mel.shape[1])), mode="constant")

  # ★重要：正規化の安定化。分母に0.1を足して静寂時のノイズ増幅を抑制
  log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 0.1)
  return log_mel

# ==========================================
# 3. リーダークラス (シリアル & マイク)
# ==========================================

class BaseAudioReader:
  def __init__(self, sr):
    self.sr = sr
    self.ai_buffer = collections.deque(maxlen=sr * 2)  # バッファサイズ調整
    self.ai_buffer.extend([0.0] * (sr * 2))
    self.playback_queue = queue.Queue()
    self.is_listening = False
    self.running = False
    self.connection_state = "WAITING"

  def get_ai_input(self):
    data = np.array(list(self.ai_buffer), dtype=np.float32)
    return data[-CONFIG["max_len"]:]

class SerialAudioReader(BaseAudioReader):
  def __init__(self, sr):
    super().__init__(sr)
    self.ser = None
    self.p = pyaudio.PyAudio()
    self.stream = self.p.open(
        format=pyaudio.paInt16, channels=1, rate=sr, output=True)

  def start(self):
    self.running = True
    threading.Thread(target=self._connection_loop, daemon=True).start()
    threading.Thread(target=self._playback_loop, daemon=True).start()

  def _connection_loop(self):
    while self.running:
      if self.ser is None:
        ports = list(serial.tools.list_ports.comports())
        port = next(
            (p.device for p in ports if CONFIG["preferred_port"] in p.device), ports[0].device if ports else None)
        if port:
          try:
            self.ser = serial.Serial(
                port, CONFIG["baud_rate"], timeout=0.1)
            self.connection_state = "CONNECTED"
            self._read_loop()
          except: self.ser = None
      time.sleep(1.0)

  def _read_loop(self):
    HEADER = b'\xAA\x55'
    while self.running and self.ser:
      try:
        if self.ser.in_waiting < 258: continue
        if self.ser.read(2) != HEADER: continue
        raw = self.ser.read(256)
        block = np.frombuffer(raw, dtype='<h').astype(
            np.float32) / 32768.0
        self.ai_buffer.extend(block)
        if self.is_listening: self.playback_queue.put(block)
      except:
        self.connection_state = "DISCONNECTED"
        self.ser = None; break

  def _playback_loop(self):
    while self.running:
      try:
        data = self.playback_queue.get(timeout=0.1)
        self.stream.write((data * 16384).astype(np.int16).tobytes())
      except: pass

  def stop(self): self.running = False; self.p.terminate()

class MicAudioReader(BaseAudioReader):
  def __init__(self, sr, device_idx=None):
    super().__init__(sr)
    self.device_idx = device_idx
    self.p = pyaudio.PyAudio()
    self.out_stream = self.p.open(
        format=pyaudio.paInt16, channels=1, rate=sr, output=True)

  def start(self):
    self.running = True
    try:
      # 16kHz, Mono, Int16でオープン（ラズパイ4最適設定）
      self.in_stream = self.p.open(
          format=pyaudio.paInt16, channels=1, rate=self.sr,
          input=True, input_device_index=self.device_idx,
          stream_callback=self._callback, frames_per_buffer=1024
      )
      self.connection_state = "CONNECTED"
    except Exception as e:
      print(f"Mic Error: {e}"); self.connection_state = "DISCONNECTED"
    threading.Thread(target=self._playback_loop, daemon=True).start()

  def _callback(self, in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(
        in_data, dtype=np.int16).astype(np.float32) / 32768.0
    self.ai_buffer.extend(audio_data)
    if self.is_listening: self.playback_queue.put(audio_data)
    return (None, pyaudio.paContinue)

  def _playback_loop(self):
    while self.running:
      try:
        data = self.playback_queue.get(timeout=0.1)
        self.out_stream.write((data * 16384).astype(np.int16).tobytes())
      except: pass

  def stop(self): self.running = False; self.p.terminate()

# ==========================================
# 4. GUIアプリ (移動平均 & Appwrite対応)
# ==========================================

class App:
  def __init__(self, root, reader):
    self.root = root
    self.root.title("Rescue AI Detector v6 (Attention + Smooth)")
    self.root.geometry("520x580")
    self.is_running, self.is_inferencing = True, False
    self.last_sync_time = 0

    # ★重要：移動平均バッファ (直近3回分)
    self.prob_history = collections.deque(maxlen=CONFIG["mv_avg_window"])

    # Appwrite初期化 (ライブラリがある場合のみ)
    if HAS_APPWRITE:
      try:
        self.client = Client().set_endpoint(APPWRITE_CONFIG["endpoint"]).set_project(
            APPWRITE_CONFIG["project_id"]).set_key(APPWRITE_CONFIG["api_key"])
        self.db_service = Databases(self.client)
      except: print("Appwrite Init Failed.")

    # モデル読み込み
    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    self.model = CRNN().to(self.device)
    if os.path.exists(MODEL_PATH):
      self.model.load_state_dict(torch.load(
          MODEL_PATH, map_location=self.device, weights_only=True))
      print(f"Model loaded on {self.device}")
    self.model.eval()

    self.reader = reader; self.reader.start()

    # UI構築
    self.status = tk.Label(root, text="● WAITING",
                           fg="orange", font=("Arial", 12, "bold"))
    self.status.pack(pady=10)

    self.listen_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="🔊 Listen Mode", variable=self.listen_var,
                   command=lambda: setattr(self.reader, "is_listening", self.listen_var.get())).pack()

    self.labels, self.bars = [], []
    colors = ["red", "gray30", "green"]
    for i, name in enumerate(ID_TO_LABEL.values()):
      f = tk.Frame(root); f.pack(fill="x", padx=30, pady=10)
      l = tk.Label(f, text=f"{name}: 0%", font=(
          "Arial", 11)); l.pack(side="left")
      self.labels.append(l)
      c = tk.Canvas(f, height=15, bg="#eee"); c.pack(
          side="right", fill="x", expand=True, padx=10)
      r = c.create_rectangle(0, 0, 0, 15, fill=colors[i], width=0)
      self.bars.append((c, r))

    self.loop()

  def push_to_cloud(self):
    """Victim検知時にAppwriteへ送信"""
    if not HAS_APPWRITE: return
    try:
      self.db_service.create_document(
          database_id=APPWRITE_CONFIG["db_id"], collection_id=APPWRITE_CONFIG["col_id"],
          document_id='unique()', data={"value": 1, "timestamp": int(time.time() * 1000)}
      )
      print("Cloud: Victim Alert Sent!")
    except Exception as e: print(f"Cloud Error: {e}")

  def loop(self):
    if not self.is_running: return
    s = self.reader.connection_state
    color = "green" if s == "CONNECTED" else "red" if s == "DISCONNECTED" else "orange"
    self.status.config(text=f"● {s}", fg=color)

    audio = self.reader.get_ai_input()
    if len(audio) == CONFIG["max_len"] and not self.is_inferencing:
      # 一定以上の音量がある場合のみ推論に回す
      if np.max(np.abs(audio)) > 0.005:
        self.is_inferencing = True
        threading.Thread(target=self.run_inference,
                         args=(audio,), daemon=True).start()

    self.root.after(int(CONFIG["update_interval"] * 1000), self.loop)

  def run_inference(self, audio):
    try:
      feat = extract_features(audio)
      x = torch.tensor(feat[None, None],
                       dtype=torch.float32).to(self.device)
      with torch.no_grad():
        logits = self.model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
      self.root.after(0, self.update_gui, probs)
    finally: self.is_inferencing = False

  def update_gui(self, probs):
    """移動平均を用いて判定とGUIを更新"""
    self.prob_history.append(probs)
    avg_probs = np.mean(self.prob_history, axis=0)  # 直近3回の平均

    now = time.time()
    # ★判定は「平均スコア」で行う
    if avg_probs[0] >= 0.95 and (now - self.last_sync_time > CONFIG["cool_down"]):
      self.last_sync_time = now
      threading.Thread(target=self.push_to_cloud, daemon=True).start()

    for i in range(3):
      p = avg_probs[i]
      self.labels[i].config(text=f"{ID_TO_LABEL[i]}: {p*100:.1f}%")
      c, r = self.bars[i]
      w = c.winfo_width() if c.winfo_width() > 10 else 300
      c.coords(r, 0, 0, int(p * w), 15)

  def close(self):
    self.is_running = False; self.reader.stop(); self.root.destroy()

if __name__ == "__main__":
  print("--- Rescue AI System v6 ---")
  print("[1] Serial (ESP32)  [2] Mic (Laptop/USB)")
  mode = input("Select (1/2): ")
  reader = MicAudioReader(
      sr=CONFIG["sr"]) if mode == '2' else SerialAudioReader(sr=CONFIG["sr"])
  root = tk.Tk()
  app = App(root, reader)
  root.protocol("WM_DELETE_WINDOW", app.close)
  root.mainloop()
