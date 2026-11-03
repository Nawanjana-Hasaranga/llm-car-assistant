import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000   # Whisper expects 16,000 sound samples per second
SECONDS = 5           # how long to record

print("Using microphone:", sd.query_devices(kind="input")["name"])

print("Loading Whisper model...")
model = WhisperModel("base.en", device="cpu", compute_type="int8")

input("Press Enter, then speak for 5 seconds...")
print("Recording... speak now!")
audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()

loudness = float(np.abs(audio).max())
print(f"Loudness: {loudness:.3f}   (silence: below 0.01, normal speech: 0.05 to 0.5)")

print("Transcribing...")
start = time.time()
segments, info = model.transcribe(audio.flatten(), language="en")
text = " ".join(segment.text for segment in segments).strip()
seconds = time.time() - start

print("You said:", text)
print(f"Transcription time: {seconds:.1f} s")