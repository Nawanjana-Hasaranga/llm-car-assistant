import time
import numpy as np
import sounddevice as sd
import pyttsx3
import ollama
from faster_whisper import WhisperModel
from agent import ask_assistant, SYSTEM_PROMPT, MODEL

SAMPLE_RATE = 16000
SECONDS = 5            # recording length
MIN_LOUDNESS = 0.02    # below this = silence, so we ignore it
WHISPER_MODEL = "base.en"

# Words a driver is likely to say: helps Whisper choose the right word
CAR_VOCABULARY = (
    "In-car voice commands: I'm freezing, it's boiling in here, I'm a bit chilly, "
    "warmer, cooler, set the temperature to twenty degrees, seat heating, "
    "make my seat really warm, open the windows, close the windows, "
    "battery level, range, how far can I drive."
)

# ---------- Startup: load both models before the driver speaks ----------
print("Loading Whisper model...")
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

print("Warming up the LLM...")
start = time.time()
ollama.generate(model=MODEL, prompt="", keep_alive="30m")
print(f"LLM ready ({time.time() - start:.1f} s)")


# ---------- Ears: microphone -> text ----------
def listen():
    print("Recording... speak now!")
    audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    loudness = float(np.abs(audio).max())
    if loudness < MIN_LOUDNESS:
        return ""  # silence: don't let Whisper invent words

    segments, info = whisper.transcribe(
        audio.flatten(),
        language="en",
        vad_filter=True,
        initial_prompt=CAR_VOCABULARY,
    )
    return " ".join(segment.text for segment in segments).strip()


# ---------- Mouth: text -> speech ----------
def speak(text):
    engine = pyttsx3.init()  # new engine each time avoids a Windows bug where it goes silent after the first sentence
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()


# ---------- The voice assistant ----------
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
print("Voice car assistant ready.")

while True:
    command = input("\nPress Enter to talk (or type 'quit'): ")
    if command.lower() == "quit":
        break

    t_start = time.time()
    text = listen()
    t_heard = time.time()

    if not text:
        print("(I heard nothing. Speak louder or closer to the microphone.)")
        continue

    print("You said:", text)
    answer = ask_assistant(text, messages)
    t_answered = time.time()

    print("Car:", answer)
    print(f"[timing] speech-to-text: {t_heard - t_start - SECONDS:.1f} s | LLM + tools: {t_answered - t_heard:.1f} s")
    speak(answer)