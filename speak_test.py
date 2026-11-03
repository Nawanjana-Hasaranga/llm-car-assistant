import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)   # speaking speed in words per minute (default is about 200)

engine.say("I've set the temperature to 23 degrees to help you warm up.")
engine.runAndWait()

print("Done speaking.")