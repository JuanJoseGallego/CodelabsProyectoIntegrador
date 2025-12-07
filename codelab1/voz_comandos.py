import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os
import wmi


SRATE = 16000     # tasa de muestreo
DUR = 5           # segundos

print("Grabando... habla ahora!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")
    cmd = texto.lower()

    if "hola" in cmd:
        print("¡Hola, bienvenido al curso!")
    elif "abrir google" in cmd:
        import webbrowser
        webbrowser.open("https://www.google.com")
    elif "hora" in cmd:
        from datetime import datetime
        print("Hora actual:", datetime.now().strftime("%H:%M"))
    elif "clima actual" in cmd:
        import webbrowser
        ciudad = "Tuluá"
        url = f"https://www.google.com/search?q=clima+{ciudad}"
        webbrowser.open(url)
    elif "temperatura cpu" in cmd:
        try:
            w = wmi.WMI(namespace="root\\wmi")
            temperatura = w.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature
            # Conversión desde décimas de Kelvin a °C
            celsius = (temperatura / 10.0) - 273.15
            print(f"Temperatura CPU: {celsius:.1f} °C")
        except Exception as e:
            print("No se pudo leer la temperatura de la CPU:", e)

        else:
            print("Comando no reconocido.")
except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

