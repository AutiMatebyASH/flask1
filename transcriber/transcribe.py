# import speech_recognition as sr

# def transcribe_audio_file(file_path):
#     """
#     Transcribes the given .wav file using Google Speech Recognition API.
#     :param file_path: Path to the .wav file
#     :return: Transcribed text
#     """
#     recognizer = sr.Recognizer()

#     with sr.AudioFile(file_path) as source:
#         print("Loading audio file...")
#         audio_data = recognizer.record(source)  # Load the audio file

#     print("Transcribing audio...")
#     transcription = recognizer.recognize_google(audio_data)  # Transcribe audio
#     return transcription
import speech_recognition as sr

def transcribe_audio_file(file_path):
    """
    Transcribes the given .wav file using Google Speech Recognition API.
    :param file_path: Path to the .wav file
    :return: Transcribed text
    """
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        print("Loading audio file...")
        audio_data = recognizer.record(source)  # Load the audio file

    print("Transcribing audio...")
    transcription = recognizer.recognize_google(audio_data)  # Transcribe audio
    return transcription
