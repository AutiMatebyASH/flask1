# import speech_recognition as sr
# import pyttsx3

# # Initialize the recognizer
# r = sr.Recognizer()

# def record_text():
#     # Loop in case of errors
#     while(1):
#         try:
#             # Use the microphone as source for input
#             with sr.Microphone() as source2:
#                 # Prepare recognizer to receive input
#                 r.adjust_for_ambient_noise(source2, duration=0.2)
                
#                 # Listens for the user's input
#                 audio2 = r.listen(source2)
                
#                 # Using Google to recognize audio

#                 MyText = r.recognize_google(audio2)
#                 return MyText
                
#         except sr.RequestError as e:
#             print("Could not request results; {0}".format(e))

#         except sr.RequestError as e:
#             print("Could not request results; {0}".format(e))

#         except sr.UnknownValueError:
#             print("Unknown error occurred")

#         return

#     def output_text(text):
#         f = open("output.txt", "a")
#         f.write(text)
#         f.write("\n")
#         f.close()
#         return

#     while(1):
#         text = record_text()
#         output_text(text)

#         print("Wrote text")
import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

def transcribe_audio_file(file_path):
    try:
        # Load the audio file
        with sr.AudioFile(file_path) as source:
            print("Loading audio file...")
            audio = r.record(source)  # Read the entire audio file
        
        # Transcribe the audio using Google's speech recognition API
        print("Transcribing audio...")
        transcription = r.recognize_google(audio)
        print("Transcription complete.")
        return transcription
    
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # Path to your .wav file
    audio_file_path = "example.wav"  # Replace with the path to your .wav file
    transcription = transcribe_audio_file(audio_file_path)
    
    if transcription:
        print("\nTranscribed Text:")
        print(transcription)
        # Optionally save to a file
        with open("transcription.txt", "w") as f:
            f.write(transcription)
