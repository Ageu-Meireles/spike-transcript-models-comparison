import speech_recognition as sr


class GoogleAdapter:
    def __init__(self):
        pass
    
    @property
    def client(self):
        return sr.Recognizer()
    
    def transcribe(self, audio_buffer: bytes):
        with sr.AudioFile(audio_buffer) as audio:
            audio_file = self.client.record(audio)
            
            try:
                transcription = self.client.recognize_google(
                    audio_file, language="pt-BR", show_all=False
                )
            except sr.UnknownValueError:
                transcription = "Não foi possível entender o áudio."
            except sr.RequestError:
                transcription = "Erro ao acessar o serviço de reconhecimento de fala."
            
            return transcription
