from enum import StrEnum
from openai import OpenAI


class OpenAITranscriptModelsEnum(StrEnum):
    WHISPER = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


class OpenAIAdapter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def client(self):
        openai = OpenAI(api_key=self.api_key)
        return openai

    def transcribe(self, audio_file, model: OpenAITranscriptModelsEnum = "whisper-1") -> str:
        transcript = self.client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="json",
            language="pt",
            temperature=0,
        )
        return transcript.text
