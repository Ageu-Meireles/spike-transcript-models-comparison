from deepgram import (
    DeepgramClient, FileSource, PrerecordedOptions, PrerecordedResponse
)


class DeepgramAdapter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def client(self):
        deepgram = DeepgramClient(
            api_key=self.api_key
        )
        return deepgram.listen.rest.v("1")

    def transcribe(self, audio_buffer: bytes, options: PrerecordedOptions):
        payload: FileSource = {
            "buffer": audio_buffer,
        }
        response: PrerecordedResponse = self.client.transcribe_file(
            payload, options
        )
        return response.results.channels[0].alternatives[0].transcript
