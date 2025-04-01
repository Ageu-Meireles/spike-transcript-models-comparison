from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError

class SpeechmaticsAdapter:
    def __init__(self, api_key):
        self.api_key = api_key            

    def transcribe(self, audio_buffer: bytes, configs: dict):
        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=self.api_key,
        )
        with BatchClient(settings) as client:
            try:
                conf = {
                    "type": "transcription",
                    "transcription_config": {
                        "language": "pt",
                        **configs
                    }
                }
                job_id = client.submit_job(('audio.wav', audio_buffer), conf)
                return client.wait_for_completion(job_id, transcription_format='txt')
            except HTTPStatusError as e:
                if e.response.status_code == 401:
                    print('Invalid API key - Check your API_KEY at the top of the code!')
                elif e.response.status_code == 400:
                    print(e.response.json()['detail'])
                else:
                    raise e