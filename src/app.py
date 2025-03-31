import io
import speech_recognition as sr
import streamlit as st

from deepgram import DeepgramClient, FileSource, PrerecordedOptions, PrerecordedResponse
from openai import OpenAI

from similarity import Similarity

st.title('Assertividade entre modelos')

open_ai_api_key = st.text_input(
    "Chave da API do OpenAI", type="password"
)
deepgram_api_key = st.text_input(
    "Chave da API do Deepgram", type="password"
)
audio_value = st.audio_input("Grave o áudio que será transcrito")
uploaded_file = st.file_uploader("Carregue um audio em formato wav")
transcription_text = st.text_input('Transcrição esperada:')

if uploaded_file:
    if not uploaded_file.name.endswith(".wav"):
        st.error("O arquivo selecionado não é um arquivo wav.")
    else:
        st.audio(uploaded_file)

if "options" not in st.session_state:
    st.session_state.options = []

selected_vocabulary = st.multiselect(
    "Vocabulário:", max_selections=100, options=st.session_state.options
)

new_vocabulary = st.text_input("Adicione novas palavras:")

if new_vocabulary:
    if new_vocabulary not in st.session_state.options:
        st.session_state.options.append(new_vocabulary)
        st.rerun()

if (
    (audio_file := audio_value or uploaded_file)
    and open_ai_api_key
    and deepgram_api_key
):
    audio_bytes = io.BytesIO(audio_file.read())

    if not audio_file.name.endswith(".wav"):
        st.error("O arquivo selecionado não é um arquivo wav.")

    similarity = Similarity()

    client = OpenAI(api_key=open_ai_api_key)

    f'Transcrição esperada: {transcription_text}'

    st.header('WHISPER:', divider=True)

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="json",
        language="pt",
        temperature=0,
    )

    st.write(f'transcription: {transcription.text}')
    similarity.get_similarities(
        'WHISPER', transcription_text, transcription.text
    )

    st.header('GPT-4O-MINI-TRANSCRIBE:', divider=True)

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="gpt-4o-mini-transcribe",
        response_format="json",
        language="pt",
        prompt=None,
        temperature=0,
    )

    st.write(f'transcription: {transcription.text}')
    similarity.get_similarities(
        'GPT-4O-MINI-TRANSCRIBE', transcription_text, transcription.text
    )

    st.header('GPT-4O-TRANSCRIBE', divider=True)

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="gpt-4o-transcribe",
        response_format="json",
        language="pt",
        prompt=None,
        temperature=0,
    )

    st.write(f'transcription: {transcription.text}')

    similarity.get_similarities(
        'GPT-4O-TRANSCRIBE', transcription_text, transcription.text
    )

    st.header('Google Speech Recognition:', divider=True)

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_bytes) as teste:
        audio = recognizer.record(teste)

    try:
        transcription = recognizer.recognize_google(
            audio, language="pt-BR", show_all=False
        )
    except sr.UnknownValueError:
        transcription = "Não foi possível entender o áudio."
    except sr.RequestError:
        transcription = "Erro ao acessar o serviço de reconhecimento de fala."

    st.write(f'transcription: {transcription}')

    similarity.get_similarities(
        'Google Speech Recognition', transcription_text, transcription
    )

    st.header('Deepgram', divider=True)

    options = PrerecordedOptions(
        model="nova-2",
        language="pt",
        smart_format=True,
    )

    payload: FileSource = {
        "buffer": audio_bytes.getvalue(),
    }

    deepgram = DeepgramClient(
        api_key=deepgram_api_key
    )
    client = deepgram.listen.rest.v("1")
    response: PrerecordedResponse = client.transcribe_file(payload, options)
    transcription = response.results.channels[0].alternatives[0].transcript

    similarity.get_similarities(
        'Deepgram', transcription_text, transcription
    )

    st.write(f'transcription: {transcription}')

    if selected_vocabulary:
        st.header('Deepgram with keywords', divider=True)

        options = PrerecordedOptions(
            model="nova-2",
            language="pt",
            smart_format=True,
            keywords=selected_vocabulary
        )

        payload: FileSource = {
            "buffer": audio_bytes.getvalue(),
        }

        response: PrerecordedResponse = client.transcribe_file(
            payload, options)
        transcription = response.results.channels[0].alternatives[0].transcript

        st.write(f'transcription: {transcription}')

        similarity.get_similarities(
            'Deepgram with keywords', transcription_text, transcription
        )

    st.header('Similaridade de Levenshtein', divider=True)
    st.bar_chart(
        data=similarity.levenshtein,
        x_label="Modelo de transcrição",
        y_label="% de similaridade",
    )

    st.header('Similaridade de Jaccard', divider=True)
    st.bar_chart(
        data=similarity.jaccard,
        x_label="Modelo de transcrição",
        y_label="% de similaridade",
    )

    st.header('Similaridade de TF-IDF', divider=True)
    st.bar_chart(
        data=similarity.tfidf,
        x_label="Modelo de transcrição",
        y_label="% de similaridade",
    )
