import io
import streamlit as st
import time

from deepgram import PrerecordedOptions

from adapters.openai import OpenAIAdapter, OpenAITranscriptModelsEnum
from adapters.deepgram import DeepgramAdapter
from adapters.google import GoogleAdapter
from adapters.speechmatics import SpeechmaticsAdapter

from similarity import Similarity

st.title("Assertividade entre modelos")

open_ai_api_key = st.text_input("Chave da API da OpenAI", type="password")
deepgram_api_key = st.text_input("Chave da API da Deepgram", type="password")
speechmatics_api_key = ""  # st.text_input(
#     "Chave da API da Speechmatics", type="password"
# )
audio_value = st.audio_input("Grave o áudio que será transcrito")
uploaded_file = st.file_uploader("Carregue um audio em formato wav")
transcription_text = st.text_input("Transcrição esperada:")

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

if audio_file := audio_value or uploaded_file:
    audio_bytes = io.BytesIO(audio_file.read())

    if not audio_file.name.endswith(".wav"):
        st.error("O arquivo selecionado não é um arquivo wav.")

    response_time = {}
    similarity = Similarity()

    f"Transcrição esperada: {transcription_text}"

    if open_ai_api_key:
        st.header("WHISPER:", divider=True)

        openai_adapter = OpenAIAdapter(open_ai_api_key)

        start = time.perf_counter()
        transcript = openai_adapter.transcribe(
            audio_file, model=OpenAITranscriptModelsEnum.WHISPER
        )
        end = time.perf_counter()

        response_time["WHISPER"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities("WHISPER", transcription_text, transcript)

        st.header("GPT-4O-MINI-TRANSCRIBE:", divider=True)

        start = time.perf_counter()
        transcript = openai_adapter.transcribe(
            audio_file, model=OpenAITranscriptModelsEnum.GPT_4O_MINI_TRANSCRIBE
        )
        end = time.perf_counter()

        response_time["GPT-4O-MINI-TRANSCRIBE"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities(
            "GPT-4O-MINI-TRANSCRIBE", transcription_text, transcript
        )

        st.header("GPT-4O-TRANSCRIBE", divider=True)

        start = time.perf_counter()
        transcript = openai_adapter.transcribe(
            audio_file, model=OpenAITranscriptModelsEnum.GPT_4O_TRANSCRIBE
        )
        end = time.perf_counter()

        response_time["GPT-4O-TRANSCRIBE"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities("GPT-4O-TRANSCRIBE", transcription_text, transcript)

    st.header("Google Speech Recognition:", divider=True)

    google = GoogleAdapter()

    start = time.perf_counter()
    transcript = google.transcribe(audio_bytes)
    end = time.perf_counter()

    response_time["Google Speech Recognition"] = end - start

    st.write(f"Tempo de processamento: {end - start:.2f} segundos")
    st.write(transcript)

    similarity.get_similarities(
        "Google Speech Recognition", transcription_text, transcript
    )

    if deepgram_api_key:
        st.header("Deepgram", divider=True)

        options = PrerecordedOptions(
            model="nova-2",
            language="pt",
            smart_format=True,
        )

        deepgram = DeepgramAdapter(deepgram_api_key)

        start = time.perf_counter()
        transcript = deepgram.transcribe(audio_bytes.getvalue(), options)
        end = time.perf_counter()

        response_time["Deepgram"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities("Deepgram", transcription_text, transcript)

        st.header("Deepgram with vocabulary", divider=True)

        options = PrerecordedOptions(
            model="nova-2",
            language="pt",
            smart_format=True,
            keywords=selected_vocabulary,
        )

        start = time.perf_counter()
        transcript = deepgram.transcribe(audio_bytes.getvalue(), options)
        end = time.perf_counter()

        response_time["Deepgram with vocabulary"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities(
            "Deepgram with vocabulary", transcription_text, transcript
        )

    if speechmatics_api_key:
        st.header("Speechmatics", divider=True)

        conf = {"type": "transcription", "transcription_config": {"language": "pt"}}

        speechmatics = SpeechmaticsAdapter(api_key=speechmatics_api_key)

        start = time.perf_counter()
        transcript = speechmatics.transcribe(audio_bytes.getvalue(), {})
        end = time.perf_counter()

        response_time["Speechmatics"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities("Speechmatics", transcription_text, transcript)

        st.header("Speechmatics with vocabulary", divider=True)

        start = time.perf_counter()
        transcript = speechmatics.transcribe(
            audio_bytes.getvalue(),
            {
                "additional_vocab": [
                    {
                        "content": vocabulary,
                    }
                    for vocabulary in selected_vocabulary
                ]
            },
        )
        end = time.perf_counter()

        response_time["Speechmatics with vocabulary"] = end - start

        st.write(f"Tempo de processamento: {end - start:.2f} segundos")
        st.write(transcript)

        similarity.get_similarities(
            "Speechmatics with vocabulary", transcription_text, transcript
        )

    st.header("Similaridade de Levenshtein", divider=True)
    st.bar_chart(
        data=similarity.levenshtein,
        x_label="Modelo de transcrição",
        y_label="% de similaridade",
    )

    st.header("Similaridade de Jaccard", divider=True)
    st.bar_chart(
        data=similarity.jaccard,
        x_label="Modelo de transcrição",
        y_label="% de similaridade",
    )

    st.header("Similaridade de TF-IDF", divider=True)
    st.bar_chart(
        data=similarity.tfidf,
        x_label="Modelo de transcrição",
        y_label="% de similaridade",
    )

    st.header("Tempo de resposta", divider=True)
    st.bar_chart(
        data=response_time,
        x_label="Modelo de transcrição",
        y_label="Tempo de resposta (segundos)",
    )
