# Spike-transcript-models-comparison

Aplicação simples, construída usando Streamlit, para comparar modelos de IA para transcrição de audios pré-gravados e obter métricas de assertividade com base no resultado esperado.

# Instalação

```bash
pip install poetry
poetry install
```

# Execução

```bash
make run-dev
```

# Usando o app

Inicialmente é necessário preencher as chaves de API necessárias (openAI e
deepgram). Depois, basta carregar um arquivo wav ou gravar um audio e aguardar a
geração do resultado. Adicionalmente, caso queira incluir nos resultados a
funcionalidade de vocabulário para o deepgram, basta incluir palavras no campo
de vocabulario, depois selecionar as que devem ser consideradas na execução,
então será possível visualizar duas barras para o deepgram em cada gráfico de
percentual de similaridade.
