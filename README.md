# PDF Q&A Demo

> **An interactive Streamlit app to upload one or more PDF documents and ask questions, powered by OpenAI and FAISS.**

&#x20;&#x20;

---

## 🚀 Table of Contents

- [PDF Q\&A Demo](#pdf-qa-demo)
  - [🚀 Table of Contents](#-table-of-contents)
  - [🎯 Introduction](#-introduction)
  - [✨ Features](#-features)
  - [📋 Prerequisites](#-prerequisites)
  - [🛠️ Installation](#️-installation)
  - [⚙️ Configuration](#️-configuration)
  - [▶️ Usage](#️-usage)
  - [📽 Demo Walkthrough](#-demo-walkthrough)
  - [🚄 Caching \& Performance](#-caching--performance)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgements](#-acknowledgements)

## 🎯 Introduction

The **PDF Q&A Demo** is a lightweight Streamlit application that allows users to upload multiple PDF documents, splits the text into manageable chunks, indexes them with FAISS, and streams answers to natural language questions via OpenAI's GPT-4o-mini. All operations happen in-browser (or via a lightweight server), making it ideal for demos, testing, and small-scale deployments.

## ✨ Features

- 📄 **Multiple PDF upload**: Select and process several PDF files at once.
- 🔍 **Intelligent chunking**: Splits large documents into 1 000-token chunks (with overlap) for reliable retrieval.
- ⚡️ **Vector search with FAISS**: Rapid similarity search over your document corpus.
- 🧠 **Streaming answers**: Real-time display of token-by-token responses.
- 🔧 **Debug mode**: Toggle sidebar option to view chunk counts and other internals.
- 🔒 **Env key check**: Validates that `OPENAI_API_KEY` is set before running.
- 🛠️ **Cache-enabled**: Heavy operations are cached for faster repeat runs.

## 📋 Prerequisites

- Python 3.9 or above
- An OpenAI API key with access to GPT-4o-mini embeddings & completions
- Supported OS: macOS, Linux, Windows

## 🛠️ Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/FinishYourQuiz/PDF-Question-Answering.git
   cd PDF-Question-Answering
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a **`.env`** file**

   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## ⚙️ Configuration

The app reads configuration from environment variables (via `python-dotenv`):

| Variable         | Description         |
| ---------------- | ------------------- |
| `OPENAI_API_KEY` | Your OpenAI API key |

Optional Streamlit settings can be toggled in the sidebar:

- **Show debug info**: Display chunk count and other internals.

---

## ▶️ Usage

Run the Streamlit server:

```bash
streamlit run app.py
```

1. Open the link provided by Streamlit in your browser (typically `http://localhost:8501`).
2. Upload one or more PDF files via the sidebar widget.
3. Enter your question in the text input.
4. Watch the answer stream in real time.
5. Expand **Sources & Context** to view which file each chunk came from.


## 📽 Demo Walkthrough

1. **Upload**: Select several PDF files in one go.
2. **Extract & Split**: The app extracts text, chunks it, and caches the result.
3. **Vectorize**: Chunks are embedded and stored in FAISS.
4. **Query**: Type a question and hit Enter.
5. **Stream**: See token-by-token generation of the answer.
6. **Cite**: Review source PDF filenames in the citations panel.


## 🚄 Caching & Performance

- `extract_and_split` is used for PDF extraction and splitting — avoids reprocessing the same files.
  - Optional: Adjust chunk size in `extract_and_split` to optimize retrieval speed vs. accuracy.
- `get_vector_store` caches the FAISS vector store for quick rebuilds.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add YourFeature"`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure code passes existing tests and includes new tests where applicable.


## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.


## 🙏 Acknowledgements

- Built with [Streamlit](https://streamlit.io)
- Powered by [OpenAI](https://openai.com)
- Vector search via [FAISS](https://github.com/facebookresearch/faiss)
- PDF parsing with [PyPDF2](https://github.com/mstamy2/PyPDF2)


*Happy PDF Q&Aing!*

