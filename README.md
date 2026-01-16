# 📚 Gemini Long-Context Chat (DocuChat AI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-8E75B2?logo=google&logoColor=white)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

## 📌 Overview
**DocuChat AI** is a streamlined document analysis tool that leverages the massive **Context Window** of Google's Gemini 2.5 Flash model. 

Unlike traditional RAG (Retrieval-Augmented Generation) systems that chunk documents into small pieces, this application takes a **"Full Context" approach**. It feeds the *entire* content of your uploaded documents (PDFs, TXT, MD) directly into the model's memory for every query.

**Why this approach?**
* **Zero Information Loss:** No risk of missing context due to poor chunking or retrieval algorithms.
* **Global Reasoning:** The AI understands the document as a whole, connecting information from Page 1 to Page 100.
* **Simplicity:** No Vector Databases (Pinecone/Chroma) or Embedding models required.

## ✨ Key Features

### 🧠 Full Context "Brute Force"
Leveraging Gemini's ability to handle millions of tokens, we send the **entire dataset** + **conversation history** with every prompt. This ensures the highest possible accuracy for complex queries involving multiple parts of a document.

### 📂 Multi-Format Support
* **PDFs:** Processed as binary streams for accurate parsing.
* **Text & Markdown:** Native handling with clear metadata tagging (`=== START OF FILE ===`) to help the AI distinguish sources.

### 📊 Token Awareness
Since this method is token-heavy, the app includes a built-in **Token Counter**. It transparently displays exactly how many tokens are being sent to Google's API for every message, helping you monitor usage.

### ⚡ Professional UX
* **Real-time Streaming:** Responses are streamed character-by-character.
* **Status Indicators:** Visual feedback for every background process (Parsing -> Synchronizing Memory -> Token Calculation -> Generating).
* **Safety Rails:** Configured with `BLOCK_ONLY_HIGH` thresholds to ensure valid documents aren't blocked by over-sensitive filters.

## 🛠️ Tech Stack
* **Engine:** Google Gemini 2.5 Flash (via `google-genai` SDK).
* **Interface:** Streamlit.
* **Language:** Python 3.9+.

## ⚠️ Limitations (The "Cost" of Full Context)
While this method provides superior accuracy, it comes with trade-offs:
1.  **Token Usage:** Sending entire documents repeatedly consumes a high number of tokens.
2.  **Latency:** Response times may increase linearly with the size of uploaded documents.
3.  **Best Use Case:** Ideal for specific tasks (analyzing 1-5 papers/contracts) rather than querying massive libraries (1000+ books).

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/viochris/Gemini-Long-Context-Chat.git](https://github.com/viochris/Gemini-Long-Context-Chat.git)
    cd Gemini-Long-Context-Chat
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 🚀 Usage Guide

1.  **Get API Key:** Obtain your free API Key from [Google AI Studio](https://aistudio.google.com/).
2.  **Configure:** Enter the key in the sidebar.
3.  **Upload:** Drag and drop your PDF, TXT, or MD files.
4.  **Chat:** Ask questions like *"Summarize the methodology in the PDF"* or *"Compare the conclusions of these two text files."*

## 📷 Gallery

### The Interface
![UI Screenshot](assets/screenshot_ui.png)
*Clean interface with sidebar file management and chat history.*

### Streaming & Status
![Streaming Screenshot](assets/screenshot_stream.png)
*Real-time token counting and step-by-step status updates.*

---
**Author:** [Silvio Christian]
*"Sometimes, the best retrieval strategy is to just read everything."*