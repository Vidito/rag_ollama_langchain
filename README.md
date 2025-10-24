# Retrieval-Augmented Generation (RAG) Web Scraper

This is a web application that uses a Retrieval-Augmented Generation (RAG) model to answer questions based on the content of a scraped webpage. It's built with Streamlit, LangChain, Ollama, and Chroma.

## How it works

The application takes a URL and a question from the user. It then:

1.  Scrapes the content of the provided URL.
2.  Splits the content into smaller chunks.
3.  Creates a vector store from the chunks using Ollama embeddings.
4.  Retrieves the most relevant chunks based on the user's question.
5.  Uses a large language model to generate an answer based on the retrieved content.

## Setup and Usage

1.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Ollama and the necessary models:**

    - Download and install Ollama from the [official website](https://ollama.ai/).
    - In your terminal, run the following commands to download the language and embedding models:

      ```bash
      ollama run llama3.2:1b
      ollama pull all-minilm
      ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

4.  **Use the application:**

    - Open the provided URL in your browser.
    - Enter the URL of the webpage you want to scrape.
    - Enter your question about the content of the webpage.
    - Click the "Get Answer" button to see the result.
