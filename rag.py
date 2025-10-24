import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

def process_url_and_question(url, question):
    """
    This function takes a URL and a question, scrapes the URL,
    and uses a RAG model to answer the question based on the scraped content.
    """
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    local_embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    context = ' '.join([doc.page_content for doc in retrieved_docs])

    llm = OllamaLLM(model="llama3.2:1b")
    response = llm.invoke(f"""Answer the question according to the context given very briefly:
               Question: {question}.
               Context: {context}
    """)
    return response

if __name__ == '__main__':
    # This block is for testing the function directly
    test_url = "https://pythonology.eu/using-pandas_ta-to-generate-technical-indicators-and-signals/"
    test_question = "what are the oversold and overbought periods?"
    answer = process_url_and_question(test_url, test_question)
    print(answer)
