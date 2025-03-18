# Install necessary libraries
# pip install requests beautifulsoup4 langchain-community langchain-text-splitters langchain-chroma langchain-ollama

# Import required libraries
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

# Define a SoupStrainer to parse only specific parts of the webpage
bs4_strainer = bs4.SoupStrainer(class_=("content-area"))

# Initialize the WebBaseLoader with the target URL and the SoupStrainer
loader = WebBaseLoader(
    web_paths=("https://pythonology.eu/using-pandas_ta-to-generate-technical-indicators-and-signals/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

# Load the documents from the webpage
docs = loader.load()

# Initialize the RecursiveCharacterTextSplitter to split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Size of each chunk
    chunk_overlap=100,  # Overlap between chunks
    add_start_index=True  # Add start index to each chunk
)

# Split the documents into smaller chunks
all_splits = text_splitter.split_documents(docs)

# Download Ollama on your machine from their official website.
# Then in your terminal use these commands to download a language model and an embedding model:
# ollama run llama3.2:1b
# ollama pull all-minilm

# Initialize OllamaEmbeddings with the specified model
local_embeddings = OllamaEmbeddings(model="all-minilm")

# Create a Chroma vector store from the document splits using the local embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# Define the question to be answered
question = "what are the oversold and overbought periods?"

# Create a retriever from the vector store with similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Retrieve the most relevant documents based on the question
retrieved_docs = retriever.invoke(question)

# Combine the content of the retrieved documents into a single context string
context = ' '.join([doc.page_content for doc in retrieved_docs])

# Initialize the OllamaLLM with the specified model
llm = OllamaLLM(model="llama3.2:1b")

# Generate a response to the question using the retrieved context
response = llm.invoke(f"""Answer the question according to the context given very briefly:
           Question: {question}.
           Context: {context}
""")

# Print the generated response
print(response)
