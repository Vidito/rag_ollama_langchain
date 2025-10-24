import streamlit as st
import rag

st.title("Retrieval-Augmented Generation (RAG) Web Scraper")

url = st.text_input("Enter the URL to scrape:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if url and question:
        with st.spinner("Processing..."):
            try:
                answer = rag.process_url_and_question(url, question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter both a URL and a question.")
