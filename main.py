from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
import os, tempfile

def main():
    # 1. Check for OpenAI API key in environment
    if "OPENAI_API_KEY" not in os.environ:
      st.error("Missing environment variable: OPENAI_API_KEY. Please set it before running the app.")
      st.stop()
      
    # 2. Set up Streamlit app
    load_dotenv()
    st.set_page_config(page_title="Ask your PDFs", layout="wide")
    st.header("📄💬 Upload PDFs and Ask Questions")
    
    # 3. Allow multiple PDF uploads
    uploads = st.file_uploader(
      label="Upload one or more PDF files", type="pdf",
      accept_multiple_files=True
    )
    question = st.text_input("Ask a question about the uploaded PDFs:")
    # extract the text
    if uploads is not None:
      pdf_texts = []
      for pdf in uploads:
          pdf_reader = PdfReader(pdf)
          text = ""
          for page in pdf_reader.pages:
      text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI(
          model="gpt-4o-mini",
          temperature=0.2,
          max_tokens=1000,
          top_p=1,
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()