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
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

def main():
    # 1. Check for OpenAI API key in environment
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
      st.error("Missing environment variable: OPENAI_API_KEY. Please set it before running the app.")
      st.stop()
           
    # 2. Set up Streamlit app
    st.set_page_config(page_title="Ask your PDFs", layout="wide")
    st.header("📄💬 Upload PDFs and Ask Questions")
    
    # 3. Allow multiple PDF uploads
    uploads = st.file_uploader(
      label="Upload one or more PDF files", type="pdf",
      accept_multiple_files=True
    )
    
    # 4. Process uploaded PDFs
    if uploads:
      # pdf_texts = [] 
      text = ""
      for uploaded in uploads:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
          tmp.write(uploaded.read())
          path = tmp.name 
          loader = PdfReader(path) 
          for page in loader.pages:
            if page.extract_text():
              # pdf_texts.append(page.extract_text())
              text += page.extract_text() + "\n"

      # 5. Split text into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text) 
      
      # 6. Create embeddings and knowledge base
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # 7. Answer questions using the knowledge base
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)

        # 8. Set up streaming LLM with callback
        output_container = st.empty()
        
        class StreamlitCallbackHandler(BaseCallbackHandler):
            def __init__(self):
                self.response = ""
            def on_llm_new_token(self, token: str, **kwargs):
                self.response += token
                output_container.markdown(self.response)

        callback_manager = CallbackManager([StreamlitCallbackHandler()])

        llm = OpenAI(
            model="gpt-4o-mini",
            streaming=True,
            callback_manager=callback_manager
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
          
if __name__ == '__main__':
    main()