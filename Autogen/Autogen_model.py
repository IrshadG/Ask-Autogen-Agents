
#conda activate POP
#streamlit run Langchain/Autogen_model.py
from keys import HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY
import streamlit as st
import pandas as pd
import numpy as np
import autogen
import autogen.retrieve_utils as utils
import chromadb
import os
from io import StringIO
from PyPDF2 import PdfReader 

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

class AutoGen_Model():
    def __init__(self):
        self.config_list = [
            {
                'model': 'gpt-3.5-turbo',
                'api_key': OPENAI_API_KEY,
            },  # OpenAI API endpoint for gpt-3.5-turbo
            {
                'model': 'gpt-3.5-turbo-16k',
                'api_key': OPENAI_API_KEY,
            },  # OpenAI API endpoint for gpt-3.5-turbo-16k
            {
                'model': 'gpt-4',
                'api_key': OPENAI_API_KEY,
            },  # OpenAI API endpoint for gpt-4
        ]
        self.conversation = ""
        # autogen.ChatCompletion.start_logging()
    
    def split_files_to_text(self,file):
        """Split a list of files into text."""
        
        _, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if file_extension == ".pdf":
            text = utils.extract_text_from_pdf(file)
        else:  # For non-PDF text-based files
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        return text

    def read_pdf(self, file):
        reader = PdfReader(file) 
        cont=""

        page = reader.pages[0] 
        for page in reader.pages:
            cont = cont +  page.extract_text() 
        cont = cont.replace("\n", " ")
        return cont

    def create_agents(self, cntx="", url=""):
        
        self.assistant = None
        self.ragproxyagent = None
        #create an RetrieveAssistantAgent instance named "assistant"
        self.assistant = RetrieveAssistantAgent(
            name="assistant", 
            system_message="You are a helpful assistant.",
            llm_config={
                "request_timeout": 600,
                "seed": 42,
                "config_list": self.config_list,
            },
        )
        #create an UserProxyAgent instance named "ragproxyagent"
        self.ragproxyagent = RetrieveUserProxyAgent(
            name="ragproxyagent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            retrieve_config={
                "task": "code",
                "docs_path": url,
                "chunk_token_size": 500,
                "model": self.config_list[0]["model"],
                "client": chromadb.PersistentClient(path="/tmp/test_retrieve_utils_chromadb.db"),
                "embedding_model": "all-mpnet-base-v2",
                "extracted_text":  cntx,
                "update_context": True,
            },
        )
        # st.write("\ncntx: "+cntx+"\n")

    def solve(self, problem):
        # reset the assistant. Always reset the assistant before starting a new conversation.
        self.assistant.reset()   
        self.ragproxyagent.reset()     
        self.ragproxyagent.initiate_chat(self.assistant, problem=problem, clear_history=True)
        last_messages = self.ragproxyagent.last_message(self.assistant)["content"]
        return last_messages


############################ GET INPUT ##############################
st.title('Apparel-AI Advanced Chatbot')
st.text('(Optional) Browse a document to query from.')
# st.text('(.pdf, .csv, .txt, .json, .tsv, .html, .xml, .yaml, .yml)')

upload_type = st.radio(
    "File upload method:",
    ["Browse File", "Paste URL"],
    captions = ["Upload a document", "Url starting from 'https://'"])

if upload_type == 'Browse File':
    uploaded_file =st.file_uploader("Choose a document", accept_multiple_files=False, type=['pdf', 'csv','txt', 'json', 'tsv', 'md', 'html', 'log', 'xml', 'yaml', 'yml'])
    st.divider()
    problem = st.text_input('Enter your query')
    if(uploaded_file and problem):
        model = AutoGen_Model()
        cntx = model.read_pdf(uploaded_file)
        model.create_agents(cntx)
        
        if st.button('Submit'):
            output = model.solve(problem)
            st.write(output)

            db_path = "/tmp/test_retrieve_utils_chromadb.db"
            if os.path.exists(db_path):
                print('Deleting temp files...')
                os.remove(db_path)  # Delete the database file after tests are finished

else:
    url = st.text_input('Enter a file url')
    st.divider()
    problem = st.text_input('Enter your query')
    if(url and problem):
        model = AutoGen_Model()
        model.create_agents(url=url)

        if st.button('Submit'):
            st.write(url)
            output = model.solve(problem + ".Only Refer " + url + " as context")
            st.write(output)

            db_path = "/tmp/test_retrieve_utils_chromadb.db"
            if os.path.exists(db_path):
                print('Deleting temp files...')
                os.remove(db_path)  # Delete the database file after tests are finished

    
