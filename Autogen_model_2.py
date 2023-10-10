
#conda activate POP
#streamlit run Langchain/Autogen_model_2.py
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
        self.llm_config={
            "request_timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        # autogen.ChatCompletion.start_logging()
    
    def split_files_to_text(self,file):
        """Split a list of files into text."""
        text = file.getvalue().decode("utf-8")

        if(len(text) > 7000):
            st.text('Uploaded doc contains '+ str(len(text)) + ' letters, limiting to 7000.')
            text = text[slice(7000)]

        return text

    def read_pdf(self, file):
        reader = PdfReader(file) 
        cont=""

        for page in reader.pages:
            cont = cont +  page.extract_text() 
        cont = cont.replace("\n", " ")
        return cont

    def create_agents(self):
        #Chatbot Agent
        self.chatbot = autogen.AssistantAgent(
        name="Chatbot",
        system_message="You are a helpful assistant. You can process a string and generate answers based on the string. You can use 'cntx' variable to access the string if you face errors. Reply TERMINATE when the task is done.",
        llm_config= self.llm_config,
        )
        #
        #Coder Agent
        self.coder = autogen.AssistantAgent(
            name="Coder",
            system_message="You are a skilled coder. You can generate code to convert text into dataframe and answer the question. Reply only with TERMINATE when the task is done.",
            llm_config=self.llm_config,
        )
        #
        # User proxy Agent
        self.user_proxy = autogen.UserProxyAgent(
            name = "Admin",
            max_consecutive_auto_reply=8,
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},
            system_message="A human admin. Reply only with TERMINATE when the task is done.",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        )
        

    def solve(self, problem, agent):
        # reset the assistant. Always reset the assistant before starting a new conversation.
        self.user_proxy.reset()   
        self.chatbot.reset()   

        #initiate chat  
        self.user_proxy.initiate_chat(
            agent,
            message="Problem:" + problem,
        )

        chat = self.user_proxy.chat_messages[agent]
        message = chat[len(chat)-3]['content']
        return message


############################ GET INPUT ##############################
st.title('Apparel-AI Advanced Chatbot')
st.text('(Optional) Browse a document to query from.')
# st.text('(.pdf, .csv, .txt, .json, .tsv, .html, .xml, .yaml, .yml)')


uploaded_file =st.file_uploader("Choose a document", accept_multiple_files=False, type=['pdf', 'csv', 'xls','txt', 'json', 'tsv', 'md', 'html', 'log', 'xml', 'yaml', 'yml'])
st.divider()
problem = st.text_input('Enter your query')
if(uploaded_file and problem):
    model = AutoGen_Model()
    model.create_agents()

    if ".pdf" in uploaded_file.name:
        cntx = model.read_pdf(uploaded_file)
        agent = model.chatbot
    elif ".csv" in uploaded_file.name:
        cntx = str(pd.read_csv(uploaded_file))
        agent = model.coder
    elif ".xls" in uploaded_file.name:
        cntx = str(pd.read_excel(uploaded_file))
        agent = model.coder
    else:
        cntx = model.split_files_to_text(uploaded_file)
        agent = model.chatbot

    problem = problem + "Reply with TERMINATE when question is answered; context: " + cntx

    if st.button('Submit'):
        output = model.solve(problem, agent=agent)
        st.write(output)
