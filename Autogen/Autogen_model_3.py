
#conda activate POP
#streamlit run Langchain/Autogen_model_3.py
#https://docs.google.com/spreadsheets/d/18IEd4E0dkDnvJ-DG-r_lnMXpuFH0UkxKgNAHVRv7v-k/edit?usp=sharing

from keys import OPENAI_API_KEY
import streamlit as st
import pandas as pd
import numpy as np
import autogen
import autogen.code_utils as code_utils
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import autogen.retrieve_utils as utils
from autogen.retrieve_utils import TEXT_FORMATS
import chromadb.utils.embedding_functions as ef
from PyPDF2 import PdfReader 
import logging
import chromadb
import os
from termcolor import colored
import json

def colored(x, *args, **kwargs):
    return x

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
logger = logging.getLogger(__name__)
uploaded_content = None

class AutoGen_Model():
    def __init__(self):
        autogen.ChatCompletion.start_logging()
        self.collection_name = "Autogen_db"
        self.docs_path = "Langchain"
        self.max_tokens = 4000
        
        self.config_list = [
            {
                'model': 'gpt-4',
                'api_key': OPENAI_API_KEY,
            },  # OpenAI API endpoint for gpt-4
            {
                'model': 'gpt-3.5-turbo-16k',
                'api_key': OPENAI_API_KEY,
            },  # OpenAI API endpoint for gpt-3.5-turbo-16k
        ]
        self.llm_config={
                "functions": [
                {
                    "name": "python",
                    "description": "run python code and print result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "str_code": {
                                "type": "string",
                                "description": "Valid Python code to execute.",
                            }
                        },
                        "required": ["str_code"],
                    },
                },
                # {
                #     "name": "sh",
                #     "description": "run a shell script and return the execution result.",
                #     "parameters": {
                #         "type": "object",
                #         "properties": {
                #             "script": {
                #                 "type": "string",
                #                 "description": "Valid shell script to execute.",
                #             }
                #         },
                #         "required": ["script"],
                #     },
                # },
            ],
            "request_timeout": 120,
            "seed": 42,
            "config_list": self.config_list,
        }

    def split_files_to_text(self,file, letter_limit = 7000):
        """Split a file into text."""
        text = file.getvalue().decode("utf-8")

        if(len(text) > letter_limit):
            st.text('Uploaded doc contains '+ str(len(text)) + ' letters, limiting to 7000.')
            text = text[slice(7000)]

        return text

    def exec_sh(self, script):
        # str_df = str(uploaded_content.to_dict())
        # code_with_data = """
        # import pandas as pd
        # data=""" + str_df +"""
        # df = pd.DataFrame.from_dict(data)                                 
        # """ + script
        print("Edited code is:\n" + script + "\n Ended print")
        return self.user_proxy.execute_code_blocks([("sh", script)])

    def execute_code(self, str_code):

        str_df = str(uploaded_content.to_dict())
        str_code = """
import pandas as pd
data=""" + str_df +"""
df = pd.DataFrame.from_dict(data)                                 
""" + str_code

        print("Edited code is:\n" + str_code + "\n Ended print")

        exec(str_code)
        # return str_code

    def read_pdf(self, file):
        reader = PdfReader(file) 
        cont=""

        for page in reader.pages:
            cont = cont +  page.extract_text() 
        cont = cont.replace("\n", " ")
        return cont
    
    def read_csv(self, path:str):
        df = pd.read_csv(path)
        return df

    def filter_token_context(self, content):
        doc_contents = ""
        current_tokens = 0

        _doc_tokens = utils.num_tokens_from_text(content)
        if _doc_tokens > self.max_tokens:
            func_print = f"Skipping content as it is too long to fit in the context. Using {_doc_tokens} tokens"
            print(colored(func_print, "green"), flush=True)
            return doc_contents
        func_print = f"Adding content to context. Using {_doc_tokens} tokens"
        print(colored(func_print, "green"), flush=True)
        current_tokens += _doc_tokens
        doc_contents += content + "\n"
        return doc_contents

    def create_agents(self):
        # create a UserProxyAgent instance named "user_proxy"
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            system_message = """
            Human Admin. You can get the code from the engineer and run the function
            """,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "Langchain",
                                   "use_docker": False
                                },
            )

        self.engineer = autogen.AssistantAgent(
            name="coder", 
            system_message="""Coder. You are a skilled Coder. You can generate python code to solve problems.
                    Dataframe columns will be provided to you, write code based on the columns to solve the coding problem.
                    Always follow the code protocols:
                    -Assume pandas is already imported as pd.
                    -Assume 'df' variable already contains a dataframe.
                    -Send the code to the admin.
                    -Reply with TERMINATE when task is done.
                    """,
            llm_config=self.llm_config,
        )

        self.user_proxy.register_function(
            function_map={
                # "sh": self.exec_sh,
                "python": self.execute_code,
            }
        )
        # self.engineer.register_function(
        #     function_map={
        #         "python": self.execute_code,
        #     }
        # )
        

    def solve(self, problem):        
        # problem += "Train 30 seconds and force cancel jobs if time limit is reached."
        self.user_proxy.initiate_chat(self.engineer, message=problem)  # search_string is used as an extra filter for the embedding

        # chat = self.ragproxyagent.chat_messages[self.assistant]
        # chat = self.groupchat.messages
        # output = chat[len(chat)-1]["content"]
        chat = self.user_proxy.chat_messages[self.engineer]
        output = chat[len(chat)-1]['content']
        return output


############################ GET INPUT ##############################
st.title('Apparel-AI Advanced Chatbot')
model = AutoGen_Model()
uploaded_file = st.file_uploader('Upload your file here')

if uploaded_file:
    if ".pdf" in uploaded_file.name:
        file_content = model.read_pdf(uploaded_file)
        uploaded_content = file_content
    elif ".csv" in uploaded_file.name:
        uploaded_content = model.read_csv(uploaded_file)
        file_content = uploaded_content.columns

    content = model.filter_token_context("Dataframe columns: \n" + str(file_content))
    model.create_agents()

    st.divider()

    problem = st.text_input('Enter your query')
    if(problem):
        problem = "Problem: "+ problem  +" \n" + content
        if st.button('Submit'):
            model.user_proxy.reset()
            model.engineer.reset()

            output = model.solve(problem)
            st.write(output)