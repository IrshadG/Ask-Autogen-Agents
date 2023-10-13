
#conda activate POP
#streamlit run Langchain/Autogen_model_2.py
from keys import OPENAI_API_KEY
import streamlit as st
import pandas as pd
import numpy as np
import autogen
import autogen.retrieve_utils as utils
import os
from PyPDF2 import PdfReader 
import asyncio

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
        ]
        self.llm_config = {
            #     "functions": [
            #     {
            #         "name": "Read_csv",
            #         "description": "Read csv file from directory.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "cell": {
            #                     "type": "string",
            #                     "description": "Valid Python cell to execute.",
            #                 }
            #             },
            #             "required": ["path"],
            #         },
            #     },
            # ],
            "config_list": self.config_list,
            "request_timeout": 120,
            "seed": 42,
            "temperature": 0,
        }
        # autogen.ChatCompletion.start_logging()
    
    async def add_data_reply(self,recipient, messages, sender, config):
        await asyncio.sleep(0.1)
        data = config["daraframe"]
        if data.done():
            result = data.result()
            if result:
                news_str = "\n".join(result)
                result.clear()
                return (
                    True,
                    f"Here is the dataframe you need. Use this to solve the problem.\n{news_str}",
                )
            return False, None
    

    def exec_sh(self, script):
        return self.user_proxy.execute_code_blocks([("sh", script)])

    def read_csv(self, path:str):
        df = pd.read_csv(path)
        return df
        
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
        self.executor = autogen.AssistantAgent(
        name="Executor",
        system_message="""Executor. You will execute the code provided by the Engineer, using python language and return the output back to the user. 
        Only reply TERMINATE when the task is done.
        """,
        llm_config= self.llm_config,
        )
        #
        #Coder Agent
        # self.coder = autogen.AssistantAgent(
        #     name="Engineer",
        #     system_message="""Engineer. You are a skilled Engineer. You can generate code to solve any problem. Always follow the procedure to write effecient code that does not require revision:
        #         -Use the current directory as default directory. Use 'PyPDF2' library to real pdf and 'pandas' for csv or xlxs formats.
        #         -Use the file name provided to you to read the file from current directory. 
        #         -Import the required libraries. Only use already installed libraries. 
        #         -Read the file using library
        #         -Write efficient code to get desired output. Use case sensitive values from the file for execution.
        #         -Finally, send the code to Executor for execution.
        #         -Your task is completed.""",
        #     llm_config=self.llm_config,
        # )
        self.coder = autogen.AssistantAgent(
            name="Engineer",
            system_message="""Engineer. You are a skilled Engineer. You can generate code to solve any problem. Always follow the procedure to write effecient code that does not require revision:
                -Use the data provided to you to solve the problem. 
                -Import the required libraries. Only use already installed libraries. 
                -Write efficient code to get desired output.
                -Generate output by executing your code and send the output to the admin.
                -Reply with 'TERMINATE' when your task is completed.""",
            llm_config=self.llm_config,
        )

        #
        # User proxy Agent
        self.user_proxy = autogen.UserProxyAgent(
            name = "Admin",
            max_consecutive_auto_reply=8,
            human_input_mode="NEVER",
            # default_auto_reply="CONTINUE",
            # function_map = {"Read_csv",self.read_csv},
            code_execution_config={"use_docker": True , "work_dir": "Langchain"},
            system_message="A human admin. Your job is to provide any help required by the Engineer and Executor. You have to recieve the output of the problem by the Executor. Reply only with TERMINATE when the answer is recieved.",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        )

        self.groupchat = autogen.GroupChat(agents=[self.user_proxy, self.coder, self.executor], messages=[], max_round=12)
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

    def solve(self, problem, agent, data):
        # reset the assistant. Always reset the assistant before starting a new conversation.
        self.user_proxy.reset()   
        self.coder.reset()   

        #initiate chat  
        self.user_proxy.initiate_chat(
            agent,
            message=problem,
            context=data,
        )
        chat = self.user_proxy.chat_messages[agent]
        output = chat[len(chat)-3]['content']
        # chat = self.groupchat.messages
        
        return output


############################ GET INPUT ##############################
st.title('Apparel-AI Advanced Chatbot')
st.text('(Optional) Browse a document to query from.')
# st.text('(.pdf, .csv, .txt, .json, .tsv, .html, .xml, .yaml, .yml)')


uploaded_file =st.file_uploader("Choose a document", accept_multiple_files=False, type=['pdf', 'csv', 'xls','txt', 'json', 'tsv', 'md', 'html', 'log', 'xml', 'yaml', 'yml'])
st.divider()
problem = st.text_input('Enter your query')
if(problem):
    model = AutoGen_Model()
    model.create_agents()


    problem = "Problem: " + problem + "Reply with TERMINATE when question is answered"

    if st.button('Submit'):
        data = model.read_csv(uploaded_file)
        model.user_proxy.register_reply(model.coder, model.add_data_reply, 1, config={"dataframe": data})
        output = model.solve(problem, agent=model.coder, data=data)
        st.write(output)
