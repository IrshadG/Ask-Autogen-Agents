import streamlit as st
import autogen
import autogen.code_utils as code_utils

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

config_list = [
            {
                'model': 'gpt-4',
                'api_key': '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',
            },  # OpenAI API endpoint for gpt-4
            {
                'model': 'gpt-3.5-turbo-16k',
                'api_key': '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',
            },  # OpenAI API endpoint for gpt-3.5-turbo-16k
        ]

uploaded_content = None

def runmodel():
    # create an AssistantAgent named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "seed": 42,  # seed for caching and reproducibility
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
        },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    )
    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },
    )
    # the assistant receives a message from the user_proxy, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message="""What date is today? Compare the year-to-date gain for META and TESLA.""",
    )

    user_proxy.send(
    recipient=assistant,
    message="""Plot a chart of their stock price change YTD and save to stock_price_ytd.png.""",
    )


st.title('LLM-AI Advanced Chatbot')
st.divider()





problem = st.text_input('Enter your query')
if(problem):
    problem = "Problem: "+ problem  +" \n" 
    if st.button('Submit'):
        runmodel()
        output = ""#model.trying_example()
        #output = model.solve(problem)
        st.write(output)