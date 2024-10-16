import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun , DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper , ArxivAPIWrapper 
from langchain.agents import initialize_agent , AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os 

from dotenv import load_dotenv


##used inbuilt tool of wikipedia 

arxiv_wrapper=WikipediaAPIWrapper(top_k_results=1 , doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=arxiv_wrapper)

##used inbuilt tool of Arxiv

api_wrapper= ArxivAPIWrapper(top_k_results=1 , doc_content_chars_max=200)
arxiv= ArxivQueryRun(api_wrapper=api_wrapper)

search= DuckDuckGoSearchRun(name="Search")



st.title("Langchain - chat with search")
"""
In this example , we are running streamlitcallbackhandler  to display
the thoughts and actions of an agent in an interactive streamlit app .

"""


st.sidebar.title("Settings")
api_key=st.text_input("enter your groq api key :" , type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"i am a chatbot who can search the web for you . how can i help you "}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
if prompt:=st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    
    llm=ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It", streaming=True)
    tools=[arxiv , wiki , search]
    
    search_agent=initialize_agent(tools , llm , agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION , handling_parsing_errors =True)
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
    