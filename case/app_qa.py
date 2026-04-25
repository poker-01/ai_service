import time
from rag import RagService
import streamlit as st
from click import prompt
import  config_data as config

st.title("ai问答助手")
st.divider()

#刚进来
if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，需要什么帮助？"}]

#只创建一次对象
if "rag" not in st.session_state:
    st.session_state["rag"]=RagService()

#显示前文
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
#输入栏
prompt=st.chat_input()

if prompt:
    #输出用户提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list=[]
    with st.spinner("ai思考中..."):
        res_stream=st.session_state["rag"].chain.stream({"input":prompt},config.session_config)

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
