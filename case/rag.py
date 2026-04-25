from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

from file_history_store import get_history
from vector_stores import VectorStoreService
import config_data as config

def print_prompt(prompt):
    print ("="*20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt



class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的已知参考资料为主，"
                 "简介专业地回答用户问题。参考资料：{context}。"),
                ("system","并且我提供用户的对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}")
            ]
        )

        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain=self.__get_Chain()

    def __get_Chain(self):
        # 获取最终执行链
        retriever=self.vector_service.get_retriever()

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str=""
            for doc in docs:
                formatted_str+=f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        chain=(
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever)|retriever|format_document
            } | RunnableLambda(format_for_prompt_template)|self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain= RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

# if __name__ == '__main__':
#     session_config={
#         "configurable":{
#             "session_id":"user_02",
#         }
#     }
#     res=RagService().chain.invoke({"input":"地球和火星哪个温度高"},session_config)
#     print(res)