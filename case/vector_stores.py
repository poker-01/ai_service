from re import search

from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        #嵌入模型传入
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    #返回向量检索器，方便加入链
    def get_retriever(self):
        return  self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})

if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res=retriever.invoke("木星有多少颗卫星")
    print(res)

# if __name__ == '__main__':
#     from langchain_community.embeddings import DashScopeEmbeddings
#     import config_data as config
#     from langchain_chroma import Chroma
#
#     # 直接连接数据库查看
#     vector_store = Chroma(
#         collection_name=config.collection_name,
#         embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
#         persist_directory=config.persist_directory,
#     )
#
#     # 获取所有文档
#     all_docs = vector_store.get()
#     print(f"数据库中存储的文档块数量: {len(all_docs['ids'])}")
#
#     for i, (doc_id, metadata, content) in enumerate(zip(all_docs['ids'], all_docs['metadatas'], all_docs['documents'])):
#         print(f"\n=== 块{i + 1} ===")
#         print(f"ID: {doc_id}")
#         print(f"来源: {metadata.get('source') if metadata else 'unknown'}")
#         print(f"内容长度: {len(content)} 字符")
#         print(f"内容预览: {content[:150]}...")