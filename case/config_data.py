from distutils.command.config import config

from langchain_community.embeddings import DashScopeEmbeddings

md5_path="./md5.text"

collection_name="rag"
persist_directory="./chroma_db"

chunk_size=200
chunk_overlap=50
separators=["\n\n","\r\n","\r\n\r\n", "\n", ".","!","？","。","！","?"," ",""]
max_split_char_number=200

similarity_threshold=2  #检索返回匹配的文档数量

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"