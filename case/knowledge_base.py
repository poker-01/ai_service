import os

from sqlalchemy.testing.suite.test_reflection import metadata

import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

#检查传入的字符串MD5是否已经处理
def check_md5(md5_str:str):
    if not os.path.exists(config.md5_path):
        #创建文件，如果文件不存在打开就能自动创建
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()  #处理前后空格和回车
            if line==md5_str:
                return True
        return False

#检查传入的字符串MD5记录到文件保存
def save_md5(md5_str:str):
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')


#检查传入的字符串转换MD5
def get_string_md5(input_str:str,encoding='utf-8'):
    #将字符串转为字节数组
    str_bytes=input_str.encode(encoding=encoding)
    #得到MD5对象
    md5_obj = hashlib.md5()

    md5_obj.update(str_bytes)
    md5_hex=md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService:
    def __init__(self):
        #文件夹不存在则创建
        os.makedirs(config.persist_directory, exist_ok=True)

        # 向量存储的实例Chroma向量库对象
        self.chroma=Chroma(
            collection_name=config.collection_name,   #数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,  #数据库本地存储文件夹
        )

        #文本分割器对象
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,         #分割后文本段最大长度
            chunk_overlap=config.chunk_overlap,   #连续文本段字符重叠数量
            separators=config.separators,         #自然段落划分符号
            length_function=len,                  #长度统计依据
        )

    #将传入的字符串向量化再存进向量数据库中
    def upload_by_str(self,data,filename):
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"
        if len(data)>config.max_split_char_number:
            knowledge_chunks:list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]

        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小明"
        }
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)
        return "[成功]内容已经成功载入数据库"


# if __name__ == '__main__':
#     service=KnowledgeBaseService()
#     r=service.upload_by_str("蔡依林","testfile")
#     print(r)

