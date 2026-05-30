from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import CharacterTextSplitter

from utils.load_api_key import get_api_key

loader = TextLoader("../../data/clothes_data.txt")
docs = loader.load()
get_api_key(llm_type="DASHSCOPE_API_KEY")
text_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator='\n\n',
    keep_separator=True
)

chunks = text_splitter.split_documents(docs)


embedding_model = DashScopeEmbeddings(model='text-embedding-v4')


config = RedisConfig(
    index_name="sale-index",
    redis_url='redis://localhost:6379',
)

vector_store = RedisVectorStore(embedding_model, config)

vector_store.add_documents(chunks)
