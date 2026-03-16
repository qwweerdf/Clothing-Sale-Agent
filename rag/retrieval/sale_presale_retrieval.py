from models.embedding import dash_scope_embedding
from utils.load_api_key import get_api_key
from langchain_redis import RedisConfig, RedisVectorStore

get_api_key()

class PresaleRetrieval:
    def __init__(self, index='sale-index'):
        config = RedisConfig(
            index_name=index,
            redis_url="redis://localhost:6379"
        )
        vector_store = RedisVectorStore(dash_scope_embedding, config=config)
        self.retriever = vector_store.as_retriever()

    def invoke(self, query, k=3) -> str:
        docs = self.retriever.invoke(query, k=k)
        context = '\n'.join([doc.page_content for doc in docs])
        return context


presale_index_retriever = PresaleRetrieval()
presale_index_qa_retriever = PresaleRetrieval('sale-index-qa')
