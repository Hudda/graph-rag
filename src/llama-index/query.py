from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv


from query_engine import GraphRAGQueryEngine
from store import GraphRAGStore

import os

load_dotenv()

os.env = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-4")

graph_store = GraphRAGStore(
    username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"), url=os.getenv("NEO4J_URI")
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    show_progress=True,
)

index.property_graph_store.build_communities()

query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)
