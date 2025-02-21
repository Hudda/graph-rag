
from typing import Any
import pandas as pd
from llama_index.core import Document
import os
import re
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

from extractor import GraphRAGExtractor
from extractor_prompt import KG_TRIPLET_EXTRACT_TMPL
from store import GraphRAGStore
from query_engine import GraphRAGQueryEngine


load_dotenv()

os.env = os.getenv("OPENAI_API_KEY")

news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)[:50]

news.head()


documents = [
    Document(text=f"{row['title']}: {row['text']}")
    for i, row in news.iterrows()
]

llm = OpenAI(model="gpt-4")


splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Note: used to be `Neo4jPGStore`
graph_store = GraphRAGStore(
    username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"), url=os.getenv("NEO4J_URI")
)

from llama_index.core import PropertyGraphIndex

index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)