import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import geocoder
import googlemaps
import numpy as np
import requests
from ansistrip import ansi_strip
from dotenv import load_dotenv
from llama_index.core import (KnowledgeGraphIndex, Settings,
                              SimpleDirectoryReader, StorageContext)
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.perplexity import Perplexity
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from memary.agent.data_types import Context, Message
from memary.agent.llm_api.tools import openai_chat_completions_request
from memary.memory import EntityKnowledgeStore, MemoryStream
from memary.synonym_expand.synonym import custom_synonym_expand_fn

MAX_ENTITIES_FROM_KG = 5
ENTITY_EXCEPTIONS = ["Unknown relation"]
# LLM token limits
CONTEXT_LENGTH = 4096
EVICTION_RATE = 0.7
NONEVICTION_LENGTH = 5
TOP_ENTITIES = 20
# History message limits - research shows degradation after ~30 messages
MAX_HISTORY_MESSAGES = 30  # Maximum number of recent chat messages to include in LLM context


def generate_string(entities):
    cypher_query = "MATCH p = (n) - [*1 .. 2] - ()\n"
    cypher_query += "WHERE n.id IN " + str(entities) + "\n"
    cypher_query += "RETURN p"

    return cypher_query


class Agent(object):
    """Agent manages the RAG model, the ReAct agent, and the memory stream."""

    def __init__(
        self,
        agent_name,
        memory_stream_json,
        entity_knowledge_store_json,
        system_persona_txt,
        user_persona_txt,
        past_chat_json,
        user_id='falkor',
        llm_model_name="llama3",
        vision_model_name="llava",
        include_from_defaults=["search", "locate", "vision", "stocks"],
        debug=True,
    ):
        load_dotenv()
        self.name = agent_name
        self.model = llm_model_name

        googlemaps_api_key = os.getenv("GOOGLEMAPS_API_KEY")
        pplx_api_key = os.getenv("PERPLEXITY_API_KEY")

        # initialize APIs
        self.load_llm_model(llm_model_name)
        self.load_vision_model(vision_model_name)
        
        # Initialize Perplexity only if API key is available
        if pplx_api_key:
            self.query_llm = Perplexity(
                api_key=pplx_api_key, model="mistral-7b-instruct", temperature=0.5
            )
        else:
            self.query_llm = None
            print("Warning: PERPLEXITY_API_KEY not found. External search functionality will be limited.")
        
        # Initialize Google Maps only if API key is available
        if googlemaps_api_key:
            self.gmaps = googlemaps.Client(key=googlemaps_api_key)
        else:
            self.gmaps = None
            print("Warning: GOOGLEMAPS_API_KEY not found. Location functionality will be limited.")
        
        # Only set Settings.llm if it wasn't already configured
        if Settings.llm is None:
            Settings.llm = self.llm
        Settings.chunk_size = 512
        
        self.falkordb_url = os.getenv("FALKORDB_URL")
        neo4j_url = os.getenv("NEO4J_URL")
        neo4j_password = os.getenv("NEO4J_PW")
        
        # Initialize graph store only if database credentials are available
        self.graph_store = None
        if self.falkordb_url is not None:
            try:
                from llama_index.graph_stores.falkordb import FalkorDBGraphStore
                # initialize FalkorDB graph resources
                self.graph_store = FalkorDBGraphStore(self.falkordb_url, database=user_id, decode_responses=True)
                print("Connected to FalkorDB graph store")
            except Exception as e:
                print(f"Warning: Failed to connect to FalkorDB: {e}")
        elif neo4j_url and neo4j_password:
            try:
                from llama_index.graph_stores.neo4j import Neo4jGraphStore
                # Neo4j credentials
                self.neo4j_username = "neo4j"
                self.neo4j_password = neo4j_password
                self.neo4j_url = neo4j_url
                database = "neo4j"

                # initialize Neo4j graph resources
                self.graph_store = Neo4jGraphStore(
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    url=self.neo4j_url,
                    database=database,
                )
                print("Connected to Neo4j graph store")
            except Exception as e:
                print(f"Warning: Failed to connect to Neo4j: {e}")
        else:
            print("Warning: No graph database configured. Please set FALKORDB_URL or both NEO4J_URL and NEO4J_PW.")

        self.vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        # Initialize storage context and query engine only if graph store is available
        if self.graph_store:
            self.storage_context = StorageContext.from_defaults(
                graph_store=self.graph_store
            )
            graph_rag_retriever = KnowledgeGraphRAGRetriever(
                storage_context=self.storage_context,
                verbose=True,
                llm=self.llm,
                retriever_mode="keyword",
                synonym_expand_fn=custom_synonym_expand_fn,
            )

            self.query_engine = RetrieverQueryEngine.from_args(
                graph_rag_retriever,
            )
        else:
            self.storage_context = None
            self.query_engine = None
            print("Warning: Knowledge graph functionality is disabled due to missing database configuration.")

        self.debug = debug
        self.tools = {}
        self._init_default_tools(default_tools=include_from_defaults)

        self.memory_stream = MemoryStream(memory_stream_json)
        self.entity_knowledge_store = EntityKnowledgeStore(entity_knowledge_store_json)

        self.message = Message(
            system_persona_txt, user_persona_txt, past_chat_json, self.model
        )

    def __str__(self):
        return f"Agent {self.name}"

    def load_llm_model(self, llm_model_name):
        # Check if Settings.llm is already configured (e.g., by main.py)
        if Settings.llm is not None:
            print(f"Using pre-configured LLM: {type(Settings.llm).__name__}")
            self.llm = Settings.llm
            return
            
        if llm_model_name == "gpt-3.5-turbo":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            self.openai_api_key = os.environ["OPENAI_API_KEY"]
            self.model_endpoint = "https://api.openai.com/v1"
            self.llm = OpenAI(model="gpt-3.5-turbo-instruct")
        else:
            try:
                self.llm = Ollama(model=llm_model_name, request_timeout=60.0)
            except:
                raise ("Please provide a proper llm_model_name.")

    def load_vision_model(self, vision_model_name):
        if vision_model_name == "gpt-4-vision-preview":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            self.openai_api_key = os.environ["OPENAI_API_KEY"]
            self.mm_model = OpenAIMultiModal(
                model="gpt-4-vision-preview",
                api_key=os.getenv("OPENAI_KEY"),
                max_new_tokens=300,
            )
        else:
            try:
                self.mm_model = OllamaMultiModal(model=vision_model_name, request_timeout=60.0)
            except:
                raise ("Please provide a proper vision_model_name.")

    def external_query(self, query: str):
        if self.query_llm is None:
            return "External search is not available. Please configure PERPLEXITY_API_KEY."
        
        messages_dict = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ]
        messages = [ChatMessage(**msg) for msg in messages_dict]
        external_response = self.query_llm.chat(messages)

        return str(external_response)

    def search(self, query: str) -> str:
        """Search the knowledge graph or perform search on the web if information is not present in the knowledge graph"""
        if self.query_engine is None:
            return self.external_query(query)
        
        response = self.query_engine.query(query)

        if response.metadata is None:
            return self.external_query(query)
        else:
            return response

    def locate(self, query: str) -> str:
        """Finds the current geographical location"""
        if self.gmaps is None:
            return "Location services are not available. Please configure GOOGLEMAPS_API_KEY."
        
        location = geocoder.ip("me")
        lattitude, longitude = location.latlng[0], location.latlng[1]

        reverse_geocode_result = self.gmaps.reverse_geocode((lattitude, longitude))
        formatted_address = reverse_geocode_result[0]["formatted_address"]
        return "Your address is" + formatted_address

    def vision(self, query: str, img_url: str) -> str:
        """Uses computer vision to process the image specified by the image url and answers the question based on the CV results"""
        query_image_dir_path = Path("query_images")
        if not query_image_dir_path.exists():
            Path.mkdir(query_image_dir_path)

        data = requests.get(img_url).content
        query_image_path = os.path.join(query_image_dir_path, "query.jpg")
        with open(query_image_path, "wb") as f:
            f.write(data)
        image_documents = SimpleDirectoryReader(query_image_dir_path).load_data()

        response = self.mm_model.complete(prompt=query, image_documents=image_documents)

        os.remove(query_image_path)  # delete image after use
        return response

    def stocks(self, query: str) -> str:
        """Get the stock price of the company given the ticker"""
        if not self.vantage_key:
            return "Stock price lookup is not available. Please configure ALPHA_VANTAGE_API_KEY."
        
        request_api = requests.get(
            r"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol="
            + query
            + r"&apikey="
            + self.vantage_key
        )
        return request_api.json()

    # def get_news(self, query: str) -> str:
    #     """Given a keyword, search for news articles related to the keyword"""
    #     request_api = requests.get(r'https://newsdata.io/api/1/news?apikey=' + self.news_data_key + r'&q=' + query)
    #     return request_api.json()

    def query(self, query: str) -> str:
        # get the response from react agent
        response = self.routing_agent.chat(query)
        self.routing_agent.reset()
        # write response to file for KG writeback
        with open("data/external_response.txt", "w") as f:
            print(response, file=f)
        # write back to the KG
        self.write_back()
        return response

    def write_back(self):
        if self.storage_context is None:
            print("Warning: Cannot write back to knowledge graph - no database configured")
            return
            
        documents = SimpleDirectoryReader(
            input_files=["data/external_response.txt"]
        ).load_data()

        KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=8,
        )

    def check_KG(self, query: str) -> bool:
        """Check if the query is in the knowledge graph.

        Args:
            query (str): query to check in the knowledge graph

        Returns:
            bool: True if the query is in the knowledge graph, False otherwise
        """
        if self.query_engine is None:
            return False
            
        response = self.query_engine.query(query)

        if response.metadata is None:
            return False
        return generate_string(
            list(list(response.metadata.values())[0]["kg_rel_map"].keys())
        )

    def _select_top_entities(self):
        entity_knowledge_store = self.message.llm_message["knowledge_entity_store"]
        entities = [entity.to_dict() for entity in entity_knowledge_store]
        entity_counts = [entity["count"] for entity in entities]
        top_indexes = np.argsort(entity_counts)[:TOP_ENTITIES]
        return [entities[index] for index in top_indexes]

    def _add_contexts_to_llm_message(self, role, content, index=None):
        """Add contexts to the llm_message."""
        if index:
            self.message.llm_message["messages"].insert(index, Context(role, content))
        else:
            self.message.llm_message["messages"].append(Context(role, content))

    def _change_llm_message_chat(self) -> dict:
        """Change the llm_message to chat format.

        Returns:
            dict: llm_message in chat format
        """
        llm_message_chat = self.message.llm_message.copy()
        llm_message_chat["messages"] = []
        top_entities = self._select_top_entities()
        logging.info(f"top_entities: {top_entities}")
        llm_message_chat["messages"].append(
            {
                "role": "user",
                "content": "Knowledge Entity Store:" + str(top_entities),
            }
        )
        llm_message_chat["messages"].extend(
            [context.to_dict() for context in self.message.get_limited_messages_for_llm()]
        )
        llm_message_chat.pop("knowledge_entity_store")
        llm_message_chat.pop("memory_stream")
        return llm_message_chat

    def _summarize_contexts(self, total_tokens: int):
        """Summarize the contexts.

        Args:
            total_tokens (int): total tokens in the response
        """
        messages = self.message.llm_message["messages"]

        # First two messages are system and user personas
        if len(messages) > 2 + NONEVICTION_LENGTH:
            messages = messages[2:-NONEVICTION_LENGTH]
            del self.message.llm_message["messages"][2:-NONEVICTION_LENGTH]
        else:
            messages = messages[2:]
            del self.message.llm_message["messages"][2:]

        message_contents = [message.to_dict()["content"] for message in messages]

        llm_message_chat = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize these previous conversations into 50 words:"
                    + str(message_contents),
                }
            ],
        }
        response, _ = self._get_chat_response(llm_message_chat)
        content = "Summarized past conversation:" + response
        self._add_contexts_to_llm_message("assistant", content, index=2)
        logging.info(f"Contexts summarized successfully. \n summary: {response}")
        logging.info(f"Total tokens after eviction: {total_tokens*EVICTION_RATE}")

    def _get_chat_response(self, llm_message_chat) -> tuple:
        """Get response from the LLM chat model.

        Args:
            llm_message_chat: query to get response for

        Returns:
            tuple: (response, total_tokens)
        """
        if self.model == "gpt-3.5-turbo":
            response = openai_chat_completions_request(
                self.model_endpoint, self.openai_api_key, llm_message_chat
            )
            total_tokens = response["usage"]["total_tokens"]
            response = str(response["choices"][0]["message"]["content"])
        else:  # Use configured LlamaIndex LLM (LlamaCPP or other)
            from llama_index.core import Settings
            from llama_index.core.llms import ChatMessage, MessageRole
            
            # Convert messages to LlamaIndex format
            messages = []
            for msg in llm_message_chat["messages"]:
                role = MessageRole.USER if msg["role"] == "user" else (
                    MessageRole.SYSTEM if msg["role"] == "system" else MessageRole.ASSISTANT
                )
                messages.append(ChatMessage(role=role, content=msg["content"]))
            
            # Use the configured LLM (LlamaCPP)
            chat_response = Settings.llm.chat(messages)
            response = str(chat_response.message.content)
            
            # Post-process response to remove any fake conversation turns
            response = self._clean_response(response)
            
            total_tokens = 0  # Token counting not implemented for local models
                
        return response, total_tokens

    def get_response(self) -> str:
        """Get response from the RAG model.

        Returns:
            str: response from the RAG model
        """
        llm_message_chat = self._change_llm_message_chat()
        response, total_tokens = self._get_chat_response(llm_message_chat)
        if total_tokens > CONTEXT_LENGTH * EVICTION_RATE:
            logging.info("Evicting and summarizing contexts")
            self._summarize_contexts(total_tokens)

        self.message.save_contexts_to_json()

        return response

    def get_routing_agent_response(self, query, return_entity=False):
        """Get response from the ReAct."""
        response = ""
        if self.debug:
            # writes ReAct agent steps to separate file and modifies format to be readable in .txt file
            with open("data/routing_response.txt", "w") as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                response = str(self.query(query))
                sys.stdout.flush()
                sys.stdout = orig_stdout
            text = ""
            with open("data/routing_response.txt", "r") as f:
                text = f.read()

            plain = ansi_strip(text)
            with open("data/routing_response.txt", "w") as f:
                f.write(plain)
        else:
            response = str(self.query(query))

        if return_entity:
            # the query above already adds final response to KG so entities will be present in the KG
            if self.query_engine:
                return response, self.get_entity(self.query_engine.retrieve(query))
            else:
                return response, []
        return response

    def get_entity(self, retrieve) -> list[str]:
        """retrieve is a list of QueryBundle objects.
        A retrieved QueryBundle object has a "node" attribute,
        which has a "metadata" attribute.

        example for "kg_rel_map":
        kg_rel_map = {
            'Harry': [['DREAMED_OF', 'Unknown relation'], ['FELL_HARD_ON', 'Concrete floor']],
            'Potter': [['WORE', 'Round glasses'], ['HAD', 'Dream']]
        }

        Args:
            retrieve (list[NodeWithScore]): list of NodeWithScore objects
        return:
            list[str]: list of string entities
        """

        entities = []
        kg_rel_map = retrieve[0].node.metadata["kg_rel_map"]
        for key, items in kg_rel_map.items():
            # key is the entity of question
            entities.append(key)
            # items is a list of [relationship, entity]
            entities.extend(item[1] for item in items)
            if len(entities) > MAX_ENTITIES_FROM_KG:
                break
        entities = list(set(entities))
        for exceptions in ENTITY_EXCEPTIONS:
            if exceptions in entities:
                entities.remove(exceptions)
        return entities

    def _init_ReAct_agent(self):
        """Initializes ReAct Agent with list of tools in self.tools."""
        tool_fns = []
        for func in self.tools.values():
            tool_fns.append(FunctionTool.from_defaults(fn=func))
        self.routing_agent = ReActAgent.from_tools(tool_fns, llm=self.llm, verbose=True)

    def _init_default_tools(self, default_tools: List[str]):
        """Initializes ReAct Agent from the default list of tools memary provides.
        List of strings passed in during initialization denoting which default tools to include.
        Args:
            default_tools (list(str)): list of tool names in string form
        """

        for tool in default_tools:
            if tool == "search":
                self.tools["search"] = self.search
            elif tool == "locate":
                self.tools["locate"] = self.locate
            elif tool == "vision":
                self.tools["vision"] = self.vision
            elif tool == "stocks":
                self.tools["stocks"] = self.stocks
        self._init_ReAct_agent()

    def add_tool(self, tool_additions: Dict[str, Callable[..., Any]]):
        """Adds specified tools to be used by the ReAct Agent.
        Args:
            tools (dict(str, func)): dictionary of tools with names as keys and associated functions as values
        """

        for tool_name in tool_additions:
            self.tools[tool_name] = tool_additions[tool_name]
        self._init_ReAct_agent()

    def remove_tool(self, tool_name: str):
        """Removes specified tool from list of available tools for use by the ReAct Agent.
        Args:
            tool_name (str): name of tool to be removed in string form
        """

        if tool_name in self.tools:
            del self.tools[tool_name]
            self._init_ReAct_agent()
        else:
            raise ("Unknown tool_name provided for removal.")

    def update_tools(self, updated_tools: List[str]):
        """Resets ReAct Agent tools to only include subset of default tools.
        Args:
            updated_tools (list(str)): list of default tools to include
        """

        self.tools.clear()
        for tool in updated_tools:
            if tool == "search":
                self.tools["search"] = self.search
            elif tool == "locate":
                self.tools["locate"] = self.locate
            elif tool == "vision":
                self.tools["vision"] = self.vision
            elif tool == "stocks":
                self.tools["stocks"] = self.stocks
        self._init_ReAct_agent()

    def _clean_response(self, response: str) -> str:
        """Clean response to remove any fake conversation turns or role labels.
        
        Args:
            response (str): Raw response from the LLM
            
        Returns:
            str: Cleaned response with fake conversations removed
        """
        import re
        
        # Remove any text that looks like fake conversation turns
        # Pattern: user: ... assistant: ... or \nuser: ... \nassistant: ...
        patterns = [
            r'\nuser:\s*.*?(?=\nassistant:|\nuser:|$)',
            r'\nassistant:\s*.*?(?=\nuser:|\nassistant:|$)',
            r'user:\s*.*?(?=assistant:|user:|$)',
            r'assistant:\s*.*?(?=user:|assistant:|$)',
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up any remaining artifacts
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)  # Remove multiple newlines
        cleaned = cleaned.strip()
        
        # If the response got completely cleaned (was all fake conversation), 
        # return just the first sentence before any fake turns
        if not cleaned or len(cleaned) < 10:
            # Extract first sentence/paragraph before any fake conversation
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not re.match(r'^(user|assistant):\s*', line, re.IGNORECASE):
                    return line
            return "I'm here to help! How can I assist you?"
        
        return cleaned

    def get_response(self) -> str:
        """Get response from the RAG model.

        Returns:
            str: response from the RAG model
        """
        llm_message_chat = self._change_llm_message_chat()
        response, total_tokens = self._get_chat_response(llm_message_chat)
        if total_tokens > CONTEXT_LENGTH * EVICTION_RATE:
            logging.info("Evicting and summarizing contexts")
            self._summarize_contexts(total_tokens)

        self.message.save_contexts_to_json()

        return response

    def get_routing_agent_response(self, query, return_entity=False):
        """Get response from the ReAct."""
        response = ""
        if self.debug:
            # writes ReAct agent steps to separate file and modifies format to be readable in .txt file
            with open("data/routing_response.txt", "w") as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                response = str(self.query(query))
                sys.stdout.flush()
                sys.stdout = orig_stdout
            text = ""
            with open("data/routing_response.txt", "r") as f:
                text = f.read()

            plain = ansi_strip(text)
            with open("data/routing_response.txt", "w") as f:
                f.write(plain)
        else:
            response = str(self.query(query))

        if return_entity:
            # the query above already adds final response to KG so entities will be present in the KG
            if self.query_engine:
                return response, self.get_entity(self.query_engine.retrieve(query))
            else:
                return response, []
        return response

    def get_entity(self, retrieve) -> list[str]:
        """retrieve is a list of QueryBundle objects.
        A retrieved QueryBundle object has a "node" attribute,
        which has a "metadata" attribute.

        example for "kg_rel_map":
        kg_rel_map = {
            'Harry': [['DREAMED_OF', 'Unknown relation'], ['FELL_HARD_ON', 'Concrete floor']],
            'Potter': [['WORE', 'Round glasses'], ['HAD', 'Dream']]
        }

        Args:
            retrieve (list[NodeWithScore]): list of NodeWithScore objects
        return:
            list[str]: list of string entities
        """

        entities = []
        kg_rel_map = retrieve[0].node.metadata["kg_rel_map"]
        for key, items in kg_rel_map.items():
            # key is the entity of question
            entities.append(key)
            # items is a list of [relationship, entity]
            entities.extend(item[1] for item in items)
            if len(entities) > MAX_ENTITIES_FROM_KG:
                break
        entities = list(set(entities))
        for exceptions in ENTITY_EXCEPTIONS:
            if exceptions in entities:
                entities.remove(exceptions)
        return entities

    def _init_ReAct_agent(self):
        """Initializes ReAct Agent with list of tools in self.tools."""
        tool_fns = []
        for func in self.tools.values():
            tool_fns.append(FunctionTool.from_defaults(fn=func))
        self.routing_agent = ReActAgent.from_tools(tool_fns, llm=self.llm, verbose=True)

    def _init_default_tools(self, default_tools: List[str]):
        """Initializes ReAct Agent from the default list of tools memary provides.
        List of strings passed in during initialization denoting which default tools to include.
        Args:
            default_tools (list(str)): list of tool names in string form
        """

        for tool in default_tools:
            if tool == "search":
                self.tools["search"] = self.search
            elif tool == "locate":
                self.tools["locate"] = self.locate
            elif tool == "vision":
                self.tools["vision"] = self.vision
            elif tool == "stocks":
                self.tools["stocks"] = self.stocks
        self._init_ReAct_agent()

    def add_tool(self, tool_additions: Dict[str, Callable[..., Any]]):
        """Adds specified tools to be used by the ReAct Agent.
        Args:
            tools (dict(str, func)): dictionary of tools with names as keys and associated functions as values
        """

        for tool_name in tool_additions:
            self.tools[tool_name] = tool_additions[tool_name]
        self._init_ReAct_agent()

    def remove_tool(self, tool_name: str):
        """Removes specified tool from list of available tools for use by the ReAct Agent.
        Args:
            tool_name (str): name of tool to be removed in string form
        """

        if tool_name in self.tools:
            del self.tools[tool_name]
            self._init_ReAct_agent()
        else:
            raise ("Unknown tool_name provided for removal.")

    def update_tools(self, updated_tools: List[str]):
        """Resets ReAct Agent tools to only include subset of default tools.
        Args:
            updated_tools (list(str)): list of default tools to include
        """

        self.tools.clear()
        for tool in updated_tools:
            if tool == "search":
                self.tools["search"] = self.search
            elif tool == "locate":
                self.tools["locate"] = self.locate
            elif tool == "vision":
                self.tools["vision"] = self.vision
            elif tool == "stocks":
                self.tools["stocks"] = self.stocks
        self._init_ReAct_agent()

    def _clean_response(self, response: str) -> str:
        """Clean response to remove any fake conversation turns or role labels.
        
        Args:
            response (str): Raw response from the LLM
            
        Returns:
            str: Cleaned response with fake conversations removed
        """
        import re
        
        # Remove any text that looks like fake conversation turns
        # Pattern: user: ... assistant: ... or \nuser: ... \nassistant: ...
        patterns = [
            r'\nuser:\s*.*?(?=\nassistant:|\nuser:|$)',
            r'\nassistant:\s*.*?(?=\nuser:|\nassistant:|$)',
            r'user:\s*.*?(?=assistant:|user:|$)',
            r'assistant:\s*.*?(?=user:|assistant:|$)',
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up any remaining artifacts
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)  # Remove multiple newlines
        cleaned = cleaned.strip()
        
        # If the response got completely cleaned (was all fake conversation), 
        # return just the first sentence before any fake turns
        if not cleaned or len(cleaned) < 10:
            # Extract first sentence/paragraph before any fake conversation
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not re.match(r'^(user|assistant):\s*', line, re.IGNORECASE):
                    return line
            return "I'm here to help! How can I assist you?"
        
        return cleaned
