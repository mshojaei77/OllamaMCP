# pip install langchain-ollama langchain-core langchain-mcp-adapters

import asyncio
import os
import logging
from typing import Optional, List, Any, Dict, Generator
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage
)
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
app_logger = logging.getLogger(__name__)

load_dotenv()  # load environment variables from .env

class BaseChatbot:
    """Base class for chatbot implementations"""

    def __init__(
        self,
        model_name: str = "llama3.2",
        system_message: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.system_message = system_message
        self.verbose = verbose
        self.memory = ConversationBufferMemory(return_messages=True)

    async def initialize(self) -> None:
        """Initialize the chatbot - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement initialize()")

    async def chat(self, message: str) -> str:
        """Process a chat message and return the response"""
        raise NotImplementedError("Subclasses must implement chat()")

    async def cleanup(self) -> None:
        """Clean up any resources used by the chatbot"""
        self.memory = ConversationBufferMemory(return_messages=True)

    def get_history(self) -> List[BaseMessage]:
        """Get the conversation history"""
        memory_variables = self.memory.load_memory_variables({})
        return memory_variables.get("history", [])

    def clear_history(self) -> None:
        """Clear the conversation history"""
        self.memory.clear()

class OllamaMCPChatbot(BaseChatbot):
    """Chatbot implementation using Ollama with MCP tools integration"""

    def __init__(
        self,
        model_name: str = "llama3.2",
        system_message: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        mcp_server_command: str = "uvx",
        mcp_server_args: List[str] = ["mcp-server-calculator"],
        verbose: bool = False,
    ):
        super().__init__(model_name, system_message, verbose)
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.top_p = top_p
        self.chat_model = None
        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args
        self.mcp_server_params = StdioServerParameters(
            command=mcp_server_command,
            args=mcp_server_args,
        )
        self.mcp_session = None
        self.mcp_read = None
        self.mcp_write = None
        self.mcp_tools = None
        self.agent = None
        self.ready = False

    async def initialize(self) -> None:
        try:
            # Initialize Ollama model
            self.chat_model = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            if self.verbose:
                app_logger.info(f"Initializing Ollama with model: {self.model_name}")

            # Initialize MCP connection
            if self.verbose:
                app_logger.info(f"Initializing MCP connection")
            
            # We need to use client and session as part of instance variables for persistence
            self.mcp_client = stdio_client(self.mcp_server_params)
            self.mcp_read, self.mcp_write = await self.mcp_client.__aenter__()
            self.mcp_session = ClientSession(self.mcp_read, self.mcp_write)
            await self.mcp_session.__aenter__()
            
            # Initialize the MCP connection
            await self.mcp_session.initialize()
            
            # Get tools
            self.mcp_tools = await load_mcp_tools(self.mcp_session)
            
            if self.verbose:
                app_logger.info("Available MCP tools:")
                for tool in self.mcp_tools:
                    app_logger.info(f"- {tool.name}: {tool.description}")
            
            # Create agent with tools
            self.agent = create_react_agent(self.chat_model, self.mcp_tools)
            
            self.ready = True

            if self.system_message:
                self.memory.chat_memory.add_message(SystemMessage(content=self.system_message))

        except Exception as e:
            app_logger.error(f"Failed to initialize OllamaMCP chatbot: {str(e)}")
            self.ready = False
            raise

    async def chat(self, message: str) -> str:
        if not self.ready:
            await self.initialize()

        if not self.ready:
            return "Chatbot is not ready. Please check the logs and try again."

        history = self.get_history()
        self.memory.chat_memory.add_message(HumanMessage(content=message))

        try:
            # Format messages for the agent
            messages_dict = {"messages": history + [HumanMessage(content=message)]}
            
            # Invoke agent
            agent_response = await self.agent.ainvoke(messages_dict)
            
            # Process and extract the response
            response_content = ""
            for message in agent_response.get("messages", []):
                if isinstance(message, AIMessage) and message.content:
                    response_content = message.content
                    self.memory.chat_memory.add_message(AIMessage(content=response_content))
                elif isinstance(message, ToolMessage) and message.content:
                    # For tool messages, we add them to history but don't consider them the final response
                    self.memory.chat_memory.add_message(message)
            
            return response_content

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            app_logger.error(error_msg)
            return error_msg

    async def cleanup(self) -> None:
        await super().cleanup()
        
        # Clean up MCP resources
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
                await self.mcp_client.__aexit__(None, None, None)
            except Exception as e:
                app_logger.error(f"Error cleaning up MCP session: {str(e)}")
        
        self.chat_model = None
        self.mcp_session = None
        self.mcp_read = None
        self.mcp_write = None
        self.mcp_tools = None
        self.agent = None
        self.ready = False

    def get_available_tools(self) -> List[Dict]:
        """Get information about available MCP tools"""
        if not self.mcp_tools:
            return []
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": getattr(tool, "args_schema", None)
            }
            for tool in self.mcp_tools
        ]

async def main():
    # Initialize the chatbot with MCP tools
    chatbot = OllamaMCPChatbot(
        model_name="llama3.2",
        system_message="You are a helpful AI assistant that can use tools to solve problems.",
        verbose=True
    )
    
    try:
        # Initialize the chatbot
        await chatbot.initialize()
        
        # Show available tools
        tools = chatbot.get_available_tools()
        print("\n=== Available MCP Tools ===")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")
        
        # Example calculation using MCP tools
        print("\n=== Calculation Query Example ===")
        response = await chatbot.chat("what's (3 + 5) x 12?")
        print(f"Bot: {response}")
        
        # Example of another query
        print("\n=== Another Query Example ===")
        response = await chatbot.chat("Can you calculate 25 * 4 + 7?")
        print(f"Bot: {response}")
        
        # Example of getting chat history
        print("\n=== Chat History ===")
        history = chatbot.get_history()
        for message in history:
            if isinstance(message, SystemMessage):
                role = "System"
            elif isinstance(message, HumanMessage):
                role = "Human"
            elif isinstance(message, AIMessage):
                role = "AI"
            elif isinstance(message, ToolMessage):
                role = f"Tool({message.tool_call_id})"
            else:
                role = message.__class__.__name__
            
            print(f"{role}: {message.content}")
    
    finally:
        # Cleanup
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 