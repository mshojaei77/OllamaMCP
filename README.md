# Ollama MCP Chatbot

A chatbot implementation that combines [Ollama](https://ollama.ai/) with MCP (Machine Control Protocol) tools support. This allows the chatbot to use tools to perform tasks like calculations while maintaining conversation memory.

## Features

- Uses Ollama for LLM capabilities with LangChain integration
- Integrates MCP tools for extended capabilities 
- Maintains conversation history
- Supports custom system prompts
- Detailed logging and error handling

## Requirements

```
pip install langchain-ollama langchain-core langchain-mcp-adapters langgraph python-dotenv
```

You'll also need:
- Ollama running locally (or specify a custom host)
- The UVX MCP server tools installed (`uvx mcp-server-calculator`)

## Usage

Run the example in `ollama_mcp_chatbot.py`:

```python
# Initialize the chatbot with MCP tools
chatbot = OllamaMCPChatbot(
    model_name="llama3.2",
    system_message="You are a helpful AI assistant that can use tools to solve problems.",
    verbose=True
)

# Chat with the bot
response = await chatbot.chat("what's (3 + 5) x 12?")
print(f"Bot: {response}")
```

## Customization

You can customize the MCP server and tools by changing the initialization parameters:

```python
chatbot = OllamaMCPChatbot(
    model_name="llama3.2",
    base_url="http://your-ollama-host:11434",
    temperature=0.5,  
    mcp_server_command="your-command",
    mcp_server_args=["your", "args"],
    verbose=True
)
```

## Architecture

This implementation:
1. Extends the `BaseChatbot` class with MCP tool support
2. Uses LangChain's ReAct agent pattern to handle tool usage
3. Manages MCP connections appropriately through async context managers
4. Provides proper cleanup of resources

## License

MIT

