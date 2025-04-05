#!/usr/bin/env python
import asyncio
import sys
import logging
from ollama_mcp_adapter import OllamaMCPChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the chatbot with MCP support
    chatbot = OllamaMCPChatbot(
        model_name="llama3.2",
        system_message="You are a helpful AI assistant who can use tools.",
        verbose=True
    )
    
    try:
        # Initialize the chatbot
        print("Initializing chatbot with MCP support...")
        
        try:
            await chatbot.initialize()
            print("Initialization completed successfully.")
        except Exception as e:
            print(f"Warning: Some MCP servers failed to initialize: {str(e)}")
            print("Continuing with available servers...")
        
        # Display available tools from active MCP servers
        tools = await chatbot.mcp_manager.list_tools()
        
        print("\n=== Available MCP Tools ===")
        if not tools:
            print("No active MCP tools available. Check your mpc_servers.json configuration.")
        else:
            for server, server_tools in tools.items():
                print(f"\nServer: {server}")
                for tool in server_tools:
                    print(f"  - {tool['name']}: {tool['description']}")
                    
                    # Print parameters if available
                    params = tool.get("parameters", {}).get("properties", {})
                    if params:
                        print("    Parameters:")
                        for param_name, param_info in params.items():
                            param_desc = param_info.get("description", "No description")
                            param_type = param_info.get("type", "unknown")
                            print(f"      - {param_name} ({param_type}): {param_desc}")
        
        # Interactive chat loop
        print("\n=== Chatbot with MCP tools is ready ===")
        print('Type "@server_name/tool_name(param1=value1, param2=value2)" to use a tool')
        print('Type "exit" to quit')
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() == "exit":
                    break
                
                # Handle streaming output
                print("Bot: ", end="", flush=True)
                async for chunk in chatbot.chat_stream(user_input):
                    if chunk is not None:
                        print(chunk, end="", flush=True)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
        
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        await chatbot.cleanup()
        print("Done. Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0) 