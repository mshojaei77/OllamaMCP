import json
from praisonaiagents import Agent, MCP

LLM = "ollama/llama3.2"

def load_mcp_config():
    with open('mpc_servers.json', 'r') as f:
        return json.load(f)

def create_agents():
    config = load_mcp_config()
    agents = {}
    
    for server_name, server_config in config['mcpServers'].items():
        if not server_config.get('active', False):
            continue
            
        # Construct MCP command
        mcp_command = [server_config['command']]
        mcp_command.extend(server_config['args'])
        mcp_command = " ".join(mcp_command)
        
        # Create agent with configuration
        agents[server_name] = Agent(
            instructions=server_config['instructions'],
            llm=LLM,
            tools=MCP(mcp_command),
            verbose=True 
        )
    
    return agents

if __name__ == "__main__":
    # Create all configured agents
    agents = create_agents()
    
    # Example usage of the Airbnb agent
    if 'airbnb' in agents:
        airbnb_agent = agents['airbnb']
        response = airbnb_agent.start("Search for Apartments in Paris for 2 nights. 04/28 - 04/30 for 2 adults.")
        print("\nSearch Results:")
        print(response)
