# Ollama MCP (Multi-Agent Control Protocol)

A Python-based multi-agent system that uses Ollama's language models to control and coordinate multiple agents through a model context protocol (MCP) architecture.

## Overview

This project implements a flexible multi-agent system where each agent can be configured to perform specific tasks using Ollama's language models. The system uses a model context protocol (MCP) pattern to manage and coordinate these agents.

## Features

- Multiple agent support with individual configurations
- Integration with Ollama language models (currently using llama3.2)
- Configurable MCP servers through JSON configuration
- Flexible command and argument structure for each agent
- Active/inactive state management for servers

## Requirements

- Python 3.x
- Ollama installed and running
- `praisonaiagents` package
- Access to Ollama language models

## Configuration

The system uses a `mpc_servers.json` file to configure different MCP servers. Example configuration structure:

```json
{
    "mcpServers": {
        "serverName": {
            "active": true,
            "command": "command_to_run",
            "args": ["arg1", "arg2"],
            "instructions": "Agent specific instructions"
        }
    }
}
```

## Usage

1. Ensure your `mpc_servers.json` is properly configured
2. Run the main script:

```bash
python ollama_mcp.py
```

## Example

The script includes an example usage with an Airbnb agent:

```python
if 'airbnb' in agents:
    airbnb_agent = agents['airbnb']
    response = airbnb_agent.start("Search for Apartments in Paris for 2 nights. 04/28 - 04/30 for 2 adults.")
```

## Project Structure

- `ollama_mcp.py`: Main script containing the MCP implementation
- `mpc_servers.json`: Configuration file for MCP servers

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

