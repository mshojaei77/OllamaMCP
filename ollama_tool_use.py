import ollama
import yfinance as yf
import requests
from typing import Dict, Any, Callable


class OllamaToolManager:
    def __init__(self):
        self.available_functions = {
            'get_stock_price': self.get_stock_price,
            'request': self.request_with_headers,
            'add_numbers': self.add_numbers,
            'subtract_numbers': self.subtract_numbers,
            'arithmetic': self.arithmetic
        }

        self.tools = [
            self._create_stock_price_tool(),
            self._create_arithmetic_tools(),
            self._create_request_tool()
        ]

    def get_stock_price(self, symbol: str) -> float:
        """Get current stock price for a symbol"""
        ticker = yf.Ticker(symbol)
        price_attrs = ['regularMarketPrice', 'currentPrice', 'price']
        
        for attr in price_attrs:
            if attr in ticker.info and ticker.info[attr] is not None:
                return ticker.info[attr]
                
        fast_info = ticker.fast_info
        if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
            return fast_info.last_price
            
        raise Exception("Could not find valid price data")

    def add_numbers(self, a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    def subtract_numbers(self, a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b
    
    def arithmetic(self, operation: str, a, b):
        """Route to the appropriate arithmetic operation"""
        # Convert string numbers to integers
        try:
            if isinstance(a, str):
                a = int(a)
            if isinstance(b, str):
                b = int(b)
                
            # Handle different operation formats
            if operation in ["add", "+"]:
                return self.add_numbers(a, b)
            elif operation in ["subtract", "-"]:
                return self.subtract_numbers(a, b)
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            return f"Error in arithmetic operation: {str(e)}"
    
    def request_with_headers(self, method: str, url: str, **kwargs):
        """Make a request with appropriate headers"""
        # Add default headers if not provided
        if 'headers' not in kwargs:
            kwargs['headers'] = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        
        # Add timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10
            
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return f"Successfully retrieved content from {url} (status code: {response.status_code})"
        except requests.exceptions.RequestException as e:
            return f"Error making request: {str(e)}"

    def _create_stock_price_tool(self):
        """Create tool definition for stock price function"""
        return {
            'type': 'function',
            'function': {
                'name': 'get_stock_price',
                'description': 'Get the current stock price for any symbol',
                'parameters': {
                    'type': 'object',
                    'required': ['symbol'],
                    'properties': {
                        'symbol': {'type': 'string', 'description': 'The stock symbol (e.g., AAPL, GOOGL)'},
                    },
                },
            },
        }

    def _create_arithmetic_tools(self):
        """Create tool definitions for arithmetic functions"""
        return {
            'type': 'function',
            'function': {
                'name': 'arithmetic',
                'description': 'Perform basic arithmetic operations',
                'parameters': {
                    'type': 'object',
                    'required': ['operation', 'a', 'b'],
                    'properties': {
                        'operation': {'type': 'string', 'enum': ['add', '+', 'subtract', '-']},
                        'a': {'type': 'integer'},
                        'b': {'type': 'integer'},
                    },
                },
            },
        }
    
    def _create_request_tool(self):
        """Create tool definition for web requests"""
        return {
            'type': 'function',
            'function': {
                'name': 'request',
                'description': 'Make a web request to any URL',
                'parameters': {
                    'type': 'object',
                    'required': ['method', 'url'],
                    'properties': {
                        'method': {'type': 'string', 'enum': ['GET', 'POST', 'PUT', 'DELETE']},
                        'url': {'type': 'string', 'description': 'The URL to request'},
                    },
                },
            },
        }

    def process_prompt(self, prompt: str, model: str = 'llama3.2'):
        """Process a user prompt using the Ollama model"""
        print('Prompt:', prompt)
        
        response = ollama.chat(
            model,
            messages=[{'role': 'user', 'content': prompt}],
            tools=self.tools
        )

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                if function_to_call := self.available_functions.get(tool.function.name):
                    print('Calling function:', tool.function.name)
                    print('Arguments:', tool.function.arguments)
                    result = function_to_call(**tool.function.arguments)
                    print('Function output:', result)
                    return result
                else:
                    print('Function', tool.function.name, 'not found')
        return None


def main():
    # Example usage
    tool_manager = OllamaToolManager()
    
    # Get stock price
    tool_manager.process_prompt("What is the current stock price of Apple?")
    
    # Arithmetic operation
    tool_manager.process_prompt("What is three plus one?")
    
    # Web request (using a reliable URL instead of ollama.com)
    tool_manager.process_prompt("get the https://httpbin.org/get webpage?")


if __name__ == "__main__":
    main()