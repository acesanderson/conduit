import json
from openai import OpenAI

# 1. Initialize the OpenAI client to point to Ollama's local server
# Make sure Ollama is running (e.g., `ollama serve`)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but can be any string for ollama
)

# 2. Define the tool(s) in OpenAI's JSON Schema format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location", "unit"],
            },
        },
    }
]

# 3. Define the messages, including a user prompt that should trigger the tool
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston in fahrenheit?",
    }
]

print("Sending request to Ollama...")

try:
    # 4. Make the API call, passing the model, messages, and tools
    # Use a model known to support tool/function calling (e.g., llama3, phi3)
    response = client.chat.completions.create(
        model="llama3.1:latest",  # Or another capable model like "phi3"
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 'auto' lets the model decide, 'required' forces a tool
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 5. Check if the model decided to call a tool
    if tool_calls:
        print("\n--- Model decided to call a tool ---")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"Function Name: {function_name}")
            print(f"Arguments: {function_args}")

            # Here, you would typically execute the function.
            # For this example, we just print the call.

    else:
        print("\n--- Model responded directly ---")
        print(response_message.content)

except Exception as e:
    print(f"\nAn error occurred:")
    print(f"Error: {e}")
    print(
        "\nPlease ensure 'ollama serve' is running and you have a tool-calling model like 'llama3' or 'phi3' installed."
    )
