clients = {
        "openai": {




            if model in model_list["openai"]:
                return "openai", "OpenAIClientSync"
            elif model in model_list["anthropic"]:
                return "anthropic", "AnthropicClientSync"
            elif model in model_list["google"]:
                return "google", "GoogleClientSync"
            elif model in model_list["ollama"]:
                return "ollama", "OllamaClientSync"
            elif model in model_list["groq"]:
                return "groq", "GroqClientSync"
            elif model in model_list["perplexity"]:
                return "perplexity", "PerplexityClientSync"
            else:
                raise ValueError(f"Model {model} not found in synchronous clients")


