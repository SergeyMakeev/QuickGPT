import os

# Remove proxy-related environment variables
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)

import sys
import json
import openai
import anthropic
import pyperclip
import subprocess
import ollama
import requests
import re
import time
from colorama import Fore, Back, Style, init
from google import genai

# Initialize colorama for cross-platform colored output
init(autoreset=True)

chatgpt_client = None
openai_model = "gpt-4.5-preview"

claude_client = None
anthropic_model = "claude-3-7-sonnet-20250219"

grok_client = None
xai_model = "grok-2-1212"

perplexity_client = None
perplexity_model = "r1-1776"

deepseek_client = None
deepseek_model = "deepseek-chat"

# ollama (local)
ollama_client = None
ollama_model = "deepseek-r1:latest"
# "ollama" : "http://localhost:11434"
ollama_host = None

google_client = None
google_model = "gemini-2.0-flash"

# chatgpt | claude | grok | perplexity | deepseek | ollama | google
agent_name = "chatgpt"


# Note: different agents are using OpenAI API (they were intentionally made compatible)
def run_openai(
        client, model: str, context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):

    if not stream:
        print("Thinking\n")
    
    messages = []
    
    # If we have conversation history, use it
    if conversation_history:
        messages = conversation_history.copy()
    else:
        # Original single-shot behavior
        if context and len(context) > 0:
            messages.append({
                "role": "system",
                "content": context
            })

        if input_text and len(input_text) > 0:
            messages.append({
                "role": "user",
                "content": input_text
            })

        if input_text2 is not None:
            messages.append({
               "role": "user",
               "content": input_text2
            })

    # If we have conversation history but need to add new user input
    if conversation_history and input_text:
        messages.append({
            "role": "user", 
            "content": input_text
        })

    response = client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      stream=stream
    )
    
    if stream:
        # Handle streaming response
        response_content = ""
        try:
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response_content += content
            print()  # New line after streaming is complete
        except Exception as e:
            print(f"\n[ERROR] Streaming failed: {e}")
            response_content = "Error: Streaming failed"
    else:
        if response.choices and len(response.choices) > 0:
            response_content = response.choices[0].message.content
        else:
            response_content = "Error: No response content received"
    
    # Add assistant response to conversation history if we're in chat mode
    if conversation_history is not None:
        # Rebuild conversation history with the new assistant response
        new_history = []
        
        # Add system message first if we have one
        if context and len(context) > 0:
            new_history.append({
                "role": "system",
                "content": context
            })
        
        # Add all messages that were sent to the API
        for msg in messages:
            new_history.append(msg)
        
        # Add the assistant response
        new_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        # Replace the conversation history
        conversation_history.clear()
        conversation_history.extend(new_history)
    
    return response_content


# Note: different agents are using Anthropic API (they were intentionally made compatible)
def run_anthropic(
        client, model: str, context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):

    if not stream:
        print("Thinking\n")
    
    temperature = min(temperature, 1.0)
    messages = []
    system_message = None
    
    # If we have conversation history, use it
    if conversation_history:
        # Extract system message and regular messages separately for Anthropic
        for msg in conversation_history:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] in ["user", "assistant"]:
                messages.append(msg)
    else:
        # Original single-shot behavior
        if context and len(context) > 0:
            system_message = context

        if input_text and len(input_text) > 0:
            messages.append({
                "role": "user",
                "content": input_text
            })

        if input_text2 is not None:
            messages.append({
               "role": "user",
               "content": input_text2
            })

    # If we have conversation history but need to add new user input
    if conversation_history and input_text:
        messages.append({
            "role": "user", 
            "content": input_text
        })

    # Filter out any empty or invalid messages
    valid_messages = []
    for msg in messages:
        if (msg.get("role") in ["user", "assistant"] and 
            msg.get("content") and 
            isinstance(msg.get("content"), str) and 
            msg.get("content").strip()):
            valid_messages.append(msg)
    
    if not valid_messages:
        return "Error: No valid messages to send to API"
    
    messages = valid_messages
    
    # Ensure first message is from user (Anthropic requirement)
    if messages[0]["role"] != "user":
        return "Error: First message must be from user for Anthropic API"

    # Prepare the API call parameters
    api_params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }
    
    # Add system message if we have one
    if system_message:
        api_params["system"] = system_message
    
    # Important: Don't add stream=True to api_params here since we handle it separately

    if stream:
        # Implement proper Anthropic streaming with defensive programming
        try:
            response_content = ""
            # Use the streaming API with proper error handling
            stream_response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                system=system_message if system_message else None,
                stream=True
            )
            
            for chunk in stream_response:
                # Handle different types of streaming chunks safely
                if hasattr(chunk, 'type'):
                    if chunk.type == 'content_block_delta':
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                            text = chunk.delta.text
                            print(text, end="", flush=True)
                            response_content += text
                    elif chunk.type == 'content_block_start':
                        if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'text'):
                            text = chunk.content_block.text
                            print(text, end="", flush=True)
                            response_content += text
            print()  # New line after streaming is complete
            
        except Exception as e:
            print(f"\n[ERROR] Streaming failed: {e}")
            # Try basic non-streaming as fallback, but with small max_tokens to avoid timeout
            try:
                fallback_response = client.messages.create(
                    model=model,
                    max_tokens=min(max_tokens, 4096),  # Reduce to avoid timeout
                    temperature=temperature,
                    messages=messages,
                    system=system_message if system_message else None
                )
                if fallback_response.content and len(fallback_response.content) > 0:
                    response_content = fallback_response.content[0].text
                    print(response_content)
                else:
                    response_content = "Error: No response content received"
            except Exception as e2:
                print(f"[ERROR] Fallback also failed: {e2}")
                response_content = f"Error: Both streaming and fallback failed: {e}"
    else:
        # For non-streaming mode, try basic create first, fall back to streaming if needed
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                system=system_message if system_message else None
            )
            if message.content and len(message.content) > 0:
                response_content = message.content[0].text
                print(response_content)
            else:
                response_content = "Error: No response content received"
        except Exception as e:
            # If non-streaming fails (likely due to long operation requirement), use streaming
            print(f"[INFO] Non-streaming failed ({e}), switching to streaming...")
            try:
                response_content = ""
                stream_response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    system=system_message if system_message else None,
                    stream=True
                )
                
                for chunk in stream_response:
                    if hasattr(chunk, 'type'):
                        if chunk.type == 'content_block_delta':
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                                response_content += chunk.delta.text
                        elif chunk.type == 'content_block_start':
                            if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'text'):
                                response_content += chunk.content_block.text
                
                print(response_content)  # Print full response for non-streaming mode
            except Exception as e2:
                print(f"[ERROR] Streaming fallback failed: {e2}")
                response_content = f"Error: All methods failed: {e}"
    
    # Add assistant response to conversation history if we're in chat mode
    if conversation_history is not None:
        # Rebuild conversation history with the new assistant response
        new_history = []
        
        # Add system message first if we have one
        if system_message:
            new_history.append({
                "role": "system",
                "content": system_message
            })
        
        # Add all messages that were sent to the API
        for msg in messages:
            new_history.append(msg)
        
        # Add the assistant response
        new_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        # Replace the conversation history
        conversation_history.clear()
        conversation_history.extend(new_history)
    
    return response_content


def run_ollama(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):
    assert (isinstance(ollama_client, ollama.Client))
    max_tokens = min(max_tokens, 32768)
    
    if not stream:
        print("Thinking\n")

    messages = []
    
    # If we have conversation history, use it
    if conversation_history:
        messages = conversation_history.copy()
    else:
        # Original single-shot behavior
        if context and len(context) > 0:
            messages.append({
                "role": "user",
                "content": context
            })

        if input_text and len(input_text) > 0:
            messages.append({
                "role": "user",
                "content": input_text
            })

        if input_text2 is not None:
            messages.append({
               "role": "user",
               "content": input_text2
            })

    # If we have conversation history but need to add new user input
    if conversation_history and input_text:
        messages.append({
            "role": "user", 
            "content": input_text
        })

    if stream:
        # Handle streaming response for Ollama
        response_content = ""
        try:
            response = ollama_client.chat(
                model=ollama_model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message'] and chunk['message']['content'] is not None:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    response_content += content
            print()  # New line after streaming is complete
            cleaned_content = re.sub(r"<think>.*?</think>\n?", "", response_content, flags=re.DOTALL)
        except Exception as e:
            print(f"\n[ERROR] Ollama streaming failed: {e}")
            # Fallback to non-streaming
            try:
                response = ollama_client.chat(
                    model=ollama_model,
                    messages=messages,
                )
                res = response['message']['content']
                cleaned_content = re.sub(r"<think>.*?</think>\n?", "", res, flags=re.DOTALL)
            except Exception as e2:
                print(f"[ERROR] Ollama fallback failed: {e2}")
                cleaned_content = "Error: Failed to get response from Ollama"
    else:
        try:
            response = ollama_client.chat(
                model=ollama_model,
                messages=messages,
            )
            res = response['message']['content']
            cleaned_content = re.sub(r"<think>.*?</think>\n?", "", res, flags=re.DOTALL)
        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            cleaned_content = "Error: Failed to get response from Ollama"
    
    # Add assistant response to conversation history if we're in chat mode
    if conversation_history is not None:
        messages.append({
            "role": "assistant",
            "content": cleaned_content
        })
        # Update the conversation history in place
        conversation_history.clear()
        conversation_history.extend(messages)
    
    return cleaned_content


def run_google(
        context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):

    if not stream:
        print("Thinking\n")
    
    messages = []
    system_instruction = context  # Default system instruction

    # If we have conversation history, use it (Google API is different)
    if conversation_history:
        # Extract system instruction and user messages from conversation history
        for msg in conversation_history:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                messages.append(msg["content"])
            # Skip assistant messages as Google API handles them differently
        
        # Add new user input if provided
        if input_text and len(input_text) > 0:
            messages.append(input_text)
    else:
        # Original single-shot behavior
        if input_text and len(input_text) > 0:
            messages.append(input_text)

        if input_text2 is not None:
            messages.append(input_text2)

    # Note: Google API streaming support would need to be implemented here
    # For now, we'll use non-streaming mode even when stream=True
    if system_instruction:
        response = google_client.models.generate_content(
            model=google_model,
            contents=messages,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature
            )
        )
        response_text = response.text
    else:
        response = google_client.models.generate_content(
            model=google_model,
            contents=messages,
            config=genai.types.GenerateContentConfig(
                temperature=temperature
            )
        )
        response_text = response.text
    
    # Print the response (this was missing for streaming mode)
    if not stream:
        print(response_text)
    else:
        # For streaming mode, we should print the response since we're not actually streaming
        print(response_text)
    
    # Add to conversation history if we're in chat mode (simplified for Google API)
    if conversation_history is not None and input_text:
        conversation_history.append({
            "role": "user",
            "content": input_text
        })
        conversation_history.append({
            "role": "assistant", 
            "content": response_text
        })
    
    return response_text


def load_json(file_path: str):
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path) as json_file:
            data = json.load(json_file)
            return data
    except IOError:
        return None
    except ValueError:
        return None


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except UnicodeDecodeError:
        print(f"The file at {file_path} is not properly UTF-8 encoded.")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_chatgpt(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):
    assert(isinstance(chatgpt_client, openai.OpenAI))
    max_tokens = min(max_tokens, 16384)
    return run_openai(chatgpt_client, openai_model, context, input_text, temperature, max_tokens, input_text2, conversation_history, stream)


def run_perplexity(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):
    assert (isinstance(perplexity_client, openai.OpenAI))
    max_tokens = min(max_tokens, 127072)
    return run_openai(perplexity_client, perplexity_model, context, input_text, temperature, max_tokens, input_text2, conversation_history, stream)


def run_deepseek(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):
    assert (isinstance(deepseek_client, openai.OpenAI))
    max_tokens = min(max_tokens, 8192)
    return run_openai(deepseek_client, deepseek_model, context, input_text, temperature, max_tokens, input_text2, conversation_history, stream)


def run_claude(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):
    assert (isinstance(claude_client, anthropic.Anthropic))
    max_tokens = min(max_tokens, 8192)
    return run_anthropic(claude_client, anthropic_model, context, input_text, temperature, max_tokens, input_text2, conversation_history, stream)


def run_grok(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None, conversation_history=None, stream=False):
    assert (isinstance(grok_client, anthropic.Anthropic))
    max_tokens = min(max_tokens, 131072)
    return run_anthropic(grok_client, xai_model, context, input_text, temperature, max_tokens, input_text2, conversation_history, stream)


# function routing table
agent_functions = {
    "chatgpt": run_chatgpt,
    "claude": run_claude,
    "grok": run_grok,
    "perplexity": run_perplexity,
    "deepseek": run_deepseek,
    "ollama": run_ollama,
    "google": run_google,
}


def run_agent_task(task_name: str, text: str, temperature: float, max_tokens: int, input_text2=None):
    task_context = read_file(os.path.join("tasks", task_name))

    if agent_name in agent_functions:
        return agent_functions[agent_name](task_context, text, temperature, max_tokens, input_text2)
    else:
        print(f"Unsupported agent name: {agent_name}")
        return text


def ask_expert(prompt, expert_file):
    temperature = 0.7
    max_tokens = 131072
    task_context = read_file(os.path.join("experts", expert_file))

    if agent_name in agent_functions:
        return agent_functions[agent_name](task_context, prompt, temperature, max_tokens)
    else:
        print(f"Unsupported agent name: {agent_name}")
        return prompt


def chat_with_expert(expert_file, expert_name):
    """
    Start a chat session with a specific expert
    """
    temperature = 0.7
    max_tokens = 131072
    task_context = read_file(os.path.join("experts", expert_file))
    
    if not task_context:
        print(f"{Fore.RED}[ERROR] Could not load expert context from {expert_file}{Style.RESET_ALL}")
        return
    
    # Initialize conversation history with system context
    conversation_history = []
    if task_context and len(task_context) > 0:
        conversation_history.append({
            "role": "system",
            "content": task_context
        })
    
    last_response = None  # Track last response for copying
    
    print(f"\n{Back.BLUE}{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Back.BLUE}{Fore.WHITE}                Chat with {expert_name} Expert                {Style.RESET_ALL}")
    print(f"{Back.BLUE}{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}>> You are now chatting with the {expert_name} expert.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}>> Type 'exit', 'quit', or 'back' to return to the main menu.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}>> Type 'clear' to clear the conversation history.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}>> Type 'copy' to copy the last response to clipboard.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}>> Responses will stream in real-time for better interactivity.{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * 60}{Style.RESET_ALL}")
    
    while True:
        # Get user input
        user_input = input(f"\n{Fore.YELLOW}[You]: {Style.RESET_ALL}").strip()
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'back']:
            print(f"{Fore.CYAN}[INFO] Ending chat with {expert_name} expert.{Style.RESET_ALL}")
            break
        
        # Check for clear command
        if user_input.lower() == 'clear':
            # Reset conversation history with just the system context
            conversation_history = []
            if task_context and len(task_context) > 0:
                conversation_history.append({
                    "role": "system",
                    "content": task_context
                })
            last_response = None
            print(f"{Fore.GREEN}[INFO] Conversation history cleared.{Style.RESET_ALL}")
            continue
        
        # Check for copy command
        if user_input.lower() == 'copy':
            if last_response:
                pyperclip.copy(last_response)
                print(f"{Fore.GREEN}[INFO] Last response copied to clipboard!{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[INFO] No response to copy yet.{Style.RESET_ALL}")
            continue
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Get response from the expert using streaming
            print(f"\n{Fore.MAGENTA}[{expert_name} Expert]: {Style.RESET_ALL}", end="", flush=True)
            
            if agent_name in agent_functions:
                # Check if the agent function supports streaming
                import inspect
                sig = inspect.signature(agent_functions[agent_name])
                supports_streaming = 'stream' in sig.parameters
                
                if supports_streaming:
                    response = agent_functions[agent_name](
                        None,  # context is now in conversation_history
                        user_input, 
                        temperature, 
                        max_tokens, 
                        None,  # input_text2
                        conversation_history,
                        stream=True  # Enable streaming
                    )
                else:
                    # Fallback for agents that don't support streaming yet
                    response = agent_functions[agent_name](
                        None,  # context is now in conversation_history
                        user_input, 
                        temperature, 
                        max_tokens, 
                        None,  # input_text2
                        conversation_history
                    )
                    print(f"{response}")
                
                last_response = response  # Store for potential copying
            else:
                print(f"{Fore.RED}[ERROR] Unsupported agent name: {agent_name}{Style.RESET_ALL}")
                break
                
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Failed to get response from {expert_name} expert: {e}{Style.RESET_ALL}")
            continue


def get_git_diff(working_dir):
    print("Generate git diff")
    if not os.path.isdir(working_dir):
        print(f"Working dir '{working_dir}' does not exist")
        return None

    original_dir = os.getcwd()
    os.chdir(working_dir)

    result = subprocess.run(
        ["git", "diff", "--staged"],
        # cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # encoding='utf-8',
        text=True
    )

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(result.stderr)

    os.chdir(original_dir)
    git_diff = result.stdout
    return f"`````\n{git_diff}\n`````\n"


def is_directory_path(path: str) -> bool:
    return os.path.isdir(path)


def return_directory_path_or_fallback(input_text, fallback):
    if len(input_text) > 250:
        return fallback

    if is_directory_path(input_text):
        return input_text
    else:
        return fallback


def check_ollama_availability(url):
    # print("Check if ollama is available")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        # print("ollama is available!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to Ollama server at {url}. {e}")
        return False


def initialize():
    global chatgpt_client
    global claude_client
    global grok_client
    global perplexity_client
    global deepseek_client
    global ollama_client
    global ollama_host
    global google_client

    # List to store initialization times (model_name, init_time)
    init_times = []

    print(f"{Fore.CYAN}>> Initializing AI models...{Style.RESET_ALL}")
    
    api_keys = load_json("api_keys.json")

    # OpenAI/ChatGPT initialization
    openai_api_key = api_keys.get('openai', None)
    if openai_api_key:
        print(f"{Fore.YELLOW}  [*] Initializing ChatGPT...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        chatgpt_client = openai.OpenAI(api_key=openai_api_key)
        init_time = time.time() - start_time
        init_times.append(("ChatGPT", init_time))
        print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")
        # models = chatgpt_client.models.list()
        # print(models)

    # Anthropic/Claude initialization
    anthropic_api_key = api_keys.get('anthropic', None)
    if anthropic_api_key:
        print(f"{Fore.YELLOW}  [*] Initializing Claude...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        init_time = time.time() - start_time
        init_times.append(("Claude", init_time))
        print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")
        # models = claude_client.models.list()
        # print(models)

    # XAI/Grok initialization
    xai_api_key = api_keys.get('xai', None)
    if xai_api_key:
        print(f"{Fore.YELLOW}  [*] Initializing Grok...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        grok_client = anthropic.Anthropic(api_key=xai_api_key, base_url="https://api.x.ai",)
        init_time = time.time() - start_time
        init_times.append(("Grok", init_time))
        print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")
        # models = grok_client.models.list()
        # print(models)

    # Perplexity initialization
    perplexity_api_key = api_keys.get('perplexity', None)
    if perplexity_api_key:
        print(f"{Fore.YELLOW}  [*] Initializing Perplexity...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        perplexity_client = openai.OpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")
        init_time = time.time() - start_time
        init_times.append(("Perplexity", init_time))
        print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")
        # models = perplexity_client.models.list()
        # print(models)

    # DeepSeek initialization
    deepseek_api_key = api_keys.get('deepseek', None)
    if deepseek_api_key:
        print(f"{Fore.YELLOW}  [*] Initializing DeepSeek...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        deepseek_client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        init_time = time.time() - start_time
        init_times.append(("DeepSeek", init_time))
        print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")
        # models = deepseek_client.models.list()
        # print(models)

    # Google/Gemini initialization
    google_api_key = api_keys.get('google', None)
    if google_api_key:
        print(f"{Fore.YELLOW}  [*] Initializing Google...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        google_client = genai.Client(api_key=google_api_key)
        init_time = time.time() - start_time
        init_times.append(("Google", init_time))
        print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")

    # Ollama initialization
    ollama_host = api_keys.get('ollama', None)
    if ollama_host:
        print(f"{Fore.YELLOW}  [*] Initializing Ollama...{Style.RESET_ALL}", end="", flush=True)
        sys.stdout.flush()  # Extra flush for problematic terminals
        start_time = time.time()
        if check_ollama_availability(ollama_host):
            ollama_client = ollama.Client(host=ollama_host)
            init_time = time.time() - start_time
            init_times.append(("Ollama", init_time))
            print(f"{Fore.GREEN} [OK]{Style.RESET_ALL}")
        else:
            init_time = time.time() - start_time
            init_times.append(("Ollama", init_time))
            print(f"{Fore.RED} [FAILED - Server not available]{Style.RESET_ALL}")

    # Print initialization times sorted from slowest to fastest
    # if init_times:
    #     print(f"\n{Fore.CYAN}>> Model initialization times (slowest to fastest):{Style.RESET_ALL}")
    #     print(f"{Fore.BLUE}{'=' * 52}{Style.RESET_ALL}")
    #     # Sort by init_time in descending order (slowest first)
    #     sorted_times = sorted(init_times, key=lambda x: x[1], reverse=True)
    #     for i, (model_name, init_time) in enumerate(sorted_times):
    #         if i == 0:  # Slowest
    #             color = Fore.RED
    #             icon = "[SLOW]"
    #         elif i == len(sorted_times) - 1:  # Fastest
    #             color = Fore.GREEN
    #             icon = "[FAST]"
    #         else:  # Middle
    #             color = Fore.YELLOW
    #             icon = "[MID] "
    #         print(f"{color}{icon} {model_name:<12}: {init_time:.4f} seconds{Style.RESET_ALL}")
    #     print(f"{Fore.BLUE}{'=' * 52}{Style.RESET_ALL}")


def print_header():
    """Print a nice header for the application"""
    print(f"\n{Back.BLUE}{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Back.BLUE}{Fore.WHITE}                     QuickGPT Assistant                     {Style.RESET_ALL}")
    print(f"{Back.BLUE}{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")


def print_clipboard_content(clipboard_text):
    """Print clipboard content in a nice format"""
    print(f"\n{Fore.CYAN}>> Clipboard Content:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")
    
    # Truncate very long text for display
    display_text = clipboard_text
    if len(clipboard_text) > 500:
        display_text = clipboard_text[:500] + f"{Fore.YELLOW}... (truncated, {len(clipboard_text)} total chars){Style.RESET_ALL}"
    
    print(f"{Fore.WHITE}{display_text}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")


def get_agent_choice(agents):
    """Display agent selection menu and return chosen agent"""
    print(f"\n{Fore.MAGENTA}>> Select AI Agent:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * 30}{Style.RESET_ALL}")
    
    for i, agent in enumerate(agents, 1):
        print(f"{Fore.WHITE}{i}. {agent.title()}{Style.RESET_ALL}")
    
    print(f"{Fore.RED}0. Exit{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'-' * 30}{Style.RESET_ALL}")
    
    while True:
        choice = input(f"{Fore.CYAN}>> Choose agent (1-{len(agents)}, 0 to exit): {Style.RESET_ALL}").strip()
        
        if choice == "0":
            return None
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(agents):
                selected_agent = agents[choice_num - 1]
                print(f"{Fore.GREEN}[OK] Selected: {selected_agent.title()}{Style.RESET_ALL}")
                return selected_agent
        except ValueError:
            pass
        
        print(f"{Fore.RED}[ERROR] Invalid choice. Please try again.{Style.RESET_ALL}")


def display_menu(dir_path):

    def fix_grammar():
        print(f"{Fore.YELLOW}[*] Fixing grammar...{Style.RESET_ALL}")
        answer = run_agent_task("grammar.txt", clipboard_text, 0.7, 1024)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def translate_ru():
        print(f"{Fore.YELLOW}[*] Translating to Russian...{Style.RESET_ALL}")
        answer = run_agent_task("ru.txt", clipboard_text, 1.3, 1024)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def translate_en():
        print(f"{Fore.YELLOW}[*] Translating to English...{Style.RESET_ALL}")
        answer = run_agent_task("en.txt", clipboard_text, 1.3, 1024)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def generate_commit_message():
        print(f"{Fore.YELLOW}[*] Generating commit message...{Style.RESET_ALL}")
        staged_changes = get_git_diff(dir_path)
        if not staged_changes:
            print(f"{Fore.RED}[ERROR] No staged changes found.{Style.RESET_ALL}")
        else:
            answer = run_agent_task("commit_message.txt", staged_changes, 0.2, 16384)
            print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
            print(answer)
            pyperclip.copy(answer)
            print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def code_review_changes():
        print(f"{Fore.YELLOW}[*] Reviewing code changes...{Style.RESET_ALL}")
        staged_changes = get_git_diff(dir_path)
        if not staged_changes:
            print(f"{Fore.RED}[ERROR] No staged changes found.{Style.RESET_ALL}")
        else:
            answer = run_agent_task("code_review.txt", staged_changes, 0.1, 16384)
            print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
            print(answer)
            pyperclip.copy(answer)
            print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def reply_to_email():
        print(f"{Fore.YELLOW}[*] Preparing email reply...{Style.RESET_ALL}")
        draft_answer = input(f"{Fore.CYAN}>> What would you like me to reply? {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Generating reply...{Style.RESET_ALL}")
        answer = run_agent_task("email_reply.txt", clipboard_text, 1.3, 1024, draft_answer)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def summarize():
        print(f"{Fore.YELLOW}[*] Summarizing content...{Style.RESET_ALL}")
        answer = run_agent_task("summarize.txt", clipboard_text, 0.7, 16384)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def explain():
        print(f"{Fore.YELLOW}[*] Explaining content...{Style.RESET_ALL}")
        answer = run_agent_task("explain.txt", clipboard_text, 0.7, 16384)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def structure_braindump():
        print(f"{Fore.YELLOW}[*] Structuring braindump...{Style.RESET_ALL}")
        answer = run_agent_task("braindump.txt", clipboard_text, 0.5, 16384)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def fix_commit_message():
        print(f"{Fore.YELLOW}[*] Fixing commit message...{Style.RESET_ALL}")
        answer = run_agent_task("fix_commit_message.txt", clipboard_text, 0.7, 1024)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def raw_prompt():
        print(f"{Fore.YELLOW}[*] Processing raw prompt...{Style.RESET_ALL}")
        answer = run_agent_task("raw.txt", clipboard_text, 1.0, 8192)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def custom_raw_prompt():
        user_raw_prompt = input(f"{Fore.CYAN}>> Type your custom prompt: {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[*] Processing custom prompt...{Style.RESET_ALL}")
        answer = run_agent_task("raw.txt", user_raw_prompt, 1.0, 8192)
        print(f"\n{Fore.GREEN}[RESULT]{Style.RESET_ALL}")
        print(answer)
        pyperclip.copy(answer)
        print(f"{Fore.CYAN}[OK] Copied to clipboard!{Style.RESET_ALL}")

    def ask_rendering_expert():
        chat_with_expert("rendering.txt", "Rendering")

    def ask_cpp_expert():
        chat_with_expert("cpp.txt", "C++")

    def ask_manager_expert():
        chat_with_expert("manager.txt", "Management")

    def ask_entrepreneur_expert():
        chat_with_expert("entrepreneur.txt", "Entrepreneur")

    # Build a dictionary of menu items: "key": (label, handler_function)
    menu_items = {
        "1": ("Fix grammar", fix_grammar),
        "2": ("Translate to Russian", translate_ru),
        "3": ("Translate to English", translate_en),
        "4": (f"Generate commit message ({dir_path})", generate_commit_message),
        "5": (f"Code review changes ({dir_path})", code_review_changes),
        "6": ("Fix commit message", fix_commit_message),
        "7": ("Reply to email", reply_to_email),
        "8": ("Summarize", summarize),
        "9": ("Explain", explain),
        "10": ("Structure (braindump)", structure_braindump),
        "11": ("Raw prompt", raw_prompt),
        "12": ("Custom prompt...", custom_raw_prompt),
        "13": ("Chat with rendering expert", ask_rendering_expert),
        "14": ("Chat with C++ expert", ask_cpp_expert),
        "15": ("Chat with management expert", ask_manager_expert),
        "16": ("Chat with entrepreneur expert", ask_entrepreneur_expert),
        "0": ("Exit", None)
    }

    while True:
        # Read fresh clipboard content and print it before showing menu
        clipboard_text = pyperclip.paste()
        print_clipboard_content(clipboard_text)
        
        # Display the menu
        # print(f"\n{Fore.MAGENTA}>> Available Actions for {agent_name.title()}:{Style.RESET_ALL}")
        # print(f"{Fore.BLUE}{'=' * 60}{Style.RESET_ALL}")
        
        # Group menu items for better readability
        text_actions = ["1", "2", "3"]
        git_actions = ["4", "5", "6"]
        content_actions = ["7", "8", "9", "10"]
        prompt_actions = ["11", "12"]
        expert_actions = ["13", "14", "15", "16"]
        
        def print_menu_group(title, items, color=Fore.WHITE):
            print(f"{color}{title}:{Style.RESET_ALL}")
            for key in items:
                if key in menu_items:
                    label, _ = menu_items[key]
                    print(f"  {Fore.WHITE}{key:>2}. {label}{Style.RESET_ALL}")
        
        print_menu_group("TEXT PROCESSING", text_actions)
        print_menu_group("GIT OPERATIONS", git_actions)
        print_menu_group("CONTENT TOOLS", content_actions)
        print_menu_group("CUSTOM PROMPTS", prompt_actions)
        print_menu_group("EXPERT CHAT", expert_actions)
        
        print(f"\n{Fore.RED}   0. Exit{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'=' * 60}{Style.RESET_ALL}")

        choice = input(f"{Fore.CYAN}>> Choose an action (0-16): {Style.RESET_ALL}").strip()

        if choice == "0":
            print(f"{Fore.YELLOW}[EXIT] Goodbye!{Style.RESET_ALL}")
            break

        if choice in menu_items:
            label, action = menu_items[choice]
            if action:
                print(f"\n{Fore.CYAN}[RUN] Executing: {label}{Style.RESET_ALL}")
                action()  # Execute the corresponding function
                
                # Expert chat functions handle their own interaction loops
                # so we don't need to ask "Continue?" for them
                if choice not in ["13", "14", "15", "16"]:
                    # Ask if user wants to continue for non-chat functions
                    print(f"\n{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")
                    continue_choice = input(f"{Fore.CYAN}>> Continue? (y/n): {Style.RESET_ALL}").strip().lower()
                    if continue_choice not in ['y', 'yes', '']:
                        break
            else:
                print(f"{Fore.RED}[ERROR] No action defined for this option.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[ERROR] Invalid choice. Please try again.{Style.RESET_ALL}")


def main():
    global agent_name

    print_header()
    initialize()

    clipboard_text = pyperclip.paste()
    dir_path = return_directory_path_or_fallback(clipboard_text, ".")

    agents = []
    if chatgpt_client:
        agents.append('chatgpt')
    if claude_client:
        agents.append('claude')
    if grok_client:
        agents.append('grok')
    if perplexity_client:
        agents.append('perplexity')
    if deepseek_client:
        agents.append('deepseek')
    if google_client:
        agents.append('google')
    if ollama_client:
        agents.append('ollama')

    if len(agents) == 0:
        print(f"{Fore.RED}[ERROR] No agents found! Please check your API keys.{Style.RESET_ALL}")
        return

    agent_name = agents[0]

    # Select an active agent if multiple agents are available
    if len(agents) > 1:
        selected_agent = get_agent_choice(agents)
        if selected_agent is None:
            print(f"{Fore.YELLOW}[EXIT] Goodbye!{Style.RESET_ALL}")
            return
        agent_name = selected_agent

    display_menu(dir_path)

    print(f"\n{Fore.GREEN}[DONE] Thank you for using QuickGPT!{Style.RESET_ALL}")


main()
