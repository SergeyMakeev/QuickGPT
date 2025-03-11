import os
import sys
import json
import openai
import anthropic
import pyperclip
import subprocess
import ollama
import requests
import re
from google import genai

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
        client, model: str, context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):

    print("Thinking\n")
    messages = []
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

    response = client.chat.completions.create(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens
    )
    return response.choices[0].message.content


# Note: different agents are using Anthropic API (they were intentionally made compatible)
def run_anthropic(
        client, model: str, context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):

    print("Thinking\n")
    temperature = min(temperature, 1.0)
    messages = []
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

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages
    )
    return message.content[0].text.strip()


def run_ollama(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):
    assert (isinstance(ollama_client, ollama.Client))
    max_tokens = min(max_tokens, 32768)
    print("Thinking\n")

    messages = []
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

    response = ollama_client.chat(
        model=ollama_model,
        messages=messages,
    )
    res = response['message']['content']
    cleaned_content = re.sub(r"<think>.*?</think>\n?", "", res, flags=re.DOTALL)
    return cleaned_content


def run_google(
        context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):

    print("Thinking\n")
    messages = []

    if input_text and len(input_text) > 0:
        messages.append(input_text)

    if input_text2 is not None:
        messages.append(input_text2)

    if context:
        response = google_client.models.generate_content(
            model=google_model,
            contents=messages,
            config=genai.types.GenerateContentConfig(
                system_instruction=context,
                temperature=temperature
            )
        )
        return response.text
    else:
        response = google_client.models.generate_content(
            model=google_model,
            contents=messages,
            config=genai.types.GenerateContentConfig(
                temperature=temperature
            )
        )
        return response.text


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


def run_chatgpt(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):
    assert(isinstance(chatgpt_client, openai.OpenAI))
    max_tokens = min(max_tokens, 16384)
    return run_openai(chatgpt_client, openai_model, context, input_text, temperature, max_tokens, input_text2)


def run_perplexity(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):
    assert (isinstance(perplexity_client, openai.OpenAI))
    max_tokens = min(max_tokens, 127072)
    return run_openai(perplexity_client, perplexity_model, context, input_text, temperature, max_tokens, input_text2)


def run_deepseek(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):
    assert (isinstance(deepseek_client, openai.OpenAI))
    max_tokens = min(max_tokens, 8192)
    return run_openai(deepseek_client, deepseek_model, context, input_text, temperature, max_tokens, input_text2)


def run_claude(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):
    assert (isinstance(claude_client, anthropic.Anthropic))
    max_tokens = min(max_tokens, 8192)
    return run_anthropic(claude_client, anthropic_model, context, input_text, temperature, max_tokens, input_text2)


def run_grok(context: str, input_text: str, temperature: float, max_tokens: int, input_text2=None):
    assert (isinstance(grok_client, anthropic.Anthropic))
    max_tokens = min(max_tokens, 131072)
    return run_anthropic(grok_client, xai_model, context, input_text, temperature, max_tokens, input_text2)


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

    api_keys = load_json("api_keys.json")

    openai_api_key = api_keys.get('openai', None)
    if openai_api_key:
        chatgpt_client = openai.OpenAI(api_key=openai_api_key)
        # models = chatgpt_client.models.list()
        # print(models)

    anthropic_api_key = api_keys.get('anthropic', None)
    if anthropic_api_key:
        claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        # models = claude_client.models.list()
        # print(models)

    xai_api_key = api_keys.get('xai', None)
    if xai_api_key:
        grok_client = anthropic.Anthropic(api_key=xai_api_key, base_url="https://api.x.ai",)
        # models = grok_client.models.list()
        # print(models)

    perplexity_api_key = api_keys.get('perplexity', None)
    if perplexity_api_key:
        perplexity_client = openai.OpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")
        # models = perplexity_client.models.list()
        # print(models)

    deepseek_api_key = api_keys.get('deepseek', None)
    if deepseek_api_key:
        deepseek_client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        # models = deepseek_client.models.list()
        # print(models)

    google_api_key = api_keys.get('google', None)
    if google_api_key:
        google_client = genai.Client(api_key=google_api_key)

    ollama_host = api_keys.get('ollama', None)
    if ollama_host and check_ollama_availability(ollama_host):
        ollama_client = ollama.Client(host=ollama_host)


def display_menu(dir_path, clipboard_text):

    def fix_grammar():
        answer = run_agent_task("grammar.txt", clipboard_text, 0.7, 1024)
        print(answer)
        pyperclip.copy(answer)

    def translate_ru():
        answer = run_agent_task("ru.txt", clipboard_text, 1.3, 1024)
        print(answer)
        pyperclip.copy(answer)

    def translate_en():
        answer = run_agent_task("en.txt", clipboard_text, 1.3, 1024)
        print(answer)
        pyperclip.copy(answer)

    def generate_commit_message():
        staged_changes = get_git_diff(dir_path)
        if not staged_changes:
            print("No staged changes found.")
        else:
            answer = run_agent_task("commit_message.txt", staged_changes, 0.2, 16384)
            print(answer)
            pyperclip.copy(answer)

    def code_review_changes():
        staged_changes = get_git_diff(dir_path)
        if not staged_changes:
            print("No staged changes found.")
        else:
            answer = run_agent_task("code_review.txt", staged_changes, 0.1, 16384)
            print(answer)
            pyperclip.copy(answer)

    def reply_to_email():
        draft_answer = input("Q: What would you like me to reply?\n\n")
        answer = run_agent_task("email_reply.txt", clipboard_text, 1.3, 1024, draft_answer)
        print(answer)
        pyperclip.copy(answer)

    def summarize():
        answer = run_agent_task("summarize.txt", clipboard_text, 0.7, 16384)
        print(answer)
        pyperclip.copy(answer)

    def explain():
        answer = run_agent_task("explain.txt", clipboard_text, 0.7, 16384)
        print(answer)
        pyperclip.copy(answer)

    def structure_braindump():
        answer = run_agent_task("braindump.txt", clipboard_text, 0.5, 16384)
        print(answer)
        pyperclip.copy(answer)

    def fix_commit_message():
        answer = run_agent_task("fix_commit_message.txt", clipboard_text, 0.7, 1024)
        print(answer)
        pyperclip.copy(answer)

    def raw_prompt():
        answer = run_agent_task("raw.txt", clipboard_text, 1.0, 8192)
        print(answer)
        pyperclip.copy(answer)

    def custom_raw_prompt():
        user_raw_prompt = input("Q: Type your prompt here\n\n")
        answer = run_agent_task("raw.txt", user_raw_prompt, 1.0, 8192)
        print(answer)
        pyperclip.copy(answer)

    def ask_rendering_expert():
        answer = ask_expert(clipboard_text, "rendering.txt")
        print(answer)
        pyperclip.copy(answer)

    def ask_cpp_expert():
        answer = ask_expert(clipboard_text, "cpp.txt")
        print(answer)
        pyperclip.copy(answer)

    def ask_manager_expert():
        answer = ask_expert(clipboard_text, "manager.txt")
        print(answer)
        pyperclip.copy(answer)

    def ask_entrepreneur_expert():
        answer = ask_expert(clipboard_text, "entrepreneur.txt")
        print(answer)
        pyperclip.copy(answer)

    # Build a dictionary of menu items: "key": (label, handler_function)
    menu_items = {
        "1": ("Fix grammar", fix_grammar),
        "2": ("Translate to RU", translate_ru),
        "3": ("Translate to EN", translate_en),
        "4": ("Generate auto-commit message {" + dir_path + "}", generate_commit_message),
        "5": ("Code review changes{" + dir_path + "}", code_review_changes),
        "6": ("Reply to the email", reply_to_email),
        "7": ("Summarize", summarize),
        "8": ("Explain", explain),
        "9": ("Structure (braindump)", structure_braindump),
        "10": ("Fix commit message", fix_commit_message),
        "11": ("Raw prompt", raw_prompt),
        "12": ("Type raw prompt...", custom_raw_prompt),
        "13": ("Ask expert (rendering)", ask_rendering_expert),
        "14": ("Ask expert (cpp)", ask_cpp_expert),
        "15": ("Ask expert (manager)", ask_manager_expert),
        "16": ("Ask expert (entrepreneur)", ask_entrepreneur_expert),
        "0": ("Exit", None)
    }

    while True:
        # Display the menu
        print("\nActions for:", agent_name)
        for key, (label, _) in menu_items.items():
            print(f"{key}. {label}")

        choice = input("\nQ: What to do?\n").strip().lower()

        if choice == "0":
            print("Exiting...")
            break

        if choice in menu_items:
            label, action = menu_items[choice]
            if action:
                action()  # Execute the corresponding function
                break
            else:
                print("No action defined for this option.")
        else:
            print("Invalid choice. Please try again.")


def main():
    global agent_name

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
        print("No agents found!")
        return

    agent_name = agents[0]

    # Select an active agent if multiple agents are available
    if len(agents) > 1:
        prompt = "\n\nQ: Agent name?\n"
        agent_number = 1
        for agent in agents:
            agent_text = str(agent_number) + "." + agent + "\n"
            prompt = prompt + agent_text
            agent_number = agent_number + 1

        prompt = prompt + "\n0.Exit\n"

        choice = input(prompt).strip().lower()

        agent_selected = False
        agent_number = 1
        for agent in agents:
            if choice == str(agent_number):
                agent_name = agent
                agent_selected = True
                break
            agent_number = agent_number + 1

        if not agent_selected:
            print("Bye!")
            return

    print("Input text ------------------------------")
    print(clipboard_text)
    print("------------------------------")

    display_menu(dir_path, clipboard_text)

    print("\nBye!")


main()
