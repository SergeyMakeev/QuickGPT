import os
import sys
import json
from openai import OpenAI
import anthropic
import pyperclip
import subprocess


chatgpt_client = None
openai_model = "chatgpt-4o-latest"

claude_client = None
anthropic_model = "claude-3-5-sonnet-20241022"

# chatgpt | claude
agent_name = "chatgpt"


def is_file_exist(file_path):
    return os.path.isfile(file_path)


def json_load(file_path: str):
    if not is_file_exist(file_path):
        return None
    try:
        with open(file_path) as json_file:
            data = json.load(json_file)
            return data
    except IOError:
        return None
    except ValueError:
        return None


def read_text_file(file_path):
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


def do_chatgpt(context: str, input_text: str, max_tokens: int, input_text2=None):
    max_tokens = min(max_tokens, 16384)
    messages = [
       {
           "role": "system",
           "content": context
       },
       {
           "role": "user",
           "content": input_text
       }
    ]

    if input_text2 is not None:
        messages.append({
           "role": "user",
           "content": input_text2
        })

    response = chatgpt_client.chat.completions.create(
      model=openai_model,
      messages=messages,
      temperature=0.7,
      max_tokens=max_tokens,
      top_p=1
    )
    return response.choices[0].message.content


def do_claude(context: str, input_text: str, max_tokens: int, input_text2=None):
    max_tokens = min(max_tokens, 8192)
    messages = [
       {
           "role": "user",
           "content": context
       },
       {
           "role": "user",
           "content": input_text
       }
    ]

    if input_text2 is not None:
        messages.append({
           "role": "user",
           "content": input_text2
        })

    message = claude_client.messages.create(
        model=anthropic_model,
        max_tokens=max_tokens,
        messages=messages
    )
    return message.content[0].text.strip()


def run_chatgpt_task(task_name: str, text: str, max_tokens: int, input_text2=None):
    task_context = read_text_file("./tasks/" + task_name)

    if agent_name == "chatgpt":
        print("ChatGPT is thinking... -< " + task_name + " >-")
        return do_chatgpt(task_context, text, max_tokens, input_text2)
    elif agent_name == "claude":
        print("Claude is thinking... -< " + task_name + " >-")
        return do_claude(task_context, text, max_tokens, input_text2)
    else:
        print("Unsupported agent name " + agent_name)
        return text


def run_task(task_name: str, text: str, max_tokens: int, input_text2=None):
    return run_chatgpt_task(task_name, text, max_tokens, input_text2)


def ask_expert(prompt, expert_file):
    task_context = read_text_file("./experts/" + expert_file)
    if agent_name == "chatgpt":
        print("ChatGPT. Expert is thinking... -< " + expert_file + " >-")
        return do_chatgpt(task_context, prompt, 16384)
    elif agent_name == "claude":
        print("Claude. Expert is thinking... -< " + expert_file + " >-")
        return do_claude(task_context, prompt, 16384)
    else:
        print("Unsupported agent name " + agent_name)
        return prompt


def run_raw_prompt_claude(prompt):
    print("Thinking -----< raw prompt >-----")
    if not claude_client:
        print("No Anthropic API key was provided")
        return

    message = claude_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return message.content[0].text


def generate_commit_message_claude(changes):
    message = claude_client.messages.create(
        model=anthropic_model,
        max_tokens=5000,
        messages=[
            {
                "role": "user",
                "content": "You are an assistant that generates helpful and concise git commit messages. "
                           "Here is the guideline to write a good commit message.\n"
                           "A good commit message describes the following:\n"
                           "- Commit message should start with a descriptive subject line (72 characters max)\n"
                           "- Why is the change being made?\n"
                           "- What is the summary of the changes being made?\n"
                           "- What are the possible consequences of the change being made on the rest of the system?\n"
                           "- If the change is performance or memory related, what is the summary of expected impact?\n"
                           "- If the change changes API, what is the expected user-observed behavior if any?\n"
                           "Basically think about this this way. "
                           "Five years later, somebody will hit a problem and trace it to your change. "
                           "They will want to understand more about the change but you may not remember the details or "
                           "may not work at the company. "
                           "What's more, there is no guarantee that the JIRA ticket linked would contain any "
                           "actionable info - it definitely would not contain some of the details mentioned!\n"
                           "Please always use active voice, and preferably just imperative.\n"
                           "Here are a few examples:\n"
                           "Bad\n"
                           "'The code was fixed to avoid a NULL pointer dereference'\n"
                           "OK(ish)\n"
                           "'This fixes a NULL pointer dereference'\n"
                           "Good (Preferable)\n"
                           "'Fix NULL pointer dereference'\n",
            },
            {
                "role": "user",
                "content": f"Generate a good commit message for the following changes\n\n{changes}",
            }
        ]
    )
    return message.content[0].text.strip()


def get_git_diff(working_dir):
    print("generate git diff")
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
    return result.stdout


def generate_good_commit_message_claude(git_repo_dir):
    print("Thinking -----< Generate Commit Message >-----")
    changes = get_git_diff(git_repo_dir)

    if not changes:
        print("No staged changes found.")
        return

    return generate_commit_message_claude(changes)


def is_directory_path(path: str) -> bool:
    return os.path.isdir(path)


def return_directory_path_or_fallback(input_text, fallback):
    if len(input_text) > 250:
        return fallback

    if is_directory_path(input_text):
        return input_text
    else:
        return fallback


def initialize():
    global chatgpt_client
    global claude_client

    api_keys = json_load("api_keys.json")

    openai_api_key = api_keys.get('openai', None)
    if openai_api_key:
        chatgpt_client = OpenAI(api_key=openai_api_key)
        # models = chatgpt_client.models.list()
        # print(models)

    anthropic_api_key = api_keys.get('anthropic', None)
    if anthropic_api_key:
        claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        # models = claude_client.models.list()
        # print(models)


def main():
    global agent_name

    initialize()

    clipboard_text = pyperclip.paste()
    print("Input text ------------------------------")
    print(clipboard_text)

    dir_path = return_directory_path_or_fallback(clipboard_text, ".")

    # Select an active agent if multiple agents are available
    if claude_client and chatgpt_client:
        choice = input("\n\nQ: Agent name?\n"
                       "1.ChatGPT\n"
                       "2.Claudie\n"
                       "\n"
                       "0.Exit\n").strip().lower()
        if choice == '1':
            agent_name = 'chatgpt'
        elif choice == '2':
            agent_name = 'claude'
        else:
            print("Bye!")
            return

    choice = input("\n\nQ: What to do? (" + agent_name + ")\n"
                   "1.Fix grammar\n"
                   "2.Translate to RU\n"
                   "3.Translate to EN\n"
                   "4.Generate auto-commit message {" + dir_path + "}\n"
                   "5.Fix commit message\n"
                   "6.Reply to the email...\n"
                   "7.Summarize\n"
                   "8.Explain\n"
                   "9.Structure (structure a braindump)\n"
                   "10.Raw prompt\n"
                   "11.Type raw prompt...\n"
                   "12.Ask expert (rendering)\n"
                   "13.Ask expert (cpp)\n"
                   "14.Ask expert (manager)\n"
                   "15.Ask expert (entrepreneur)\n"
                   "\n"
                   "0.Exit\n").strip().lower()

    if choice == '1':
        answer = run_task("grammar.txt", clipboard_text, 1024)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '2':
        answer = run_task("ru.txt", clipboard_text, 1024)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '3':
        answer = run_task("en.txt", clipboard_text, 1024)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '4':
        staged_changes = get_git_diff(dir_path)
        if not staged_changes:
            print("No staged changes found.")
        else:
            answer = run_task("commit_message.txt", staged_changes, 16384)
            print(answer)
            pyperclip.copy(answer)
    elif choice == '5':
        answer = run_task("fix_commit_message.txt", clipboard_text, 1024)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '6':
        draft_answer = input("Q: What would you like me to reply?\n\n")
        answer = run_task("email_reply.txt", clipboard_text, 1024, draft_answer)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '7':
        answer = run_task("summarize.txt", clipboard_text, 16384)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '8':
        answer = run_task("explain.txt", clipboard_text, 16384)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '9':
        answer = run_task("braindump.txt", clipboard_text, 16384)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '10':
        answer = run_task("raw.txt", clipboard_text, 8192)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '11':
        user_raw_prompt = input("Q: Type your prompt here\n\n")
        answer = run_task("raw.txt", user_raw_prompt, 8192)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '12':
        answer = ask_expert(clipboard_text, "rendering.txt")
        print(answer)
        pyperclip.copy(answer)
    elif choice == '13':
        answer = ask_expert(clipboard_text, "cpp.txt")
        print(answer)
        pyperclip.copy(answer)
    elif choice == '14':
        answer = ask_expert(clipboard_text, "manager.txt")
        print(answer)
        pyperclip.copy(answer)
    elif choice == '15':
        answer = ask_expert(clipboard_text, "entrepreneur.txt")
        print(answer)
        pyperclip.copy(answer)
    print("\nBye!")


main()
