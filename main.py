import os
import sys
from openai import OpenAI
import anthropic
import pyperclip
import subprocess

openai_api_key = ""
if len(sys.argv) > 1:
    param = sys.argv[1]
    openai_api_key = str(param)
else:
    print("No API Key was provided.")
    exit(-1)

client = OpenAI(
  api_key=openai_api_key
)


anthropic_api_key = ""
claude_client = None
if len(sys.argv) > 2:
    param = sys.argv[2]
    anthropic_api_key = str(param)
    claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
    # print("Anthropic models")
    # models = claude_client.models.list()
    # print(models)

# print("OpenAI models")
# models = client.models.list()
# print(models)

anthropic_model = "claude-3-5-sonnet-20241022"
openai_model = "chatgpt-4o-latest"


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


def run_raw_prompt(prompt):
    print("Thinking -----< raw prompt >-----")
    response = client.chat.completions.create(
      model=openai_model,
      messages=[
        {
          "role": "user",
          "content": prompt
        }
      ],
      temperature=0.7,
      max_tokens=1024,
      top_p=1
    )
    return response.choices[0].message.content


def fix_grammar(text):
    print("Thinking -----< fix grammar >-----")
    response = client.chat.completions.create(
      model=openai_model,
      messages=[
        {
          "role": "system",
          "content": "You will be provided with statements; your task is to convert them to standard English, "
                     "fix the grammar, and make them sound like a native US speaker. "
                     "Keep it simple and make it sound like a software engineer. "
                     "If you see something marked with tilde that's a code snippet, do not change it and "
                     "keep opening and closing tildes. "
                     "Try to keep the original style if possible. "
                     "Provide a new corrected text as an answer without any additional ideas or comments. "
                     "No further explanation needed."
        },
        {
          "role": "user",
          "content": text
        }
      ],
      temperature=0.7,
      max_tokens=256,
      top_p=1
    )
    return response.choices[0].message.content


def translate_to_ru(text):
    print("Thinking -----< translate to RU >-----")
    response = client.chat.completions.create(
      model=openai_model,
      messages=[
        {
          "role": "system",
          "content": "You will be provided with statements; your task is to translate that text to Russian. "
                     "Provide a new corrected text as an answer without any additional ideas or comments. "
                     "No further explanation needed."
        },
        {
          "role": "user",
          "content": text
        }
      ],
      temperature=0.7,
      max_tokens=256,
      top_p=1
    )
    return response.choices[0].message.content


def translate_to_en(text):
    print("Thinking -----< translate to EN >-----")
    response = client.chat.completions.create(
      model=openai_model,
      messages=[
        {
          "role": "system",
          "content": "You will be provided with statements; your task is to translate that text to US English. "
                     "Provide a new corrected text as an answer without any additional ideas or comments. "
                     "No further explanation needed."
        },
        {
          "role": "user",
          "content": text
        }
      ],
      temperature=0.7,
      max_tokens=256,
      top_p=1
    )
    return response.choices[0].message.content


def structure_braindump(text):
    print("Thinking -----< Structurizing >-----")
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            dict(role="system",
                 content="You are an assistant who structure notes and brain dumps using simple English. "
                         "You will be provided with an unstructured text, and you need to structure and summarize it."
                         " Find common topics and put them into a separate paragraphs. Add a paragraph with summary "
                         "in the end. "
                         "Assume that the reader is a software engineer familiar with basic math, physics, etc."),
            {
                "role": "user",
                "content": f"Explain the following.\n\n{text}",
            },
        ],
        max_tokens=2500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def explain(text):
    print("Thinking -----< Explaining >-----")
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who explains things using simple language. "
                           "You will be provided with a text, and you need to explain what is written in that text." 
                           "Consider that the explanation should be in simple English and "
                           "assume that the reader is a software engineer familiar with basic math, physics, etc."
            },
            {
                "role": "user",
                "content": f"Explain the following.\n\n{text}",
            },
        ],
        max_tokens=500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def summarize(text):
    print("Thinking -----< Summarizing >-----")
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that summarizes long texts. "
                           "You will be provided with a text, and you need to write a clear "
                           "yet comprehensive summary of that text."
            },
            {
                "role": "user",
                "content": f"Summarize the following text.\n\n{text}",

            },
        ],
        max_tokens=500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def reply_to_email(text, rough_answer):
    print("Thinking -----< Replying to email >-----")
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that replies to emails on behalf of Sergei Makeev. "
                           "You will be provided with an email text and very draft response. "
                           "Given provided draft response you need to write a clear, and polite answer to that email. "
                           "Provide a new corrected text as an answer without any additional ideas or comments. "
                           "No further explanation needed."
            },
            {
                "role": "user",
                "content": f"Source email\n\n{text}\n\nDraft response\n\n{rough_answer}",
            },
        ],
        max_tokens=500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def fix_commit_message(text):
    print("Thinking -----< Fix Commit Message >-----")
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
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
                "content": f"Rewrite the provided commit message to conform "
                           f"to the commit message guidelines.\n\n{text}",
            },
        ],
        max_tokens=500,  # Adjust as needed
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def generate_commit_message(changes):
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
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
            },
        ],
        max_tokens=5000,  # Adjust as needed
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def get_git_diff(working_dir):

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
        text=True,
        encoding='utf-8'
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


def generate_good_commit_message(git_repo_dir):
    print("Thinking -----< Generate Commit Message >-----")
    changes = get_git_diff(git_repo_dir)

    if not changes:
        print("No staged changes found.")
        return

    return generate_commit_message(changes)


def is_directory_path(path: str) -> bool:
    return os.path.isdir(path)


def return_directory_path_or_fallback(input_text, fallback):
    if len(input_text) > 250:
        return fallback

    if is_directory_path(input_text):
        return input_text
    else:
        return fallback


def ask_expert(prompt, expert_file):
    expert_def = read_text_file("./experts/" + expert_file)
    print("Expert is thinking...")
    response = client.chat.completions.create(
      model=openai_model,
      messages=[
          {
              "role": "system",
              "content": expert_def
          },
          {
              "role": "user",
              "content": prompt
        }
      ],
      temperature=0.7,
      max_tokens=8192,
      top_p=1
    )
    return response.choices[0].message.content


def main():
    clipboard_text = pyperclip.paste()
    print("Input text ------------------------------")
    print(clipboard_text)

    dir_path = return_directory_path_or_fallback(clipboard_text, ".")

    choice = input("\n\nQ: What to do?\n"
                   "1.Fix grammar\n"
                   "2.Translate to RU\n"
                   "3.Translate to EN\n"
                   "4.Generate auto-commit message {" + dir_path + "}\n"
                   "5.Fix commit message\n"
                   "6.Reply to the email...\n"
                   "7.Summarize\n"
                   "8.Explain\n"
                   "9.Structure (structure a braindump)\n"
                   "Q.Run clipboard as a raw prompt\n"
                   "W.Type raw prompt...\n"
                   "E.Type raw prompt (Claude AI) ...\n"
                   "R.Generate auto-commit message  (Claude AI) {" + dir_path + "}\n"
                   "T.Ask expert (rendering)\n"
                   "Y.Ask expert (cpp)\n"
                   "\n"
                   "0.Exit\n").strip().lower()

    if choice == '1':
        answer = fix_grammar(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '2':
        answer = translate_to_ru(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '3':
        answer = translate_to_en(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '4':
        answer = generate_good_commit_message(dir_path)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '5':
        answer = fix_commit_message(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '6':
        rough_answer = input("Q: What would you like me to reply?\n\n")
        answer = reply_to_email(clipboard_text, rough_answer)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '7':
        answer = summarize(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '8':
        answer = explain(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == '9':
        answer = structure_braindump(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == 'Q' or choice == 'q':
        answer = run_raw_prompt(clipboard_text)
        print(answer)
        pyperclip.copy(answer)
    elif choice == 'W' or choice == 'w':
        user_raw_prompt = input("Q: Type your prompt here\n\n")
        answer = run_raw_prompt(user_raw_prompt)
        print(answer)
        pyperclip.copy(answer)
    elif choice == 'E' or choice == 'e':
        user_raw_prompt = input("Q: Type your prompt here\n\n")
        answer = run_raw_prompt_claude(user_raw_prompt)
        print(answer)
        pyperclip.copy(answer)
    elif choice == 'R' or choice == 'r':
        answer = generate_good_commit_message_claude(dir_path)
        print(answer)
        pyperclip.copy(answer)
    elif choice == 'T' or choice == 't':
        answer = ask_expert(clipboard_text, "rendering.txt")
        print(answer)
        pyperclip.copy(answer)
    elif choice == 'Y' or choice == 'y':
        answer = ask_expert(clipboard_text, "cpp.txt")
        print(answer)
        pyperclip.copy(answer)
    print("\nBye!")


main()
