import sys
from openai import OpenAI
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
  api_key = openai_api_key
)

# models = client.models.list()
# print(models)


def fix_grammar(text):
    print("Thinking -----< fix grammar >-----")
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {
          "role": "system",
          "content": "You will be provided with statements; your task is to convert them to standard English, "
                     "fix the grammar, and make them sound like a native US speaker. "
                     "Keep it simple and make it sound like a software engineer. "
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
      model="gpt-4o-mini",
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
      model="gpt-4o-mini",
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


def generate_commit_message(changes):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
                           "actionable info - it definitely would not contain some of the details mentioned!",
            },
            {
                "role": "user",
                "content": f"Generate a good commit message for the following changes\n\n{changes}",
            },
        ],
        max_tokens=500,  # Adjust as needed
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def get_git_diff():
    result = subprocess.run(
        ["git", "diff", "--staged"], stdout=subprocess.PIPE, text=True
    )
    return result.stdout


def generate_good_commit_message():
    print("Thinking -----< Generate Commit Message >-----")
    changes = get_git_diff()

    if not changes:
        print("No staged changes found.")
        return

    return generate_commit_message(changes)


def main():
    clipboard_text = pyperclip.paste()
    print("Input text ------------------------------")
    print(clipboard_text)

    choice = input("\n\nWhat to do?\n"
                   "1.Fix grammar\n"
                   "2.Translate to RU\n"
                   "3.Translate to EN\n"
                   "4.Generate auto-commit message (run this script directly from repo)\n").strip().lower()

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
        answer = generate_good_commit_message()
        print(answer)
        pyperclip.copy(answer)
    print("\nBye!")


main()
