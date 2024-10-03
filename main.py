import sys
from openai import OpenAI
import pyperclip

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


def main():
    clipboard_text = pyperclip.paste()
    print("Input text ------------------------------")
    print(clipboard_text)

    choice = input("\n\nContinue?\nY/n and press Enter: ").strip().lower()
    if choice != 'y':
        print("Done")
        return

    answer = fix_grammar(clipboard_text)
    print(answer)

    answer = translate_to_ru(answer)
    print(answer)

    answer = translate_to_en(answer)
    print(answer)

    print("Done")


main()
