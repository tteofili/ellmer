import os
import openai
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value .
openai.api_key = os.getenv("OPENAI_API_KEY")

conversation=[{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input()
    conversation.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
        messages=conversation
    )

    conversation.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
    print("\n" + response['choices'][0]['message']['content'] + "\n")