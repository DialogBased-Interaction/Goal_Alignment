from openai import OpenAI, ChatCompletion
client = OpenAI()

def chat_with_gpt4(messages):
    completion = client.chat.completions.create(
      model="gpt-4",
      messages=messages
    )
    return completion.choices[0].message.content


# Start the conversation with the system message
messages = [
    {"role": "system", "content": "Given an initial task description, your task is to ask follow-up questions and parse the user’s response for answer type and value to be stored into <mem>. Only ask one question at a time. If you are done, reply with <finish>. Please reply only with the question and <mem> if any.\n\nFirst Example:\nUser: Book me a flight\nAgent: Where are you going?\n\nSecond Example:\nUser: Subscribe to newsletter\nAgent: newsletter name to subscribe to?\nUser: Daily Fitness Tips \nAgent: <mem>Newsletter Name: Daily Fitness Tips </mem> What email address should be used?\nUser: john.fitnessfan@example.com\nAgent: <mem>Email Address: john.fitnessfan@example.com </mem><Finish>\nNow complete the following task: Book a restaurant for me"},
]

i=0
while True:
    if i>0 :
      user_message = input("You: ")
      messages.append({"role": "user", "content": user_message})
    response = chat_with_gpt4(messages)
    print("AI: ", response)
    messages.append({"role": "assistant", "content": response})
    i+=1