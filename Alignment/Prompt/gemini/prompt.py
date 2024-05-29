import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))
# The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
model = genai.GenerativeModel('gemini-1.5-flash')

chat = model.start_chat(history=[])


def build_baseline_prompt() :
    prompt = """Given an initial task description, your task is to ask follow-up questions and parse the user's response. Only ask one question at a time. If you are done, reply with <Finish>. Please reply only with the question.

First Example:
User: Book me a flight
Agent: Where are you going?

Second Example:
User: Subscribe to newsletter
Agent: newsletter name to subscribe to? 
User: Daily Fitness Tips
Agent: What email address should be used? 
User: john.fitnessfan@example.com
Agent: <Finish>


Now complete the following task:
"""
    return prompt

def build_memory_bank_prompt() :
    prompt = """Given an initial task description, your task is to ask follow-up questions and parse the use's response for answer type and value to be stored into <mem>type: value</mem>. Only ask one question at a time. If you are done, reply with <Finish>. Please reply only with the question and <mem> if any.

First Example:
User: Book me a flight
Agent: Where are you going?

Second Example:
User: Subscribe to newsletter
Agent: newsletter name to subscribe to? 
User: Daily Fitness Tips
Agent: <mem> Newsletter Name: Daily Fitness Tips </mem> What email address should be used? 
User: john.fitnessfan@example.com
Agent: <mem> Email Address: john.fitnessfan@example.com </mem><Finish>

Now complete the following task:
"""
    return prompt


def build_CoT_memory_bank_prompt() :
    prompt = """Given an initial task description, your task is to ask follow-up questions and parse the user's response for answer type and value to be stored into <mem>type: value</mem>. Only ask one question at a time. If you are done, reply with <Finish>. Please include your question in <Q> tag and parsed answer in <mem> tag.

First Example:
User: Book me a flight
Agent: Let's think step by step. To book a flight, we need to know the departure, arrival location, and time. I will first ask about the departure location. <Q>Where are you going? <Q>

Second Example:
User: Subscribe to newsletter
Agent: newsletter name to subscribe to? 
User: Daily Fitness Tips 
Agent: <mem> Newsletter Name: Daily Fitness Tips </mem> What email address should be used? 
User: john.fitnessfan@example.com
Agent: Let's think step by step. The user wants to subscribe to a newsletter. We need the newsletter name and email address for it. Since all the information is already asked, I will finish now and store the email address from the last user reply. <mem>Email Address: john.fitnessfan@example.com </mem><Finish>

Now complete the following task:
"""
    return prompt


def build_ReACT_memory_bank_prompt() : 
    prompt = """Given an initial task description, your task is to ask follow-up questions and parse the user's response for answer type and value to be stored into <mem>type: value</mem>. Only ask one question at a time and include your thought and action. If you are done, reply with <Finish>. Please include your question in <Q> tag and parsed answer in <mem> tag. 

First Example: 
User: Book me a flight 
Agent: Thought: To book a flight, we need to know the departure, arrival location, and time. I will first ask about the departure location. Action: <Q> Where are you going? <Q> 

Second Example: 
User: Subscribe to newsletter 
Agent: Thought: To subscribe, I need to know the newsletter name first. Action: <Q> Newsletter name to subscribe to? </Q> 
User: Daily Fitness Tips 
Agent: Thought: I will put the user response into the mem bank as a newsletter name. I also need to ask about their email address. Action: <mem> Newsletter Name: Daily Fitness Tips </mem> <Q> What email address should be used? </Q>
User: john.fitnessfan@example.com 
Agent: Thought: Since all the information is already asked, I will finish now and store the email address from the last reply. Action: <mem> Email Address: john.fitnessfan@example.com </mem><Finish>

Now complete the following task:
"""
    return prompt


def inference_with_baseline() :
    # choose prompt type
    prompt = build_baseline_prompt()
    it=0
    while True:
        user_message = input("User: ")
        if it==0 :
            response = chat.send_message(prompt+user_message)
        else :
            response = chat.send_message(user_message)
        print("Agent: ", response.text)
        it+=1
        if "<Finish>" in response.text :
            break


# inferencing with memory bank
def inference_with_memory_bank() :
    # choose prompt type
    prompt = build_memory_bank_prompt()
    it=0
    while True:
        user_message = input("User: ")
        if it==0 :
            response = chat.send_message(prompt+user_message)
        else :
            response = chat.send_message(user_message)
        print("Agent: ", response.text)
        it+=1
        if "<Finish>" in response.text :
            break

# inferencing with CoT memory bank
def inference_with_CoT_memory_bank() :
    # choose prompt type
    prompt = build_CoT_memory_bank_prompt()
    it=0
    while True:
        user_message = input("User: ")
        if it==0 :
            response = chat.send_message(prompt+user_message)
        else :
            response = chat.send_message(user_message)
        agent_response = response.text
        # check if "<mem>" in response.text
        if "<mem>" in agent_response :
            agent_response = re.sub(r'.*?(<mem>)', r'\1', agent_response)
            if "<Q>" in agent_response :
                agent_response = re.sub(r'<Q>|</Q>', '', agent_response)
        elif "<Q>" in agent_response :
            agent_response = re.sub(r'.*?(<Q>)', r'\1', agent_response)
            agent_response = re.sub(r'<Q>|</Q>', '', agent_response)
        print("Agent: ", agent_response)
        it+=1
        if "<Finish>" in agent_response :
            break

# inferencing with ReACT memory bank
def inference_with_ReACT_memory_bank() :
    # choose prompt type
    prompt = build_ReACT_memory_bank_prompt()
    it=0
    while True:
        user_message = input("User: ")
        if it==0 :
            response = chat.send_message(prompt+user_message)
        else :
            response = chat.send_message(user_message)
        agent_response = response.text
        # check if "<mem>" in response.text
        if "<mem>" in agent_response :
            agent_response = re.sub(r'.*?(<mem>)', r'\1', agent_response)
            if "<Q>" in agent_response :
                agent_response = re.sub(r'<Q>|</Q>', '', agent_response)
        elif "<Q>" in agent_response :
            agent_response = re.sub(r'.*?(<Q>)', r'\1', agent_response)
            agent_response = re.sub(r'<Q>|</Q>', '', agent_response)
        print("Agent: ", agent_response)
        it+=1
        if "<Finish>" in agent_response :
            break


# custom inferencing test
def inference() :
    # choose prompt type
    prompt = build_ReACT_memory_bank_prompt()
    it=0
    while True:
        user_message = input("User: ")
        if it==0 :
            response = chat.send_message(prompt+user_message)
        else :
            response = chat.send_message(user_message)
        print("Agent: ", response.text)
        it+=1
        if "<Finish>" in response.text :
            break

inference()