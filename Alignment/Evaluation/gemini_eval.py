import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
import argparse
import numpy as np
import time
import requests
import google.generativeai as genai
import re
import os
# Import Colab Secrets userdata module
from google.colab import userdata

# Set OpenAI API key
import os
os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-1.5-pro-latest')


safety_config = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

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
User: Book me a flight.
Agent: Let's think step by step. To book a flight, we need to know the departure, arrival location, and time. I will first ask about the departure location. <Q>Where are you going? <Q>

Second Example:
User: Subscribe to newsletter.
Agent: Let's think step by step. To subscribe, I need to know the newsletter name first. <Q>What is the newsletter name to subscribe to?</Q>
User: Daily Fitness Tips.
Agent: Let's think step by step. I will put the user response into the mem bank as a newsletter name. I also need to ask about their email address. <mem> Newsletter Name: Daily Fitness Tips </mem> <Q>What email address should be used?</Q>
User: john.fitnessfan@example.com.
Agent: Let's think step by step. Since all the information is already asked, I will finish now and store the email address from the last reply. <mem>Email Address: john.fitnessfan@example.com </mem><Finish>

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

class BufferedFileWriter:
    def __init__(self, filename, buffer_byte_size=40960):  # 40 KB buffer size
        self.filename = filename
        self.buffer_byte_size = buffer_byte_size
        self.buffer = []
        self.current_buffer_size = 0

    def write(self, data):
        data_bytes = data.encode('utf-8')
        self.current_buffer_size += len(data_bytes)
        self.buffer.append(data)

        if self.current_buffer_size >= self.buffer_byte_size:
            self.flush()

    def flush(self):
        if self.buffer:  # Write only if buffer has data
            with open(self.filename, 'a') as f:
                f.write("\n".join(self.buffer) + "\n")
            self.buffer = []
            self.current_buffer_size = 0

    def close(self):
        self.flush()  # Ensure remaining data is written to file


# Initialize similarity the model
model2 = SentenceTransformer('all-MiniLM-L6-v2')

def load_eval_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def find_most_similar_model_output(input_string, ground_truth_conversations, model2, similarity_threshold=0.7):
    input_string = input_string.split("</mem>")[-1]
    ground_truth_texts = [conv['value'] for conv in ground_truth_conversations if conv['from'] == 'gpt']
    new_ground_truth_texts = [text.split("</mem>")[-1] for text in ground_truth_texts]

    embeddings = model2.encode([input_string] + new_ground_truth_texts)
    cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()

    best_match_index = cosine_similarities.argmax()

    if float(cosine_similarities[best_match_index]) < float(similarity_threshold):
        return "Not specified", ground_truth_texts[best_match_index]

    return best_match_index * 2 + 1, ground_truth_texts[best_match_index]


def evaluate_conversations(eval_data, file_writer):
    bleu_scorer = BLEU()
    all_bleu_scores, all_bert_scores = [], []
    all_bleu_scores_real, all_bert_scores_real = [], []
    turn_counts, results = {}, []
    prompt=build_baseline_prompt()

    for convo in eval_data:
        log_entry = f"identity -> {convo['id']}"
        file_writer.write(log_entry)
        print(log_entry)

        chat = model.start_chat(history=[])

        # messages = [{"role": "user", "content": prompt+convo['conversations'][0]['value']}]
        human_response = prompt+convo['conversations'][0]['value']

        extracted_variable_count = convo['conversations'][0]['value'].count(':')
        turn_counts.setdefault(extracted_variable_count, [])
        conversation_end, turn_of_conversation = False, 0
        false_negative, false_positive = 0, 0

        curr_bleu_scores, curr_bert_scores = [], []
        curr_bleu_scores_real, curr_bert_scores_real = [], []

        asked_questions = set()
        ground_truth_questions = set(i for i, message in enumerate(convo['conversations']) if message['from'] == 'gpt')

        while not conversation_end:
            turn_of_conversation += 1
            max_retries = 10
            retries = 0


            # response = client.chat.completions.create(
            #   model="gpt-4o",
            #   messages=messages,
            #   temperature=0.7,
            #   top_p=0.95
            # )

            # response = chat.send_message(human_response)
            # print(response.text)

            model_output = ""
            while retries < max_retries:
                try:
                    response = chat.send_message(human_response, safety_settings=safety_config)
                    print(response.text)
                    model_output = response.text
                    break
                except Exception as e:
                    retries += 1
                    print(f"Error in model response: {e}, retrying ({retries}/{max_retries})...")
                    time.sleep(2 ** retries)  # Exponential backoff

            if retries == max_retries:
                print(f"Failed after {max_retries} retries. Skipping this conversation.")
                conversation_end = True
                continue


            if "<mem>" in model_output :
              _, _, model_output = model_output.partition('<mem>')
              model_output = '<mem>' + model_output
              if "<Q>" in model_output :
                  model_output = re.sub(r'<Q>|</Q>', '', model_output)
            elif "<Q>" in model_output :
              _, _, model_output = model_output.partition('<Q>')
              model_output = '<Q>' + model_output
              model_output = re.sub(r'<Q>|</Q>', '', model_output)
            elif "<Finish>" in model_output :
              model_output = "<Finish>"


            # messages.append({"role": "assistant", "content": model_output})
            if '<Finish>' in model_output or turn_of_conversation >= 10:
                conversation_end = True

            idx, similar_str = find_most_similar_model_output(model_output, convo['conversations'], model2)
            if idx == "Not specified":
                false_positive += 1
            else:
                asked_questions.add(idx)

            if not conversation_end:
                if idx == "Not specified" or idx + 1 >= len(convo['conversations']):
                    human_response = "Not specified"
                else :
                    human_response = convo['conversations'][idx + 1]['value']
                print("human response -> "+human_response)
                # messages.append({"role": "user", "content": human_response})

            if "</mem>" in similar_str:
                _, _, similar_str = similar_str.partition('</mem>')

            bleu_score1 = bleu_scorer.sentence_score(model_output, [similar_str]).score

            _, _, bert_f1 = bert_score([model_output], [similar_str], lang="en")

            curr_bleu_scores.append(bleu_score1)
            curr_bert_scores.append(bert_f1.item())

            log_entry = (
                f"generated -> {model_output}\n"
                f"best_match -> {similar_str}\n"
                f"user response -> {human_response}\n"
                f"BLEU_SCORE {bleu_score1}\n"
                f"BERT SCORE {bert_f1.item()}\n"
            )
            print(log_entry)
            file_writer.write(log_entry)

            if idx != "Not specified":
                bleu_score1 = bleu_scorer.sentence_score(model_output, [similar_str]).score
                _, _, bert_f1 = bert_score([model_output], [similar_str], lang="en")

                curr_bleu_scores_real.append(bleu_score1)
                curr_bert_scores_real.append(bert_f1.item())

            # Delay for the specified number of seconds
            print(f"Waiting for {10} seconds...")
            time.sleep(2)

        result_entry = convo.copy()
        # result_entry['conversations'] = messages

        result_entry['bleu_score_with_false_positive'] = sum(curr_bleu_scores) / len(curr_bleu_scores)
        result_entry['bert_score_with_false_positive'] = sum(curr_bert_scores) / len(curr_bert_scores)

        result_entry['bleu_score_without_false_positive'] = 0
        result_entry['bert_score_without_false_positive'] = 0

        if len(curr_bleu_scores_real) > 0:
            result_entry['bleu_score_without_false_positive'] = sum(curr_bleu_scores_real) / len(curr_bleu_scores_real)
        if len(curr_bert_scores_real) > 0:
            result_entry['bert_score_without_false_positive'] = sum(curr_bert_scores_real) / len(curr_bert_scores_real)

        result_entry['false_negative'] = len(ground_truth_questions - asked_questions)
        result_entry['false_positive'] = false_positive

        result_entry['turn_of_conversation'] = turn_of_conversation
        result_entry['extracted_variable_count'] = extracted_variable_count
        results.append(result_entry)

        turn_counts[extracted_variable_count].append(turn_of_conversation)

        all_bleu_scores.append(result_entry['bleu_score_with_false_positive'])
        all_bleu_scores_real.append(result_entry['bleu_score_without_false_positive'])

        all_bert_scores.append(result_entry['bert_score_with_false_positive'])
        all_bert_scores_real.append(result_entry['bert_score_without_false_positive'])

    avg_bleu_with_false_positive = sum(all_bleu_scores) / len(all_bleu_scores)
    avg_bleu_without_false_positive = sum(all_bleu_scores_real) / len(all_bleu_scores_real)

    avg_bert_with_false_positive = sum(all_bert_scores) / len(all_bert_scores)
    avg_bert_without_false_positive = sum(all_bert_scores_real) / len(all_bert_scores_real)

    return results, avg_bleu_with_false_positive, avg_bleu_without_false_positive, avg_bert_with_false_positive, avg_bert_without_false_positive, turn_counts


eval_data = load_eval_data('./drive/MyDrive/filtered_test_data/filtered_cross_domain.json')

# Create a buffered file writer with a buffer size of 40 KB (40960 bytes)
file_writer = BufferedFileWriter('./drive/MyDrive/filtered_test_data/filtered_cross_domain_baseline.txt', buffer_byte_size=40960)

# Perform evaluation
results, avg_bleu_with_false_positive, avg_bleu_without_false_positive, avg_bert_with_false_positive, avg_bert_without_false_positive, turn_counts = evaluate_conversations(eval_data, file_writer)

# Print and log the summary
summary = (
    f"Average BLEU Score with false positive: {avg_bleu_with_false_positive}\n"
    f"Average BLEU Score without false positive: {avg_bleu_without_false_positive}\n"
    f"Average BERT Score with false positive: {avg_bert_with_false_positive}\n"
    f"Average BERT Score without false positive: {avg_bert_without_false_positive}\n"
    f"Turn counts: {turn_counts}\n"
)
print(summary)
file_writer.write(summary)

# Ensure all data is written to the file
file_writer.close()