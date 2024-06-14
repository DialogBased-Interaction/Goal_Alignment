from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import openai
import torch
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "checkpoint-2245"
prompt = "Once upon a time"

completion = openai.completions.create(model=model, prompt=prompt, max_tokens=64)



model2 = SentenceTransformer('all-MiniLM-L6-v2')

def find_most_similar_model_output(input_string, ground_truth_conversations):
    input_string = input_string.split("</mem>")[-1]

    ground_truth_texts = [conv['value'] for conv in ground_truth_conversations if conv['from'] == 'gpt']

    new_ground_truth_texts = []

    for i in range(len(ground_truth_texts)):
        new_ground_truth_texts.append(ground_truth_texts[i].split("</mem>")[-1])

    embeddings = model2.encode([input_string] + new_ground_truth_texts)

    cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()

    best_match_index = cosine_similarities.argmax()

    return best_match_index*2+1, ground_truth_texts[best_match_index]


def load_eval_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def calculate_perplexity(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    softmax_logits = torch.softmax(logits, dim=-1).squeeze(dim=0)
    word_ids = input_ids.squeeze(dim=0)
    perplexity = 0
    for i, word_id in enumerate(word_ids):
        perplexity -= torch.log(softmax_logits[i, word_id])
    perplexity = torch.exp(perplexity / len(word_ids))
    return perplexity.item()

def evaluate_conversations2(eval_data):
    bleu_scorer = BLEU()
    all_scores1 = []
    all_scores2 = []
    all_bert_scores = []
    all_perplexities = []
    turn_counts = {}
    it=0
    fileStr = ""
    for convo in eval_data:
        it+=1
        # file.write("EVAL DATA -> " + str(it)+"\n")
        fileStr = fileStr + "EVAL DATA -> " + str(it)+"\n"
        messages = []
        extracted_variable_count = convo['conversations'][0]['value'].count(':')
        if extracted_variable_count not in turn_counts:
            turn_counts[extracted_variable_count] = []
        messages.append({"role": "user", "content": convo['conversations'][0]['value']})
        # input_text = f"human: {convo['conversations'][0]['value']}"
        print("Initial conversation" )  # Print the initial input_text
        print("human: ", convo['conversations'][0]['value'])
        conversation_end = False
        turn_of_conversation = 0;
        while not conversation_end:
            turn_of_conversation+=1
            response = openai.chat.completions.create(
                model=model,
                messages=messages
            )

            model_output = response.choices[0].message.content
            # print("gpt: ", model_output)
            messages.append({"role": "assistant", "content": model_output})

            if '<Finish>' in model_output:
                conversation_end = True


            # Find the most similar ground truth question from amader eevaldata
            idx, similar_str = find_most_similar_model_output(model_output, convo['conversations'])

            # Prepare the next input text
            if not conversation_end:
                human_response = ""
                if(idx+1 == len(convo['conversations'])) : human_response = "none"
                else : human_response = convo['conversations'][idx+1]['value']
                messages.append({"role": "user", "content": human_response})
                # print("human: ", human_response)  # Print the updated input_text after each response

            # Add to BLEU evaluation
            # file.write("With comparing mem\n")
            fileStr = fileStr + "With comparing mem\n"

            # file.write("generated -> "+model_output+"\n")
            fileStr = fileStr + "generated -> "+model_output+"\n"

            # file.write("best_match -> "+similar_str+"\n")
            fileStr = fileStr + "best_match -> "+similar_str+"\n"


            bleu_score1 = bleu_scorer.sentence_score(model_output, [similar_str]).score
            # file.write("BLEU_SCORE with mem"+str(bleu_score1)+"\n\n")
            fileStr = fileStr + "BLEU_SCORE with mem"+str(bleu_score1)+"\n\n"

            if(model_output != " <Finish>") :
                # file.write("Without comparing mem\n")
                fileStr = fileStr + "Without comparing mem\n"


                model_output = model_output.split("</mem>")[-1]
                similar_str = similar_str.split("</mem>")[-1]
                # file.write("generated -> "+model_output+"\n")
                fileStr = fileStr + "generated -> "+model_output+"\n"

                # file.write("best_match -> "+similar_str+"\n")
                fileStr = fileStr + "best_match -> "+similar_str+"\n"

                bleu_score2 = bleu_scorer.sentence_score(model_output, [similar_str]).score
                # file.write("BLEU_SCORE without mem"+str(bleu_score2)+"\n\n")
                fileStr = fileStr + "BLEU_SCORE without mem"+str(bleu_score2)+"\n\n"

            # print(bleu_score1)
            # print(bleu_score2)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            all_scores1.append(bleu_score1)
            all_scores2.append(bleu_score2)

            # Calculate BERTScore
            _, _, bert_f1 = bert_score([model_output], [similar_str], lang="en")
            all_bert_scores.append(bert_f1.item())

            # Calculate Perplexity
            perplexity = calculate_perplexity(gpt2_model, gpt2_tokenizer, model_output)
            all_perplexities.append(perplexity)

        turn_counts[extracted_variable_count].append(turn_of_conversation)

    # graph draw
    avg_turn_counts = {count: sum(turns) / len(turns) for count, turns in turn_counts.items()}
    plt.bar(avg_turn_counts.keys(), avg_turn_counts.values())
    plt.xlabel('# of variable in retrieved memory bank from Vector Database')
    plt.ylabel('# Turn of conversation')
    plt.title('Average Turn of Conversation vs Extracted Variable Count')
    plt.show()

    # print(avg_turn_counts)
    # Calculate and return the average BLEU score
    average_bleu1 = sum(all_scores1) / len(all_scores1)
    average_bleu2 = sum(all_scores2) / len(all_scores2)
    average_bert_score = sum(all_bert_scores) / len(all_bert_scores)
    average_perplexity = sum(all_perplexities) / len(all_perplexities)
    return {
        'average_bleu1': average_bleu1,
        'average_bleu2': average_bleu2,
        'average_bert_score': average_bert_score,
        'average_perplexity': average_perplexity,
        'turn_counts': turn_counts,
        "fileStr": fileStr
    }



path = './data/tempuu.json'
evaluate_conversations2(load_eval_data(path))


# plot_combined_turn_counts(results.turn_counts)