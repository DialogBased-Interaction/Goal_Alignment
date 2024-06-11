import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

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

def find_most_similar_model_output(input_string, ground_truth_conversations, model2):
    input_string = input_string.split("</mem>")[-1]
    ground_truth_texts = [conv['value'] for conv in ground_truth_conversations if conv['from'] == 'gpt']
    new_ground_truth_texts = [text.split("</mem>")[-1] for text in ground_truth_texts]
    embeddings = model2.encode([input_string] + new_ground_truth_texts)
    cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    best_match_index = cosine_similarities.argmax()
    return best_match_index * 2 + 1, ground_truth_texts[best_match_index]


def evaluate_conversations(eval_data, model, tokenizer, gpt2_model, gpt2_tokenizer, model2, generation_params):
    bleu_scorer = BLEU()
    all_scores1, all_scores2, all_bert_scores, all_perplexities = [], [], [], []
    turn_counts, results = {}, []

    for convo in eval_data:
        messages = [{"role": "user", "content": convo['conversations'][0]['value']}]
        extracted_variable_count = convo['conversations'][0]['value'].count(':')
        turn_counts.setdefault(extracted_variable_count, [])
        conversation_end, turn_of_conversation = False, 0

        while not conversation_end:
            turn_of_conversation += 1

            
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

            
            response = model.generate(
                model_inputs,
                max_new_tokens=2048,  # barate komate parbo
                temperature=generation_params.get('temperature', 1.0),
                top_p=generation_params.get('top_p', 1.0),
                top_k=generation_params.get('top_k', 50),
                do_sample=True,
                num_return_sequences=1  
            )
            model_output = tokenizer.decode(response[0], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": model_output})
            if '<Finish>' in model_output:
                conversation_end = True

            idx, similar_str = find_most_similar_model_output(model_output, convo['conversations'], model2)
            if not conversation_end:
                human_response = convo['conversations'][idx + 1]['value'] if idx + 1 < len(convo['conversations']) else "none"
                messages.append({"role": "user", "content": human_response})

            bleu_score1 = bleu_scorer.sentence_score(model_output, [similar_str]).score
            model_output_trimmed = model_output.split("</mem>")[-1]
            similar_str_trimmed = similar_str.split("</mem>")[-1]
            bleu_score2 = bleu_scorer.sentence_score(model_output_trimmed, [similar_str_trimmed]).score

            _, _, bert_f1 = bert_score([model_output_trimmed], [similar_str_trimmed], lang="en")
            perplexity = calculate_perplexity(gpt2_model, gpt2_tokenizer, model_output)

            result_entry = convo.copy()
            result_entry['bleu_score_with_mem'] = bleu_score1
            result_entry['bleu_score_without_mem'] = bleu_score2
            result_entry['bert_score'] = bert_f1.item()
            results.append(result_entry)

            all_scores1.append(bleu_score1)
            all_scores2.append(bleu_score2)
            all_bert_scores.append(bert_f1.item())
            all_perplexities.append(perplexity)

        turn_counts[extracted_variable_count].append(turn_of_conversation)

    avg_bleu1 = sum(all_scores1) / len(all_scores1)
    avg_bleu2 = sum(all_scores2) / len(all_scores2)
    avg_bert_score = sum(all_bert_scores) / len(all_bert_scores)
    avg_perplexity = sum(all_perplexities) / len(all_perplexities)

    return results, avg_bleu1, avg_bleu2, avg_bert_score, avg_perplexity, turn_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("test_file", type=str, help="Path to the test JSON file")
    parser.add_argument("model_dir", type=str, help="Path to the fine-tuned model directory")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")

    args = parser.parse_args()

    eval_data = load_eval_data(args.test_file)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model2 = SentenceTransformer('all-MiniLM-L6-v2')

    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k
    }

    results, avg_bleu1, avg_bleu2, avg_bert_score, avg_perplexity, turn_counts = evaluate_conversations(
        eval_data, model, tokenizer, gpt2_model, gpt2_tokenizer, model2, generation_params
    )

    with open(args.output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Average BLEU Score with mem: {avg_bleu1}")
    print(f"Average BLEU Score without mem: {avg_bleu2}")
    print(f"Average BERT Score: {avg_bert_score}")
    print(f"Average Perplexity: {avg_perplexity}")



