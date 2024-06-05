import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import argparse
import numpy as np

def load_eval_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def calculate_perplexity(model, tokenizer, text, device='cuda'):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Reshape loss to (batch_size, sequence_length - 1)
    loss = loss.view(input_ids.size(0), input_ids.size(1) - 1)
    
    # Mask the loss for padding tokens
    attention_mask = attention_mask[..., 1:].contiguous()
    loss = loss * attention_mask
    
    # Calculate perplexity
    perplexity = torch.exp(loss.sum() / attention_mask.sum())
    
    return perplexity.item()


def find_most_similar_model_output(input_string, ground_truth_conversations, model2, similarity_threshold=0.7):
    input_string = input_string.split("</mem>")[-1]
    ground_truth_texts = [conv['value'] for conv in ground_truth_conversations if conv['from'] == 'gpt']
    new_ground_truth_texts = [text.split("</mem>")[-1] for text in ground_truth_texts]
    
    embeddings = model2.encode([input_string] + new_ground_truth_texts)
    cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    
    best_match_index = cosine_similarities.argmax()
    
    if cosine_similarities[best_match_index] < similarity_threshold:
        return "not mentioned", ground_truth_texts[best_match_index]
    
    return best_match_index * 2 + 1, ground_truth_texts[best_match_index]


def evaluate_conversations(eval_data, model, tokenizer, gpt2_model, gpt2_tokenizer, model2, generation_params):
    bleu_scorer = BLEU()
    all_bleu_scores, all_bert_scores, all_perplexities = [], [], []
    all_bleu_scores_real, all_bert_scores_real, all_perplexities_real = [], [], []
    turn_counts, results = {}, []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for convo in eval_data:
        messages = [{"role": "user", "content": convo['conversations'][0]['value']}]
        extracted_variable_count = convo['conversations'][0]['value'].count(':')
        turn_counts.setdefault(extracted_variable_count, [])
        conversation_end, turn_of_conversation = False, 0
        false_negative, false_positive = 0, 0

        curr_bleu_scores, curr_bert_scores, curr_perplexities = [], [], []
        curr_bleu_scores_real, curr_bert_scores_real, curr_perplexities_real = [], [], []

        asked_questions = set()
        ground_truth_questions = set(i for i, message in enumerate(convo['conversations']) if message['from'] == 'gpt')

        while not conversation_end:
            turn_of_conversation += 1

            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            response = model.generate(
                model_inputs,
                max_new_tokens=2048,
                temperature=generation_params.get('temperature', 0.7),
                top_p=generation_params.get('top_p', 0.95),
                top_k=generation_params.get('top_k', 50),
                do_sample=True,
                num_return_sequences=1
            )
            model_output = tokenizer.decode(response[0], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": model_output})
            if '<Finish>' in model_output:
                conversation_end = True

            idx, similar_str = find_most_similar_model_output(model_output, convo['conversations'], model2)
            if idx == "not mentioned":
                false_positive += 1
            else:
                asked_questions.add(idx)

            if not conversation_end:
                human_response = convo['conversations'][idx + 1]['value'] if isinstance(idx, int) and idx + 1 < len(convo['conversations']) else "not mentioned"
                messages.append({"role": "user", "content": human_response})

            bleu_score1 = bleu_scorer.sentence_score(model_output, [similar_str]).score

            _, _, bert_f1 = bert_score([model_output], [similar_str], lang="en")
            perplexity = calculate_perplexity(gpt2_model, gpt2_tokenizer, model_output, device=device)

            curr_bleu_scores.append(bleu_score1)
            curr_bert_scores.append(bert_f1.item())
            curr_perplexities.append(perplexity)

            if idx != "not mentioned":
                bleu_score1 = bleu_scorer.sentence_score(model_output, [similar_str]).score
                _, _, bert_f1 = bert_score([model_output], [similar_str], lang="en")
                perplexity = calculate_perplexity(gpt2_model, gpt2_tokenizer, model_output, device=device)

                curr_bleu_scores_real.append(bleu_score1)
                curr_bert_scores_real.append(bert_f1.item())
                curr_perplexities_real.append(perplexity)

        result_entry = convo.copy()
        result_entry['conversations'] = messages

        result_entry['bleu_score_with_false_positive'] = sum(curr_bleu_scores) / len(curr_bleu_scores)
        result_entry['bert_score_with_false_positive'] = sum(curr_bert_scores) / len(curr_bert_scores)
        result_entry['perplexity_score_with_false_positive'] = sum(curr_perplexities) / len(curr_perplexities)

        result_entry['bleu_score_without_false_positive'] = 0
        result_entry['bert_score_without_false_positive'] = 0
        result_entry['perplexity_score_without_false_positive'] = 0

        if len(curr_bleu_scores_real) > 0:
            result_entry['bleu_score_without_false_positive'] = sum(curr_bleu_scores_real) / len(curr_bleu_scores_real)
        if len(curr_bert_scores_real) > 0:
            result_entry['bert_score_without_false_positive'] = sum(curr_bert_scores_real) / len(curr_bert_scores_real)
        if len(curr_perplexities_real) > 0:
            result_entry['perplexity_score_without_false_positive'] = sum(curr_perplexities_real) / len(curr_perplexities_real)

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

        all_perplexities.append(result_entry['perplexity_score_with_false_positive'])
        all_perplexities_real.append(result_entry['perplexity_score_without_false_positive'])

    avg_bleu_with_false_positive = sum(all_bleu_scores) / len(all_bleu_scores)
    avg_bleu_without_false_positive = sum(all_bleu_scores_real) / len(all_bleu_scores_real)

    avg_bert_with_false_positive = sum(all_bert_scores) / len(all_bert_scores)
    avg_bert_without_false_positive = sum(all_bert_scores_real) / len(all_bert_scores_real)

    avg_perplexity_with_false_positive = sum(all_perplexities) / len(all_perplexities)
    avg_perplexity_without_false_positive = sum(all_perplexities_real) / len(all_perplexities_real)

    return results, avg_bleu_with_false_positive, avg_bleu_without_false_positive, avg_bert_with_false_positive, avg_bert_without_false_positive, avg_perplexity_with_false_positive, avg_perplexity_without_false_positive, turn_counts



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
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model2 = SentenceTransformer('all-MiniLM-L6-v2')

    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k
    }

    results, avg_bleu_with_false_positive, avg_bleu_without_false_positive, avg_bert_with_false_positive, avg_bert_without_false_positive, avg_perplexity_with_false_positive, avg_perplexity_without_false_positive, turn_counts = evaluate_conversations(
        eval_data, model, tokenizer, gpt2_model, gpt2_tokenizer, model2, generation_params
    )

    with open(args.output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Average BLEU Score with false positive: {avg_bleu_with_false_positive}")
    print(f"Average BLEU Score without false positive: {avg_bleu_without_false_positive}")

    print(f"Average BERT Score with false positive: {avg_bert_with_false_positive}")
    print(f"Average BERT Score without false positive: {avg_bert_without_false_positive}")

    print(f"Average Perplexity with false positive: {avg_perplexity_with_false_positive}")
    print(f"Average Perplexity without false positive: {avg_perplexity_without_false_positive}")
