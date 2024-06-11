from collections import defaultdict
import json
import re


with open('total_cross_website.json', 'r') as file:
    data = json.load(file)



def has_memory_element(content):
    match = re.search(r'<mem_bank>\s*(.*?)\s*</mem_bank>', content)
    return bool(match and match.group(1).strip())


filtered_data = [
    obj for obj in data 
    if has_memory_element(obj["conversations"][0]["content"])
]

output_file_name = 'memful_cross_website.json'
with open(output_file_name, 'w') as outfile:
    json.dump(filtered_data, outfile, indent=4)




total_bleu_with_fp = 0
total_bert_with_fp = 0
total_perplexity_with_fp = 0
total_bleu_without_fp = 0
total_bert_without_fp = 0
total_perplexity_without_fp = 0
total_turns = 0


variable_count_analysis = defaultdict(lambda: {"total_turns": 0, "count": 0})


num_filtered_objects = len(filtered_data)


for obj in filtered_data:
    total_bleu_with_fp += obj["bleu_score_with_false_positive"]
    total_bert_with_fp += obj["bert_score_with_false_positive"]
    total_perplexity_with_fp += obj["perplexity_score_with_false_positive"]
    total_bleu_without_fp += obj["bleu_score_without_false_positive"]
    total_bert_without_fp += obj["bert_score_without_false_positive"]
    total_perplexity_without_fp += obj["perplexity_score_without_false_positive"]
    total_turns += obj["turn_of_conversation"]

    extracted_variable_count = obj["extracted_variable_count"]
    variable_count_analysis[extracted_variable_count]["total_turns"] += obj["turn_of_conversation"]
    variable_count_analysis[extracted_variable_count]["count"] += 1


if num_filtered_objects > 0:
    avg_bleu_with_fp = total_bleu_with_fp / num_filtered_objects
    avg_bert_with_fp = total_bert_with_fp / num_filtered_objects
    avg_perplexity_with_fp = total_perplexity_with_fp / num_filtered_objects
    avg_bleu_without_fp = total_bleu_without_fp / num_filtered_objects
    avg_bert_without_fp = total_bert_without_fp / num_filtered_objects
    avg_perplexity_without_fp = total_perplexity_without_fp / num_filtered_objects
    avg_turns = total_turns / num_filtered_objects
else:
    avg_bleu_with_fp = avg_bert_with_fp = avg_perplexity_with_fp = 0
    avg_bleu_without_fp = avg_bert_without_fp = avg_perplexity_without_fp = 0
    avg_turns = 0


avg_turns_per_variable_count = {
    count: analysis["total_turns"] / analysis["count"]
    for count, analysis in variable_count_analysis.items()
}


print(f"Average BLEU score with false positive: {avg_bleu_with_fp}")
print(f"Average BERT score with false positive: {avg_bert_with_fp}")
print(f"Average Perplexity score with false positive: {avg_perplexity_with_fp}")
print(f"Average BLEU score without false positive: {avg_bleu_without_fp}")
print(f"Average BERT score without false positive: {avg_bert_without_fp}")
print(f"Average Perplexity score without false positive: {avg_perplexity_without_fp}")
print(f"Average turn of conversation: {avg_turns}")

print("\nAverage turn of conversation for each extracted variable count:")
for count, avg_turns in avg_turns_per_variable_count.items():
    print(f"Extracted variable count {count}: {avg_turns} turns")
