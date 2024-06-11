import json
import re

# Load the JSON data from a file
with open('total_cross_website.json', 'r') as file:
    data = json.load(file)



def has_empty_memory_element(content):
    match = re.search(r'<mem_bank>\s*(.*?)\s*</mem_bank>', content)
    return bool(match and not match.group(1).strip())

# Filter JSON objects where the <mem_bank> in the first conversation is empty
filtered_data = [
    obj for obj in data 
    if has_empty_memory_element(obj["conversations"][0]["content"])
]

output_file_name = 'memless_cross_website.json'
with open(output_file_name, 'w') as outfile:
    json.dump(filtered_data, outfile, indent=4)


total_bleu_with_fp = 0
total_bert_with_fp = 0
total_perplexity_with_fp = 0
total_bleu_without_fp = 0
total_bert_without_fp = 0
total_perplexity_without_fp = 0
total_turns = 0

# Count the number of filtered objects
num_filtered_objects = len(filtered_data)

# Iterate through each filtered JSON object and sum the values
for obj in filtered_data:
    total_bleu_with_fp += obj["bleu_score_with_false_positive"]
    total_bert_with_fp += obj["bert_score_with_false_positive"]
    total_perplexity_with_fp += obj["perplexity_score_with_false_positive"]
    total_bleu_without_fp += obj["bleu_score_without_false_positive"]
    total_bert_without_fp += obj["bert_score_without_false_positive"]
    total_perplexity_without_fp += obj["perplexity_score_without_false_positive"]
    total_turns += obj["turn_of_conversation"]

# Calculate the average for each field
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

# Print the results
print(f"Average BLEU score with false positive: {avg_bleu_with_fp}")
print(f"Average BERT score with false positive: {avg_bert_with_fp}")
print(f"Average Perplexity score with false positive: {avg_perplexity_with_fp}")
print(f"Average BLEU score without false positive: {avg_bleu_without_fp}")
print(f"Average BERT score without false positive: {avg_bert_without_fp}")
print(f"Average Perplexity score without false positive: {avg_perplexity_without_fp}")
print(f"Average turn of conversation: {avg_turns}")
