import json

# Load the JSON data from a file
with open('total_cross_website.json', 'r') as file:
    data = json.load(file)

# Initialize variables to store the sum of each field
total_bleu_with_fp = 0
total_bert_with_fp = 0
total_perplexity_with_fp = 0
total_bleu_without_fp = 0
total_bert_without_fp = 0
total_perplexity_without_fp = 0

# Count the number of objects
num_objects = len(data)

# Iterate through each JSON object and sum the values
for obj in data:
    total_bleu_with_fp += obj["bleu_score_with_false_positive"]
    total_bert_with_fp += obj["bert_score_with_false_positive"]
    total_perplexity_with_fp += obj["perplexity_score_with_false_positive"]
    total_bleu_without_fp += obj["bleu_score_without_false_positive"]
    total_bert_without_fp += obj["bert_score_without_false_positive"]
    total_perplexity_without_fp += obj["perplexity_score_without_false_positive"]

# Calculate the average for each field
avg_bleu_with_fp = total_bleu_with_fp / num_objects
avg_bert_with_fp = total_bert_with_fp / num_objects
avg_perplexity_with_fp = total_perplexity_with_fp / num_objects
avg_bleu_without_fp = total_bleu_without_fp / num_objects
avg_bert_without_fp = total_bert_without_fp / num_objects
avg_perplexity_without_fp = total_perplexity_without_fp / num_objects

# Print the results
print(f"Average BLEU score with false positive: {avg_bleu_with_fp}")
print(f"Average BERT score with false positive: {avg_bert_with_fp}")
print(f"Average Perplexity score with false positive: {avg_perplexity_with_fp}")
print(f"Average BLEU score without false positive: {avg_bleu_without_fp}")
print(f"Average BERT score without false positive: {avg_bert_without_fp}")
print(f"Average Perplexity score without false positive: {avg_perplexity_without_fp}")
