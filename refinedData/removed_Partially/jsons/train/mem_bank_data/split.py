import json
from sklearn.model_selection import train_test_split

# Load your data
with open('./whole/mem_bank_whole.json') as f:
    data = json.load(f)

# Split the data
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)


# Write train data to file
with open("./splitted/train.json", "w") as file:
    json.dump(train_data, file)

# Write eval data to file
with open("./splitted/eval.json", "w") as file:
    json.dump(eval_data, file)
