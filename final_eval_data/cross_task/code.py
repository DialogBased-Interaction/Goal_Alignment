import json

# Step 1: Load the JSON data from both files
with open('filtered_cross_task.json', 'r') as f1, open('cross_task.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)


ids_file1 = [item['id'] for item in data1]


filtered_data2 = [item for item in data2 if any(item['id'].startswith(id1) for id1 in ids_file1)]


output_file_name = 'filtered_cross_task_mem.json'
with open(output_file_name, 'w') as outfile:
    json.dump(filtered_data2, outfile, indent=4)

print(f'Filtered data saved to {output_file_name}')
