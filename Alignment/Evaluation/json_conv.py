import json
import re

def parse_block(block):
    lines = block.strip().split('\n')
    identity = lines[0].strip().split(' -> ')[1]
    generated_segments = []
    best_match_segments = []
    
    for line in lines[1:]:
        if line.startswith("generated ->"):
            generated_segments.append(line.split(" -> ")[1].strip())
        elif line.startswith("best_match ->"):
            best_match_segments.append(line.split(" -> ")[1].strip())
    
    return {
        "identity": identity,
        "generated": generated_segments,
        "best_match": best_match_segments
    }

def convert_txt_to_json_array(file_path, output_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    blocks = re.split(r'\nidentity -> ', content)
    json_array = []
    
    for block in blocks:
        if block.strip():  # Skip empty blocks
            if not block.startswith('identity -> '):
                block = 'identity -> ' + block  # Add the removed part back if missing
            json_obj = parse_block(block)
            json_array.append(json_obj)
    
    with open(output_path, 'w') as json_file:
        json.dump(json_array, json_file, indent=4)

# Example usage
file_path = '../Results/Alignment/GEMINI/filtered_cross_website_react.txt'
output_path = 'filtered_cross_website_react.json'
convert_txt_to_json_array(file_path, output_path)