import json
import re

def extract_mem_parts(conversations):
    mem_parts = []
    mem_pattern = re.compile(r'<mem>(.*?)</mem>', re.DOTALL)
    
    for conversation in conversations:
        content = conversation.get("content", "")
        matches = mem_pattern.findall(content)
        mem_parts.extend(matches)
    
    return mem_parts

def extract_mem_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    all_mem_parts = []
    
    for item in data:
        identity = item.get("id")
        conversations = item.get("conversations", [])
        mem_parts = extract_mem_parts(conversations)
        all_mem_parts.append([identity, mem_parts])
    
    return all_mem_parts

# Example usage
file_path = '../Data/cross_domain_output.json'
mem_parts = extract_mem_from_json(file_path)
print(mem_parts)