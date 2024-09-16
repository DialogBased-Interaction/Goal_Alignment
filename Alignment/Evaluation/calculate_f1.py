import json
import re
from fuzzywuzzy import fuzz
import sys
import spacy

# Load a pre-trained spaCy language model
nlp = spacy.load("en_core_web_md")

def extract_mem_parts(segments):
    mem_parts = []
    mem_pattern = re.compile(r'<mem>(.*?)</mem>', re.DOTALL)
    
    for segment in segments:
        matches = mem_pattern.findall(segment)
        mem_parts.extend(matches)
    
    return mem_parts

def extract_mem_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    all_mem_parts = []
    
    for item in data:
        identity = item.get("identity")
        generated_segments = item.get("generated", [])
        mem_parts = extract_mem_parts(generated_segments)
        all_mem_parts.append([identity, mem_parts])
    
    return all_mem_parts

# Example usage
file_path1 = 'output.json'
mem_parts1 = extract_mem_from_json(file_path1)

print("HOLLAAAAAAAAAA\n\n\n")

def extract_mem_parts_2(conversations):
    mem_parts = []
    mem_pattern = re.compile(r'<mem>(.*?)</mem>', re.DOTALL)
    
    for conversation in conversations:
        value = conversation.get("value", "")
        matches = mem_pattern.findall(value)
        mem_parts.extend(matches)
    
    return mem_parts

def extract_mem_from_json_2(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    all_mem_parts = []
    
    for item in data:
        identity = item.get("identity")
        conversations = item.get("conversations", [])
        mem_parts = extract_mem_parts_2(conversations)
        all_mem_parts.append([identity, mem_parts])
    
    return all_mem_parts

# Example usage
file_path2 = '../Data/filtered_cross_domain.json'
mem_parts2 = extract_mem_from_json_2(file_path2)

# calcuating f1 score for memory part1 with memory part2
def calculate_f1(mem_parts1, mem_parts2, f, threshold=0.8):
    # iterate over the list of lists and calculate the f1 score for each list
    print(len(mem_parts1) == len(mem_parts2))
    type_f1_scores = []
    value_f1_scores = []
    type_precision_scores = []
    type_recall_scores = []
    value_precision_scores = []
    value_recall_scores = []

    for i in range(len(mem_parts1)):
        mem_parts1[i] = [mem_parts1[i][0], set(mem_parts1[i][1])]
        mem_parts2[i] = [mem_parts2[i][0], set(mem_parts2[i][1])]

    print(f"----------------------- TYPE PART SCORES -----------------\n", file=f)

    for i in range(len(mem_parts1)):
        mem1 = mem_parts1[i][1]
        mem2 = mem_parts2[i][1]

        # here my mem parts are like this form Location : NY. So, i want to calculate f1 for Location part only

        # Remove entities where the entity value is 'not mentioned' ignoring case
        mem1 = [m for m in mem1 if ":" in m and m.split(":")[1].strip().lower() != 'not specified']
        mem2 = [m for m in mem2 if ":" in m and m.split(":")[1].strip().lower() != 'not specified']

        mem1 = list(mem1)
        mem2 = list(mem2)

        mem1 = [m.split(":")[0].strip() for m in mem1]
        mem2 = [m.split(":")[0].strip() for m in mem2]

        mem1 = set(mem1)
        mem2 = set(mem2)
        
        mem2.discard('')

        # printing to file f
        print(mem_parts1[i][0], file=f)
        print(mem1, file=f)
        print(mem2, file=f)

        best_match = {}
        for m1 in mem1:
            best_match[m1] = None
            pred = nlp(m1)
            for m2 in mem2:
                similarity = pred.similarity(nlp(m2))
                if similarity >= threshold:
                    best_match[m1] = m2
                    break
        
        # Calculate true positives, false positives, and false negatives
        true_positives = sum(1 for m1 in mem1 if best_match[m1] is not None)
        false_positives = len(mem1) - true_positives
        false_negatives = len(mem2) - true_positives

        print(f"True Positives: {true_positives}", file=f)
        print(f"False Positives: {false_positives}", file=f)
        print(f"False Negatives: {false_negatives}", file=f)

        
        if true_positives + false_positives == 0 or true_positives + false_negatives == 0:
            f1 = 0.0
            precision = 0.0
            recall = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            # if precision + recall is 0 then f1 is 0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        type_precision_scores.append(precision)
        type_recall_scores.append(recall)
        type_f1_scores.append(f1)

        print(f"Precision: {precision}", file=f)
        print(f"Recall: {recall}\n", file=f)
        

    print("NEWWWW\n\n")

    print(f"----------------------- VALUE PART SCORES -----------------\n", file=f)

    for i in range(len(mem_parts1)):
        mem1 = mem_parts1[i][1]
        mem2 = mem_parts2[i][1]

        # here my mem parts are like this form Location : NY. So, i want to calculate f1 for NY part only

        # Remove entities where the entity value is 'not mentioned' ignoring case
        mem1 = [m for m in mem1 if ":" in m and m.split(":")[1].strip().lower() != 'not specified']
        mem2 = [m for m in mem2 if ":" in m and m.split(":")[1].strip().lower() != 'not specified']

        mem1 = list(mem1)
        mem2 = list(mem2)

        # Clean up mem_parts
        mem1 = [m.split(":")[1].strip() for m in mem1 if ":" in m]
        mem2 = [m.split(":")[1].strip() for m in mem2 if ":" in m]

        mem1 = set(mem1)
        mem2 = set(mem2)

        print(mem_parts1[i][0], file=f)
        print(mem1, file=f)
        print(mem2, file=f)

        best_match = {}
        for m1 in mem1:
            best_match[m1] = None
            pred = nlp(m1)
            for m2 in mem2:
                similarity = pred.similarity(nlp(m2))
                if similarity >= threshold:
                    best_match[m1] = m2
                    break
        
        # Calculate true positives, false positives, and false negatives
        true_positives = sum(1 for m1 in mem1 if best_match[m1] is not None)
        false_positives = len(mem1) - true_positives
        false_negatives = len(mem2) - true_positives

        print(f"True Positives: {true_positives}", file=f)
        print(f"False Positives: {false_positives}", file=f)
        print(f"False Negatives: {false_negatives}", file=f)
        
        if true_positives + false_positives == 0 or true_positives + false_negatives == 0:
            f1 = 0.0
            precision = 0.0
            recall = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            # if precision + recall is 0 then f1 is 0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        value_precision_scores.append(precision)
        value_recall_scores.append(recall)
        value_f1_scores.append(f1)

        print(f"Precision: {precision}", file=f)
        print(f"Recall: {recall}\n", file=f)
        
        
    
    return type_precision_scores, type_recall_scores, type_f1_scores, value_precision_scores, value_recall_scores, value_f1_scores

with open('output_domain_gpt.txt', 'w') as f:
    type_precision_scores, type_recall_scores, type_f1_scores, value_precision_scores, value_recall_scores, value_f1_scores = calculate_f1(mem_parts1, mem_parts2, f)
    # calculating the average f1 score
    type_precision = sum(type_precision_scores) / len(type_precision_scores)
    type_recall = sum(type_recall_scores) / len(type_recall_scores)
    value_precision = sum(value_precision_scores) / len(value_precision_scores)
    value_recall = sum(value_recall_scores) / len(value_recall_scores)

    print(f"Type Precision: {type_precision}", file=f)
    print(f"Type Recall: {type_recall}", file=f)
    print(f"Value Precision: {value_precision}", file=f)
    print(f"Value Recall: {value_recall}", file=f)