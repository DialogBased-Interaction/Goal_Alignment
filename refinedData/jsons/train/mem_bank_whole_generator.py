import json


from itertools import combinations
file_path = './train.json'

def generate_mem_bank_subsets(conversations):
    # Extract memory statements excluding the first one (usually at index 1)
    mem_entries = [msg['value'][msg['value'].index("<mem>") + 5:msg['value'].index("</mem>")] 
                   for i, msg in enumerate(conversations) if "<mem>" in msg['value'] and i != 1]
    
    # Generate all non-empty subsets of the memory entries
    subsets = []
    for r in range(0, len(mem_entries) + 1):
        for subset in combinations(mem_entries, r):
            subsets.append(", ".join(subset))
    
    return subsets

# Load the JSON data from the file
cnt = 0

def process_conversations(data):
    
    for item in data:
        conversations = item['conversations']
        all_subsets = generate_mem_bank_subsets(conversations)

        # print(all_subsets)
        m_cnt = -1
        for subset in all_subsets:
            global cnt
            m_cnt = m_cnt+1
            
            tempoString = "{\n" + "\t" + f'"id"' + ": " + f'"{item["id"]}_{m_cnt}"' + ",\n\t" + f'"conversations"' + ": [\n"

            cnt = cnt+1

            abs_task = conversations[0]['value']

            tempoString += "\t{\n" + "\t\t" + f'"from"' + ": " + f'"human"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"<mem_bank> {subset} </mem_bank> {abs_task}"' + "\n\t}"

            prev_memory = conversations[1]['value'].split('</mem>')[0].split('<mem>')[-1].strip()
            if len(subset) > 0 and len(prev_memory)>0:
                prev_memory += ", "+subset.strip()
            elif len(subset) > 0:
                prev_memory += subset.strip()
            prev_ques = conversations[1]['value'].split('</mem>')[-1].strip()
            prev_ans = conversations[0]['value']

            loca_cnt = 0
            accepted = True
            for msg in conversations:
                loca_cnt = loca_cnt + 1
                if loca_cnt <= 1:
                    continue
                if msg['from'] == 'gpt':
                    curr_mem = msg['value'].split('</mem>')[0].split('<mem>')[-1].strip()
                    if curr_mem.strip() in subset:
                        prev_ans=""
                    else:
                        if loca_cnt == 2:
                            continue                   
                        agent_value = f"<mem> {prev_memory} </mem> {prev_ques}" 
                        tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"gpt"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{agent_value}"' + "\n\t}"
                        human_ans = prev_ans
                        
                        tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"human"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{human_ans}"' + "\n\t}"
                        # prev_memory = ""
                        prev_memory = curr_mem
                    
                    prev_ques = msg['value'].split('</mem>')[-1].strip()
                            
                        
                elif msg['from'] == 'human':
                    prev_ans = msg['value']

            tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"gpt"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"<mem> {prev_memory} </mem> <Finish>"' +  "\n\t}"


            tempoString += "]\n},"
            with open("./mem_bank_data/whole/mem_bank_whole.json", "a") as file:
                file.write(tempoString)

with open(file_path, 'r') as file:
    data = json.load(file)

process_conversations(data)
# print(data)

