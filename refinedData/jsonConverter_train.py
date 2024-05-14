import pandas as pd
import re


cnt = 0
m_cnt = 0
total_mismatch = 0
df = pd.read_csv('./gpt_generated/refined_train_response.csv')

def create_json_format(text):
    global cnt
    global m_cnt
    global df
    global total_mismatch
    abs_match = re.search(r'<Abs>(.*?)</Abs>', text, re.DOTALL)
    questions_match = re.findall(r'<Q>(.*?)</Q>\s*<A>(.*?)</A>\s*<mem>(.*?)</mem>', text, re.DOTALL)

    conversations = []

    # tempoString = "{\n" + "\t" + f'"id"' + ": " + f'"identity_{df.loc[cnt, 'index']}"' + ",\n\t" + f'"conversations"' + ": [\n" 
    tempoString = "{\n" + "\t" + f'"id"' + ": " + f'"identity_{df.loc[cnt, "index"]}"' + ",\n\t" + f'"conversations"' + ": [\n"

    cnt = cnt+1
    m_cnt = 0
    curr_abstract = ""
    if abs_match:
        conversations.append(f'{{"from": "human", "value": "{abs_match.group(1).strip()}" }}')
        curr_abstract = abs_match.group(1).strip()
        tempoString += "\t{\n" + "\t\t" + f'"from"' + ": " + f'"human"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{abs_match.group(1).strip()}"' + "\n\t}"

    prev_memory = ""
    abstract_mem_bank=""

    for question, answer, memory in questions_match:
        if memory.strip().lower().split(':')[-1].strip() in curr_abstract.lower():
            if abstract_mem_bank == "":
                abstract_mem_bank += memory.strip()
            else:
                abstract_mem_bank += ", " + memory.strip()


    
    abstract_mem_bank = "<mem> " + abstract_mem_bank + " </mem>"

    
    for question, answer, memory in questions_match:
        if memory.strip().lower().split(':')[-1].strip() in curr_abstract.lower():
            m_cnt = m_cnt+1
        else:
            agent_value = f"{prev_memory} {question.strip()}" if prev_memory else f"{abstract_mem_bank} {question.strip()}"
            conversations.append(f'{{"from": "gpt", "value": "{agent_value}"}}')
            tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"gpt"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{agent_value}"' + "\n\t}"
            conversations.append(f'{{"from": "human", "value": "{answer.strip()}"}}')
            tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"human"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{answer.strip()}"' + "\n\t}"
            prev_memory = f"<mem> {memory.strip()} </mem> "


    
    if prev_memory:
        conversations.append(f'{{"from": "gpt", "value": "{prev_memory} <Finish>"}}')
        tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"gpt"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{prev_memory} <Finish>"' +  "\n\t}"
    else:
        conversations.append(f'{{"from": "gpt", "value": "{abstract_mem_bank} <Finish>"}}')
        tempoString += ",\n\t{\n" + "\t\t" + f'"from"' + ": " + f'"gpt"' + ",\n" + "\t\t" + f'"value"' + ": " + f'"{abstract_mem_bank} <Finish>"' + "\n\t}"

    tempoString += "]\n},"
    with open("./jsons/train/train.json", "a") as file:
        file.write(tempoString)

    # print("matched case count", m_cnt)
    total_mismatch += m_cnt

    if m_cnt > 0:
        print("Index:", df.loc[cnt-1, 'index'], "; Abstract: ", abs_match.group(1).strip(),"; Match count: ", m_cnt)  
        
    

    return " 'conversations': [" + ', '.join(conversations) + ']'





df['json format'] = df['final_conversation'].apply(create_json_format)


print("total refined case: ", total_mismatch)
print("percentage: ", (total_mismatch/cnt)*100.0)

