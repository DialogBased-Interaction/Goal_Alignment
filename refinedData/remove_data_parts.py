import json
import re
from copy import deepcopy
import os

remove = [
    r"[^,>]+:.?[nN]ot specified.?[,(\s)]",
    r"[^,>]+:.?[nN]ot provided.?[,(\s)]",
    r"[^,>]+:.?[uU]nknown.?[,(\s)]",
    r"[^,>]+:.?N/A.?[,(\s)]",
    r"[^,>]+:.?\[[^\[\]]+\].?[,(\s)]?"
]

def addConv(conv):
    if len(conv)>2:
        return True
    gpt = conv[1]["value"]
    split1 = "<mem>"
    split2 = "</mem>"
    mem = gpt[gpt.find(split1)+len(split1):gpt.find(split2)]
    if len(mem.strip())>0:
        return True
    return False

def mod_mem(memPart):
    if len(memPart)==0:
        return " "
    if memPart[0]!=' ':
        memPart = " " + memPart
    if memPart[-1]==",":
        memPart = memPart[:-1]
    if memPart[-1]!=' ':
        memPart = memPart + " "
    return memPart

def modify(data):
    convs = []
    removed = []
    for dpoint in data:
        conv = deepcopy(dpoint["conversations"])
        mod = []
        #1st pair
        for i in range(2):
            curr = deepcopy(conv[i]["value"])
            add = True
            for rem in remove:
                if len(re.findall(rem, curr)):
                    curr = re.sub(rem, "", curr)
                    add = False
            if add:
                mod.append(conv[i])
                continue
            split1 = "<mem>"
            split2 = "</mem>"
            if "mem_bank" in curr:
                split1 = "<mem_bank>"
                split2 = "</mem_bank>"
            memPart = curr[curr.find(split1)+len(split1):curr.find(split2)]
            memPart = split1 + mod_mem(memPart) + split2
            lastPart = curr[curr.find(split2)+len(split2):]
            mod.append({"from":conv[i]["from"], "value":memPart + lastPart})
        #remaining
        for i in range(3, len(conv), 2):
            curr = deepcopy(conv[i]["value"])
            add = True
            for rem in remove:
                if len(re.findall(rem, curr)):
                    add = False
                    curr = re.sub(rem, "", curr)
            if add:
                mod += [conv[i-1], conv[i]]
                continue
            split1 = "<mem>"
            split2 = "</mem>"
            memPart = mod_mem(curr[curr.find(split1)+len(split1):curr.find(split2)])
            if len(memPart.strip())==0:
                #don't add this pair and replace gpt question part of prev with curr
                prev = mod[-1]["value"]
                comb = prev[:prev.find(split2)+len(split2)] + curr[curr.find(split2)+len(split2):]
                mod[-1]["value"] = comb
                continue
            curr = split1 + memPart + curr[curr.find(split2):]
            mod += [conv[i-1], {"from":conv[i]["from"], "value":curr}]
        if addConv(mod):
            convs.append({"id":dpoint["id"], "conversations":mod})
        else:
            removed.append({"id":dpoint["id"], "conversations":mod})
    return convs, removed

out_ = "removed_Partially"

if not os.path.exists(out_):
    os.mkdir(out_)

for root, _, files in os.walk("jsons"):
    path_split = "/"
    for file in files:
        if file.endswith("json"):
            dirs = root.split("/")
            outdir = os.path.join(out_, dirs[0])
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            for dr in dirs[1:]:
                outdir = os.path.join(outdir, dr)
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
            inPath = os.path.join(root, file)
            print(inPath)
            with open(inPath, "r") as f:
                inpData = json.load(f)
            outData, remData = modify(inpData)
            print(f"{len(remData)} samples removed from total {len(inpData)} samples... {len(remData)/len(inpData)*100}%")
            with open(os.path.join(outdir, file), "w") as f:
                json.dump(outData, f)
            with open(os.path.join(outdir, "removed_"+file), "w") as f:
                json.dump(remData, f)