We have used the scripts from the [action prediction stage](https://github.com/OSU-NLP-Group/Mind2Web/tree/main/src/action_prediction) with the following changes:
    * **Training**: The file [train.py](scripts/train.py) with added code snippet for saving LoRA model adapters\
    * **Evaluation**: 
      - [evaluate.py](scripts/evaluate.py) with added code snippet for loading LoRA model adapters.
      - [metric.py](scripts/metric.py) also contains the overall Success Rate metric calculation.
      - [llm_prompt.json](scripts/llm_prompt.json) has the modified input task in the 3-shot prompt for LLM as required for the execution phase of MemAgent.
      - [evaluate_gemini.py](scripts/evaluate_gemini.py) runs evaluation for Gemini-pro.