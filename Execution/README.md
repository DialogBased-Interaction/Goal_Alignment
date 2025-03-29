# MemAgent Execution
For the execution part, we have mainly used the scripts from the [Mind2Web action prediction stage](https://github.com/OSU-NLP-Group/Mind2Web/tree/main/src/action_prediction) with minor [changes](Changelist.md). Please refer to the [mind2web official repository](https://github.com/OSU-NLP-Group/Mind2Web) for full details. Our experiment results are in this [folder](Results).
* Clone the Mind2Web repository:
    ```
    git clone https://github.com/OSU-NLP-Group/Mind2Web.git
    ```
* To download the training data of Mind2Web, clone from [HuggingFace](https://huggingface.co/datasets/osunlp/Mind2Web):
    ```
    git clone git@hf.co:datasets/osunlp/Mind2Web
    ```
* Download the test set from this [link](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EUkdc16xUC1EplDiXS1bvSEBOddFLgOyJNkWJOxdltNEGA?e=8N1D9S) and unzip it with the password `mind2web`
* We have used Mind2Web's off-the-shelf deberta model outputs from the candidate generation step. Please save the candidate rankings file from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EZllMua3lABAhXQnCN7-pr4BIP4YV8xPfbgyP5FXT18wag?e=yXkK8k)

## Requirements
```
pip install -r reqirements.txt
```

### Fine-tuning
For training, set the parameters in the [config files](scripts/conf):
```
data:
  data_path: DATA_PATH
  train_split_file: data/train/*.json
  test_split_files:
    test_task: data/test_task/*.json
    ...
  score_file: CANDIDATE_SCORE_FILE_PATH
...
hydra:
  run:
    dir: LOG_PATH
```
and run:
```
python scripts/train.py
```
If lora is set to True, the fine-tuned model adapter is saved in hydra runtime dir LOG_PATH/lora

### Evaluation
For evaluation with fine-tuned models, set the parameters in the [config files](scripts/conf) accordingly.
If fine-tuning was done using LoRA, set the lora_adapter parameter in the [model config file](scripts/conf/model):
```
name: flan-t5-large
model_name_or_path : "osunlp/MindAct_ActionPrediction_flan-t5-large"
lora_adapter : ADAPTER_PATH
...
```
and run:
```
python scripts/evaluate.py
```
### Evaluation with LLM
We have reported results on 50 samples per split for LLM evaluation. We selected the samples based on the minimum number of action steps where the Mind2Web preprocessing steps and the Candidate Generator model missed a positive element for an action step.
* The cross-domain split has 50 random samples with no missing positive candidates for all steps.
* The cross-test split has 43 samples with positive candidates for all steps and 7 random samples with missing candidates for 1 action step.
* The cross-website split has 29 samples with positive candidates for all steps and 21 random samples with missing candidates for 1 action step.

To create this subset of data for evaluation with LLM, run:
```
python selectDataForLLM.py --top_k 10 --total 50 --score <CANDIDATE_SCORE_FILE> --data <TEST_DATA_DIR> --output <OUTPUT_DIR>
```
For evaluation with LLM using [OpenAI API](https://platform.openai.com/docs/overview), set an environment variable named `OPENAI_API_KEY` to your [OpenAI key](https://platform.openai.com/api-keys) and parameters in the config file, then run:
```
python scripts/evaluate_llm.py\
  +llm_prompt=scripts/llm_prompt.json\
  +llm=gpt-4o\
  +llm_rate_limit=4
```
Similarly, for evaluation with [Google Gemini API](https://ai.google.dev/), set an environment variable named `GEMINI_API_KEY` to your [API key](https://aistudio.google.com/app/apikey) and parameters in the config file, then run:
```
python scripts/evaluate_gemini.py\
  +llm_prompt=scripts/llm_prompt.json\
  +llm=gemini-1.5-pro-latest\
  +llm_rate_limit=4
```
Where, `llm`: OpenAI/Google AI model, `llm_prompt`: prompt file and `llm_rate_limit`: number of calls to the model's generate function per minute. Adjust this value within the API limit.
