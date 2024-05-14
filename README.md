# Goal Alignment via Dialogue based Interaction

## Overview
The **Goal Alignment via Dialogue based Interaction** project aims to align user goals through multilevel conversations using the FastChat platform. By fine-tuning different language models, such as Vicuna 7B and Vicuna 13B, we facilitate effective communication and understanding between users and the system.

## Features
- **Multilevel Conversations**: The project leverages multilevel dialogues to ensure comprehensive goal alignment.
- **FastChat Platform**: We utilize the FastChat platform for efficient and context-aware interactions.
- **Model Fine-Tuning**: Fine-tuning of language models (e.g., Vicuna 7B, Vicuna 13B ...).
- **Mind2Web Dataset**: Our training data comes from the Mind2Web dataset, which provides diverse conversational examples for generalized tasks.

## Getting Started
1. Clone this repository from [here](https://github.com/lm-sys/FastChat)
2. Have the checkpoints from [here](https://drive.google.com/drive/folders/1nR1GXj_BbIAwS5eMQLMpBEJJjNezTsQv?usp=sharing).
3. Mind2Web dataset can be found in this repo "./mind2web dataset". From this dataset and with the help of gpt-4 (model: gpt-4-1106-preview), we generated our train data. Find the code in \gpt4datagen\DataGen directory
5. You may Fine tune with our given command script

## Fine tuning
scripts for fine tuning can be found in the ./scripts directory

## Inferencing with the fine tuned model
1. Initialize the FastChat system. Install dependencies
2. Run with the following command:
```
    python3 -m fastchat.serve.cli --model-path path_to_directory  
```   
3. You can also run inferencing with a specific checkpoint.

## Evaluation Details
- **Vicuna Models**: We fine-tuned Vicuna 7B model.
- **BLEU_Score**: We have calculated bleu score with [sacrebleu](https://pypi.org/project/sacrebleu/).
- **BERT_score**


# Evaluation scores

## With Mem-bank
| Evaluation Criteria       | Cross-Task             | Cross-Website          |  Cross-Domain          |
|---------------------------|------------------------|------------------------|------------------------|
| AVERAGE BLEU SCORE        | 47.17                  | 50.07                  | 50.41                  |
| AVERAGE BERT SCORE        | 0.95                   | 0.96                   | 0.96                   |
| AVERAGE # of conversation | 2.52                   | 2.44                   | 2.67                   |

## Without Mem-bank
| Evaluation Criteria       | Cross-Task             | Cross-Website          |  Cross-Domain          |
|---------------------------|------------------------|------------------------|------------------------|
| AVERAGE BLEU SCORE        | 37.64                  | 39.31                  | 39.87                  |
| AVERAGE BERT SCORE        | 0.91                   | 0.92                   | 0.94                   |
| AVERAGE # of conversation | 3.49                   | 3.40                   | 3.66                   |

## Improvements
> We have developed the data generation for the training purpose
> We have introduced RAG pipeline 
