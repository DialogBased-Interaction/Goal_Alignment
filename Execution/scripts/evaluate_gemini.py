import google.generativeai as genai
import google.ai.generativelanguage_v1beta.types.content as ai_content

import json
import logging
import pdb
import pickle
import os
import time

import hydra
from dataloader import MultiChoiceDataset, get_data_split
from hydra.core.hydra_config import HydraConfig
from metric import ActionEvaluatorMultiChoice
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class Gemini():
    def __init__(
        self,
        api_key=None,
        model_name=None,
        rate_limit=-1,
        top_p=0.5,
        temperature=0.5,
        system_instruction = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        
        genai.configure(api_key=api_key)
        
        if system_instruction is not None:
            self.model = genai.GenerativeModel(model_name=model_name, system_instruction= system_instruction, safety_settings=safety_settings)
        else:
            self.model = genai.GenerativeModel(model_name=model_name, safety_settings=safety_settings)
        
        self.temperature = temperature
        self.top_p = top_p
        self.generation_config=genai.types.GenerationConfig(candidate_count=1, top_p=self.top_p, temperature=self.temperature)
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0

    def generate(self, prompt, max_new_tokens=50):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
            self.request_interval > 0
            and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)

        prompt[-1]["content"] += "\nGive your answer in the format:\nAnswer: <option>A|B|C|D|E|F</option>\nAction: <action>CLICK|SELECT|TYPE</action>\nValue: <value>if Action is SELECT|TYPE</value>"
        in_prompt=[]
        for pair in prompt[1:]:
            if pair["role"]=="user":
                part = {"role": "user", 'parts': [pair['content']]}
            else:
                part = ai_content.Content({
                        "parts": [
                            ai_content.Part({ # Create a Part object here
                                "text": pair['content']
                            })
                        ],
                        "role": "model"
                    })
            in_prompt.append(part)
        
        response = self.model.generate_content(in_prompt, generation_config = self.generation_config)
        
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                max(start_time, self.next_avil_time[self.current_key_idx])
                + self.request_interval
            )
        return [response.text]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Model {cfg.llm}")
    tokenizer = None#AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    candidate_results = None
    if cfg.data.score_file is not None:
        with open(cfg.data.score_file, "rb") as f:
            candidate_results = pickle.load(f)

    test_dataset_dict = {}
    for test_key, test_split_file in cfg.data.test_split_files.items():
        test_data = get_data_split(
            cfg.data.data_path,
            test_split_file,
            candidate_results=candidate_results,
        )
        test_dataset_dict[test_key] = MultiChoiceDataset(
            test_data,
            tokenizer,
            neg_ratio=cfg.train.neg_ratio,
            num_candidates=cfg.train.num_candidates,
            max_context_len=cfg.train.max_context_len,
        )
    with open(cfg.llm_prompt, "r") as f:
        llm_prompt = json.load(f)

    if cfg.get("output_path"):
        output_path = cfg.output_path
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
    else:
        output_path = HydraConfig.get().runtime.output_dir
    
    model = Gemini(
        model_name=cfg.llm,
        rate_limit=cfg.llm_rate_limit,
        system_instruction=llm_prompt[0]["content"],
        top_p=cfg.top_p,
        temperature=cfg.temperature
    )
    
    evaluator = ActionEvaluatorMultiChoice(tokenizer)
    for test_key, test_dataset in test_dataset_dict.items():
        logger.info(f"Start evaluation for {test_key}")
        result = evaluator.evaluate_dataset_llm(
            test_dataset,
            model,
            output_path=output_path,
            name=test_key,
            prompt_template=llm_prompt,
            top_k=cfg.top_k,
        )
        logger.info(f"Results for {test_key}: {result}")


if __name__ == "__main__":
    main()