import os
import pdb
import yaml
import json

from pathlib import Path

def load_prompts_generales(prompt_type: str) -> str:
  prompt_files_path = Path(__file__).parent.parent / "utils"
  prompt_file_name: str = "prompts_gemini.yaml"
  prompt_file_path = str(prompt_files_path / prompt_file_name)
  with open(prompt_file_path, "r", encoding="utf-8") as file:
    prompts_variables = yaml.safe_load(file)
    
  return prompts_variables.get(prompt_type, "")

def load_llm_parameters(model_name: str) -> dict:
  llm_parameters_path = Path(__file__).parent.parent / "config"
  llm_parameters_file_name: str = "llm_parameters.json"
  llm_parameters_file_path = str(llm_parameters_path / llm_parameters_file_name)
  print(llm_parameters_file_path)
  with open(llm_parameters_file_path, 'r') as file:
    llm_parameters = json.load(file)
    
  return llm_parameters.get(model_name, {})

