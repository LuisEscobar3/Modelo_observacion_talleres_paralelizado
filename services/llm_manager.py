import os
import logging
from ia_transversal_langchain_python_lib.llm.llm_middleware import LlmMiddleware
from app.services.miscelaneous import load_llm_parameters

# Configurar logging
logging.basicConfig(level=logging.INFO)

def load_llms():
    """Inicializa y devuelve los modelos LLM configurados."""

    middleware = LlmMiddleware()

    modelos = {
        "gpt": {
            "config": load_llm_parameters("gpt-4o-mini").get("model_config", {}),
            "params": load_llm_parameters("gpt-4o-mini").get("model_parameters", {})
        },
        "gemini_pro": {
            "config": load_llm_parameters("gemini-1.5-pro").get("model_config", {}),
            "params": load_llm_parameters("gemini-1.5-pro").get("model_parameters", {})
        },
        "gemini_flash": {
            "config": load_llm_parameters("gemini-1.5-flash").get("model_config", {}),
            "params": load_llm_parameters("gemini-1.5-flash").get("model_parameters", {})
        }
    }

    llms = {
        "gpt": middleware.get_chat(
            platform=modelos["gpt"]["config"]["plataform"],
            provider=modelos["gpt"]["config"]["provider"],
            model_name=modelos["gpt"]["config"]["model_name"],
            model_parameters=modelos["gpt"]["params"]
        ),
        "gemini_pro": middleware.get_chat(
            platform=modelos["gemini_pro"]["config"]["plataform"],
            provider=modelos["gemini_pro"]["config"]["provider"],
            model_name=modelos["gemini_pro"]["config"]["model_name"],
            model_parameters=modelos["gemini_pro"]["params"]
        ),
        "gemini_flash": middleware.get_chat(
            platform=modelos["gemini_flash"]["config"]["plataform"],
            provider=modelos["gemini_flash"]["config"]["provider"],
            model_name=modelos["gemini_flash"]["config"]["model_name"],
            model_parameters=modelos["gemini_flash"]["params"]
        )
    }

    logging.info("✅ Modelos cargados desde llm_manager.py")
    return llms
