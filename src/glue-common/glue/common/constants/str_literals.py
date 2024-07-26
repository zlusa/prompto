from dataclasses import dataclass

# TODO: add comments  for class definition and variable definition
# This file has classes storing constant literals


@dataclass
class FileConstants:
    logfile_name = "glue_logs.log"
    logfile_prefix = "glue_logs_"


@dataclass
class OAILiterals:
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_API_BASE = "OPENAI_API_BASE"
    OPENAI_API_TYPE = "OPENAI_API_TYPE"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"
    AZ_OPEN_AI_OBJECT = "AZ_OPEN_AI_OBJECT"


@dataclass
class LLMOutputTypes:
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    MULTI_MODAL = "multimodal"


@dataclass
class InstallLibs:
    LLAMA_LLM_AZ_OAI = "llama-index-llms-azure-openai==0.1.5"
    LLAMA_EMB_AZ_OAI = "llama-index-embeddings-azure-openai==0.1.6"
    LLAMA_MM_LLM_AZ_OAI = "llama-index-multi-modal-llms-azure-openai==0.1.4"
    AZURE_CORE = "azure-core==1.30.1"
    TIKTOKEN = "tiktoken"
    AZ_CONTENT_SAFETY = "azure.ai.contentsafety"
    AZ_IDENTITY = "azure-identity"
    AZ_MGMT_COG_SERVICE = "azure-mgmt-cognitiveservices==13.4.0"


@dataclass
class LLMLiterals:
    EMBEDDING_TOKEN_COUNT = "embedding_token_count"
    PROMPT_LLM_TOKEN_COUNT = "prompt_llm_token_count"
    COMPLETION_LLM_TOKEN_COUNT = "completion_llm_token_count"
    TOTAL_LLM_TOKEN_COUNT = "total_llm_token_count"


@dataclass
class DirNames:
    MODEL_DIR = "custom_models"
    PACKAGE_BASE_DIR = "copilot_platform"


@dataclass
class URLs:
    AZ_CREDENTIAL_URL= "https://management.azure.com/.default"
    AZ_COGNITIVE_SERVICES_URL = "https://cognitiveservices.azure.com/.default"
