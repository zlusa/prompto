from typing import Dict
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
from ..base_classes import LLMConfig
from ..constants.str_literals import InstallLibs, OAILiterals, \
    OAILiterals, LLMLiterals, LLMOutputTypes
from .llm_helper import get_token_counter
from ..exceptions import GlueLLMException
from ..utils.runtime_tasks import install_lib_if_missing
from ..utils.logging import get_glue_logger
from ..utils.runtime_tasks import str_to_class
import os
logger = get_glue_logger(__name__)

def call_api(messages):
    from openai import OpenAI
    from azure.identity import get_bearer_token_provider, AzureCliCredential
    from openai import AzureOpenAI
    import google.generativeai as genai
    import json

    if os.environ.get('USE_OPENAI_API_KEY') == 'True':
        try:
            if "gemini" in os.environ["OPENAI_MODEL_NAME"].lower():
                # Try using OpenAI compatibility layer first
                client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_API_BASE")
                )

                try:
                    response = client.chat.completions.create(
                        model=os.environ["OPENAI_MODEL_NAME"],
                        messages=messages,
                        temperature=0.0,
                    )
                    return response.choices[0].message.content
                except Exception as openai_error:
                    logger.warning(f"OpenAI compatibility layer failed: {str(openai_error)}")
                    logger.info("Falling back to native Gemini API...")
                    
                    # Fallback to native Gemini API
                    genai.configure(api_key=os.environ["OPENAI_API_KEY"])
                    
                    # Convert OpenAI message format to Gemini format
                    parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        
                        if role == "system":
                            parts.append(f"Instructions: {content}\n\n")
                        elif role == "user":
                            parts.append(f"User: {content}\n")
                        elif role == "assistant":
                            parts.append(f"Assistant: {content}\n")
                    
                    # Try Gemini 2.0 Flash first
                    try:
                        model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    except Exception as e:
                        logger.warning(f"Gemini 2.0 Flash failed: {str(e)}")
                        logger.info("Falling back to Gemini 1.5 Flash...")
                        model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    response = model.generate_content(
                        "".join(parts),
                        generation_config={
                            "temperature": 0.0,
                            "top_p": 1,
                            "top_k": 1,
                            "max_output_tokens": 8192,  # Maximum for Flash models
                        }
                    )
                    
                    if response.prompt_feedback:
                        logger.info(f"Prompt feedback: {response.prompt_feedback}")
                    
                    if hasattr(response, 'error'):
                        raise GlueLLMException("Unable to generate response", response.error)
                    
                    return response.text
            else:
                # Regular OpenAI API call
                client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_API_BASE")
                )
                response = client.chat.completions.create(
                    model=os.environ["OPENAI_MODEL_NAME"],
                    messages=messages,
                    temperature=0.0,
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            raise GlueLLMException("API error", e)
    else:
        # Existing Azure code
        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = AzureOpenAI(
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider
        )
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=messages,
            temperature=0.0,
        )
        prediction = response.choices[0].message.content
        return prediction


class LLMMgr:
    @staticmethod
    def chat_completion(messages: Dict):
        llm_handle = os.environ.get("MODEL_TYPE", "AzureOpenAI")
        try:
            if(llm_handle == "AzureOpenAI"): 
                # Code to for calling LLMs
                return call_api(messages)
            elif(llm_handle == "LLamaAML"):
                # Code to for calling SLMs
                return 0
        except Exception as e:
            print(e)
            return "Sorry, I am not able to understand your query. Please try again."
            # raise GlueLLMException(f"Exception when calling {llm_handle.__class__.__name__} "
            #                        f"LLM in chat mode, with message {messages} ", e)
        

    @staticmethod
    def get_all_model_ids_of_type(llm_config: LLMConfig, llm_output_type: str):
        res = []
        if llm_config.azure_open_ai:
            for azure_model in llm_config.azure_open_ai.azure_oai_models:
                if azure_model.model_type == llm_output_type:
                    res.append(azure_model.unique_model_id)
        if llm_config.custom_models:
            if llm_config.custom_models.model_type == llm_output_type:
                res.append(llm_config.custom_models.unique_model_id)
        return res

    @staticmethod
    def get_llm_pool(llm_config: LLMConfig) -> Dict[str, LLM]:
        """
        Create a dictionary of LLMs. key would be unique id of LLM, value is object using which
        methods associated with that LLM service can be called.

        :param llm_config: Object having all settings & preferences for all LLMs to be used in out system
        :return: Dict key=unique_model_id of LLM, value=Object of class llama_index.core.llms.LLM
        which can be used as handle to that LLM
        """
        llm_pool = {}
        az_llm_config = llm_config.azure_open_ai

        if az_llm_config:
            install_lib_if_missing(InstallLibs.LLAMA_LLM_AZ_OAI)
            install_lib_if_missing(InstallLibs.LLAMA_EMB_AZ_OAI)
            install_lib_if_missing(InstallLibs.LLAMA_MM_LLM_AZ_OAI)
            install_lib_if_missing(InstallLibs.TIKTOKEN)

            import tiktoken
            # from llama_index.llms.azure_openai import AzureOpenAI
            from openai import AzureOpenAI
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

            az_token_provider = None
            # if az_llm_config.use_azure_ad:
            from azure.identity import get_bearer_token_provider, AzureCliCredential
            az_token_provider = get_bearer_token_provider(AzureCliCredential(),
                                                        "https://cognitiveservices.azure.com/.default")

            for azure_oai_model in az_llm_config.azure_oai_models:
                callback_mgr = None
                if azure_oai_model.track_tokens:
                    
                    # If we need to count number of tokens used in LLM calls
                    token_counter = TokenCountingHandler(
                        tokenizer=tiktoken.encoding_for_model(azure_oai_model.model_name_in_azure).encode
                        )
                    callback_mgr = CallbackManager([token_counter])
                    token_counter.reset_counts()
                    # ()

                if azure_oai_model.model_type in [LLMOutputTypes.CHAT, LLMOutputTypes.COMPLETION]:
                    # ()
                    llm_pool[azure_oai_model.unique_model_id] = \
                        AzureOpenAI(
                            # use_azure_ad=az_llm_config.use_azure_ad,
                                    azure_ad_token_provider=az_token_provider,
                                    # model=azure_oai_model.model_name_in_azure,
                                    # deployment_name=azure_oai_model.deployment_name_in_azure,
                                    api_key=az_llm_config.api_key,
                                    azure_endpoint=az_llm_config.azure_endpoint,
                                    api_version=az_llm_config.api_version,
                                    # callback_manager=callback_mgr
                                    )
                    # ()
                elif azure_oai_model.model_type == LLMOutputTypes.EMBEDDINGS:
                    llm_pool[azure_oai_model.unique_model_id] =\
                        AzureOpenAIEmbedding(use_azure_ad=az_llm_config.use_azure_ad,
                                             azure_ad_token_provider=az_token_provider,
                                             model=azure_oai_model.model_name_in_azure,
                                             deployment_name=azure_oai_model.deployment_name_in_azure,
                                             api_key=az_llm_config.api_key,
                                             azure_endpoint=az_llm_config.azure_endpoint,
                                             api_version=az_llm_config.api_version,
                                             callback_manager=callback_mgr
                                             )
                elif azure_oai_model.model_type == LLMOutputTypes.MULTI_MODAL:

                    llm_pool[azure_oai_model.unique_model_id] = \
                        AzureOpenAIMultiModal(use_azure_ad=az_llm_config.use_azure_ad,
                                              azure_ad_token_provider=az_token_provider,
                                              model=azure_oai_model.model_name_in_azure,
                                              deployment_name=azure_oai_model.deployment_name_in_azure,
                                              api_key=az_llm_config.api_key,
                                              azure_endpoint=az_llm_config.azure_endpoint,
                                              api_version=az_llm_config.api_version,
                                              max_new_tokens=4096
                                              )

        if llm_config.custom_models:
            for custom_model in llm_config.custom_models:
                # try:
                custom_llm_class = str_to_class(custom_model.class_name, None, custom_model.path_to_py_file)

                callback_mgr = None
                if custom_model.track_tokens:
                    # If we need to count number of tokens used in LLM calls
                    token_counter = TokenCountingHandler(
                        tokenizer=custom_llm_class.get_tokenizer()
                        )
                    callback_mgr = CallbackManager([token_counter])
                    token_counter.reset_counts()
                llm_pool[custom_model.unique_model_id] = custom_llm_class(callback_manager=callback_mgr)
                # except Exception as e:
                    # raise GlueLLMException(f"Custom model {custom_model.unique_model_id} not loaded.", e)
        return llm_pool

    @staticmethod
    def get_tokens_used(llm_handle: LLM) -> Dict[str, int]:
        """
        For a given LLM, output the number of tokens used.

        :param llm_handle: Handle to a single LLM
        :return: Dict of token-type and count of tokens used
        """
        token_counter = get_token_counter(llm_handle)
        if token_counter:
            return {
                LLMLiterals.EMBEDDING_TOKEN_COUNT: token_counter.total_embedding_token_count,
                LLMLiterals.PROMPT_LLM_TOKEN_COUNT: token_counter.prompt_llm_token_count,
                LLMLiterals.COMPLETION_LLM_TOKEN_COUNT: token_counter.completion_llm_token_count,
                LLMLiterals.TOTAL_LLM_TOKEN_COUNT: token_counter.total_llm_token_count
                }
        return None
