import ast
import json
import os
import ssl
from typing import Any, List, Optional, Sequence
import urllib.request

from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

from llama_index.core.llms.callbacks import (
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response
)

from glue.common.llm.custom_llm import GlueLLM


class LLamaAML(CustomLLM, GlueLLM):
    """
    Use Llama model that is deployed on AML endpoint.
    """
    context_window: int = 4096
    temperature: float = 0.1
    model_name: str = "llama13b-chat"
    is_chat_model = True
    url: str = "https://**.eastus.inference.ml.azure.com/score"
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer **'),
               'azureml-model-deployment': '**' }

    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        super().__init__(callback_manager=callback_manager)

        self.context_window: int = 4096
        self.temperature: float = 0.1
        self.model_name: str = "llama13b-chat"
        self.is_chat_model = True
        self.url: str = "https://**.eastus.inference.ml.azure.com/score"
        self.headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer **'),
                        'azureml-model-deployment': '**' }

        if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model
        )

    def messages_to_dict(self, messages: List[ChatMessage]):
        message_dict_list = []

        for message in messages:
            message_dict = {}
            if message.role == "system":
                message_dict["role"] = "system"
            elif message.role == "user":
                message_dict["role"] = "user"
            elif message.role == "assistant":
                message_dict["role"] = "assistant"
            message_dict["content"] = message.content
            message_dict_list.append(message_dict)
        return message_dict_list

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dict_list = self.messages_to_dict(messages)
        completion_response = self.complete(message_dict_list, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, message_dict_list: str, **kwargs: Any) -> CompletionResponse:
        data = {"input_data": {
                    "input_string": message_dict_list,
                    "parameters": {
                      "temperature": self.temperature,
                      "top_p": 0.9,
                      "do_sample": True,
                      "max_new_tokens": self.context_window
                    }
                  }
                }

        body = str.encode(json.dumps(data))
        req = urllib.request.Request(self.url, body, self.headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()
            result = ast.literal_eval(result.decode('utf-8'))["output"]
            return CompletionResponse(text=result)
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))


    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)
