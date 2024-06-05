<div align="center">

![](https://img.shields.io/badge/Task-Code_Related-blue)
![](https://img.shields.io/badge/Code_License-MIT-green)

</div>

Here's the link to our paper on arxiv: [📜 paper on arxiv](https://arxiv.org/abs/2405.18369)

## 🚀 Steps to run:
### 1) Setup
- Create new conda environment or virtual-environment for python. We have tested our framework on Python 3.10.13. It may work on other python versions as well, but we haven't tested on other versions. 
- Install libs listed in [./requirements.txt](./requirements.txt)
- If you need content moderation enabled install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli)
- If you are planning to use [Azure Open AI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) to access LLMs like GPT-4 etc., create required LLM deployments in your service.

### 2) Update llm_config.yaml 
Update LLM endpoint credentials in [./configs/llm_config.yaml](./configs/llm_config.yaml). We have used [Azure Open AI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) endpoints to access Gpt-4. If you are planning to use the Azure Open AI service, update values for `api_key`,   `api_version`, `azure_endpoint`, `deployment_name_in_azure` . Provide your LLM endpoint a unique id/name by updating value for `unique_model_id`.   
  <br />
    If you are planning to use custom LLM, then you can inherit llama-index's [CustomLLM class](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html) and [GlueLLM class](../glue-common/glue/common/llm/custom_llm.py). Here's a script for demo where we use custom LLM from [Azure ML's model catalogue](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-catalog?view=azureml-api-2), which is hosted as web-service in Azure ML [./scripts/custom_llms/aml_model.py](./scripts/custom_llms/aml_model.py). In this case update parameter custom_models/path_to_py_file in [./configs/llm_config.yaml](./configs/llm_config.yaml).

### 3) Update promptopt_config.yaml 
Hyperparameters related to prompt optimization can be edited in [./configs/promptopt_config.yaml](./configs/promptopt_config.yaml).
   - `unique_model_id` that you specify in promptopt_config.yaml should be defined in  [./configs/llm_config.yaml](./configs/llm_config.yaml)
   - Update values for `task_description`, `base_instruction`, `answer_format`. Refer to comments provided for each of these parameters in promptopt_config.yaml to understand their meaning/usage.
   - **[Advanced]** To understand meaning of rest of the parameters in  promptopt_config.yaml, you will have to go through our [paper](https://arxiv.org/abs/2405.18369)

### 4) Update setup_config.yaml 
Experiment management related configurations can be set in [./configs/setup_config.yaml](./configs/setup_config.yaml).
   - `unique_model_id` that you specify in setup_config.yaml should be defined in  [./configs/llm_config.yaml](./configs/llm_config.yaml) 
   - [Optional] `content_moderation`. We use [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview) to prevent use of our prompt-optimization framework for malicious tasks. It scans the inputs and outputs and aborts the run if malicious task is detected. You need to update `subscription_id`, `resource_group`. You can go with defaults provided for `name`, `location`, `sku_name`. It'll automatically create a AACS service in your azure account. You also need to install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/get-started-with-azure-cli) for authenticating to Azure services.  
   <br />**NOTE:** If you don't wish to use AACS content moderation, remove `content_moderation` field from *setup_config.yaml* to avoid system errors.
   - Specify values for other fields in *setup_config.yaml*. Definition of these fields is specified as comments. 

### 5) Running framework on custom dataset
You can run this framework on any dataset. You need to specify dataset related information e.g. question-format, answer-format etc. to the framework. Scripts to run our prompt optimization framework over some the standard datasets are present in directory [./src/glue-promptopt/scripts](./src/glue-promptopt/scripts). These scripts can be used as reference for creating data handling code for your custom-dataset.
For this one needs to implement `DatasetSpecificProcessing` class and define its abstract methods.
   - Train and test dataset are given to framework using .jsonl files. One needs to define `dataset_to_jsonl` method on how to load the data from any source and put it in .jsonl files. This method can be called twice, once for creating .jsonl file for training dataset and once for test dataset.
   - Define `extract_final_answer` method. It takes as input the raw output of LLM for your question. You need to write the code on how to parse the LLM's output and extract your final answer.
   - [Optional] `access_answer` method takes as input LLM's raw output and ground truth answer. It calls `extract_final_answer` method over raw-output of LLM to extract final answer and compares it with ground truth answer and returns True/False based on whether predicted answer matches with ground truth or not. This method can be overridden based on your scenario.

### 6) [Optional | Advanced] Customize prompt templates in prompt pool.
All the prompt templates used in this framework are defined in [./src/glue-promptopt/glue/promptopt/techniques/critique_n_refine/prompt_pool.yaml]([./src/glue-promptopt/glue/promptopt/techniques/critique_n_refine/prompt_pool.yaml]). To understand the use of each prompt template one might have to go through our [paper](https://arxiv.org/abs/2405.18369) & understand the flow of code.
   - Feel free to experiment with different prompt templates.
   - We encourage you to use the `thinking_styles` defined in [PromptBreeder by Google's DeepMind](https://github.com/vaughanlove/PromptBreeder/blob/main/pb/thinking_styles.py). Copy the thinking styles and paste it under `thinking_styles` field in prompt_pool.yaml

### 7) Demo: Running the framework on [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
```sh
cd src/glue-promptopt/scripts
bash python run_gsm8k.py
```

## 📖 License

This code repository is licensed under the MIT License.

## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@article{PromptWizardFramework,
  title={PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework},
  author={Eshaan Agarwal∗, Vivek Dani∗, Tanuja Ganu, Akshay Nambi},
  journal={arXiv preprint arXiv:2405.18369},
  year={2024}
}
∗Equal Contributions
```

## 🍀 Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

Resources:

- [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
- [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
- Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns