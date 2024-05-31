[Here's the link to our paper on archive](https://arxiv.org/abs/2405.18369)

## Steps to run:
1) Install libs listed in [./requirements.txt](./requirements.txt)

2) Update LLM endpoint credentials in [./configs/llm_config.yaml](./configs/llm_config.yaml). We have used Azure Open AI endpoints to access Gpt-4. Update values for `api_key`,   `api_version`, `azure_endpoint`, `deployment_name_in_azure`.
If you are planning to use custom LLM, then you can define the methods in llama-index's CustomLLM class in [./scripts/custom_llms/aml_model.py](./scripts/custom_llms/aml_model.py). In this case update parameter custom_models/path_to_py_file in [./configs/llm_config.yaml](./configs/llm_config.yaml).
3) Hyperparameters related to prompt optimization can be edited in [./configs/promptopt_config.yaml](./configs/promptopt_config.yaml). Ensure that `unique_model_id` specified is defined in  [./configs/llm_config.yaml](./configs/llm_config.yaml)

4) Experiment management related configs can be set in [./configs/setup_config.yaml](./configs/setup_config.yaml). Ensure that `unique_model_id` specified is defined in  [./configs/llm_config.yaml](./configs/llm_config.yaml)

5) Scripts to preprocess datasets & run our prompt optimization framework that we used in our work over some the standard datasets are present in directory [./src/glue-promptopt/scripts](./src/glue-promptopt/scripts)
