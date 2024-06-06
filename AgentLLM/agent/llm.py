from abc import ABC, abstractmethod
import logging
import os
import time
import random
import re

import openai
import tiktoken
from openai import OpenAI

from AgentLLM.utils.llm_cost import CostManager
from AgentLLM.utils.logging import CustomAdapter


class BaseLLM(ABC):
    """Base class for all LLM classes. It defines the api to use the LLMs"""

    def __init__(self, prompt_token_cost: float, response_token_cost: float, max_tokens: int,
                 max_tokens_ratio_per_input: float = 0.7):
        """Constructor for the BaseLLM class
        Args:
            prompt_token_cost (float): Cost of a token in the prompt
            response_token_cost (float): Cost of a token in the response
            max_tokens (int): Maximum number of tokens
            max_tokens_ratio_per_input (int): Maximum ratio of tokens per input in the prompt, to avoid the LLM to use all the tokens in the prompt for just the input
        """
        self.cost_manager = CostManager(prompt_token_cost, response_token_cost)
        self.max_tokens = max_tokens
        self.max_tokens_ratio_per_input = max_tokens_ratio_per_input
        self.logger = logging.getLogger(__name__)
        self.logger = CustomAdapter(self.logger)

    @abstractmethod
    def _calculate_tokens(self, prompt: str) -> int:
        """Abstract method for calculating the number of tokens in the prompt
        Args:
            prompt (str): Prompt
        Returns:
            int: Number of tokens in the prompt
        """
        pass

    def _update_costs(self, prompt_tokens: int, response_tokens: int):
        """Update the cost of the prompt and response
        Args:
            prompt_tokens (int): Number of tokens in the prompt
            response_tokens (int): Number of tokens in the response
        Returns:
            tuple(int, int): Tuple containing the tokens number of the prompt and response
        """
        self.cost_manager.update_costs(prompt_tokens, response_tokens)

    @staticmethod
    def retry_with_exponential_backoff(
            func,
            logger: logging.Logger,
            errors: tuple,
            initial_delay: float = 1,
            exponential_base: float = 1,
            jitter: bool = True,
            max_retries: int = 5,
    ):
        """Retry a function with exponential backoff.

        Args:
            func (function): Function to retry
            logger (logging.Logger): Logger
            errors (tuple): Tuple of type of errors to retry
            initial_delay (float, optional): Initial delay. Defaults to 1.
            exponential_base (float, optional): Exponential base. Defaults to 2.
            jitter (bool, optional): Add jitter to the delay. Defaults to True.
            max_retries (int, optional): Maximum number of retries. Defaults to 5.

        Raises:
            Exception: Maximum number of retries exceeded
            Exception: Any other exception raised by the function that is not specified in the errors tuple

        Returns:
            function: Function to retry with exponential backoff
        """

        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    logger.warning("Error in the llm: %s. Retrying for the %s time. Waiting %.2f seconds", e,
                                   num_retries, delay)

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    @abstractmethod
    def _completion(self, prompt: str, **kwargs) -> tuple[str, int, int]:
        """Abstract method for the completion api
        Args:
            prompt (str): Prompt for the completion
        Returns:
            tuple(str, int, int): A tuple with the completed text, the number of tokens in the prompt and the number of tokens in the response
        """
        pass

    def _load_prompt(self, prompt: str) -> str:
        """Load the prompt from a file or return the prompt if it is a string
        Args:
            prompt_file (str): Prompt file or string
        Returns:
            str: Prompt
        """
        prompt_file = os.path.join("prompts", prompt)

        # Check if the prompt is a string or a file
        if not os.path.isfile(prompt_file):
            if prompt_file.endswith(".txt"):
                logging.error(f"Prompt file: {prompt_file} not found, using the prompt as a string")
                raise ValueError("Prompt file not found")
            return prompt

        with open(prompt_file, "r") as f:
            prompt = f.read()
        return prompt

    def _replace_inputs_in_prompt(self, prompt: str, inputs: list[str] = []) -> str:
        """Replace the inputs in the prompt. The inputs are replaced in the order they are passed in the list.
        Args:
            prompt (str): Prompt. For example: "This is a <input1> prompt with <input2> two inputs"
            inputs (list[str]): List of inputs
        Returns:
            str: Prompt with the inputs
        """
        for i, input in enumerate(inputs):
            if input is None:
                input = 'None'
            # Delete the line if the input is empty
            if str(input).strip() == "":
                regex = rf"^\s*{re.escape(f'<input{i + 1}>')}[ \t\r\f\v]*\n"
                prompt = re.sub(regex, "", prompt, flags=re.MULTILINE)

            prompt = prompt.replace(f"<input{i + 1}>", str(input))

        # Check if there are any <input> left
        if "<input" in prompt:
            raise ValueError("Not enough inputs passed to the prompt")
        return prompt

    def completion(self, prompt: str, **kwargs) -> str:
        """Method for the completion api. It updates the cost of the prompt and response and log the tokens and prompts
        Args:
            prompt (str): Prompt file or string for the completion
            inputs (list[str]): List of inputs to replace the <input{number}> in the prompt. For example: ["This is the first input", "This is the second input"]
        Returns:
            str: Completed text
        """

        prompt = self._load_prompt(prompt)
        prompt = self._replace_inputs_in_prompt(prompt, kwargs.get("inputs", []))

        # Check that the prompt is not too long
        if self._calculate_tokens(prompt) > self.max_tokens * self.max_tokens_ratio_per_input:
            raise ValueError("Prompt is too long")

        self.logger.info(f"Prompt: {prompt}")
        kwargs.pop("inputs", None)  # Remove the inputs from the kwargs to avoid passing them to the completion api
        response, prompt_tokens, response_tokens = self._completion(prompt, **kwargs)
        self.logger.info(f"Response: {response}")

        self._update_costs(prompt_tokens, response_tokens)
        self.logger.info(f"Prompt tokens: {prompt_tokens}")
        self.logger.info(f"Response tokens: {response_tokens}")

        return response


class LLMModels():
    """Class to define the available LLM models"""

    def __new__(self):
        """Constructor for the LLMModels class"""
        # Singleton pattern
        if not hasattr(self, 'instance'):
            self.instance = super(LLMModels, self).__new__(self)
            self.instance.llm_models: dict[str, BaseLLM] = {
                "glm-4": GLM4(),
            }
            self.instance.main_model = "glm-4"
        return self.instance

    def get_main_model(self) -> BaseLLM:
        """Get the main model
        Returns:
            BaseLLM: Main model
        """
        return self.llm_models[self.main_model]

    def get_embedding_model(self) -> BaseLLM:
        """Get the embedding model
        Returns:
            BaseLLM: Embedding model
        """
        return self.llm_models[self.embedding_model]

    def get_longer_context_fallback(self) -> BaseLLM:
        """Get the longer context fallback model
        Returns:
            BaseLLM: Longer context fallback model
        """
        return self.llm_models[self.longer_context_fallback]

    def get_best_model(self) -> BaseLLM:
        """Get the best model
        Returns:
            BaseLLM: Best model
        """
        return self.llm_models[self.best_model]

    def get_costs(self) -> dict:
        """Get the costs of the models
        Returns:
            dict: Costs of the models
        """
        costs = {}
        total_cost = 0
        for model_name, model in self.llm_models.items():
            model_cost = model.cost_manager.get_costs()['total_cost']
            costs[model_name] = model_cost
            total_cost += model_cost

        costs['total'] = total_cost

        return costs

    def get_tokens(self) -> dict:
        """Get the tokens used by the models
        Returns:
            dict: Tokens used by model
        """
        tokens = {}
        total_tokens = 0
        for model_name, model in self.llm_models.items():
            model_tokens = model.cost_manager.get_tokens()['total_tokens']
            tokens[model_name] = model_tokens
            total_tokens += model_tokens

        tokens['total'] = total_tokens

        return tokens


class GLM4(BaseLLM):
    """Class for the GLM-4 model from OpenAI with 8.000 tokens of context"""

    def __init__(self):
        """Constructor for the GLM4 class
        Args:
            prompt_token_cost (float): Cost of a token in the prompt
            response_token_cost (float): Cost of a token in the response
        """
        super().__init__(0.01 / 1000, 0.03 / 1000, 128000, 0.7)

        self.logger.info("Loading GLM-4 model from the OPENAI API...")
        # Load the model
        self.client = OpenAI(api_key="9c7c1ccc16255db2cfda0aa351de9989.A1nLcAimMHTEhrMK", base_url="https://open.bigmodel.cn/api/paas/v4/")
        self.deployment_name = "glm-4"
        self.logger.info("Deployment name: " + self.deployment_name)
        # Encoding to estimate the number of tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4-turbo-preview")

        self.logger.info("GLM-4 model loaded")

    def _format_prompt(self, prompt: str, role: str = 'user') -> list[dict[str, str]]:
        """Format the prompt to be used by the GPT-4 model
        Args:
            prompt (str): Prompt
        Returns:
            list: List of dictionaries containing the prompt and the role of the speaker
        """
        return [
            {"content": prompt, "role": role}
        ]

    def __completion(self, prompt: str, **kwargs) -> tuple[str, int, int]:
        """Completion api for the GPT-4 model
        Args:
            prompt (str): Prompt for the completion
        Returns:
            tuple(str, int, int): A tuple with the completed text, the number of tokens in the prompt and the number of tokens in the response
        """
        prompt = self._format_prompt(prompt)

        # Check if there is a system prompt
        if "system_prompt" in kwargs:
            system_prompt = self._format_prompt(kwargs["system_prompt"], role="system")
            prompt = system_prompt + prompt
            del kwargs["system_prompt"]

        response = self.client.chat.completions.create(model=self.deployment_name, messages=prompt, **kwargs)
        completion = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        response_tokens = response.usage.completion_tokens

        return completion, prompt_tokens, response_tokens

    def _completion(self, prompt: str, **kwargs) -> tuple[str, int, int]:
        """Wrapper for the completion api with retry and exponential backoff

        Args:
            prompt (str): Prompt for the completion

        Returns:
            tuple(str, int, int): A tuple with the completed text, the number of tokens in the prompt and the number of tokens in the response
        """
        wrapper = BaseLLM.retry_with_exponential_backoff(self.__completion, self.logger, errors=(
        openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError))
        return wrapper(prompt, **kwargs)

    def _calculate_tokens(self, prompt: str) -> int:
        """Calculate the number of tokens in the prompt
        Args:
            prompt (str): Prompt
        Returns:
            int: Number of tokens in the prompt
        """
        num_tokens = 0
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += len(self.encoding.encode(prompt))
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens