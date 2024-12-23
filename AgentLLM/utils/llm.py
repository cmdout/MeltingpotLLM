import re
import json
from chromadb import Documents, EmbeddingFunction, Embeddings

# from llm import LLMModels


def extract_answers(response: str) -> dict[str, str]:
    """Extracts the answers from the response. The answers are extracted by parsing the json part of the response.
    
    Args:
        response (str): Response from the LLM. The response should have a json code snippet with the answers. For example: ```json{"plan": "go to the kitchen", "goals": "go to the kitchen"}```

    Returns:
        dict[str, str]: Dictionary with the answers.
    """
    patt = re.compile(r'\s*```json\s*([\w\W\n\r\t]*?)\s*```\s*', re.MULTILINE)
    try:
        response = response.replace('\n', ' ')  # Remove new lines to avoid errors on multiline double quotes
        answers = re.findall(patt, response)[0].strip()  # Get the first json part of the response
        answers = re.sub(r'(:\s*"[^"]+")\s*("[^"]+"\s*:)', r'\1, \2', answers)  # Add missing commas between items
        answers = re.sub(r'"\s*,\s*}', '"}', answers)  # Remove commas before the closing bracket
        parsed_answers = json.loads(answers)
    except:
        parsed_answers = {}
    return parsed_answers


def extract_text(response: str) -> str:
    """Extracts the answers from the response. The answers are extracted by parsing the ```text ``` part of the response.
    
    Args:
        response (str): Response from the LLM. The response should have a plain text snippet with the answers. For example: ```text\nHello, this is a text answer.\n```

    Returns:
        dict[str, str]: Dictionary with the answers.
    """
    patt = re.compile(r'\s*```text\s*([\w\W\n\r\t]*?)\s*```\s*', re.MULTILINE)
    try:
        response = re.findall(patt, response)[0].strip()
    except:
        response = ''
    return response


def extract_tags(response: str) -> dict:
    """Extract tags from the response. A tag is represented as <tag>content<\tag>, where in this case a dict would be return with the key anything and the value content.
    
    Args:
        response (str): Response from the LLM. The response should have tags like XML tags.

    Returns:
        dict: Dictionary with the tags as keys and the content as values.
    """
    patt = r'<(\w+)>(.*)?</\1>'
    return {k: v.strip() for k, v in re.findall(patt, response, re.DOTALL)}


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model = LLMModels().get_embedding_model()

    def __call__(self, texts: Documents):
        return texts
        # return self.model.get_embeddings(texts)
