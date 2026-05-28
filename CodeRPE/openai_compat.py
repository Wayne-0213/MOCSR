import requests


api_key = ""
base_url = ""


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def _chat_completion_create(**kwargs):
    if not api_key:
        raise RuntimeError("openai_compat.api_key is empty")
    if not base_url:
        raise RuntimeError("openai_compat.base_url is empty")

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages") or [],
        "temperature": kwargs.get("temperature", 1),
        "top_p": kwargs.get("top_p", 1),
    }
    if kwargs.get("max_completion_tokens") is not None:
        payload["max_completion_tokens"] = kwargs.get("max_completion_tokens")
    if kwargs.get("response_format") is not None:
        payload["response_format"] = kwargs.get("response_format")
    if kwargs.get("frequency_penalty") is not None:
        payload["frequency_penalty"] = kwargs.get("frequency_penalty")
    if kwargs.get("presence_penalty") is not None:
        payload["presence_penalty"] = kwargs.get("presence_penalty")
    if kwargs.get("store") is not None:
        payload["store"] = kwargs.get("store")

    response = requests.post(
        endpoint,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return _Response(data["choices"][0]["message"]["content"])


class _ChatCompletions:
    @staticmethod
    def create(**kwargs):
        return _chat_completion_create(**kwargs)


class _Chat:
    completions = _ChatCompletions()


chat = _Chat()


class ChatCompletion:
    @staticmethod
    def create(**kwargs):
        response = _chat_completion_create(**kwargs)
        return {"choices": [{"message": {"content": response.choices[0].message.content}}]}


class Completion:
    @staticmethod
    def create(**_kwargs):
        raise NotImplementedError("Completion.create is not implemented in openai_compat.")
