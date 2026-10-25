from openai import OpenAI, AzureOpenAI
import os, time, json, re

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

class OpenAI_Model:
    def __init__(self):
        if "AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ:
            self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_version="2024-10-01-preview")
        else:
            assert "OPENAI_API_KEY" in os.environ
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def cost_calculator(self, model, usage, is_batch_model=False):
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        else:
            # Unknown model — fall back to gpt-4o rates as a rough cost estimate.
            # The cost field is only logged, never used to drive simulation logic.
            print(f"[cost] Warning: unknown pricing for {model}, estimating with gpt-4o rates")
            inp_token_cost, out_token_cost = 0.0025, 0.01

        cache_discount = 0.5 # cached tokens are half the price
        batch_discount = 0.5 # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(self, messages, model="gpt-4o-mini", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}, reasoning_effort=None):
        kwargs = {}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            if temperature != 1.0:
                print(f"[reasoning] {model} with reasoning_effort requires temperature=1.0 (got {temperature}); forcing 1.0")
                temperature = 1.0
        N = 0

        messages = format_messages(messages, variables)

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if model.startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        while True:
            try:
                response = self.client.chat.completions.create(model=model, messages=messages, timeout=timeout, max_completion_tokens=max_tokens, temperature=temperature, **kwargs)
                break
            except:
                N += 1
                if N >= max_retries:
                    raise Exception("Failed to get response from OpenAI")
                else:
                    time.sleep(4)

        response = response.to_dict()
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = 0
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']

        if not return_metadata:
            return response_text
        return {"message": response_text, "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "prompt_tokens_cached": prompt_tokens_cached, "completion_tokens": usage['completion_tokens'], "total_usd": total_usd}

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        # Some backends (e.g. glm-5p2 via the fireworks shim) do not reliably
        # honour response_format=json_object and either wrap the JSON in
        # ```json ... ``` markdown fences or return empty content. The raw
        # json.loads below would then raise "Expecting value: line 1 column 1
        # (char 0)" — the SAME error for empty, fenced, or bare-backtick input,
        # which previously mis-attributed the failure to "empty content". Strip
        # fences and retry on parse failure so a single bad response does not
        # kill the whole simulation.
        parse_retries = kwargs.pop("parse_retries", 3)
        last_err = None
        for attempt in range(parse_retries):
            response = self.generate(messages, model, is_json=True, **kwargs)
            raw = response.get("message")
            if not raw or not raw.strip():
                last_err = ValueError(f"empty content from {model}")
                print(f"[generate_json] empty content (attempt {attempt+1}/{parse_retries}) from {model}")
                continue
            s = raw.strip()
            # strip ```json ... ``` / ``` ... ``` markdown fences
            if s.startswith("```"):
                s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
                idx = s.rfind("```")
                if idx != -1:
                    s = s[:idx]
                s = s.strip()
            if not s:
                last_err = ValueError(f"empty content from {model} after fence-strip")
                continue
            try:
                response["message"] = json.loads(s)
                return response
            except Exception as e:
                last_err = e
                print(f"[generate_json] parse fail (attempt {attempt+1}/{parse_retries}) "
                      f"from {model}: {repr(s[:80])} -> {e}")
        raise last_err if last_err else ValueError("generate_json exhausted retries")


model = OpenAI_Model()
generate = model.generate
generate_json = model.generate_json





if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke about UC Berkeley."},
        {"role": "assistant", "content": '{"joke": '}
    ]

    model = "gpt-4o"
    response = generate(messages, model=model, return_metadata=True)
    
    print(response)

    # batch_id = client.generate_w_delay(messages, model="gpt-4o-mini")
    # print(batch_id)
