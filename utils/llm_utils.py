from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm_client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)


def query_llm(system_prompt, user_prompt, temperature=0.1, top_p=0.8, max_tokens=None):
    model_name = os.getenv("DEFAULT_MODEL")

    # 构建API调用参数，消除代码重复
    params = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "extra_body": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    # 只有当max_tokens不为None时才添加该参数
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    
    completion = llm_client.chat.completions.create(**params)
    
    return completion.choices[0].message.content


def query_llm_with_top_chunks(top_chunks, query):
    system_prompt = """
    你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。"""

    user_prompt = "\n".join(
        [f"上下文内容 {i + 1} :\n{result["text"]}\n========\n"
         for i, result in enumerate(top_chunks)])

    user_prompt = f"{user_prompt}\n\n Question: {query}"

    # 7. 调用LLM模型，生成回答
    return query_llm(system_prompt, user_prompt)


