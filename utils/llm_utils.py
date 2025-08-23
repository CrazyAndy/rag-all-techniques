from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm_client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)


def query_llm(system_prompt, user_prompt):
    model_name = os.getenv("DEFAULT_MODEL")

    completion = llm_client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        # stream=True,  # 启用流式输出
        extra_body={"enable_thinking": False},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.choices[0].message.content
