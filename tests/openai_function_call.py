import openai
from openai import OpenAI
import openai.types

with open(".env", "r") as file:
    # It's like OPENAI_API_KEY=...\n
    # So we need to split by = and take the second element
    keys = {}
    for line in file:
        key, value = line.strip().split("=")
        keys[key] = value

client = OpenAI(
    api_key=keys["OPENAI_API_KEY"],
)


from pydantic import BaseModel

class TestFunctionCall(BaseModel):
    name: str
    description: str


def test_function_call():
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What is the capital of the moon?"}
        ],
        tools=[
            openai.pydantic_function_tool(TestFunctionCall)
        ],
        tool_choice="required",
        stream=True,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.tool_calls)


if __name__ == "__main__":
    test_function_call()
