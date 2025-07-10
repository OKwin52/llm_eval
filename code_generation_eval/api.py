import argparse
from openai import OpenAI
import numpy as np
import os
import re
from dotenv import load_dotenv
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
def get_model_response(input, model, k=1, temperature=0.8):
    load_dotenv()
    if "gpt" in model:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": input
            }],
            n=k,
            temperature=temperature
        )
        return completion.choices[0].message.content
    elif "o1" in model or "o3" in model:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": input
            }],
        )
        return completion.choices[0].message.content
    elif "DeepSeek" in model or "Doubao" in model:
        if "v3" in model:
            model = os.getenv("DeepSeek-v3")
        elif "r1" in model:
            model = os.getenv("DeepSeek-r1")
        elif "pro" in model:
            model = os.getenv("Doubao-1.5-pro")
        else:
            model = os.getenv("Doubao-1.5-lite")
        client = OpenAI(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key
            api_key=os.getenv("ARK_API_KEY"),
        )
        completion = client.chat.completions.create(
            model=model,  # your model endpoint ID
            messages=[{
                "role": "user",
                "content": input
            }],
            temperature=temperature,
            n=k
        )
        return completion.choices[0].message.content
    elif "ernie" in model:
        client = OpenAI(
            base_url='https://qianfan.baidubce.com/v2',
            api_key=os.getenv("ERNIE_API_KEY")
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": input
            }],
            n=k,
            temperature=temperature
        )
        return completion.choices[0].message.content
def extract_boxed_answer(text):
    """
    从文本中提取最后一个 \boxed{} 内的数值
    支持整数、小数、负数（假设模型严格遵循格式）
    """
    # 匹配所有 \boxed{} 模式
    matches = re.findall(r"\\boxed{(-?\d+\.?\d*)}", text)

    if matches:
        try:
            # 取最后一个答案（修正链式思考中可能的中间步骤）
            last_answer = matches[-1]
            # 处理可能存在的逗号（如 12,345 → 12345）
            cleaned = last_answer.replace(",", "")
            return float(cleaned)
        except:
            return None
    return None
def parse_args():
    parser = argparse.ArgumentParser(description="Large model response acquisition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="DeepSeek-v3", help="Select your model",
                        choices=["gpt-3.5-turbo-0125", "gpt-4-0613", "gpt-4-0314", "gpt-4o-2024-11-20",
                                 "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09",
                                 "gpt-4-0125-preview", "gpt-4-0125-preview", "gpt-4-1106-vision-preview",
                                 "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct", "o1-mini-2024-09-12",
                                 "o1-2024-12-17", "o1-preview-2024-09-12", "o1-mini-2024-09-12",
                                 "gpt-4o-mini-2024-07-18","DeepSeek-v3", "DeepSeek-r1", "Doubao-1.5-lite",
                                 "Doubao-1.5-pro", "ernie-x1-32k-preview", "ernie-4.5-8k-preview",
                                 "ernie-4.0-8k-latest", "ernie-4.0-turbo-8k-latest", "ernie-3.5-8k",
                                 "ernie-4.0-8k"])
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="Temperature parameter(Must be a floating-point number)",
                        choices=np.arange(0, 2.1, 0.1))
    parser.add_argument("--k", default=1, type=int, help="The number of answers returned",
                        choices=list(range(0, 7)))
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    print(args.model)
    question = "hello"

    print(get_model_response(question, args.model, args.k, args.temperature))

