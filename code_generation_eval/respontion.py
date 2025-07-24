from data import write_jsonl, read_problems
from api import get_model_response, parse_args
import numpy as np
import time
import os
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

"""def args():
    parser = argparse.ArgumentParser(description="Large model response acquisition")
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
    parser.add_argument("--system_prompt", type=str, default=(
    "You are a code assistant.\n"
    "In every response, you are given a Python function header (signature).\n"
    "You may respond however you like, including explanations, comments, or other content.\n"
    "However, you must include a valid and executable function body **after** the marker:\n"
    "[code]\n"
    "The code after [code] must:\n"
    "- Be a properly indented function body (do not include the function header);\n"
    "- Be syntactically valid Python code;\n"
    "- Be semantically consistent with the function header provided;\n"
    "- Be executable when directly appended to the function header.\n"
    "This requirement is mandatory and will be evaluated strictly."), help="System prompt")
    return parser.parse_args()"""
args = parse_args()
problems = read_problems()
num_samples_per_task = 20
task_ids = list(problems.keys())

for task_id in task_ids[1:]:
    samples_batch = []
    for _ in range(num_samples_per_task):
        start_time = time.time()
        completion = get_model_response(input=problems[task_id]["prompt"], system_prompt=args.system_prompt, model=args.model, k=args.k, temperature=args.temperature)
        end_time = time.time()
        samples_batch.append(
            dict(task_id=task_id, completion=completion, time=end_time - start_time)
        )
        print(len(samples_batch))
        if len(samples_batch) == 40:
            write_jsonl("samples_{}.jsonl".format(args.model), samples_batch, append=True)
            samples_batch = []
        else:
            continue
# If your model encounters a disconnection during inference that results in an error, 
# you can use this module to independently control and return responses for a specific task, achieving precise control.
"""samples = []
for _ in range(num_samples_per_task):
    start_time = time.time()
    copletion = get_model_response(input=problems["HumanEval/0"]["prompt"], system_prompt=args.system_prompt, model=args.model, k=args.k, temperature=args.temperature)
    end_time = time.time()
    samples.append(dict(task_id="HumanEval/0", completion=copletion, time=end_time - start_time))
    print(len(samples))
    if len(samples) == 10:
        write_jsonl("samples_{}.jsonl".format(args.model), samples, append=True)
        samples = []
    else:
        continue"""





