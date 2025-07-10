from human_eval.data import write_jsonl, read_problems
from human_eval.api import get_model_response
from argparse import ArgumentParser
import numpy as np
import time
def args():
    parser = ArgumentParser(description="Large model response acquisition")
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
    parser.add_argument("--temperature", default=0.8, type=float,
                        help="Temperature parameter(Must be a floating-point number)",
                        choices=np.arange(0, 2.1, 0.1))
    parser.add_argument("--k", default=1, type=int, help="The number of answers returned",
                        choices=list(range(0, 7)))
    return parser.parse_args()
args = args()
problems = read_problems()
num_samples_per_task = 200
task_ids = list(problems.keys())
time_list = []
for task_id in task_ids:
    for _ in range(num_samples_per_task):
        start_time = time.time()
        samples = []
        samples = [
            dict(task_id=task_id, completion=get_model_response(problems[task_id]["prompt"], args.model, args.k, args.temperature))
        ]
        end_time = time.time()
        time_list.append(end_time - start_time)
        write_jsonl("samples_{}.jsonl".format(args.model), samples, append=True)
        time.sleep(1)






