from dotenv import load_dotenv
load_dotenv()

from ast import literal_eval
import multiprocessing
from multiprocessing import Pool
import os
from pathlib import Path
import random
import re
import time
from tqdm import tqdm
import json
import argparse
# from environment import FreeBaseEnv
from GoG.GoG_env_tools import KGEnv
import pandas as pd
# from FlagEmbedding import FlagReranker
# from sentence_transformers import SentenceTransformer

# from evaluate import eval_results
# from kb_interface.freebase_func import convert_name_to_id
from GoG.GoG_llms import run_llm
from threading import Lock
# from datasets import load_dataset
from GoG.utils import (
    convert_jsonl_to_json,
    format_prompt,
    parse_llm_output_to_list,
    read_file,
    proxy_load_dataset,
    convert_list_to_str,
)
from loguru import logger
import traceback
import sys
import subprocess


multiprocessing.set_start_method('spawn', force=True)

lock = Lock()


def answer_question_without_kg(env: KGEnv, prompt, args):
    answers = env.synthesize_answer(prompt)
    # logger.info(env.llm_output)
    return answers


def write_results(data, env: KGEnv, prediction, args, error: str = None):
    with lock:
        with open(args.output_file, "a") as f:
            res = {
                "index": data["id"],
                "question": data["question"],
                "prediction": prediction,
                "answers": data["answer"],
                "hard_answer": data['hard_answer'],
                # "ground_truth": list(set(data["answer"] + data['hard_answer'])),
                "generate_call_count": env.generate_call_count,
                "records": env.records,
                "error": error,
            }
            if env.llm_output:
                res["llm_output"] = env.llm_output

            f.write(json.dumps(res) + "\n")
        logger.info(
            f"Finished query {data['id']}: generate_call_count={env.generate_call_count}"
        )


def find_answer(process_idx, idxes_to_process, args, datas, env):
    logger.debug(f"{process_idx}, {idxes_to_process[0]}")

    # if args.wiki:
    #     instruction = format_prompt(read_file("prompts2/instruction_wiki"))
    #     example = format_prompt(read_file("prompts2/examples_wiki"))
    # else:
    #     instruction = format_prompt(read_file("prompts2/instruction"))
    #     instruction = format_prompt(read_file("prompts2/instruction"))

    # instruction = format_prompt(read_file("prompts2/instruction"))
    if args.no_kg:
        example = format_prompt(read_file(f"{args.prompt_dir}/examples_no-kg"))
    else:
        example = format_prompt(read_file(f"{args.prompt_dir}/examples"))

    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        # if n > 9: 
        #     break
        if (n + 1) % 10 == 0:
            t2 = time.time()
            logger.debug(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        try:
            logger.info("-----------")
            logger.info(f"Process query {idx} ...")
            data = datas[idx]
            # print("data: ", data)
            if "question" not in data:
                data["question"] = data["ProcessedQuestion"]

            topic_entity_names = sorted(data["q_entity"])
            topic_entity_names_str = "[" + " | ".join(topic_entity_names) + "]"

            base_prompt = (
                example
                + f'Question: {data["question"]}\nTopic Entity: {topic_entity_names_str}\n'
            )

            # print("base_prompt: ", base_prompt)
            # exit()
            logger.info(f"Question: {data['question']}")

            env.assign_query(data["q_entity"], data['question'])
            # env.mid_crucial_triples = data["mid_crucial_triples"]

            n_calls, n_badcalls, n_expand = 0, 0, 0
            done = False
            prompt = base_prompt

            if args.no_kg:
                prediction = answer_question_without_kg(env, base_prompt, args)
                write_results(data, env, prediction, args)
                continue

            for _ in range(6):
                i = len(env.records) + 1

                n_calls += 1
                thought_action = run_llm(
                    prompt + f"Thought {i}: ",
                    args.temperature,
                    args.max_length,
                    args.opeani_api_keys,
                    args.LLM_type,
                    f"\nObservation",
                )

                legal = False
                n_retry = 0
                while not legal:
                    n_retry += 1
                    if n_retry == 6:
                        thought, action = None, None
                        break

                    try:
                        thought_action = f"Thought {i}: " + thought_action
                        
                        thought_pattern = r'Thought \d+: (.+)'
                        action_pattern = r'Action \d+: (.+)'

                        thought_match = re.search(thought_pattern, thought_action)
                        action_match = re.search(action_pattern, thought_action)

                        thought = thought_match.group(1)
                        action = action_match.group(1)

                        legal = True
                    except:
                        # output final answers directly
                        if 'Finish' in thought_action:
                            thought = action = thought_action
                            legal = True
                        else:
                            logger.debug(f"ohh... {thought_action}")
                            continue

                logger.debug(f"Thought {i}: {thought}")
                logger.debug(f"Action {i}: {action}")

                match = re.search("Finish(\[.*\])", action)
                if match:
                    logger.debug("Match  ", match)
                    prediction = match.group(1)
                    prediction = parse_llm_output_to_list(prediction)
                    # print("prediction: ", prediction)
                    done = True
                    # if prediction[0].lower() == "unknown":
                    #     # Finish["unkown"]
                    #     # not enough information even after generation
                    #     # expand to 2-hop sub-graph
                    #     logger.debug("roll back and expand")
                    #     while "generate" in env.last_action.lower():
                    #         env.records.pop()
                    #     assert "search" in env.last_action.lower()

                    #     i = len(env.records) + 1

                    #     action = "Search[ALL]"
                    #     n_expand += 1
                    #     if n_expand >= args.max_n_expand:
                    #         logger.debug("max n_expand, break")
                    #         prediction = answer_question_without_kg(env, base_prompt, args)
                    #         done = True
                    # else:
                        # done = True

                env.records.append({"i": i, "thought": thought, "action": action})
                if done:
                    logger.info(
                        f"Finish query {idx} with KG, prediction: {prediction}"
                    )
                    write_results(data, env, prediction, args)
                    break

                obs = env.step(action)
                obs = obs.replace("\\n", "")
                env.records[-1]["observation"] = obs

                logger.debug(obs)

                records_str = env.convert_records_to_str()
                prompt = base_prompt + records_str
                logger.debug(records_str)

            if not done:
                logger.warning(
                    f"Finish query {idx} without KG..."
                )
                prediction = answer_question_without_kg(env, prompt, args)
                write_results(data, env, prediction, args)

            logger.info(f"ground truth: {data['answer']}")
        except Exception as e:
            # logger.error(f"{traceback.print_exc()}, trying get answer without kg")
            # prediction = answer_question_without_kg(env, base_prompt, args)
            # write_results(data, env, prediction, args)
            # stack_trace = traceback.format_exc()
            # print(stack_trace)
            write_results(data, env, None, args, error=str(e))
            logger.exception(f"Error processing query {idx} and leave it missing: {e}")

    logger.info(f"{process_idx} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="brink_dataset/family/test_passed.tsv",
        help="choose the dataset.",
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="the max length of LLMs output."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="the temperature in exploration stage.",
    )
    parser.add_argument("--width", type=int, default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int, default=3, help="choose the search depth of ToG.")
    parser.add_argument(
        "--remove_unnecessary_rel",
        type=bool,
        default=True,
        help="whether removing unnecessary relations.",
    )
    parser.add_argument(
        "--LLM_type", type=str, default="gpt-4o-mini", help="base LLM model."
    )
    parser.add_argument(
        "--opeani_api_keys",
        type=str,
        default=None,
        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.",
    )
    parser.add_argument(
        "--num_retain_entity",
        type=int,
        default=5,
        help="Number of entities retained during entities search.",
    )
    parser.add_argument(
        "--prune_tools",
        type=str,
        default="llm",
        help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n_process", type=int, default=1)
    parser.add_argument("--no_kg", action="store_true")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_n_expand", default=3)
    parser.add_argument("--n_related_triples", type=int, default=10)
    parser.add_argument("--wiki", action="store_true")
    parser.add_argument("--wiki_num", default=3, type=int)
    parser.add_argument("--prompt_dir", default='GoG/prompts_v3', type=str)
    parser.add_argument("--sc_num", type=int, default=1,
                        help="choose the number of self-consistency check.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="save results to test_predictions.jsonl (overwrites on each run).")
    
    # parser.add_argument("start_idx", type=int, default=0, help="the start index of the dataset to process.")

    args = parser.parse_args()

    LOG_LEVEL = "DEBUG" if args.debug else "INFO"
    os.environ["LOG_LEVEL"] = LOG_LEVEL
    logger.remove()
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    )
    logger.debug(f"{args}")

    datas = pd.read_csv(args.dataset, sep="\t").to_dict(orient="records")
    for data in datas:
        for k in data.keys():
            if type(data[k]) == str and data[k].startswith("[") and data[k].endswith("]"):
                data[k] = literal_eval(data[k])
        # print(data)
                
    # exit()
    # print(datas[0])
    postfix = '_no-kb' if args.no_kg else ""
    dataset_name = args.dataset.split("/")[1]
    
    if args.test:
        output_file = Path(f"./{args.output_dir}/{args.LLM_type.split('/')[-1]}/{dataset_name}/test_predictions.jsonl")
    else:
        output_file = (
            Path(f"./{args.output_dir}/{args.LLM_type.split('/')[-1]}/{dataset_name}")
            / f"{args.sc_num}_{args.n_related_triples}_{args.max_n_expand}_{args.temperature}_{Path(args.dataset).stem + postfix}_predictions.jsonl"
        )

    args.output_file = output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        sink=output_file.parent / f"{args.temperature}_{Path(args.dataset).stem}.log",
        mode="w",
    )

    # print(output_file)

    if args.test:
        # In test mode, always overwrite the file
        with open(output_file, "w") as f:
            pass
    elif os.path.exists(output_file) and not args.force:
        with open(output_file, "r") as f:
            output_datas = [json.loads(line) for line in f.readlines()]
        processed_idxes = set([data["id"] for data in output_datas])

        datas = [data for data in datas if data['id'] not in processed_idxes]

        logger.debug(len(datas))
    else:
        with open(output_file, "w") as f:
            pass

    
    #REMOVE LATER
    # failed_cases = []
    # k = 0
    # f = json.load(open("failed_cases.json", "r"))
    # for q in f:
    #     failed_cases.append(q["index"])
    #     # k += 1
    #     # if k >= 10:  # Limit to first 10 failed cases
    #     #     break
    # datas = [data for data in datas if data['id'] in failed_cases]
    print(f"Number of datas to process: {len(datas)}")
    datas = datas[1:4]
    idxes_to_process = range(len(datas))



    num_samples = len(idxes_to_process)
    logger.debug(num_samples)

    n_process = min(args.n_process, num_samples)
    logger.debug(n_process)

    # args.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    # args.reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)
    ## Produce a sample_args file
    import pickle as pkl
    with open("sample_args_fb15k_237.pkl", "wb") as f:
        pkl.dump(args, f)

    env = KGEnv(args)

    if n_process > 1:
        with Pool(processes=n_process) as pool:
            num_samples_in_chunk = num_samples // n_process
            jobs = []
            st = 0
            for i in range(n_process):
                ed = st + num_samples_in_chunk
                ed = min(ed, num_samples)
                jobs.append([i, idxes_to_process[st:ed], args, datas, env])

                st = ed

            results = pool.starmap(find_answer, jobs)
    elif n_process == 1:
        
        find_answer(0, idxes_to_process, args, datas, env)

    json_filepath = convert_jsonl_to_json(output_file)
    # eval_results(json_filepath)
    
    # Automatically extract failed cases from the predictions
    logger.info(f"Extracting failed cases from {json_filepath}")
    
    if os.path.exists(json_filepath):
        try:
            # Get the absolute path to extract_fail_cases.py
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            extract_script = os.path.join(parent_dir, "extract_fail_cases.py")
            
            result = subprocess.run(
                [sys.executable, extract_script, "--prediction_path", str(json_filepath)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                logger.info(f"Extract output: {result.stdout}")
            
            if result.returncode != 0:
                logger.error(f"Failed to extract failed cases. Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
            else:
                logger.info("Successfully extracted failed cases")
        except Exception as e:
            logger.error(f"Error extracting failed cases: {e}")
    else:
        logger.warning(f"Predictions file not found: {json_filepath}")
    
    # Automatically run evaluation on the predictions
    logger.info(f"Running evaluation on {json_filepath}")
    
    if os.path.exists(json_filepath):
        try:
            # Generate evaluation output filename based on predictions filename
            json_path = Path(json_filepath)
            eval_filename = json_path.stem.replace("_predictions", "_evaluation") + ".json"
            eval_output_path = json_path.parent / eval_filename
            
            # Get the absolute path to evaluation.py
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            eval_script = os.path.join(parent_dir, "evaluation.py")
            
            result = subprocess.run(
                [sys.executable, eval_script, "--prediction_file", str(json_filepath), 
                 "--output_file", str(eval_output_path)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                logger.info(f"Evaluation output: {result.stdout}")
            
            if result.returncode != 0:
                logger.error(f"Failed to run evaluation. Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
            else:
                logger.info(f"Successfully completed evaluation. Results saved to {eval_output_path}")
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
    else:
        logger.warning(f"Predictions file not found: {json_filepath}")
