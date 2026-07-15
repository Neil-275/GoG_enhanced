import argparse
from ast import literal_eval
from dotenv import load_dotenv
import multiprocessing
from multiprocessing import Pool
import os
from pathlib import Path
import random
import re
import time
import json
import pandas as pd
from threading import Lock
from loguru import logger
import sys

from GoG.GoG_env_tools import KGEnv
from GoG.GoG_llms import run_llm
from GoG.utils import (
    format_prompt,
    parse_llm_output_to_list,
    read_file
)
from postprocess_predictions import postprocess_prediction_jsonl


load_dotenv()

multiprocessing.set_start_method('spawn', force=True)
lock = Lock()


def answer_question_without_kg(env: KGEnv, prompt, args):
    answers = env.synthesize_answer(prompt)
    logger.info(answers)
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


def find_answer(process_idx, idxes_to_process, args, datas, env: KGEnv):
    logger.debug(f"{process_idx}, {idxes_to_process[0]}")

    # Skip due to Wikidata-5m skipped
    # if args.wiki:
    #     instruction = format_prompt(read_file("prompts2/instruction_wiki"))
    #     example = format_prompt(read_file("prompts2/examples_wiki"))
    # else:
    #     instruction = format_prompt(read_file("prompts2/instruction"))
    #     instruction = format_prompt(read_file("prompts2/instruction"))

    # First of all, determine the base prompt path matched with the current arguments
    example_path: Path = Path(args.prompt_dir)
    if args.no_kg:
        example_path /= "examples_no-kg"
    elif args.ablate_collect:
        example_path /= "examples_no-collect"
    elif args.ablate:
        example_path /= "ablate_examples.txt"
    else:
        example_path /= "examples"

    if not example_path:
        raise ValueError("Cannot determine prompt path with current arguments.")

    if not example_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {example_path}")

    example = format_prompt(read_file(str(example_path)))

    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        if (n + 1) % 10 == 0:
            t2 = time.time()
            logger.debug(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        try:
            logger.info("-----------")
            logger.info(f"Process query {idx} ...")
            data = datas[idx]
            if "question" not in data:
                data["question"] = data["ProcessedQuestion"]

            topic_entity_names = sorted(data["q_entity"])
            topic_entity_names_str = "[" + " | ".join(topic_entity_names) + "]"

            base_prompt = (
                example
                + f'Question: {data["question"]}\nTopic Entity: {topic_entity_names_str}\n'
            )

            logger.info(f"Question: {data['question']}")

            env.assign_query(data)

            n_calls, n_badcalls, n_expand = 0, 0, 0
            done = False
            prompt = base_prompt

            if args.no_kg:
                prediction = answer_question_without_kg(env, base_prompt, args)
                write_results(data, env, prediction, args)
                continue
            prediction_pool = []
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
                is_collect = 0
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
                        if 'Collect' in thought_action:
                            thought = action = thought_action
                            legal = True
                        elif 'Finish' in thought_action:
                            done = True
                            legal = True
                        else:
                            logger.debug(f"ohh... {thought_action}")
                            continue

                logger.debug(f"Thought {i}: {thought}")
                logger.debug(f"Action {i}: {action}")

                if not args.ablate_collect:
                    obs = None
                    match = re.search(r"Collect(?:ed)?(\[.*\])", action)

                    if match:
                        logger.debug("Match  ", match)
                        prediction = match.group(1)
                        prediction_pool.extend(parse_llm_output_to_list(prediction))
                        obs = f"Collected the answers: {prediction}"
                        logger.info(f"Collected the answers: {prediction}")
                        is_collect = 1

                finish_match = re.search("Finish", action)
                if finish_match:
                    prediction_match = re.search(r"Finish(\[.*\])", action)
                    if prediction_match:
                        prediction = prediction_match.group(1)
                        parsed_prediction = parse_llm_output_to_list(prediction)
                        if parsed_prediction and parsed_prediction != ["unknown"]:
                            prediction_pool.extend(parsed_prediction)
                        obs = f"Finished with answers: {prediction}"
                    else:
                        obs = "Finished the search."
                    done = True

                env.records.append({"i": i, "thought": thought, "action": action})
                if done:
                    env.records[-1]["observation"] = obs
                    break
                if args.ablate_collect and action.lower().startswith("collect"):
                    obs = "Collect disabled."
                elif not is_collect:
                    obs = env.step(action, collect_enabled=not args.ablate_collect)
                obs = obs.replace("\\n", "")
                env.records[-1]["observation"] = obs

                logger.debug(obs)

                records_str = env.convert_records_to_str()
                prompt = base_prompt + records_str
                logger.debug(records_str)

            if prediction_pool:
                prediction_pool = list(set(prediction_pool))
                logger.info(f"Finish query {idx} with KG, prediction: {prediction_pool} ...")
                write_results(data, env, prediction_pool, args)
            else:
                logger.warning(f"Finish query {idx} without KG...")
                prediction = answer_question_without_kg(env, prompt, args)
                write_results(data, env, prediction, args)

            logger.info(f"ground truth: {data['answer']}")
        except Exception as e:
            write_results(data, env, None, args, error=str(e))
            logger.exception(f"Error processing query {idx} and leave it missing: {e}")

    logger.info(f"{process_idx} finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="brink_dataset/family/test_is_collect.tsv",
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
    parser.add_argument("--run_fail_case", default=None)
    parser.add_argument("--hard_only", action="store_true", help="prune all ez answer in each query")
    parser.add_argument("--ablate", action="store_true", help="ablate the generation module")
    parser.add_argument("--ablate_collect", action="store_true", help="ablate the collect module")
    parser.add_argument("--ver", default=None, type=str, help="version identifier for the experiment")
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
            if isinstance(data[k], str) and data[k].startswith("[") and data[k].endswith("]"):
                data[k] = literal_eval(data[k])

    postfix = '_no-kb' if args.no_kg else ""
    dataset_name = args.dataset.split("/")[1]

    # if args.test:
    #     output_file = Path(f"./{args.output_dir}/{args.LLM_type.split('/')[-1]}/{dataset_name}/test_predictions.jsonl")
    # else:
    if args.ver is not None:
        output_prefix: Path = Path(f"./{args.output_dir}/{args.LLM_type.split('/')[-1]}/{dataset_name}")
        output_prefix.mkdir(parents=True, exist_ok=True)
        if args.ablate:
            if args.ablate_collect:
                output_file = output_prefix / f"{args.ver}_no-collect_no-gnn.jsonl"
            else:
                output_file = output_prefix / f"{args.ver}_ablate_2_predictions.jsonl"
        else:
            if args.ablate_collect:
                output_file = output_prefix / f"{args.ver}_no-collect_predictions.jsonl"
            else:
                output_file = output_prefix / f"{args.ver}_predictions.jsonl"
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

    if args.test:
        # In test mode, always overwrite the file
        with open(output_file, "w") as f:
            pass
    elif os.path.exists(output_file) and not args.force:
        with open(output_file, "r") as f:
            output_datas = [json.loads(line) for line in f.readlines()]
        processed_idxes = set([data["index"] for data in output_datas])

        datas = [data for data in datas if data['id'] not in processed_idxes]

        logger.debug(len(datas))
    else:
        with open(output_file, "w") as f:
            pass

    seed = 42
    random.seed(seed)
    random.shuffle(datas)
    if args.test:
        val_id = [1053, 2745, 5144, 3717, 869, 4464, 1589, 2767]
        # datas = random.sample(datas, min(1, len(datas)))  # Randomly sample 3 cases for testing
        datas = [data for data in datas if data['id'] in val_id]
        # datas = datas[35:23]  # Limit to first 3 cases for testing
    if args.run_fail_case:
        failed_cases = []

        with open(args.run_fail_case, "r") as f:
            failed_cases = json.load(f)
        failed_id = set([case["index"] for case in failed_cases])
        datas = [data for data in datas if data['id'] in failed_id]
        logger.info(f"Running on {len(datas)} failed cases...")
    print(f"Number of datas to process: {len(datas)}")
    idxes_to_process = range(len(datas))

    num_samples = len(idxes_to_process)
    logger.debug(num_samples)

    n_process = min(args.n_process, num_samples)
    logger.debug(n_process)

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

    try:
        postprocess_outputs = postprocess_prediction_jsonl(output_file)
        logger.info(f"Post-processed predictions: {postprocess_outputs}")
    except Exception as e:
        logger.error(f"Error post-processing predictions: {e}")
