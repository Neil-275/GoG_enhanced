from collections import defaultdict
from copy import deepcopy
from logging import Formatter
import random
import re
# import spacy
from string import Formatter
import traceback
import asyncio
from loguru import logger
from GoG.kg_interface import KGInterface
from GoG.utils import (
    convert_list_to_str,
    format_prompt,
    parse_llm_output_to_list,
    read_file,
    shorten_relation,
    convert_triples_to_str,
    extract_numbers_from_string,
    parse_generated_relation_directions
)
from GoG.GoG_llms import run_llm
from GoG.gnn_interface import OneShotInterface
import pandas as pd
import pickle as pkl
import os
import sys
# from rank_bm25 import BM25Okapi

# Note: Some legacy methods are not available in the new KGInterface
# They will be handled gracefully or passed
# try:
#     from bm25_name2ids import retrieve_id2types_by_name
# except ImportError:
#     logger.warning("bm25_name2ids not available, some methods will be skipped")
#     def retrieve_id2types_by_name(entity_name):
#         logger.warning(f"retrieve_id2types_by_name not available for {entity_name}")
#         return {}


logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
)


class KGEnv:
    def __init__(self, args) -> None:
        self.args = args
        # with open("sample_args_family.pkl", "wb") as f:
        #     pkl.dump(self.args, f)
        self.dataset_name = args.dataset.split("/")[1]

        # Initialize KGInterface

        self.kg: KGInterface = KGInterface(self.dataset_name)
        ## NBFNet ##
        # self.gnn: GNNInterface = GNNInterface(self.dataset_name)
        # self.gnn.assign_graph(self.kg.pyg_data)
        ## One-shot subgraph 
        self.gnn: OneShotInterface = OneShotInterface(self.dataset_name, self.kg.n_ent, self.kg.n_rel)
        self.gnn.assign_graph(self.kg)
        print(f"Dataset Name: {self.dataset_name}")
        logger.info(f"Initialized KGInterface with dataset: {self.dataset_name}")

        self.records = []

        self.triples = []
        self.abbr_rel_to_rel = {}

        self.id_to_name = {}
        self.name_to_id = {}
        # use self.kg.entities to update name_to_id and id_to_name
        if self.kg:
            for entity_id in self.kg.entities:
                self.name_to_id[entity_id.lower()] = entity_id
                self.id_to_name[entity_id] = entity_id.lower()


        # self.update_id_to_name(topic_entities)

        # self.doc_to_vec = spacy.load("en_core_web_lg")

        self.explored_entities = set()

        self.mid_crucial_triples = None
        self.n_related_triples = self.args.n_related_triples
        # only used in answer without kg
        self.llm_output = None
        self.generate_call_count = 0
        self.topic_entities = None
        self.question = None

    # def update_name_to_id(self, name_to_id):
    #     name_to_id = {name.lower(): id for name, id in name_to_id.items()}
    #     self.name_to_id.update(name_to_id)
    #     self.id_to_name.update({id: label for label, id in name_to_id.items()})

    # def update_id_to_name(self, id_to_name):
    #     # id_to_name = {id: name.lower() for id, name in id_to_name.items()}
    #     # self.id_to_name.update(id_to_name)
    #     # self.name_to_id.update({label: id for id, label in id_to_name.items()})
    #     id_to_name = {id:id for id in id_to_name}
    #     self.id_to_name.update(id_to_name)
    #     self.name_to_id.update({label: id for id, label in id_to_name.items()})

    def assign_query(self, topic_entities, question):
        self.topic_entities = topic_entities
        self.question = question
        self.records = []

        self.triples = []
        self.abbr_rel_to_rel = {}
        self.explored_entities = set()

        self.mid_crucial_triples = None
        self.n_related_triples = self.args.n_related_triples
        # only used in answer without kg
        self.llm_output = None
        self.generate_call_count = 0


    def convert_records_to_str(self):
        string = ""
        for record in self.records:
            string += "Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {observation}\n".format(
                i=record["i"],
                action=record["action"],
                thought=record["thought"],
                observation=record["observation"],
            )
        return string

    def synthesize_answer(self, prompt):
        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/answer_synthesis")
        synthesis_prompt = format_prompt(prompt_path)

        records_str = self.convert_records_to_str()
        prompt = (
            synthesis_prompt
            + "\n"
            + records_str
            + "\nAnswer"
        )
        # print("Synthesis prompt:", prompt)
        output = run_llm(
            prompt,
            self.args.temperature,
            512,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
        )
        # print("LLM output:", output)
        self.llm_output = output

        if not output:
            return ["unknown"]

        match = re.search(r"Answers(\[.*\])", output)
        if match:
            answers = parse_llm_output_to_list(match.group(1))
        elif "[" in output:
            answers = parse_llm_output_to_list(output[output.index("["):])
        else:
            answers = None

        if not answers:
            return ["unknown"]

        answers = [answer for answer in answers if answer]
        if not answers:
            return ["unknown"]

        return answers

    @property
    def last_thought(self):
        return self.records[-1]["thought"]

    @property
    def last_action(self):
        return self.records[-1]["action"]

    def step(self, action_str=None):
        logger.debug(action_str)

        pattern = r"(\w+)(\[.+\])"
        result = re.match(pattern, action_str)

        logger.debug(f'Result: {result}')
        action = result.group(1).lower()
        parameter = result.group(2)

        logger.info(f"Action: {action}, Parameter: {parameter}")

        if action == "search":
            if parameter == "[ALL]":
                # repeat last search, but search two-hop
                self.records[-1]["thought"] = self.records[-2]["thought"]
                entity_str = convert_list_to_str(self.records[-2]["new_entities"])
                return self.search(entity_str)
            else:
                return self.search(parameter)
        elif action == "generate":
            self.generate_call_count += 1
            return self.generate(parameter)

    def generate(self, thought):
        # [...]
        if thought.startswith("["):
            thought = thought[1:-1]

        entities = extract_numbers_from_string(thought)
        related_triples_df = [self.kg.get_1hop_triples(str(entity_id)) 
                           for entity_id in entities if str(entity_id) in self.kg.entities]
        related_triples = [triple for df in related_triples_df for triple in df.values.tolist()]
        related_triples = random.sample(related_triples, k=min(len(related_triples), 3))
        related_triple_str = convert_triples_to_str(related_triples)
        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/generate_triples")
        prompt = format_prompt(prompt_path)

        candidate_relations = self.kg.get_best_relation_match(thought, k=5) if self.kg.rels else []
        candidate_relations_str = "\n".join(candidate_relations)
    
        n = self.args.sc_num
        prompt = (
            prompt.format(
                thought=thought,
                # existing_triples=related_triple_str,
                candidate_relations=candidate_relations_str,
            )
            + "\nAnswer: "
        )

        logger.debug(f"Generate prompt:{prompt}")
        # print("Generate prompt:", prompt)
        responses = run_llm(
            prompt,
            self.args.temperature,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
            n=n
        )
        # print("LLM responses:")
        # print(responses)

        parsed_generations = parse_generated_relation_directions(responses)

        result = []
        for start_entity, (relation_text, direction) in parsed_generations.items():
            relation, _ = self.kg.get_best_relation_match(relation_text)
            if not relation:
                continue

            print(f"Found: {start_entity}: {relation}: {direction}")

            candidates = self.gnn.predict_topk(start_entity, relation, direction, k=3, known=False)

            relation_paths = {}
            for candidate in candidates:
                if direction == 0:
                    relation_path = self.kg.get_shortest_path_with_relations(str(start_entity), str(candidate))
                else:
                    relation_path = self.kg.get_shortest_path_with_relations(str(candidate), str(start_entity))
                if relation_path == None:
                    continue
                relation_path_str = ""
                for i, ent, rel in zip(range(len(relation_path["relations"])), relation_path["path"][1:], relation_path["relations"]):
                    relation_path_str += f"[{shorten_relation(rel)}]-"
                if direction == 0:
                    relation_path_str = f"{candidate}: {start_entity} -{relation_path_str}> {candidate}"
                else:
                    relation_path_str = f"{candidate}: {candidate} -{relation_path_str}> {start_entity}"
                print("relation_path_str:", relation_path_str)
                relation_paths[candidate] = relation_path_str
            verified_candidates = self.verify(start_entity, shorten_relation(relation), relation_paths, direction)
            for candidate in verified_candidates:
                if direction == 0:
                    triple = [str(start_entity), str(relation), str(candidate)]
                else:
                    triple = [str(candidate), str(relation), str(start_entity)]
                triple_str = convert_triples_to_str([triple])
                # print("Generated triple:", triple_str)
                self.records[-1]["generated_triples"] = triple_str
                result.append(triple_str)
                
        if len(result) == 0:
            # print("No valid triples generated.")
            return "No valid triples generated."
        print("Generated triples:")
        print("\n".join(result))
        return "\n".join(result)

    def get_template_variables(self, template_string):
    # Field names can be None for raw text chunks, so filter those out
        return [field_name for _, field_name, _, _ in Formatter().parse(template_string) if field_name is not None]

    def verify(self, start_entity, relation, relation_paths, direction):
        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/verify_triples")
        prompt = format_prompt(prompt_path)
        # print("type of prompt:", type(prompt))
        # print("Format str:", self.get_template_variables(prompt))
        if direction == 0:
            prompt = prompt.format(subject=start_entity, relation=relation, object="?")
        else:
            prompt = prompt.format(subject="?", relation=relation, object=start_entity)
        relation_path_str = [f"{relation_path}" for candidate, relation_path in relation_paths.items()]
        relation_path_str = "\n".join(relation_path_str)
        
        prompt = prompt + "Candidates:\n" + relation_path_str + "\nDeduction space: "
        # print(123)
        # print("Verify prompt:", prompt)
        # verified_triples = []
        response = run_llm(
            prompt,
            self.args.temperature,
            # self.args.max_length,
            768,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
        )
        # print("Verify response:", response)
        candidate = extract_numbers_from_string(response, last_line=True)
        self.records[-1]['verified_candidates'] = candidate
        return candidate
                # for line in response.split("\n"):
        #     try:
        #         h, r, t = [item.strip() for item in line.split('\t')]
        #         verified_triples.append([h, r, t])
        #     except Exception as e:
        #         logger.error(traceback.format_exc())
        #         logger.error(line)

        # self.records[-1]['verified_triples'] = verified_triples
        # return verified_triples

    def filter_crucial_triples(self, triples):
        filtered_triples = [triple for triple in triples if triple not in self.mid_crucial_triples]
        relations = list(set([triple[1] for triple in filtered_triples]))

        return filtered_triples, relations

    def search(self, entity_names):
        # print(f"Search entity names: {entity_names}", type(entity_names))
        entity_names = parse_llm_output_to_list(entity_names)

        all_related_triples = []
        for entity_name in entity_names:
            entity_id = self.convert_name_to_id(entity_name)
            # Use KGInterface to get 1-hop triples
            if self.kg:
                try:
                    # outgoing, incoming = self.kg.get_1hop_triples(entity_id)
                    # # Convert DataFrames to list format: [head, relation, tail]
                    # incoming_triples = incoming.values.tolist() if len(incoming) > 0 else []
                    # outgoing_triples = outgoing.values.tolist() if len(outgoing) > 0 else []
                    # incoming_triples = [(1, triple) for triple in incoming_triples]  # Mark incoming triples with 1
                    # outgoing_triples = [(0, triple) for triple in outgoing_triples]  # Mark outgoing triples with 0
                    # triples = incoming_triples + outgoing_triples
                    # relations = list(set([triple[1][1] for triple in triples]))
                    triples_df = self.kg.get_1hop_triples(entity_id)
                    # Convert DataFrame to list format: [head, relation, tail]
                    triples = triples_df.values.tolist() if len(triples_df) > 0 else []
                    relations = list(set([triple[1] for triple in triples]))
                    # logger.debug(f"Relations after get_1hop_triples: {relations}")
                except Exception as e:
                    logger.error(f"Failed to get 1-hop triples for {entity_id}: {e}")
                    triples = []
                    relations = []
            else:
                triples = []
                relations = []
            
            # logger.debug(f"Relations after filter_crucial_triples: {relations}")

            for i in range(len(relations)):
                # only remain the last two parts
                abbr_rel = shorten_relation(relations[i])
                self.abbr_rel_to_rel[abbr_rel] = relations[i]
                relations[i] = abbr_rel

            for i in range(len(triples)):
                if len(triples[i]) == 2:
                    abbr_rel = shorten_relation(triples[i][1][1])
                    self.abbr_rel_to_rel[abbr_rel] = triples[i][1][1]
                    triples[i][1][1] = abbr_rel
                elif len(triples[i]) == 3:
                    abbr_rel = shorten_relation(triples[i][1])
                    self.abbr_rel_to_rel[abbr_rel] = triples[i][1]
                    triples[i][1] = abbr_rel

            relations = sorted(relations)
            # logger.debug(f"Relations after abbreviation and sorting: {relations}")
            # print(f"Relations after abbreviation and sorting: {relations}")
            ## Call LLM to filter relations for each entity
            filtered_relations = self.filter_relations(entity_name, relations, self.last_thought)
            # logger.debug(f"Relations after filter_relations (LLM filtered): {filtered_relations}")
            # print(f"Relations after filter_relations (LLM filtered): {filtered_relations}")

            related_triples = self.sample_triples_by_relation(triples, filtered_relations)
            
            # Extract relations from sampled triples
            # sampled_relations = sorted(list(set([triple[1] for triple in related_triples])))

            all_related_triples.extend(related_triples)

        all_related_triples = sorted(all_related_triples)
        # print("All related triples:")
        # for triple in all_related_triples:
        #     print(f"  {triple}")

        self.triples.extend(deepcopy(all_related_triples))

        self.records[-1]["triples"] = all_related_triples
        self.records[-1]["entity_names"] = entity_names
        self.records[-1]["one_hop_relations"] = filtered_relations

        new_entities = set()
        for triple in all_related_triples:
            if len(triple) == 2:
                if triple[1][0].lower() in self.name_to_id:
                    new_entities.add(triple[1][0])
                if triple[1][2].lower() in self.name_to_id:
                    new_entities.add(triple[1][2])
            elif len(triple) == 3:
                if triple[0].lower() in self.name_to_id:
                    new_entities.add(triple[0])
                if triple[2].lower() in self.name_to_id:
                    new_entities.add(triple[2])
        new_entities -= self.explored_entities
        self.records[-1]["new_entities"] = list(new_entities)

        self.explored_entities.update(new_entities)

        return convert_triples_to_str(all_related_triples)

    def filter_relations(self, entity_name, relations, thought):
        # logger.debug(f"{thought}\n{entity_name}")

        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/filter_relations")
        prompt = format_prompt(prompt_path)
        random.shuffle(relations)
        # relations = sorted(relations)
        # logger.debug(f"original relations {relations}")

        prompt = (
            prompt + f"Thought: {thought}\n"
            f"Entity: {entity_name}\n"
            f"Relation: [{', '.join(relations)}]\n"
            f"Answer: "
        )
        # print(f"Prompt: {prompt}")
        filtered_relations = run_llm(
            prompt,
            self.args.temperature,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
        )

        # filtered_relations = [rel.strip() for rel in filtered_relations.split(",")]
        # TODO: could generate relations not appeared in the relation list
        filtered_relations = parse_llm_output_to_list(filtered_relations, sep=',')

        return filtered_relations

    def select_entity_id_by_types(self, question, entity_name, id_to_types):
        # Note: This method uses id_to_types which requires retrieve_id2types_by_name
        # That function is not available in the new KGInterface
        # For now, we'll return the first available entity ID or pass
        
        if not id_to_types:
            logger.warning(f"No types available for entity {entity_name}, cannot select")
            return entity_name
        
        try:
            prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/select_entity")
            prompt = format_prompt(prompt_path)

            id_to_types = sorted(id_to_types.items(), key=lambda x: len(x[-1]), reverse=True)
            id_to_types = dict(id_to_types[:10])
            candidite_entities = [f"{k}: {', '.join(v)}" for k, v in id_to_types.items()]
            candidite_entities = "\n".join(candidite_entities)

            prompt = (
                prompt + f"Question: {question}\n"
                f"Entity Name: {entity_name}\n"
                f"Candidate Entities:\n{candidite_entities}\n"
                f"Answer: "
            )

            entity_id = run_llm(
                prompt,
                self.args.temperature,
                self.args.max_length,
                self.args.opeani_api_keys,
                self.args.LLM_type,
                stop="\n",
            )

            return entity_id
        except Exception as e:
            logger.error(f"Failed to select entity ID: {e}. Returning entity_name as fallback")
            return entity_name

    def sample_triples_by_relation(self, triples, filtered_relations):
        # only remain related triples
        relation_to_triples = defaultdict(list)
        for triple in triples:
            if len(triple) == 2:
                relation = triple[1][1]
                if relation in filtered_relations:
                    relation_to_triples[relation].append(triple)
            elif len(triple) == 3:
                relation = triple[1]
                if relation in filtered_relations:
                    relation_to_triples[relation].append(triple)

        related_triples = []
        for rel, triples in relation_to_triples.items():
            if len(triples) >= 5:
                relation_to_triples[rel] = random.sample(triples, k=5)
            related_triples.extend(relation_to_triples[rel])
        return related_triples

    def convert_name_to_id(self, entity_name):
        if entity_name.lower() in self.name_to_id:
            return self.name_to_id[entity_name.lower()]
        else:
            logger.warning(f"Entity name {entity_name} not found in name_to_id mapping, returning original name")
            return entity_name
    # def expand(self):
    #     # from one-hop to two hop
    #     entity_names = self.records[-2]["entity_names"]
    #     one_hop_relations = self.records[-2]["one_hop_relations"]

    #     all_triples, all_relations = [], []

    #     for entity_name in entity_names:
    #         id = self.convert_name_to_id(entity_name)
    #         triples, relations = get_2hop_triples(
    #             id, [self.abbr_rel_to_rel[rel] for rel in one_hop_relations]
    #         )

    #         all_triples.extend(triples)
    #         all_relations.extend(relations)

    #     for i in range(len(all_relations)):
    #         # only remain the last two parts
    #         abbr_rel = shorten_relation(all_relations[i])
    #         self.abbr_rel_to_rel[abbr_rel] = all_relations[i]
    #         all_relations[i] = abbr_rel

    #     # drop relations that have been considered
    #     relations = list(set(all_relations) - set(one_hop_relations))
    #     relations = sorted(relations)

    #     two_hop_relations = self.filter_relations(entity_names, relations, self.last_thought)
    #     self.records[-1]["two_hop_relations"] = two_hop_relations

    #     for i in range(len(all_triples)):
    #         abbr_rel = shorten_relation(all_triples[i][1])
    #         self.abbr_rel_to_rel[abbr_rel] = all_triples[i][1]
    #         all_triples[i][1] = abbr_rel

    #     related_triples = self.sample_triples_by_relation(
    #         all_triples, one_hop_relations + two_hop_relations
    #     )

    #     related_triples, id_to_label = convert_id_to_name_in_triples(
    #         related_triples, return_map=True
    #     )
    #     self.update_id_to_name(id_to_label)

    #     related_triples = sorted(related_triples)
    #     self.records[-1]["triples"] = related_triples

    #     return convert_triples_to_str(related_triples)


if __name__ == "__main__":
    id2types = retrieve_id2types_by_name(
        "Libya",
    )
    print(id2types)
