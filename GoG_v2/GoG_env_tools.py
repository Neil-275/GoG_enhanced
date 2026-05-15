from collections import defaultdict
from copy import deepcopy
from itertools import islice
import random
import re
# import spacy
import traceback
import asyncio
from typing import Dict, List, Optional, Sequence

import networkx as nx
from loguru import logger
from GoG_v2.kg_interface import KGInterface
from GoG_v2.utils import (
    convert_list_to_str,
    format_prompt,
    parse_llm_output_to_list,
    read_file,
    shorten_relation,
    convert_triples_to_str,
    extract_numbers_from_string,
    parse_generated_relations
)
from GoG_v2.GoG_llms import run_llm
from GoG_v2.gnn_interface import OneShotInterface
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
        self.reasoning_workspace = {}
        self.answer_candidates = set()
        self.last_reasoning_signature = None

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
        self.reasoning_workspace = {}
        self.answer_candidates = set()
        self.last_reasoning_signature = None
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
        elif action == "path":
            return self.record_trace_action("Path", parameter)
        elif action == "rule":
            return self.record_trace_action("Rule", parameter)

        raise ValueError(f"Unsupported action: {action_str}")

    def record_trace_action(self, action_name: str, parameter: str) -> str:
        """Record a trace-only reasoning step without touching KG state."""
        observation = parameter.strip()
        if self.records:
            self.records[-1][f"{action_name.lower()}_summary"] = observation
        return observation

    def _parse_triple_lines(self, triple_blob: str) -> List[List[str]]:
        triples = []
        if not triple_blob:
            return triples
        for line in str(triple_blob).splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                triples.append(parts[:3])
        return triples

    def get_known_triples(self) -> List[List[str]]:
        """Collect the triples discovered in the current run."""
        seen = set()
        triples = []

        def _add(triple: Sequence[str]):
            if len(triple) < 3:
                return
            item = tuple(str(part).strip() for part in triple[:3])
            if item not in seen:
                seen.add(item)
                triples.append(list(item))

        for triple in self.triples:
            _add(triple)

        for record in self.records:
            for triple in record.get("triples", []) or []:
                _add(triple)
            for triple in self._parse_triple_lines(record.get("generated_triples", "")):
                _add(triple)

        return triples

    def _is_cvt_like(self, entity: str) -> bool:
        return entity.startswith("m.") or entity.startswith("g.")

    def _inverse_relation(self, relation: str) -> str:
        relation = relation.strip()
        if relation.startswith("inverse(") and relation.endswith(")"):
            return relation[len("inverse("):-1]
        return f"inverse({relation})"

    def _unpack_edge_relation(self, edge_data):
        if not edge_data:
            return None
        if isinstance(edge_data, dict):
            if "relation" in edge_data:
                return edge_data["relation"]
            first_key = sorted(edge_data.keys())[0]
            payload = edge_data[first_key]
            if isinstance(payload, dict):
                return payload.get("relation")
        return None

    def _build_reasoning_graphs(self, triples: List[List[str]]):
        forward_graph = nx.MultiDiGraph()
        reasoning_graph = nx.MultiDiGraph()

        for head, relation, tail in triples:
            head = str(head)
            relation = str(relation)
            tail = str(tail)
            forward_graph.add_edge(head, tail, relation=relation)
            reasoning_graph.add_edge(head, tail, relation=relation, direction="forward")
            reasoning_graph.add_edge(
                tail,
                head,
                relation=self._inverse_relation(relation),
                direction="inverse",
            )

        return forward_graph, reasoning_graph

    def _path_score(self, path: Dict) -> float:
        relations = path.get("relations", [])
        inverse_count = sum(1 for rel in relations if rel.startswith("inverse("))
        novelty_bonus = 1.5 if path.get("target") not in self.explored_entities else 0.0
        answer_bonus = 1.0 if path.get("target") in self.answer_candidates else 0.0
        return 100.0 - (len(relations) * 10.0) - (inverse_count * 2.0) + novelty_bonus + answer_bonus

    def _path_to_text(self, path: Dict) -> str:
        entities = path.get("entities", [])
        relations = path.get("relations", [])
        if not entities:
            return ""
        parts = [entities[0]]
        for relation, entity in zip(relations, entities[1:]):
            parts.append(f"-[{relation}]-> {entity}")
        return " ".join(parts)

    def _summarize_path_candidates(self, path_candidates: List[Dict], limit: int = 5) -> str:
        if not path_candidates:
            return "No candidate reasoning paths found."
        lines = []
        for idx, candidate in enumerate(islice(path_candidates, limit), start=1):
            lines.append(
                f"{idx}. {self._path_to_text(candidate)}"
                f" | target={candidate['target']} | score={candidate['score']:.2f}"
            )
        return "\n".join(lines)

    def _summarize_rule_candidates(self, path_candidates: List[Dict], limit: int = 5) -> str:
        if not path_candidates:
            return "No candidate rules found."
        lines = []
        for idx, candidate in enumerate(islice(path_candidates, limit), start=1):
            rules = " | ".join(candidate.get("rules", []))
            lines.append(f"{idx}. {rules}")
        return "\n".join(lines)

    def _collect_seed_entities(self) -> List[str]:
        seeds = []
        if self.topic_entities:
            for entity in self.topic_entities:
                seeds.append(self.convert_name_to_id(str(entity)))

        for triple in self.get_known_triples():
            seeds.extend([triple[0], triple[2]])

        unique = []
        seen = set()
        for entity in seeds:
            entity = str(entity).strip()
            if not entity or entity in seen:
                continue
            seen.add(entity)
            unique.append(entity)
        return unique

    def _workspace_signature(self, candidates: List[Dict], key: str) -> str:
        if not candidates:
            return ""
        preview = []
        for candidate in candidates[:5]:
            value = candidate.get(key, [])
            if isinstance(value, list):
                preview.append("->".join(value))
            else:
                preview.append(str(value))
        return " || ".join(preview)

    def build_reasoning_workspace(self, max_hops: int = 3, max_candidates: int = 18) -> Dict:
        """Build path/rule candidates from the known triples collected so far."""
        known_triples = self.get_known_triples()
        forward_graph, reasoning_graph = self._build_reasoning_graphs(known_triples)

        seed_entities = self._collect_seed_entities()
        candidate_paths = []
        seen_paths = set()

        if not seed_entities:
            workspace = {
                "known_triples": known_triples,
                "path_candidates": [],
                "rule_candidates": [],
                "expandable_entities": [],
                "answer_candidates": [],
                "path_signature": "",
                "rule_signature": "",
            }
            self.reasoning_workspace = workspace
            return workspace

        path_nodes = list(reasoning_graph.nodes())
        for source in seed_entities:
            if source not in reasoning_graph:
                continue

            for target in path_nodes:
                if target == source:
                    continue
                try:
                    all_paths = nx.all_simple_paths(reasoning_graph, source, target, cutoff=max_hops)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                for entity_path in all_paths:
                    relations = []
                    rules = []
                    valid = True
                    for head, tail in zip(entity_path[:-1], entity_path[1:]):
                        edge_data = reasoning_graph.get_edge_data(head, tail)
                        relation = self._unpack_edge_relation(edge_data)
                        if relation is None:
                            valid = False
                            break
                        relations.append(relation)
                        rules.append(
                            self._inverse_relation(relation) if relation.startswith("inverse(") else relation
                        )
                    if not valid:
                        continue

                    path_key = tuple(entity_path)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    candidate = {
                        "source": source,
                        "target": entity_path[-1],
                        "entities": entity_path,
                        "relations": relations,
                        "rules": rules,
                    }
                    candidate["score"] = self._path_score(candidate)
                    candidate_paths.append(candidate)

                    if len(candidate_paths) >= max_candidates:
                        break
                if len(candidate_paths) >= max_candidates:
                    break
            if len(candidate_paths) >= max_candidates:
                break

        candidate_paths.sort(key=lambda item: (-item["score"], len(item["relations"]), item["entities"]))

        answer_candidates = []
        seen_answers = set()
        for candidate in candidate_paths:
            target = candidate["target"]
            if target in seen_answers:
                continue
            if target in self.explored_entities:
                continue
            if self._is_cvt_like(target):
                continue
            if forward_graph.out_degree(target) == 0:
                seen_answers.add(target)
                answer_candidates.append(target)

        if not answer_candidates:
            for candidate in candidate_paths:
                target = candidate["target"]
                if target in seen_answers or self._is_cvt_like(target):
                    continue
                seen_answers.add(target)
                answer_candidates.append(target)

        expandable_entities = []
        seen_expandable = set()
        for candidate in candidate_paths:
            target = candidate["target"]
            if target in seen_expandable or target in self.explored_entities or self._is_cvt_like(target):
                continue
            seen_expandable.add(target)
            expandable_entities.append(target)

        workspace = {
            "known_triples": known_triples,
            "path_candidates": candidate_paths,
            "rule_candidates": candidate_paths,
            "expandable_entities": expandable_entities,
            "answer_candidates": answer_candidates,
            "path_signature": self._workspace_signature(candidate_paths, key="entities"),
            "rule_signature": self._workspace_signature(candidate_paths, key="rules"),
        }
        self.reasoning_workspace = workspace
        return workspace

    def append_reasoning_records(self, workspace: Optional[Dict] = None):
        """Append synthetic Path/Rule records for the current reasoning state."""
        workspace = workspace or self.reasoning_workspace or self.build_reasoning_workspace()
        if not workspace.get("known_triples"):
            return
        signature = workspace.get("path_signature", "") + "##" + workspace.get("rule_signature", "")
        if signature == self.last_reasoning_signature:
            return
        self.last_reasoning_signature = signature

        path_summary = self._summarize_path_candidates(workspace.get("path_candidates", []))
        rule_summary = self._summarize_rule_candidates(workspace.get("rule_candidates", []))

        next_index = len(self.records) + 1
        self.records.append(
            {
                "i": next_index,
                "thought": "Decompose the known triples into candidate reasoning paths before deciding the next action.",
                "action": f"Path[{len(workspace.get('path_candidates', []))}]",
                "observation": path_summary,
            }
        )
        self.records.append(
            {
                "i": next_index + 1,
                "thought": "Normalize the candidate paths into inversion-aware rules before deciding the next action.",
                "action": f"Rule[{len(workspace.get('rule_candidates', []))}]",
                "observation": rule_summary,
            }
        )

        self.reasoning_workspace = workspace

    def get_expansion_entities(self, workspace: Optional[Dict] = None) -> List[str]:
        workspace = workspace or self.reasoning_workspace or self.build_reasoning_workspace()
        candidates = []
        seen = set()
        for entity in workspace.get("expandable_entities", []):
            entity = str(entity).strip()
            if not entity or entity in seen:
                continue
            seen.add(entity)
            candidates.append(entity)
        return candidates

    def get_supported_answer_candidates(self, workspace: Optional[Dict] = None) -> List[str]:
        workspace = workspace or self.reasoning_workspace or self.build_reasoning_workspace()
        candidates = []
        seen = set()
        merged = list(self.answer_candidates) + list(workspace.get("answer_candidates", []))
        for candidate in merged:
            candidate = str(candidate).strip()
            if not candidate or candidate in seen or self._is_cvt_like(candidate):
                continue
            seen.add(candidate)
            candidates.append(candidate)
        return sorted(candidates)

    def generate(self, thought):
        # [...]
        if thought.startswith("["):
            thought = thought[1:-1]

        entities = extract_numbers_from_string(thought)
        related_triples_df = [self.kg.get_1hop_triples(str(entity_id)) 
                           for entity_id in entities if str(entity_id) in self.kg.entities]
        related_triples = [triple for df in related_triples_df for triple in df.values.tolist()]
        # triples = triples_df.values.tolist() if len(triples_df) > 0 else []
        related_triples = random.sample(related_triples, k=min(len(related_triples), 3))
        # print("Related triples for generation:")
        # for triple in related_triples:
        #     print(triple)
        # print("length of related_triples:", len(related_triples))
        related_triple_str = convert_triples_to_str(related_triples)
        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/generate_triples")
        # prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/generate_triples_wo_ctx")
        prompt = format_prompt(prompt_path)
    
        n = self.args.sc_num
        # neighbors = self.kg.get_1hop_triples(self.name_to_id[self.topic_entities[0].lower()])
        sep = "\t"
        prompt = (
            prompt + f"Thought: {thought}\n"
            f"Known Triples: {related_triple_str}\n"
            f"Missing relations: "
        )

        # logger.debug(f"Generate prompt:{prompt}")
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

        # print("Parsed generated relations:")
        parsed_relations = parse_generated_relations(responses)

        print(parsed_relations)
        result = []
        for start_entity, relation in parsed_relations.items():
            relation = relation[0] # LLM đề xuất 1 relation cho mỗi entity
            
            relation, _ = self.kg.get_best_relation_match(relation)
            # print(f"Relation: {relation}")
            # relation_id = self.kg.rel2id.get(relation)
            
            candidates = self.gnn.predict_topk(start_entity, relation, k=10)
            relation_paths = {}
            for candidate in candidates:
                relation_path = self.kg.get_shortest_path_with_relations(str(start_entity), str(candidate))
                if relation_path == None or len(relation_path["relations"]) > 6:
                    continue
                print("candidate:", candidate, end="\t")
                print("relation_path:", relation_path)
                # relation_path is a dict with key path and relation_path
                relation_path_str = f"{start_entity}-"
                for i, ent, rel in zip(range(len(relation_path["relations"])), relation_path["path"][1:], relation_path["relations"]):
                    relation_path_str += f"[{rel}]-> "

                # print("relation_path_str:", relation_path_str)
                relation_paths[candidate] = relation_path_str + str(candidate)
            verified_candidates = self.verify(thought, relation_paths)
            for candidate in verified_candidates:
                triple = [str(start_entity), str(relation), str(candidate)]
                triple_str = convert_triples_to_str([triple])
                # print("Generated triple:", triple_str)
                existing_triples = self.records[-1].get("generated_triples")
                if existing_triples:
                    self.records[-1]["generated_triples"] = existing_triples + "\n" + triple_str
                else:
                    self.records[-1]["generated_triples"] = triple_str
                self.answer_candidates.add(str(candidate))
                result.append(triple_str)

        


        # if n == 1:
        #     responses = [responses]
        # self.records[-1]["generated_triples"] = {}
        # for i, response in enumerate(responses):
        #     generated_triples = []
        #     logger.debug(responses)
        #     for line in response.split("\n"):
        #         try:
        #             h, r, t = [item.strip() for item in line.split(sep)]
        #             generated_triples.append([h, r, t])
        #         except Exception as e:
        #             logger.error(traceback.format_exc())
        #             logger.error(line)

        #     generated_triples = sorted(generated_triples)
        #     self.records[-1]["generated_triples"][i + 1] = generated_triples

        # if n > 1:
        #     verified_triples = self.verify(thought)
        #     result = convert_triples_to_str(verified_triples)
        # else:
        #     result = convert_triples_to_str(generated_triples)

        return "\n".join(result)

    def verify(self, thought, relation_paths):
        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/verify_triples")
        prompt = format_prompt(prompt_path)

        relation_path_str = [f"{relation_path}" for candidate, relation_path in relation_paths.items()]
        relation_path_str = "\n".join(relation_path_str)
        
        prompt = (
            prompt + f"Question: {thought}\n"
            f"Proposed Candidates: {relation_path_str}\n"
            f"Answer: "
        )
        # print("Verify prompt:", prompt)
        # verified_triples = []
        response = run_llm(
            prompt,
            self.args.temperature,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
        )
        # print("Verify response:", response)
        candidate = extract_numbers_from_string(response)
        self.records[-1]['verified_candidates'] = candidate
        for item in candidate:
            self.answer_candidates.add(str(item))
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
            
            if self.mid_crucial_triples:
                triples, relations = self.filter_crucial_triples(triples)
            # logger.debug(f"Relations after filter_crucial_triples: {relations}")

            for i in range(len(relations)):
                # only remain the last two parts
                abbr_rel = shorten_relation(relations[i])
                self.abbr_rel_to_rel[abbr_rel] = relations[i]
                relations[i] = abbr_rel

            for i in range(len(triples)):
                abbr_rel = shorten_relation(triples[i][1])
                self.abbr_rel_to_rel[abbr_rel] = triples[i][1]
                triples[i][1] = abbr_rel

            relations = sorted(relations)
            # logger.debug(f"Relations after abbreviation and sorting: {relations}")
            ## Call LLM to filter relations for each entity
            filtered_relations = self.filter_relations(entity_name, relations, self.last_thought)
            logger.debug(f"Relations after filter_relations (LLM filtered): {filtered_relations}")

            related_triples = self.sample_triples_by_relation(triples, filtered_relations)
            
            # Extract relations from sampled triples
            sampled_relations = sorted(list(set([triple[1] for triple in related_triples])))
            # logger.debug(f"Relations in sampled triples: {sampled_relations}")

            # Note: convert_id_to_name_in_triples not available in new interface
            # IDs are already in the triples from the KGInterface
            # Skipping conversion step - triples should already have readable names
            # self.update_id_to_name(id_to_label)  # Passing this step

            all_related_triples.extend(related_triples)

        all_related_triples = sorted(all_related_triples)

        self.triples.extend(deepcopy(all_related_triples))

        self.records[-1]["triples"] = all_related_triples
        self.records[-1]["entity_names"] = entity_names
        self.records[-1]["one_hop_relations"] = filtered_relations

        new_entities = set()
        for triple in all_related_triples:
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
