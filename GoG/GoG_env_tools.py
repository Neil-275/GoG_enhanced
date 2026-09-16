from collections import defaultdict
from copy import deepcopy
from logging import Formatter
import json
import random
import re
# import spacy
from string import Formatter
import traceback
import asyncio
from loguru import logger
from GoG.kg_interface import KGInterface
from GoG.utils import (
    format_prompt,
    get_edges,
    parse_json_list,
    parse_llm_output_to_list,
    read_file,
    extract_numbers_from_string,
    parse_generated_relation_directions,
)
from GoG.GoG_llms import run_llm
from GoG.gnn_interface import OneShotInterface
import pandas as pd
import pickle as pkl
import os
import sys
from ast import literal_eval
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
        self.explored_triples = []
        # entity_id -> {parent_entity_id, relation, direction, step_index}.
        # Written at first production only; see _record_provenance.
        self.provenance = {}
        self.topic_entity_set = set()
        self.mid_crucial_triples = None
        self.n_related_triples = self.args.n_related_triples
        # only used in answer without kg
        self.llm_output = None
        self.generate_call_count = 0
        self.topic_entities = None
        self.question = None
        self.crucial_rel = None
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

    def find_crucial_rel(self, data):
        # print(data, data["q_entity"], data["hard_answer"])
        # print(type(data["q_entity"]), type(data["hard_answer"]))
        q_entity = data["q_entity"][0]
        hard_answer = data["hard_answer"][0]
        self.ez_answers = [ans for ans in data["a_entity"] if ans != hard_answer]
        crucial_edges = get_edges(self.kg.drop_edges, q_entity, hard_answer)
        if crucial_edges.empty or len(crucial_edges) > 1:
            return None
        return crucial_edges

    def assign_query(self, data):
        self.topic_entities = data["q_entity"]
        self.question = data["question"]
        self.records = []

        self.explored_triples = []
        self.explored_entities = set()
        self.provenance = {}
        # Path tracing terminates at any topic entity: they are the roots and
        # carry no provenance record.
        self.topic_entity_set = {str(entity) for entity in self.topic_entities}

        if self.args.hard_only:
            self.crucial_rel = self.find_crucial_rel(data)
        else:
            self.crucial_rel = None
        self.n_related_triples = self.args.n_related_triples
        # only used in answer without kg
        self.llm_output = None
        self.generate_call_count = 0
        # q_entity = literal_eval(case["q_entity"])[0]
        # hard_answer = literal_eval(case["hard_answer"])[0]
        # crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        # if crucial_edges.empty or len(crucial_edges) > 1:
            # continue


    def convert_records_to_str(self):
        string = ""
        for record in self.records:
            string += "Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {observation}\n".format(
                i=record["i"],
                action=record["action"],
                thought=record["thought"],
                observation=record.get("observation", ""),
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

    def step(self, action_str=None, *, collect_enabled: bool = False):
        logger.debug(action_str)

        pattern = r"(\w+)(\[.+\])"
        result = re.match(pattern, action_str)

        logger.debug(f'Result: {result}')
        action = result.group(1).lower()
        parameter = result.group(2)

        logger.info(f"Action: {action}, Parameter: {parameter}")

        if action == "propose":
            # Kept as `generate_call_count` on purpose: it is a declared field
            # of PredictionEntry in evaluation.py, and renaming it would make
            # new result files structurally incomparable with existing ones.
            # It now counts Propose calls.
            self.generate_call_count += 1
            return self.propose_action(parameter)
        elif action == "collect":
            if collect_enabled:
                return f"Collected {parameter}"
            else:
                raise ValueError("Action 'collect' is not enabled")

        raise ValueError(f"Unsupported action: {action_str}")

    def construct_neighbor_relation_set(self, entities, thought):
        """Fetch the 1-hop schema around `entities` from G_inc.

        Returns the relevance-filtered local-context relations for each
        direction (spec 2/2a), the start entity, and `observed_index`: a
        (relation, direction) -> [far-end entity id] map built from the *same*
        G_inc snapshot, so the local context and the observed candidates of
        spec 3 step 1 can never disagree. Direction is relative to the entity:
        outgoing means it is the head and the far end is the tail.
        """
        related_triples_df = [
            (str(entity_id), self.kg.get_1hop_triples(str(entity_id)))
            for entity_id in entities if str(entity_id) in self.kg.entities
        ]
        outgoing_neighbor_relation_set = set()
        incoming_neighbor_relation_set = set()
        observed_index = defaultdict(list)
        for entity_id, df in related_triples_df:
            for triple in df.values.tolist():
                if len(triple) == 2:
                    direction, wrapped_triple = triple
                    if direction == 0:
                        outgoing_neighbor_relation_set.add(wrapped_triple[1])
                        observed_index[(wrapped_triple[1], "outgoing")].append(str(wrapped_triple[2]))
                    elif direction == 1:
                        incoming_neighbor_relation_set.add(wrapped_triple[1])
                        observed_index[(wrapped_triple[1], "incoming")].append(str(wrapped_triple[0]))
                elif len(triple) == 3:
                    head, relation, tail = triple
                    if str(head) == entity_id:
                        outgoing_neighbor_relation_set.add(relation)
                        observed_index[(relation, "outgoing")].append(str(tail))
                    if str(tail) == entity_id:
                        incoming_neighbor_relation_set.add(relation)
                        observed_index[(relation, "incoming")].append(str(head))
        start_entity = entities[0]
        outgoing_related_relations = self.kg.get_best_relation_match(
            thought, rel_set=list(outgoing_neighbor_relation_set), k=5, threshold=0.1
        ) if self.kg.rels and outgoing_neighbor_relation_set else []
        incoming_related_relations = self.kg.get_best_relation_match(
            thought, rel_set=list(incoming_neighbor_relation_set), k=5, threshold=0.1
        ) if self.kg.rels and incoming_neighbor_relation_set else []
        return outgoing_related_relations, incoming_related_relations, start_entity, dict(observed_index)

    def parse_json_list_responses(self, responses):
        if isinstance(responses, str):
            responses = [responses]

        parsed_items = []
        for response in responses:
            if not response:
                continue
            parsed_response = parse_json_list(response)
            if isinstance(parsed_response, list):
                parsed_items.extend(
                    item for item in parsed_response if isinstance(item, dict)
                )
        return parsed_items

    def propose(self, q_sub, e_current):
        """Relation selection + direction specification for one entity.

        `e_current` arrives already parsed by the caller, so `q_sub` is the
        clean sub-question: the entity id is no longer glued to the front of
        the text fed to the bi-encoder and the two sub-prompts. This method
        does not call Predict -- the action wrapper does that per relation.
        """
        thought = q_sub
        outgoing_related_relations, incoming_related_relations, start_entity, observed_index = \
            self.construct_neighbor_relation_set([e_current], thought)

        relation_selection_prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/relation_selection.txt")
        relation_selection_prompt = format_prompt(relation_selection_prompt_path)

        candidate_relations = self.kg.get_best_relation_match(thought, k=5) if self.kg.rels else []
        candidate_relations_str = "[{}]".format(", ".join(candidate_relations))

        n = self.args.sc_num
        # print(f"Candidate relations: {candidate_relations_str}")
        relation_selection_prompt = (
            relation_selection_prompt.format(
                thought=thought,
                outgoing_neighboring_relations="[" + ", ".join(outgoing_related_relations) + "]",
                incoming_neighboring_relations="[" + ", ".join(incoming_related_relations) + "]",
                candidate_relations=candidate_relations_str,
            )
            + "\nAnswer: "
        )

        logger.debug(f"Relation selection prompt:{relation_selection_prompt}")
        relation_selection_responses = run_llm(
            relation_selection_prompt,
            self.args.temperature,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
            n=n
        )
        # print("Relation selection LLM responses:")
        # print(relation_selection_responses)

        selected_relation_items = self.parse_json_list_responses(relation_selection_responses)
        selected_relations = []
        updated_selected_relation_items = []

        for item in selected_relation_items:
            relation = item.get("relation")
            for gd_relation in candidate_relations:
                # print("gd_relation:", gd_relation, "relation:", relation)
                if relation in gd_relation:
                    relation = gd_relation
                    break
            if relation and relation not in selected_relations:
                selected_relations.append(relation)
                updated_selected_relation_items.append({"relation": relation})
        # Predict now runs once per selected relation, so the number of
        # relations is the GNN cost multiplier -- cap it.
        max_relations = self.args.max_selected_relations 
        if len(selected_relations) > max_relations:
            logger.debug(
                f"Capping selected relations {selected_relations} to first {max_relations}"
            )
            selected_relations = selected_relations[:max_relations]
            updated_selected_relation_items = updated_selected_relation_items[:max_relations]
        selected_relation_items = updated_selected_relation_items
        ## recreate selected_relation_items with updated relations

        if not selected_relations:
            logger.debug("Relation selection returned no relations; skipping direction step.")
            return [], [], start_entity, observed_index, outgoing_related_relations, incoming_related_relations

        direction_specification_prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/direction_specification.txt")
        direction_specification_prompt = format_prompt(direction_specification_prompt_path)
        # selected_relations_str = "{}".format(", ".join(selected_relation_items))
        selected_relations_str = str(selected_relation_items)

        direction_specification_prompt = (
            direction_specification_prompt.format(
                thought=thought,
                topic_entity=str(start_entity),
                outgoing_neighboring_relations="[" + ", ".join(outgoing_related_relations) + "]",
                incoming_neighboring_relations="[" + ", ".join(incoming_related_relations) + "]",
                selected_relations=selected_relations_str,
            )
            + "\nAnswer: "
        )
        # print(f"Direction specification prompt: {direction_specification_prompt}"   )
        # logger.debug(f"Direction specification prompt:{direction_specification_prompt}")
        direction_specification_responses = run_llm(
            direction_specification_prompt,
            self.args.temperature,
            self.args.max_length,
            self.args.opeani_api_keys,
            self.args.LLM_type,
            stop=None,
            n=n
        )
        # print("Direction specification LLM responses:")
        # print(direction_specification_responses)

        parsed_generations = self.parse_json_list_responses(direction_specification_responses)
        logger.debug(f"Parsed direction specifications: {parsed_generations}")
        return (
            selected_relations,
            parsed_generations,
            start_entity,
            observed_index,
            outgoing_related_relations,
            incoming_related_relations,
        )

    def predict(self, start_entity, relation, direction):
        candidates = self.gnn.predict_topk(
            str(start_entity), relation, direction, k=self.args.predict_topk, known=False
        )
        return candidates

    @staticmethod
    def _entity_sort_key(entity_id):
        """Deterministic ordering for capping. Numeric ids sort numerically."""
        text = str(entity_id)
        return (0, int(text), "") if text.lstrip("-").isdigit() else (1, 0, text)

    @staticmethod
    def _format_triple(start_entity, relation, direction, other):
        if direction == "outgoing":
            return f"{start_entity}, {relation}, {other}"
        return f"{other}, {relation}, {start_entity}"

    def parse_propose_argument(self, parameter):
        """Parse the argument of `Propose[e_current | q_sub]`.

        Falls back to the pre-Propose behaviour (pull the first number out of
        the whole string) when the delimiter is missing, because the driver's
        retry loop never re-prompts on a parse failure -- erroring here would
        burn a trajectory step with no way to recover.
        """
        text = str(parameter).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        text = text.strip()

        if "|" in text:
            e_current, _, q_sub = text.partition("|")
            return e_current.strip(), q_sub.strip()

        entities = extract_numbers_from_string(text)
        logger.warning(
            f"Propose argument has no '|' delimiter: {parameter!r}; "
            f"falling back to number extraction"
        )
        return (str(entities[0]) if entities else ""), text

    def propose_action(self, parameter):
        """Handle one `Propose[e_current | q_sub]` action.

        Runs relation selection + direction specification for a single entity,
        then -- unconditionally, with no activation policy -- looks up observed
        edges and runs Predict for every selected relation, and renders both
        groups with explicit labels.
        """
        e_current, q_sub = self.parse_propose_argument(parameter)
        # return (self.propose(q_sub, e_current))
        (
            selected_relations,
            parsed_generations,
            start_entity,
            observed_index,
            outgoing_context,
            incoming_context,
        ) = self.propose(q_sub, e_current)

        record = self.records[-1]
        record["e_current"] = str(start_entity)
        record["q_sub"] = q_sub
        record["local_context"] = {
            "outgoing": list(outgoing_context),
            "incoming": list(incoming_context),
        }

        lines = [
            f"Local context for {start_entity}:",
            "  outgoing: [" + ", ".join(outgoing_context) + "]",
            "  incoming: [" + ", ".join(incoming_context) + "]",
        ]

        selected_log = []
        observed_log = []
        predicted_log = []
        cap = self.args.n_related_triples

        for item in parsed_generations:
            relation = item.get("relation")
            direction = item.get("direction")
            if not relation:
                logger.debug(f"Skipping parsed generation without relation: {item}")
                continue
            if relation not in selected_relations:
                matched_relations = [
                    gd_relation for gd_relation in selected_relations
                    if relation in gd_relation or gd_relation in relation
                ]
                if matched_relations:
                    relation = matched_relations[0]
                else:
                    logger.debug(
                        f"Skipping parsed generation with unselected relation: {item}; "
                        f"selected_relations={selected_relations}"
                    )
                    continue
            if isinstance(direction, str):
                direction = direction.strip().lower()
            if direction not in {"incoming", "outgoing"}:
                logger.debug(f"Skipping parsed generation with invalid direction: {item}")
                continue

            selected_log.append({"relation": relation, "direction": direction})
            lines.append("")
            lines.append(f"Selected: {relation} ({direction})")

            # --- observed: direct edges already in G_inc ---
            observed_entities = sorted(
                set(observed_index.get((relation, direction), [])),
                key=self._entity_sort_key,
            )
            kept, suppressed = [], 0
            for other in observed_entities:
                triple = self._format_triple(start_entity, relation, direction, other)
                if triple in self.explored_triples:
                    suppressed += 1
                    continue
                kept.append((other, triple))
            kept = kept[:cap]
            for other, triple in kept:
                self.explored_triples.append(triple)
                self.explored_entities.add(str(other))
                self._record_provenance(other, start_entity, relation, direction)
            observed_log.append({
                "relation": relation,
                "direction": direction,
                "entities": [other for other, _ in kept],
                "suppressed_as_already_shown": suppressed,
            })

            if kept:
                lines.extend(self._render_group("(observed, direct edge)", [t for _, t in kept]))
            elif suppressed:
                lines.append(
                    f"  (observed, direct edge):     all {suppressed} observed edge(s) already shown above"
                )
            else:
                lines.append("  (observed, direct edge):     none")

            # --- predicted: GNN candidates, existence-filtered ---
            candidates = self.predict(start_entity, relation, direction)
            if not candidates:
                logger.debug(
                    f"No GNN candidates for entity={start_entity}, relation={relation}, direction={direction}"
                )
            observed_set = set(observed_entities)
            overlap = [c for c in candidates if str(c) in observed_set]
            if overlap:
                # Commit 1 makes predict_topk filter these out; if any survive,
                # the observed/predicted labels would be lying.
                logger.warning(
                    f"Predicted candidates overlap observed edges for "
                    f"({start_entity}, {relation}, {direction}): {overlap}"
                )
            candidates = [c for c in candidates if str(c) not in observed_set]
            # Provenance is recorded for predicted candidates too: the
            # observed/predicted distinction is applied at collect time, by
            # whether the path can be replayed against G_inc, not here.
            for candidate in candidates:
                self._record_provenance(candidate, start_entity, relation, direction)
            predicted_log.append({
                "relation": relation,
                "direction": direction,
                "entities": [str(c) for c in candidates],
            })

            if candidates:
                lines.extend(self._render_group(
                    "(predicted, unverified)",
                    [self._format_triple(start_entity, relation, direction, c) for c in candidates],
                ))
            else:
                lines.append("  (predicted, unverified):     none")

        record["selected_relations"] = selected_log
        record["observed"] = observed_log
        record["predicted"] = predicted_log

        if not selected_log:
            lines.append("")
            lines.append("No relation was selected for this entity.")

        return "\n".join(lines)

    @staticmethod
    def _render_group(label, triples):
        """Render a labelled group, aligning continuation lines under the first."""
        prefix = f"  {label}:".ljust(31)
        rendered = [prefix + triples[0]]
        rendered.extend(" " * 31 + triple for triple in triples[1:])
        return rendered

    # ------------------------------------------------------------------
    # Collect: relation-path fan-out
    # ------------------------------------------------------------------

    def _record_provenance(self, entity, parent_entity, relation, direction):
        """Persist the triple that first produced `entity`.

        First production wins: a later re-appearance via a different path must
        not overwrite the record, because that is what makes path tracing
        well-defined. Topic entities are roots and never get a record.
        """
        entity = str(entity)
        parent_entity = str(parent_entity)
        if entity in self.provenance or entity in self.topic_entity_set:
            return
        if entity == parent_entity:
            # A self-loop would make trace_path non-terminating.
            logger.debug(f"Skipping self-referential provenance for {entity}")
            return
        self.provenance[entity] = {
            "parent_entity_id": parent_entity,
            "relation": relation,
            "direction": direction,
            "step_index": self.records[-1]["i"] if self.records else None,
        }

    def _trace_path(self, entity):
        """Walk provenance back to a root, returning (path, root).

        `path` is the ordered list of (relation, direction) pairs from root to
        `entity` -- relation *types* only, with the specific intermediate
        entities deliberately discarded. Returns (None, None) if the chain
        cycles. An entity with no provenance record is its own root, which
        yields an empty path and collapses to a single-entity collect.
        """
        entity = str(entity)
        path = []
        current = entity
        visited = {current}
        while current not in self.topic_entity_set:
            prov = self.provenance.get(current)
            if prov is None:
                break
            path.insert(0, (prov["relation"], prov["direction"]))
            current = str(prov["parent_entity_id"])
            if current in visited:
                logger.warning(f"Cycle in provenance chain while tracing {entity}")
                return None, None
            visited.add(current)
        return path, current

    def _observed_neighbors(self, node, relation, direction):
        """Far ends of real G_inc edges matching (relation, direction) at `node`.

        Direction is relative to `node`: outgoing means it is the head. Reads
        the same incomplete graph that `observed_index` is built from, so a
        hop that was surfaced as observed always replays here.
        """
        graph = self.kg.incomplete_graph_nx
        node = str(node)
        if node not in graph:
            return set()
        adjacency = graph.succ[node] if direction == "outgoing" else graph.pred[node]
        return {
            str(neighbor)
            for neighbor, edges in adjacency.items()
            for edge_data in edges.values()
            if edge_data.get("relation") == relation
        }

    def _replay_path(self, root, path):
        """Fan a relation-type path out from `root` across G_inc.

        Returns (frontier, log). `frontier` is None when the replay fails and
        the caller must fall back to a single-entity collect. Depends only on
        (root, path), so `collect_action` caches it: entities sharing a traced
        path replay once.
        """
        max_hops = getattr(self.args, "collect_fanout_max_hops", 0) or 0
        if max_hops > 0 and len(path) > max_hops:
            return None, {"reason": "path_longer_than_max_hops", "n_hops": len(path)}

        max_entities = getattr(self.args, "collect_fanout_max_entities", 0) or 0
        frontier = {str(root)}
        hop_sizes = []
        for relation, direction in path:
            next_frontier = set()
            for node in frontier:
                next_frontier |= self._observed_neighbors(node, relation, direction)
            if not next_frontier:
                # No real edge realises this hop from anywhere in the frontier:
                # the signal that it was Predict-derived rather than observed.
                return None, {
                    "reason": "unreplayable_hop",
                    "failed_hop": [relation, direction],
                    "hop_sizes": hop_sizes,
                }
            if max_entities > 0 and len(next_frontier) > max_entities:
                logger.warning(
                    f"Fan-out frontier hit {len(next_frontier)} entities at "
                    f"({relation}, {direction}), over collect_fanout_max_entities="
                    f"{max_entities}; falling back to single-entity collect"
                )
                return None, {
                    "reason": "frontier_over_max_entities",
                    "frontier_size": len(next_frontier),
                    "hop_sizes": hop_sizes,
                }
            frontier = next_frontier
            hop_sizes.append(len(frontier))
        return frontier, {"hop_sizes": hop_sizes}

    def _collect_one(self, entity, replay_cache=None):
        """Expand one collected entity into the full set to add to R_t.

        Replays `entity`'s relation-type path from its root, fanning out across
        the whole frontier at every hop rather than following the specific
        intermediates that produced `entity`. Any replay failure falls back to
        collecting `entity` alone.
        """
        entity = str(entity)
        path, root = self._trace_path(entity)
        if path is None:
            return {entity}, {"entity": entity, "fan_out": False, "reason": "cycle"}

        key = (root, tuple(path))
        if replay_cache is not None and key in replay_cache:
            frontier, log = replay_cache[key]
        else:
            frontier, log = self._replay_path(root, path)
            if replay_cache is not None:
                replay_cache[key] = (frontier, log)

        if frontier is None:
            logger.debug(f"Fan-out for {entity} failed ({log['reason']}); collecting it alone")
            return {entity}, dict(log, entity=entity, fan_out=False, path=path)

        if entity not in frontier:
            # The path replayed, but not onto `entity` itself -- possible when
            # the final hop was predicted while other real edges of the same
            # type exist. Keep `entity`: dropping the answer the agent actually
            # named would be a regression against the old Collect.
            logger.warning(
                f"Collected entity {entity} is not in its own replayed frontier "
                f"(path={path}); keeping it alongside the fan-out"
            )
            frontier = frontier | {entity}

        return frontier, dict(
            log,
            entity=entity,
            fan_out=len(path) > 0,
            path=path,
            root=root,
            n_collected=len(frontier),
        )

    def collect_action(self, action_str):
        """Handle one `Collect[...]` action, returning (entities, observation).

        The agent's interface is unchanged -- it still just names entities it
        judges to be answers. R_t itself stays owned by the driver; this
        returns the expanded set for it to union in.
        """
        match = re.search(r"Collect(?:ed)?(\[.*\])", action_str)
        parameter = match.group(1) if match else str(action_str)
        requested = parse_llm_output_to_list(parameter) or []
        requested = [str(entity).strip() for entity in requested if str(entity).strip()]

        collected = []
        seen = set()
        logs = []
        # Entities that trace to the same (root, path) share one replay.
        replay_cache = {}
        for entity in requested:
            entities, log = self._collect_one(entity, replay_cache)
            logs.append(log)
            for collected_entity in sorted(entities, key=self._entity_sort_key):
                if collected_entity not in seen:
                    seen.add(collected_entity)
                    collected.append(collected_entity)

        if self.records:
            self.records[-1]["collect"] = logs

        # The fan-out members are deliberately not listed back to the agent:
        # they go into R_t only, and must not become candidates for a future
        # e_current. Only the ids the agent already named are echoed.
        lines = [f"Collected the answers: {parameter}"]
        extra = len(seen) - len({entity for entity in requested})
        if extra > 0:
            lines.append(
                f"Also recorded {extra} further answer(s) reachable from the topic "
                f"entity by the same relation path."
            )
        return collected, "\n".join(lines)

    def get_template_variables(self, template_string):
    # Field names can be None for raw text chunks, so filter those out
        return [field_name for _, field_name, _, _ in Formatter().parse(template_string) if field_name is not None]

    def verify(self, topic_entity, question,  verify_candidates, threshold=0.5):
        prompt_path = read_file(f"{self.args.prompt_dir}/primitive_tasks/verify_triples")
        prompt = format_prompt(prompt_path)
        # print("type of prompt:", type(prompt))
        # print("Format str:", self.get_template_variables(prompt))
        for cand in verify_candidates:
            if cand["direction"] == "outgoing":
                # print(f"Verifying candidate triples for: {topic_entity} -[{relation}]-> ?")
                triple = str(topic_entity) +  ", " + cand["relation"] + ", " + str(cand["candidate_id"])
            if cand["direction"] == "incoming":
                # print(f"Verifying candidate triples for: ? -[{relation}]-> {topic_entity}")
                triple = str(cand["candidate_id"]) + ", " + cand["relation"] + ", " + str(topic_entity)
            evidence = "\n".join(cand["evidence"])
            prompt = prompt + f"Proposed triple: ({triple})\nEvidence: {evidence}\n\n"

        # relation_path_str = []
        # for candidate, relation_path in relation_paths.items():
        #     paths = "\n".join(relation_path)
        #     relation_path_str.append(f"Candidate: {candidate}\nEvidence: {paths}")
        #     # print(relation_path_str[-1])
        # relation_path_str = "\n".join(relation_path_str)
        
        # prompt = prompt + "Candidates:\n" + relation_path_str  +"\nAnswer: "
        # prompt = prompt + "Candidates:\n" + relation_path_str + "\nReasoning space:" 
        # print(123)
        print("Verify prompt:", prompt)
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
        print("Verify LLM output:", response, flush=True)

        if not response:
            self.records[-1]['verified_candidates'] = []
            return []

        verified_cands = []
        response_text = response.strip()

        parsed_response = parse_json_list(response_text)
        # print("Parsed verify response:", parsed_response, flush=True)
        if isinstance(parsed_response, list):
            for item in parsed_response:
                if not isinstance(item, dict):
                    continue
                if "triple" in item and "score" in item and item["score"] >= threshold:
                    triple = [str(item["triple"][0]), str(item["triple"][1]), str(item["triple"][2])]
                    item["triple"] = triple
                    verified_cands.append(item)

        self.records[-1]['verified_candidates'] = verified_cands
        return verified_cands
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
