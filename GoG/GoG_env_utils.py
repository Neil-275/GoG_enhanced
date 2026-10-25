"""Helpers split out of `GoG_env_tools.py` that carry no agent state.

Everything here is a pure function of its arguments: Observation string
formatting, parsing of LLM/action text, and the graph-path algorithms behind
Collect's relation-path fan-out. `KGEnv` keeps the per-question mutable state
(`records`, `explored_triples`, `provenance`, ...) and the LLM/KG calls; this
module holds the parts that can be read, tested, and changed without knowing
anything about the ReAct loop.
"""

from string import Formatter

from loguru import logger

from GoG.utils import extract_numbers_from_string, parse_json_list


# ----------------------------------------------------------------------
# Observation formatting
# ----------------------------------------------------------------------

#: Column the group bodies are aligned to in a rendered Observation.
GROUP_LABEL_WIDTH = 31


def entity_sort_key(entity_id):
    """Deterministic ordering for capping. Numeric ids sort numerically."""
    text = str(entity_id)
    return (0, int(text), "") if text.lstrip("-").isdigit() else (1, 0, text)


def format_triple(start_entity, relation, direction, other):
    """One triple, oriented so `start_entity` sits on the side `direction` says."""
    if direction == "outgoing":
        return f"{start_entity}, {relation}, {other}"
    return f"{other}, {relation}, {start_entity}"


def format_triple_group(start_entity, relation, direction, others):
    """Render a whole (relation, direction) group as one bracketed triple.

    The far ends collapse into a list on the side of the triple they actually
    occupy, so the line still reads as a literal triple and the direction is
    self-evident from where `start_entity` sits.
    """
    joined = ", ".join(str(other) for other in others)
    if direction == "outgoing":
        return f"{start_entity}, {relation}, [{joined}]"
    return f"[{joined}], {relation}, {start_entity}"


def render_group(label, body):
    """Render one labelled group as a single aligned line."""
    return f"  {label}:".ljust(GROUP_LABEL_WIDTH) + body


def render_representative(start_entity, relation, direction, entity, n_options):
    """Render the single unlabelled row of --representative_only.

    `direction` is the representative's own, so the triple reads the right way
    round even when the pool mixed both orientations.
    """
    if entity is None:
        return "  No new edges retrieved"
    noun = "option" if n_options == 1 else "options"
    triple = format_triple_group(start_entity, relation, direction, [entity])
    return f"  {triple}  ({n_options} {noun})"


# ----------------------------------------------------------------------
# Parsing of action arguments and LLM responses
# ----------------------------------------------------------------------

def parse_propose_argument(parameter):
    """Parse the argument of `Propose[e_current | q_sub]`.

    Falls back to the pre-Propose behaviour (pull the first number out of the
    whole string) when the delimiter is missing, because the driver's retry
    loop never re-prompts on a parse failure -- erroring here would burn a
    trajectory step with no way to recover.
    """
    text = str(parameter).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.strip()

    if "|" in text:
        e_current, _, q_sub = text.partition("|")
        return e_current.strip(), q_sub.strip()

    entities = extract_numbers_from_string(text)
    # logger.warning(
    #     f"Propose argument has no '|' delimiter: {parameter!r}; "
    #     f"falling back to number extraction"
    # )
    return (str(entities[0]) if entities else ""), text


def parse_json_list_responses(responses):
    """Flatten one or more LLM responses into the list of dicts they contain."""
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


def get_template_variables(template_string):
    """Names of the `{}` fields in a prompt template."""
    # Field names can be None for raw text chunks, so filter those out
    return [
        field_name
        for _, field_name, _, _ in Formatter().parse(template_string)
        if field_name is not None
    ]


# ----------------------------------------------------------------------
# Collect fan-out: relation-path replay over G_inc
# ----------------------------------------------------------------------

def observed_neighbors(graph, node, relation, direction):
    """Far ends of real G_inc edges matching (relation, direction) at `node`.

    Direction is relative to `node`: outgoing means it is the head. `graph` is
    the same incomplete graph that `observed_index` is built from, so a hop
    surfaced as observed always replays here.
    """
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


def trace_path(provenance, topic_entity_set, entity):
    """Walk provenance back to a root, returning (path, root).

    `path` is the ordered list of (relation, direction) pairs from root to
    `entity` -- relation *types* only, with the specific intermediate entities
    deliberately discarded. Returns (None, None) if the chain cycles. An entity
    with no provenance record is its own root, which yields an empty path and
    collapses to a single-entity collect.
    """
    entity = str(entity)
    path = []
    current = entity
    visited = {current}
    while current not in topic_entity_set:
        prov = provenance.get(current)
        if prov is None:
            break
        path.insert(0, (prov["relation"], prov["direction"]))
        current = str(prov["parent_entity_id"])
        if current in visited:
            logger.warning(f"Cycle in provenance chain while tracing {entity}")
            return None, None
        visited.add(current)
    return path, current


def replay_path(graph, root, path, max_hops=0, max_entities=0, predict_fn=None, threshold=0.0):
    """Fan a relation-type path out from `root` across G_inc.

    Returns (frontier, log). `frontier` is None when the replay fails and the
    caller must fall back to a single-entity collect. Depends only on
    (root, path), so `collect_action` caches it: entities sharing a traced path
    replay once.

    With `predict_fn(node, relation, direction) -> [(entity, p)]`, every hop
    also follows predicted edges. Each frontier entity carries the product of
    edge confidences along its best path (observed edge = 1, predicted edge =
    p), merged by max; predicted arrivals below `threshold` are dropped. This
    recovers answers behind a dropped edge at any hop, not only the ones on
    the agent's own trajectory.
    """
    max_hops = max_hops or 0
    if max_hops > 0 and len(path) > max_hops:
        return None, {"reason": "path_longer_than_max_hops", "n_hops": len(path)}

    max_entities = max_entities or 0
    frontier = {str(root): 1.0}
    predicted_scores = {}   # entity -> path score, for entities whose best path uses a predicted edge
    hop_sizes = []
    for relation, direction in path:
        next_frontier, via_prediction = {}, {}
        for node, score in frontier.items():
            for other in observed_neighbors(graph, node, relation, direction):
                if score > next_frontier.get(other, -1.0):
                    next_frontier[other] = score
                    via_prediction[other] = node in predicted_scores
            if predict_fn is None:
                continue
            for other, p in predict_fn(node, relation, direction):
                other_score = score * p
                if other_score >= threshold and other_score > next_frontier.get(other, -1.0):
                    next_frontier[other] = other_score
                    via_prediction[other] = True
        predicted_scores = {e: s for e, s in next_frontier.items() if via_prediction[e]}
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
    log = {"hop_sizes": hop_sizes}
    if predict_fn is not None:
        log["predicted_scores"] = predicted_scores
    return set(frontier), log
