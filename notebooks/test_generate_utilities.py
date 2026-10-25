"""Helper functions for notebooks/test_generate.ipynb (Propose / Predict / relation-inference evaluation).

Import from the notebook (cwd = repo root):
    from notebooks.test_generate_utilities import *
"""
from ast import literal_eval

import pandas as pd

from GoG.GoG_env_tools import KGEnv
from relation_inference.relation_inference_interface import RelationInferenceInterface

def get_edges(df, head, tail =None, both=False):
    if isinstance(head, str):
        head = [head]
    if isinstance(tail, str):
        tail = [tail]

    if both==True and tail is None:
        edges = df[(df["head"].isin(head)) | (df["tail"].isin(head))]
        # print(123)
        return edges.reset_index(drop=True)
    if tail is None and both == False:
        edges = df[df["head"].isin(head)]
        return edges
    if tail and both == True:
        # print(123)
        edges_a = df[(df["head"].isin(head)) & (df["tail"].isin(tail))]
        edges_b = df[(df["tail"].isin(head)) & (df["head"].isin(tail))]
        edges = pd.concat([edges_a, edges_b])
        return edges.reset_index(drop=True)
    if tail and both == False:
        edges = df[(df["head"].isin(head)) & (df["tail"].isin(tail))]
        return edges


def cal_relation_inference_performance(kg_env, val, model_inteface: RelationInferenceInterface, drop_edges, indices=None):
    total = 0
    hit = 0
    corrupts = []
    for i, case in enumerate(val.to_dict(orient="records")):
        if indices and case['id'] not in indices:
                    continue
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        if crucial_edges.empty or len(crucial_edges) > 1:
            continue
        total += 1
        target_direction = "outgoing" if crucial_edges["head"].item() == q_entity else "incoming"
        target_relation = crucial_edges["relation"].values[0] if not crucial_edges.empty else None
        try:
            cands = model_inteface.query(q_entity, k=3)
        except:
            corrupts.append(case)
        for cand in cands:
            # print(cand['relation'], cand['direction'])
            if cand['relation'] == target_relation and cand['direction'] == target_direction:
                hit += 1
                break
    print("Accuracy = ", hit / total)
    return corrupts


def cal_propose_accuracy(kg_env: KGEnv, val, drop_edges, analyze=False, indices=None):
    accuracy_rel = 0
    accuracy = 0
    accuracy_rel = 0
    accuracy_dir = 0
    total = 0
    corrupts = []
    hit1, hit3, hit5, hit10 = 0, 0, 0, 0
    for i, case in enumerate(val.to_dict(orient="records")):
        if indices and case['id'] not in indices:
            continue
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        if crucial_edges.empty or len(crucial_edges) > 1:
            continue
        total += 1
        
        target_direction = "outgoing" if crucial_edges["head"].item() == q_entity else "incoming"
        target_relation = crucial_edges["relation"].values[0] if not crucial_edges.empty else None
        # if target_relation != "nephew_of":
        #     continue
        if analyze:
            print(f"Question: {case['question']}")
            print(f"Target relation: {target_relation},\nTriple: ({crucial_edges['head'].item()}, {target_relation}, {crucial_edges['tail'].item()})")
        kg_env.records = []
        kg_env.records.append({"i": 1, "thought": "helo"})
        _, candidate_relations, _, _, _, _ = kg_env.propose_action(case["question"])
        # print(candidate_relations)
        corrupted = False
        matched = False
        try:
            for i, item in enumerate(candidate_relations):
                
                pred_relation = item.get("relation")
                pred_direction = item.get("direction")
                if analyze:
                    print(f"Predicted relation {i+1}:", pred_relation)
                    # print(f"Predicted direction {i+1}:", pred_direction)
                    if pred_direction == 'outgoing':
                        print(f"Predicted triple {i+1}: Triple: ({crucial_edges['head'].item()}, r, {crucial_edges['tail'].item()})" )
                    else:
                        print(f"Predicted triple {i+1}: Triple: ({crucial_edges['tail'].item()}, r, {crucial_edges['head'].item()})" )
                    print("predictions in candidate:", pred_relation in candidate_relations)
                if pred_relation == target_relation:
                    accuracy_rel += 1
                if pred_relation == target_relation and pred_direction == target_direction and matched == False:
                    accuracy += 1
                    matched = True
                
            # print(f"Matched: {matched}")
        except:
                    # Handle the case where q_entity is not in res
            corrupts.append(i)
            corrupted = True
                    # continue
        ## Check if target_relation is among candidate_relations
        if target_relation in candidate_relations:
            rank = candidate_relations.index(target_relation) + 1
            if rank == 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1
            if rank <= 5:
                hit5 += 1
            # if rank <= 10:
            #     hit10 += 1
            if analyze:
                print("Hit at rank:", rank)
        if analyze:
            print("_________________________________________________________________")
    print("Stat:")
    print(f"Accuracy: {accuracy}/{total} = {accuracy/total:.4f}")
    print(f"Accuracy (relation): {accuracy_rel}/{total} = {accuracy_rel/total:.4f}")
    # print(f"Accuracy (direction): {accuracy_dir}/{total} = {accuracy_dir/total:.4f}")
    print(f"Hit@1: {hit1}/{total} = {hit1/total:.4f}")
    print(f"Hit@3: {hit3}/{total} = {hit3/total:.4f}")
    print(f"Hit@5: {hit5}/{total} = {hit5/total:.4f}")
    # print(f"Hit@10: {hit10}/{total} = {hit10/total:.4f}")
    print(f"Corrupts: {corrupts}")

    # print("Accuracy:", accuracy / total)
    return corrupts


def cal_prob_propose_accuracy(kg_env: KGEnv, val, drop_edges, analyze=False, indices=None):
    avg_pred_cands, avg_prop_cands = 0, 0
    accuracy_prop = 0
    accuracy_pred = 0
    normalized_pred_precision = 0 
    total = 0
    corrupts = []
    for i, case in enumerate(val.to_dict(orient="records")):
        if indices and case['id'] not in indices:
            continue
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        if crucial_edges.empty or len(crucial_edges) > 1:
            continue
        total += 1
        
        target_direction = "outgoing" if crucial_edges["head"].item() == q_entity else "incoming"
        target_relation = crucial_edges["relation"].values[0] if not crucial_edges.empty else None
        # if target_relation != "nephew_of":
        #     continue
        if analyze:
            print(f"Question: {case['question']}")
            print(f"Target relation: {target_relation},\nTriple: ({crucial_edges['head'].item()}, {target_relation}, {crucial_edges['tail'].item()})")
        kg_env.records = []
        kg_env.explored_entities = set()
        kg_env.explored_triples = []
        kg_env.records.append({"i": 1, "thought": "helo"})
        _, candidate_relations = kg_env.prob_propose_action(q_entity, case["question"])
        avg_prop_cands += len(candidate_relations)
        # print(len(candidate_relations))
        # return
        # print(candidate_relations)
        corrupted = False
        matched_prop, matched_pred = False, False
        try:
            for i, item in enumerate(candidate_relations):
                pred_relation = item[1].relation
                pred_direction = item[1].direction
                pred_entities = [e for e, p in item[0]]
                avg_pred_cands += len(pred_entities)
                if analyze:
                    print(f"Predicted relation {i+1}:", pred_relation)
                    # print(f"Predicted direction {i+1}:", pred_direction)
                    if pred_direction == 'outgoing':
                        print(f"Predicted triple {i+1}: Triple: ({crucial_edges['head'].item()}, r, {crucial_edges['tail'].item()})" )
                    else:
                        print(f"Predicted triple {i+1}: Triple: ({crucial_edges['tail'].item()}, r, {crucial_edges['head'].item()})" )
                    print("predictions in candidate:", pred_relation in candidate_relations)
                # if pred_relation == target_relation:
                #     accuracy_rel += 1
                if pred_relation == target_relation and pred_direction == target_direction and matched_prop == False:
                    accuracy_prop += 1
                    matched_prop = True
                if hard_answer in pred_entities and matched_pred == False:
                    accuracy_pred += 1
                    normalized_pred_precision += 1 / len(pred_entities)
                    matched_pred = True
            # print(f"Matched: {matched}")
        except:
                    # Handle the case where q_entity is not in res
            corrupts.append(i)
            corrupted = True
                    # continue

        if analyze:
            print("_________________________________________________________________")
    print("Stat:")
    # print(f"Accuracy (relation): {accuracy_rel}/{total} = {accuracy_rel/total:.4f}")
    print(f"Accuracy (propose): {accuracy_prop}/{total} = {accuracy_prop/total:.4f}")
    print(f"Accuracy (predict): {accuracy_pred}/{total} = {accuracy_pred/total:.4f}")
    print(f"Average proposed candidates: {avg_prop_cands/total:.4f}")
    print(f"Average predicted candidates: {avg_pred_cands/total:.4f}")
    print(f"Normalized predicted precision: {normalized_pred_precision/total:.4f}")
    # print(f"Accuracy (direction): {accuracy_dir}/{total} = {accuracy_dir/total:.4f}")
    # print(f"Hit@10: {hit10}/{total} = {hit10/total:.4f}")
    print(f"Corrupts: {corrupts}")

    # print("Accuracy:", accuracy / total)
    return corrupts


def cal_predict_verify_accuracy(kg_env: KGEnv, val, drop_edges, analyze=False, indices=None, verify=True, reverse=False):
    accuracy = 0
    total = 0
    corrupts = []
    hit1, hit3, hit5, hit10 = 0, 0, 0, 0
    mrr = 0
    for i, case in enumerate(val.to_dict(orient="records")):
        if indices and case['id'] not in indices:
            continue
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        if analyze:
            print("Target triple:", crucial_edges.values)
        if crucial_edges.empty   or len(crucial_edges) > 1:
            continue
        total += 1
        if not reverse:
            target_direction = "outgoing" if crucial_edges["head"].item() == q_entity else "incoming"
        else: 
            target_direction = "outgoing" if crucial_edges["tail"].item() == q_entity else "incoming"
        
        target_relation = crucial_edges["relation"].values[0] if not crucial_edges.empty else None
        cands = kg_env.gnn.predict_topk(q_entity, target_relation, target_direction, k=10, known=False)
        # print("cands: ", cands)
        
        if hard_answer in cands:
            rank = cands.index(hard_answer) + 1
            mrr += 1 / rank
            if rank == 1:
                hit1 += 1
                # print(case['id'], "Hit@1", "relation:", target_relation, "direction:", target_direction, "hard_answer:", hard_answer)
            if rank <= 3:
                hit3 += 1
            if rank <= 5:
                hit5 += 1
            if rank <= 10:
                hit10 += 1
        if analyze:
            print("________________________________________________________________")
    print("Total valid cases:", total)
    print("cal_predict_accuracy:")
    print(f"MRR: {mrr/total:.4f}")
    print(f"HIT@1: {hit1}/{total} = {hit1/total:.4f}")
    print(f"HIT@3: {hit3}/{total} = {hit3/total:.4f}")
    print(f"HIT@5: {hit5}/{total} = {hit5/total:.4f}")
    print(f"HIT@10: {hit10}/{total} = {hit10/total:.4f}")

    if verify:
        print(f"Accuracy after verification: {accuracy}/{hit10} = {accuracy/hit10:.4f}")


def test_propose_predict(kg_env: KGEnv, val, drop_edges, analyze=False):
    corrupts = []
    accuracy, normalized_precision = 0, 0
    total = 0
    for i, case in enumerate(val.to_dict(orient="records")):
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        if crucial_edges.empty or len(crucial_edges) > 1:
                    continue
        total += 1
        target_direction = "outgoing" if crucial_edges["head"].item() == q_entity else "incoming"
        target_relation = crucial_edges["relation"].values[0] if not crucial_edges.empty else None
        # if target_relation != "nephew_of":
        #     continue
        if analyze:
            print(f"Question: {case['question']}")
            print(f"Target relation: {target_relation},\nTriple: ({crucial_edges['head'].item()}, {target_relation}, {crucial_edges['tail'].item()})")
        kg_env.records = []
        kg_env.records.append({"i": 1, "thought": "helo"})
        _, candidate_relations, _, _, _, _ = kg_env.propose_action(case["question"])
        try:
            predicted_cands=[]
            for i, item in enumerate(candidate_relations):
                pred_relation = item.get("relation")
                pred_direction = item.get("direction")
                cands = kg_env.gnn.predict_topk(q_entity, pred_relation, pred_direction, k=3, known=False)
                predicted_cands.extend(cands)
            if hard_answer in predicted_cands:
                accuracy += 1
                normalized_precision += 1 / len(predicted_cands)
        except:
            corrupts.append(case)
    accuracy /= total
    normalized_precision /= total
    print("Accuracy: ", accuracy)
    print("Normalized precison: ", normalized_precision)
    print("Total valid cases: ", total)
    return corrupts


def test_full_generate(
    kg_env: KGEnv, val, drop_edges, analyze=False, indices=None, type_check=True
):
    accuracy, recall = 0, 0
    corrupts = []
    for i, case in enumerate(val.to_dict(orient="records")):
        if indices and case['id'] not in indices:
            continue
        q_entity = literal_eval(case["q_entity"])[0]
        hard_answer = literal_eval(case["hard_answer"])[0]
        crucial_edges = get_edges(drop_edges, q_entity, hard_answer)
        question = case['question']
        candidates = None
        try: 
            candidates = kg_env.propose_action(question, type_check=type_check)
        except Exception as e:
            corrupts.append(case)
        # print(candidates, log)
        if candidates and hard_answer in candidates:
            accuracy += 1/len(candidates)
    print(f"Normalized Precision: {accuracy/len(val)}")
    return corrupts
