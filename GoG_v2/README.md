Tạo một folder lớn cho project, `cd` folder đó.
+ clone thành folder tên GoG:
```
git clone https://github.com/Neil-275/GoG_enhanced GoG
```
+ Load folder brink_dataset vào  
Cấu trúc folder:
```
folder
|- brink_dataset
|- GoG
```
+ Build environment  

## Idea

`GoG_v2` uses a multi-path reasoning loop instead of committing to a single chain too early.

The main idea is:

- collect all known triples from the current question history
- build candidate reasoning paths from those triples
- convert each path into an inversion-aware rule view
- expand every plausible path before accepting a final answer
- finish with all supported answer candidates when the evidence is strong enough

Why this helps:

- it reduces early overconfidence on one path
- it keeps alternative branches alive when there are multiple valid routes
- it handles reversed or inverse relations more naturally
- it makes the final answer set more complete when the graph supports more than one candidate

Reasoning stages:

1. `Search`
   - fetches one-hop triples for topic entities or newly discovered entities
   - adds the fetched triples to the known triple pool
2. `Path`
   - gathers candidate entity paths from the known triples
   - ranks them so the controller can see which branches are most promising
3. `Rule`
   - rewrites each path into a relation-level rule
   - marks inverse traversal when the path runs backward through a known triple
4. `Generate`
   - proposes new triples from the current thought and known context
   - may reveal missing links between intermediate entities
5. `Finish`
   - returns one or more answer candidates
   - if the evidence is still weak, the controller expands more paths before accepting it

Examples:

```text
Question: Who are the people connected to 410 through the dating nodes?
Topic Entity: [410]
Known triples:
410, celebrity.dated, 801
801, dated.participant, 901
410, celebrity.dated, 802
802, dated.participant, 902
410, celebrity.dated, 803
803, dated.participant, 903

Path candidates:
410 -> 801 -> 901
410 -> 802 -> 902
410 -> 803 -> 903

Rule candidates:
inverse(celebrity.dated) -> inverse(dated.participant)
inverse(celebrity.dated) -> inverse(dated.participant)
inverse(celebrity.dated) -> inverse(dated.participant)

Finish:
Finish[901 | 902 | 903]
```

```text
Question: What state is the education institution that has sports team 561 in?
Topic Entity: [561]

Search[561]
Observation:
561, school_sports_team.school, 125

Path:
561 -> 125 -> 44

Rule:
school_sports_team.school -> location.located_in

Finish:
Finish[44]
```

Practical note:

- the first iteration skips path/rule gathering if there are no known triples yet
- path/rule records appear only after search or generation has produced evidence
- the no-KG mode stays direct and can still finish with a candidate list

TODO: chuyển `Path` và `Rule` thành `reasoning chains` để prompt hiểu cách kết hợp các triples một cách logic theo các luật.

+ Chạy:
```
python -m GoG_v2.GoG --dataset {dataset_path}
```
