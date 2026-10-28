# M5: Creative director — findings (2026-10-27)

**What it is:** `autodj direct plan.json` (or `set --director`). One Claude Opus 5.5 request per set,
with structured JSON output and `fallbacks: "default"` (server-side refusal fallback).

## Design
- **Claude chooses; code guarantees.** For each transition, the planner builds a menu of up to 6
  options: the plan's current pick, the best of each transition type, then the next best overall.
  Each option carries its bars, where A leaves and B enters (section and energy), what B reaches
  after the transition, the planner's score, and the planner's reasons. Claude also sees a
  per-track section summary (energy, and vocals if stems were run).
- Claude returns `{title, narrative, choices[{transition, option_id, why}]}`, validated with
  pydantic against the same JSON schema sent as `output_config.format`.
- **Repairs in code:** a missing or unknown option keeps the planner's pick. A pick that would start
  leaving a track before its incoming transition has finished is replaced by the best valid option.
  Every repair is listed in the output and stored in `plan.director.repairs`.
- The directed plan is written next to the original (`*.directed.json`) for A/B listening. The
  director's reason is the first note on each transition, and the viewer shows the title and narrative.

## First run (3 tracks: Ur Alive → dashstar → how2fly)
- 5,200 input + 845 output tokens, **about $0.04**.
- Kept the planner's filter sweep (Ur Alive's late break into dashstar's first drop, "keeps the vocal
  overlap brief and masks the key gap").
- **Changed** transition 2: echo out of dashstar *straight into how2fly's first drop* (bar 17) instead
  of its intro, making it "the set's big escalation moment". That's a choice the rule-based planner
  scored lower: it penalizes skipping 12% of B and doesn't model set-level drama.

## Limits / next
- The order is still fixed by the planner. The director could also choose between the top few orders.
- No eval yet. Judging needs listening: render both versions and compare.
- Cost scales with tracks × options (~250 tokens per option); 20 tracks ≈ 30k input tokens.
