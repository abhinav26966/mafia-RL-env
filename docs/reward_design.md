# Reward design + anti-hacking

## Four rubrics, registered independently

We use one DOMINANT outcome signal so GRPO has a clean ranking signal, plus
three SHAPING signals that are bounded and hard to game in isolation.

| Rubric      | Weight | Range | Signal kind     |
| ----------- | -----: | ----- | --------------- |
| outcome     | 0.50  | {0,1} | terminal binary |
| calibration | 0.25  | [0,1] | dense (per-suspicion accuracy) |
| survival    | 0.15  | [0,1] | dense (fraction of days survived) |
| format      | 0.10  | [0,1] | dense (penalises violations) |

GRPO sees `0.50·outcome + 0.25·calibration + 0.15·survival + 0.10·format`,
but Trackio shows each on its own curve.

## Anti-hacking checks

These are enforced inside the engine + parser, not the reward function. The
model can't accumulate reward by exploiting any of them.

1. **Role-leak filter.** Speech containing the literal `"i am the werewolf"`,
   `"i am the seer"` (case-insensitive substring) — counts a format
   violation. Stops the model from finding shortcuts that out itself
   trivially during exploration.
2. **Vote-self penalty.** Voting your own seat — format violation AND
   default-vote a random alive non-self seat. Stops survival-gaming via
   self-elimination at strategic moments.
3. **Repeated-speech penalty.** Speech identical (case-insensitive, stripped)
   to a prior speech by the same player — format violation. Stops
   degenerate "I suspect Player 3" loops.
4. **Length floor.** Speech under 5 tokens — format violation. Stops the
   model from gaming `format_reward` by saying nothing.
5. **Calibration uses ground truth from `/state`, not from the model's own
   claims.** A werewolf model that accuses *everyone* of being a werewolf
   gets calibration 0, since none of those accusations match
   ground-truth-opposing-faction.

## What "obviously improving" looks like

The README should claim improvement only when MULTIPLE of these move:

- Win rate when assigned WEREWOLF rises from ~30 % (random impostor caught
  easily) to ~55 %+.
- Average format violations per game drops from ~3 to <0.5.
- Calibration when assigned VILLAGER rises from ~0.5 (random suspicion) to
  ~0.7+.
- Speeches qualitatively reference earlier speeches by player ID. Show this
  via highlighted transcript snippets in the README.

## Reward weighting — when to retune

After the 20-step debug run on T4, look at the *scale* of each rubric's
gradient contribution. If `format_reward` swamps everything (it's the
easiest signal), drop its weight to 0.05 and re-balance.

If `calibration_reward` is flat at 0.5 (model never accuses anyone), the
prompt is missing instructions to make speeches *specific*. Patch the
system prompt before retraining.

## Things judges will probe

- Can a degenerate "always vote Player 4" agent get high reward? No — it
  loses to villagers about half the time and never gets calibration credit.
- Can a "say nothing" agent get high reward? No — length-floor format
  violation drops format_reward to 0; survival rises only marginally.
- Does the seer get a free reward channel? No — the seer's private check
  is *information*, not action; calibration is rewarded on stated
  suspicions in speeches, which everyone has equal opportunity to make.
