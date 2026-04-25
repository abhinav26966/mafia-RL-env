# 90-second video shooting script

Per MASTER_PLAN §13. Total runtime target: 1:50–2:00. Voiceover-driven.

| Time      | Visual                                                                           | Voiceover (≤ ~25 wpm at this length)                                                                                          |
| --------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 0:00–0:15 | Split-screen of two Day-1 werewolf speeches. Left: untrained Qwen3. Right: ours. | "Two LLMs were asked to pretend to be a villager while secretly being the werewolf. The one on the right learned how to lie." |
| 0:15–0:35 | Slide: 5-seat diagram, role distribution, phase cycle.                           | "DECEIT is an OpenEnv environment for hidden-role social deduction. Five players, one werewolf, one seer, three villagers."   |
| 0:35–1:05 | 4 reward curves overlaid; baseline-vs-trained win-rate bars.                     | "We trained with GRPO using four independent reward rubrics: outcome, calibration, survival, format. All four moved up."      |
| 1:05–1:35 | Auto-scroll a full game transcript. Pause on the moment of redirected suspicion. | "Watch the trained werewolf redirect suspicion in Day 2. The model is constructing a case — that's theory of mind."           |
| 1:35–1:55 | Title card: "Why this matters" + paper citation (OMAR, arxiv 2602.03109).        | "We now have a reproducible OpenEnv for theory-of-mind training. Anyone can run it, train on it, extend it."                  |
| 1:55–2:00 | Outro: GitHub URL, HF Space URL, team names.                                     | "Code, env, video — all linked below."                                                                                        |

## Production notes

- Record voice locally with a USB mic; loud-norm to -16 LUFS.
- Use OBS to capture transcript scroll at 24fps.
- Plot text must be readable at 720p — set figure DPI 150 in `compare_runs.py`.
- Final cut in iMovie / DaVinci Resolve free. Export 1080p, MP4, ≤ 50 MB.
- Upload **unlisted** to YouTube; link from project README only.
