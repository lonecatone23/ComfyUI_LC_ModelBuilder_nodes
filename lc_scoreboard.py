"""
LC Scoreboard
-------------
Head-to-head scoreboard for A / B tests between #1 and #2. Three buttons on the node (web/lc_scoreboard.js):
#1 Win, Tie, #2 Win. A win for one side is a loss for the other, so one press scores both. Reset all clears it. The node shows who leads and by how much. The three counts are ordinary widgets, so
they save with the workflow; running the graph is not needed, the outputs are there to wire or log the score.
"""

from __future__ import annotations


def summary(wins_1: int, wins_2: int, ties: int) -> str:
    total = wins_1 + wins_2 + ties
    if total == 0:
        return "No results yet"
    p1, p2, pt = (round(100 * x / total) for x in (wins_1, wins_2, ties))
    lead = "Even" if wins_1 == wins_2 else (f"#1 leads ({p1}%)" if wins_1 > wins_2 else f"#2 leads ({p2}%)")
    return f"{lead}: #1 {wins_1}-{wins_2}-{ties} ({p1}%), #2 {wins_2}-{wins_1}-{ties} ({p2}%), ties {pt}% of {total}"


class LCScoreboard:
    @classmethod
    def INPUT_TYPES(cls):
        count = {"default": 0, "min": 0, "max": 100000, "step": 1}
        return {
            "required": {
                "wins_1": ("INT", {**count, "tooltip": "#1's wins (= #2's losses). Use the buttons; type here to fix a mistake."}),
                "ties": ("INT", {**count, "tooltip": "Ties. Use the buttons; type here to fix a mistake."}),
                "wins_2": ("INT", {**count, "tooltip": "#2's wins (= #1's losses). Use the buttons; type here to fix a mistake."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("wins_1", "ties", "wins_2", "summary")
    FUNCTION = "score"
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Head-to-head scoreboard for A / B tests. Press #1 Win, Tie or #2 Win; a win for one side is a "
                   "loss for the other. Shows who leads and by how much. The score saves with the workflow.")

    def score(self, wins_1, ties, wins_2):
        return (wins_1, ties, wins_2, summary(wins_1, wins_2, ties))


NODE_CLASS_MAPPINGS = {"LCScoreboard": LCScoreboard}
NODE_DISPLAY_NAME_MAPPINGS = {"LCScoreboard": "LC Scoreboard 🏆"}
