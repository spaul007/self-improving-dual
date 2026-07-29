"""Tools specific to the Coordinator agent -- not shared by any other agent,
so they live here rather than in common_tools/. (The Coordinator also uses
`common_tools.immutable.query_db.QUERY_DB_TOOL` directly -- that one *is*
common, since every specialist uses it too; it's just imported from there
instead of redefined here.)

Both tools below are still mutable in the harness-optimizer sense (fair game
to retune, since neither is part of the benchmark's actual environment
contract) -- they're just scoped to this one agent instead of being common
across several.
"""

ASK_SPECIALIST_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_specialist",
        "description": "Ask exactly one specific specialist agent one clarifying follow-up question. You may only do this once in total for this task.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent_id of the specialist to ask.",
                },
                "question": {"type": "string"},
            },
            "required": ["agent_id", "question"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit your final decision on the root cause(s) of the database performance issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_root_causes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exactly number_of_labels_pred labels, chosen from the candidate label list.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of why these labels were chosen, grounded in the specialist evidence.",
                },
            },
            "required": ["predicted_root_causes", "reasoning"],
            "additionalProperties": False,
        },
    },
}
