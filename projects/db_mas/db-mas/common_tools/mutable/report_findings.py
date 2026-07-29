"""report_findings: shared by all 5 specialist agents -- their common terminal action.

Mutable: this tool's schema/description is fair game for an automated
prompt/tool optimizer to rewrite. It's an internal coordination-protocol tool
(how a specialist hands its result to the Coordinator), not part of the
benchmark's actual environment contract, so tuning it can't change what's
being tested -- only how well the agents perform at it.
"""

REPORT_FINDINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "report_findings",
        "description": "Submit your final structured investigation result for your assigned label. Call this exactly once, after you have run at least one query.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "The candidate root-cause label you investigated.",
                },
                "supports_label": {
                    "type": "boolean",
                    "description": "Whether the evidence you gathered supports this label as a true root cause.",
                },
                "evidence": {
                    "type": "string",
                    "description": "Concrete evidence cited: specific query results, numbers, table/index names.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence (0.0-1.0) that this label is a true root cause.",
                },
            },
            "required": ["label", "supports_label", "evidence", "confidence"],
            "additionalProperties": False,
        },
    },
}
