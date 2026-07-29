from agents.specialists._base import SpecialistAgent
from agents.specialists.redundant_index.prompt import LABEL, SYSTEM_PROMPT_TEMPLATE


class RedundantIndexSpecialist(SpecialistAgent):
    LABEL = LABEL
    SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE
