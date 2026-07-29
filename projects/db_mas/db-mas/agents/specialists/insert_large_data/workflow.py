from agents.specialists._base import SpecialistAgent
from agents.specialists.insert_large_data.prompt import LABEL, SYSTEM_PROMPT_TEMPLATE


class InsertLargeDataSpecialist(SpecialistAgent):
    LABEL = LABEL
    SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE
