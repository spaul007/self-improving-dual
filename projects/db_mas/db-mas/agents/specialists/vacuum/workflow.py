from agents.specialists._base import SpecialistAgent
from agents.specialists.vacuum.prompt import LABEL, SYSTEM_PROMPT_TEMPLATE


class VacuumSpecialist(SpecialistAgent):
    LABEL = LABEL
    SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE
