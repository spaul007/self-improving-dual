from agents.specialists._base import SpecialistAgent
from agents.specialists.lock_contention.prompt import LABEL, SYSTEM_PROMPT_TEMPLATE


class LockContentionSpecialist(SpecialistAgent):
    LABEL = LABEL
    SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE
