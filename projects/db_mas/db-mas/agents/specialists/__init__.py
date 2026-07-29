"""The 5 statically-defined specialist roles, one per real anomaly type used in
the benchmark. Every task instantiates exactly these 5 -- never a variable set."""
from agents.specialists.fetch_large_data.workflow import FetchLargeDataSpecialist
from agents.specialists.insert_large_data.workflow import InsertLargeDataSpecialist
from agents.specialists.lock_contention.workflow import LockContentionSpecialist
from agents.specialists.redundant_index.workflow import RedundantIndexSpecialist
from agents.specialists.vacuum.workflow import VacuumSpecialist

SPECIALIST_CLASSES = [
    InsertLargeDataSpecialist,
    LockContentionSpecialist,
    VacuumSpecialist,
    RedundantIndexSpecialist,
    FetchLargeDataSpecialist,
]


def build_specialists(task_content: str):
    return [
        cls(agent_id=cls.LABEL.lower(), task_content=task_content)
        for cls in SPECIALIST_CLASSES
    ]
