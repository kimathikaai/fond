from src.networks.fond import FOND, FOND_N, FOND_NC
from src.networks.base import ERM

ALGORITHMS = {
    "ERM": ERM,
    "FOND": FOND,
    "FOND_NC": FOND_NC,
    "FOND_N": FOND_N,
}
