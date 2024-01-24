import os

from src.datasets.base import MultipleEnvironmentImageFolder


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = sorted(["A", "C", "P", "R"])
    NUM_CLASSES = 65
    OVERLAP_CONFIG = {
        "0": [list(range(0, 22)), list(range(22, 44)), list(range(44, 65))],
        "low": [list(range(0, 30)), list(range(14, 44)), list(range(35, 65))],  # 25/65
        "high": [list(range(0, 38)), list(range(5, 44)), list(range(27, 65))],  # 50/65
        "low_linked_only": [
            list(range(0, 14)),
            list(range(30, 35)),
            list(range(44, 65)),
        ],
        "high_linked_only": [list(range(0, 5)), [], list(range(44, 65))],
        "100": [list(range(65)), list(range(65)), list(range(65))],
    }

    def __init__(
        self,
        root,
        test_envs,
        hparams,
        overlap_type,
        overlap_seed=None,
    ):
        self.dir = os.path.join(root, "office_home/")
        self._num_source_domains = 3
        self._num_classes = 65

        super().__init__(
            self.dir,
            test_envs,
            hparams["data_augmentation"],
            hparams,
            OfficeHome.OVERLAP_CONFIG[overlap_type],
        )
