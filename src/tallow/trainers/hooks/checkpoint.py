# import logging
# import os
# import re

# from ..trainer import Trainer
# from .hook import Hook
# from .model_save import DiskManager

# logger = logging.getLogger(__name__)

# CHECKPOINT_RE = re.compile(r"(.*[/\\])?e(\d+)_b(\d+)\.ckpt\.pt")
# CHECKPOINT_TEMPLATE = "e{epoch}_b{batch}.ckpt.pt"


# def create_time(state_file: str):
#     return os.stat(state_file).st_ctime_ns


# class CheckpointManager(DiskManager):
#     """
#     - root
#     |- e10_b30.ckpt.pt
#     |- model_config.json

#     """

#     def __init__(self, save_folder_path: str, num_saves: int = 1):
#         super().__init__(save_folder_path, num_saves, sort_key=create_time)

#     def _is_save_file(self, file: str) -> bool:
#         return CHECKPOINT_RE.match(file)

#     def _get_save_file(self, trainer: Trainer) -> str:
#         state_dict = trainer.state_dict()
#         num_epochs = state_dict["num_epochs"]
#         num_batches = state_dict["num_batches"]
#         checkpoint_file = CHECKPOINT_TEMPLATE.format(
#             epoch=num_epochs, batch=num_batches
#         )
#         return checkpoint_file

#     @property
#     def latest_checkpoint(self):
#         return self.saved_models[-1]


# class CheckpointHook(Hook):
#     def __init__(self, save_folder_path: str, reload_on_begin: bool) -> None:
#         super().__init__()
#         self._saver = CheckpointManager(save_folder_path)
#         self._reload_on_begin = reload_on_begin

#     @property
#     def latest_checkpoint(self):
#         return self._saver.latest_checkpoint

#     def on_train_begin(self, trainer: Trainer):
#         if self._reload_on_begin:
#             if self._saver.saved_models:
#                 logger.info(
#                     "Restore checkpoint from %s", self.latest_checkpoint
#                 )
#                 self._saver.load(trainer)

#     def on_epoch_end(self, trainer: Trainer):
#         ckpt_path = self._saver.save_or_substitute(trainer)
#         logger.info("Save checkpoint to %s", ckpt_path)
