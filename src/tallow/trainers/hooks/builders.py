from .checkpoint import CheckpointHook
from .hook import Hook, HookManager


class HooksBuilder:
    def __init__(self) -> None:
        self._hooks = []

    def _new_hook(self, hook: Hook):
        self._hooks.append(hook)
        return self

    def build_manager(self):
        return HookManager(self._hooks)

    def checkpoint(self, save_folder_path: str, reload_on_begin: bool):
        new_hook = CheckpointHook(
            save_folder_path, reload_on_begin=reload_on_begin
        )
        return self._new_hook(new_hook)

    pass
