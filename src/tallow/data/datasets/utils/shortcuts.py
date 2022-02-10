from .. import batch


class ShortcutsMixin:
    def batch(self, batch_size: int, collate_fn=batch.default_collate):
        from ..iterable import BatchDataset

        return BatchDataset(self, batch_size, collate_fn)

    def shuffle(self, shuffle_size: int = None):
        from ..iterable import BufferShuffledDataset

        shuffle_size = shuffle_size or len(self)
        return BufferShuffledDataset(self, shuffle_size)

    def prefetch(self, fetch_size: int):
        from ..iterable import PrefetchDataset

        return PrefetchDataset(self, fetch_size=fetch_size)

    def map(self, transform_fn, *args, **kwargs):
        from ..iterable import TransformDataset

        return TransformDataset(
            self, transform_fn=transform_fn, *args, **kwargs
        )

    def to_tensor(self, pin_memory: bool = True):
        from ..iterable import ToTensorDataset

        return ToTensorDataset(self, pin_memory=pin_memory)

    def with_length(self, length: int):
        from ..iterable import ReSizedDataset

        return ReSizedDataset(self, length)

    def repeat(self, n: int = None):
        from ..iterable import RepeatDataset

        return RepeatDataset(self, n)
