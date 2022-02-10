import io
import tarfile
from typing import Iterable, Mapping

import tqdm

from ..utils import example
from . import handlers


def create_tarfile_dataset(
    examples: Iterable[example.Example],
    tar_file: str,
    fname_template: str = "{:d}.tar",
    start: int = 0,
    additional_handlers: Mapping[str, handlers.DumpsFn] = None,
) -> int:

    additional_handlers = additional_handlers or {}

    compression = tar_file.endswith("gz")
    tarmode = "w|gz" if compression else "w|"
    tarstream = tarfile.open(tar_file, tarmode)

    dump_handlers = handlers.get_default_dumps()
    dump_handlers.update(additional_handlers)

    total = 0
    example_tqdm = tqdm.tqdm(examples)
    ex: example.Example
    for index, ex in enumerate(example_tqdm, start=start):
        example_tqdm.set_description(f"Saving {index:d}")

        for key, value in ex.items():
            _, ext = key.split(".", maxsplit=1)

            handler = dump_handlers[ext]
            data = handler(value)

            ti = tarfile.TarInfo(fname_template.format(index=index) + "/" + key)
            ti.size = len(data)
            tarstream.addfile(ti, io.BytesIO(data))
            total += ti.size

    tarstream.close()
    return total
