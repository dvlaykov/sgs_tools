from pathlib import Path


def add_extension(input: str | Path, extension: str) -> Path:
    assert extension.startswith(".")
    output = Path(input)
    if output.suffix != extension:
        output = Path(str(output) + extension)
    return output
