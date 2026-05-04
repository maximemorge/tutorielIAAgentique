# utils.py

def debug_print(label: str, text: str, width: int = 100) -> None:
    """
    Displays a formatted block of text while preserving line breaks.
    Each line is wrapped at `width` characters if it exceeds that length.
    """
    border = "─" * width
    print(f"┌{border}┐")
    print(f"│ {label}")
    print(f"├{border}┤")
    for line in text.splitlines():
        while len(line) > width - 2:
            print(f"│ {line[:width - 2]}")
            line = line[width - 2:]
        print(f"│ {line}")
    print(f"└{border}┘")
