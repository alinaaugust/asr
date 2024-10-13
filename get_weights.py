import os
from pathlib import Path

import gdown


def main():
    saved_dir = Path(__file__).absolute().resolve().parent
    saved_dir = saved_dir / "saved"
    saved_dir.mkdir(exist_ok=True, parents=True)

    gdown.download(id="15pn3OgJAjncQJjAFgHyLqj-rN-_sBI7r")  # model weights
    gdown.download(id="1G4HUjT_l9EvRkGuVMvkK7VOQFUxhy6Ca")  # lm model

    os.rename("model_best.pth", str(saved_dir) + "/model_best.pth")
    os.rename(
        "lowercase_3-gram.pruned.1e-7.arpa",
        str(saved_dir) + "/lowercase_3-gram.pruned.1e-7.arpa",
    )


if __name__ == "__main__":
    main()
