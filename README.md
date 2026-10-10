## setup
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python -m scripts.check_env


## Run

    < run demo >
    . .venv/bin/activate
    python -m jacket.fashion
    python -m jacket.clip

    < people detector: body (pose), head, or both fused >
    python -m ren.ui --detector both
    python -m jacket.fashion --detector head

    < crops >
    python -m jacket.fashion --save-crops

    < scoring >
    python -m jacket.eval

    < checks >
    python -m scripts.check_env
    python -m pytest -q


