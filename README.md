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

    < UI: data page >
    python -m ren.ui
    drop a photo anywhere on the window: everyone in it is labelled on the live page
    torso crops panel, "save crops": saves what is shown to data/raw/
    data page: drag crops onto warm / light / discard (data/eval/warm, data/eval/light, data/discarded)

    < crops >
    python -m jacket.fashion --save-crops

    < scoring >
    python -m jacket.eval

    < checks >
    python -m scripts.check_env
    python -m pytest -q


