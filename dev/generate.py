from conduit.batch import (
    AsyncConduit,
    ModelAsync,
    Prompt,
    Response,
    Verbosity,
    ConduitCache,
)
from rich.console import Console
from pathlib import Path
from asyncio import Semaphore

CONSOLE = Console()
CACHE = ConduitCache("conduit")
PREFERRED_MODEL = "gemini3"
ModelAsync.console = CONSOLE
ModelAsync.conduit_cache = CACHE
ESSAY_DIR = Path(__file__).parent / "essays"
PROMPT_STR = """
You are a talented writer. You will generate well structure and highly detailed essays of a given length.
The purpose of these essays is to train and evaluate a text summarization model.

Please write an essay of the following length:
<length>
{{length}}
</length>
""".strip()

# generate meaningful lengths for assessing summary quality by LLMs (the hundreds, the thousands, 8192 is at the high end, we want to go past that as well, all the way up to 32k
token_lengths = [
    100,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    15000,
]
# Convert to word lengths (assuming 1 token ~ 0.75 words)
lengths = [int(tl / 0.75) for tl in token_lengths]


def generate_essays() -> list[str]:
    prompt = Prompt(PROMPT_STR)
    model = ModelAsync(PREFERRED_MODEL)
    conduit = AsyncConduit(model=model, prompt=prompt)
    input_variables_list = [{"length": str(length)} for length in lengths]
    semaphore = Semaphore(5)
    responses = conduit.run(
        input_variables_list=input_variables_list,
        verbose=Verbosity.PROGRESS,
        semaphore=semaphore,
    )
    assert all(isinstance(r, Response) for r in responses)
    essays = [str(r.content) for r in responses]
    return essays


if __name__ == "__main__":
    model = ModelAsync("gpt")
    essays = generate_essays()
    ESSAY_DIR.mkdir(exist_ok=True)
    for length, essay in zip(lengths, essays):
        essay_path = ESSAY_DIR / f"essay_{length}_words.txt"
        with open(essay_path, "w", encoding="utf-8") as f:
            f.write(essay)
        token_length = model.tokenize(essay)
        CONSOLE.log(f"Saved essay of {token_length} tokens to {essay_path}")
