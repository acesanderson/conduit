import pytest
from Chain.chain.chain import Chain
from Chain.model.model import Model
from Chain.prompt.prompt import Prompt
from Chain.parser.parser import Parser
from Chain.compose.chainml import ChainML
from pathlib import Path
import json

dir_path = Path(__file__).parent
compose_files = list(dir_path.glob("*.json"))


@pytest.mark.parametrize("compose_file", compose_files)
def test_compose_files(compose_file):
    for index, compose_file in enumerate(compose_files):
        print(f"{index+1} of {len(compose_files)}")
        compose_string = compose_file.read_text()
        compose_dict = json.loads(compose_string)
        chainml_object = ChainML(**compose_dict)
        print(chainml_object)
