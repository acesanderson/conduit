from Chain.chain.chain import Chain
from Chain.model.model import Model
from Chain.prompt.prompt import Prompt
from Chain.compose.chainml import ChainML
from pathlib import Path
from pydantic import BaseModel
import json

dir_path = Path(__file__).parent
compose_file = dir_path / "chain-compose-simple.json"

compose_string = compose_file.read_text()
compose_dict = json.loads(compose_string)
chainml = ChainML(**compose_dict)

workflow = chainml.workflow
outputs = workflow.outputs
steps = workflow.steps

"""
{
    "workflow": {
        "name": "Simple Query",
        "description": "Basic single-step LLM query with no dependencies",
        "inputs": {"topic": {"type": "string", "description": "Topic to explain"}},
        "outputs": {
            "explanation": {
                "from": "explain_topic.output",
                "description": "Simple explanation of the topic",
            }
        },
        "steps": {
            "explain_topic": {
                "model": "gpt-4o-mini",
                "description": "Generate a simple explanation of the given topic",
                "prompt": "Explain {{inputs.topic}} in simple terms that anyone can understand.",
                "depends_on": [],
            }
        },
    }
}
"""


class Inputs(BaseModel): ...


class Outputs(BaseModel): ...


class Step(BaseModel):
    name: str
    model: str
    description: str
    prompt: str
    depends_on: list


class Workflow(BaseModel):
    name: str
    description: str
    inputs: dict
    outputs: dict
    steps: list[Step]


class ChainML(BaseModel):
    workflow: Workflow


def simple_query(workflow: BaseModel, input: str):
    # docstring = f"""
    # Name: {workflow.name}
    # Description: {workflow.description}
    # Inputs: topic ({workflow.type}) - {workflow.description}
    # """

    def explain_topic(step: BaseModel):
        model = step.model
        description = step.description
        prompt = step.prompt
        depends_on = step.depends_on

        model_obj = Model(model)
        prompt_obj = Prompt(prompt)
        chain = Chain(model=model_obj, prompt=prompt_obj)
        response = chain.run(input)
        return response
