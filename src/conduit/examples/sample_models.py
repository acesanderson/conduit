from pydantic import BaseModel

class MyTestPydanticContent(BaseModel):
    value: str
    number: int

class PydanticTestAnimal(BaseModel):
    name: str
    species: str
    age: int
    habitat: str

class PydanticTestFrog(BaseModel):
    species: str
    name: str
    legs: int
    color: str
