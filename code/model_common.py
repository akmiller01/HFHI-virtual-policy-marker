from pydantic import BaseModel
from typing import Literal

SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies development and humanitarian activity titles and descriptions.\n"
    "You are looking for matches with an expanded definition of the housing sector that encompasses a continuum defined by the possible classes below.\n"
    "The possible classes you are looking for are:\n"
    "{}\n"
    "The definitions of each possible class are:\n"
    "{}\n"
    "Think carefully and do not jump to conclusions: ground your response on the given text.\n"
    "Respond in JSON format, first giving your complete thoughts about all the possible matches with the above classes and definitions in the 'thoughts' key "
    "and then listing all of the classes that match in the 'classifications' key."
)
DEFINITIONS = {
    "Housing": "explicitly describes provision of housing, provision of shelter in emergencies, upgrading inadequate housing, provision of basic services in inadequate housing, construction of housing, urban development, housing policy, technical assistance for housing, or finance for housing",
    "Homelessness": "explicitly describes tents for the homeless, encampments for the homeless, or homeless shelters",
    "Transitional": "explicitly describes emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "Incremental": "explicitly describes housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "Social": "explicitly describes community land trusts, cooperative housing, or public housing",
    "Market": "explicitly describes home-rental, mortgages, rent-to-own housing, or market-rate housing",
    "Urban": "explicitly describes activities in specific urban locations",
    "Rural": "explicitly describes activities in specific rural locations"
}
SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: when the text {value}' for key, value in DEFINITIONS.items()]),
)

class ThoughtfulClassification(BaseModel):
    thoughts: str
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]
