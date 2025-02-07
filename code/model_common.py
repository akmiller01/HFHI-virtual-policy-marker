from pydantic import BaseModel
from typing import Literal

SYSTEM_PROMPT = (
    "You are a highly precise assistant that classifies development and humanitarian activity titles and descriptions based strictly on explicit textual evidence.\n"
    "You are looking for matches within an expanded definition of the housing sector, which includes the following classes:\n"
    "{}\n"
    "Definitions & Criteria for Classification\n"
    "Each classification must be based only on explicit descriptions in the text. Do not infer or assume relevance beyond what is clearly stated. The definitions are:\n"
    "{}\n"
    "Strict Classification Rules\n"
    "1. No assumptions or indirect reasoning. Do not assume a connection to housing unless it is explicitly described.\n"
    "2. Context matters only when directly related to housing. Broader urban or social development programs do not qualify unless housing is specifically mentioned.\n"
    "3. Classifications must be justified. If a class is assigned, the reason must reference specific text that matches the definition.\n"
    "Your response must be in JSON format:\n"
    "{{\n"
    "'thoughts': 'Explain your reasoning, explicitly referencing the text that supports each classification.',\n"
    "'classifications': ['List only the explicitly matched classes']\n"
    "}}"
)
DEFINITIONS = {
    "Housing": "provision of housing, provision of shelter in emergencies, upgrading inadequate housing, provision of basic services in inadequate housing, construction of housing, urban development, housing policy, technical assistance for housing, or finance for housing",
    "Homelessness": "tents for the homeless, encampments for the homeless, or homeless shelters",
    "Transitional": "emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "Incremental": "housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration as it relates to housing",
    "Social": "community land trusts, cooperative housing, or public housing",
    "Market": "home-rental, mortgages, rent-to-own housing, or market-rate housing",
    "Urban": "activities in specific urban locations",
    "Rural": "activities in specific rural locations"
}
SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: The text must explicitly describe {value}' for key, value in DEFINITIONS.items()]),
)

class ThoughtfulClassification(BaseModel):
    thoughts: str
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]
