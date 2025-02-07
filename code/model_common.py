from pydantic import BaseModel
from typing import Literal

SYSTEM_PROMPT = (
    "You are a highly precise assistant that classifies development and humanitarian activity titles and descriptions based on explicit textual evidence and strong contextual relevance.\n"
    "You are looking for matches within an expanded definition of the housing sector, which includes the following classes:\n"
    "{}\n"
    "Definitions & Criteria for Classification\n"
    "Each classification should be based primarily on explicit descriptions in the text. However, if a strong contextual clue clearly suggests alignment with a definition, it may be considered—but only if it is directly relevant to the topic of housing or shelter.\n"
    "{}\n"
    "Classification Rules\n"
    "1. Prioritize explicit mentions, but allow strong contextual clues when clearly housing-related. If a classification is based on context rather than explicit wording, explain why the connection is valid.\n"
    "2. Avoid indirect social connections. Broader urban or social development projects should not be classified unless housing or shelter is a key component.\n"
    "3. Justify every classification. Provide reasoning that directly references relevant text.\n"
    "Your response must be in JSON format:\n"
    "{{\n"
    "'thoughts': 'Explain your reasoning, referencing explicit text or strong contextual clues.',\n"
    "'classifications': ['List only the clearly matched classes']\n"
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
    "\n".join([f'- {key}: The text should explicitly describe or strongly imply {value}' for key, value in DEFINITIONS.items()]),
)

class ThoughtfulClassification(BaseModel):
    thoughts: str
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]
