from pydantic import BaseModel
from typing import Literal
from common import SYSTEM_PROMPT


SUFFIX = "_hfhi"


RETRIEVAL_TOPICS = {
    "Housing general": [
        "housing provision", "shelter provision", "housing construction", "shelter construction",
        "eco-friendly building", "green building", "sustainable housing", "housing policy",
        "technical assistance for housing", "housing finance", "affordable home loans",
        "residential construction", "housing development", "urban housing", "rural housing",
        "resilient housing", "inclusive housing", "housing access", "dwelling construction"
    ],
    "Homelessness": [
        "homeless shelters", "emergency shelters", "homeless encampments", "tent cities",
        "tents for the homeless", "transitional housing for homeless", "unsheltered population",
        "temporary housing for homeless", "street homelessness", "permanent supportive housing",
        "housing first programs", "shelter programs for homeless"
    ],
    "Transitional housing": [
        "emergency housing", "crisis housing", "disaster shelter", "disaster housing",
        "temporary housing", "refugee shelters", "humanitarian shelters",
        "temporary supportive housing", "emergency accommodation", "post-disaster housing",
        "shelters for displaced persons", "temporary accommodation for refugees"
    ],
    "Incremental housing": [
        "housing improvement", "home repairs", "slum upgrading", "basic services for housing",
        "housing technical assistance", "incremental housing", "informal settlement upgrading",
        "water services for housing", "sanitation for housing", "energy services for housing",
        "upgrading informal housing"
    ],
    "Social housing": [
        "Community Land Trust", "CLT", "cooperative housing", "public housing",
        "affordable housing", "low-cost housing", "social housing", "nonprofit housing",
        "subsidized housing", "government-funded housing", "permanently affordable housing",
        "housing cooperatives", "shared-equity housing", "municipal housing",
        "state-funded housing"
    ],
    "Market housing": [
        "home rental", "rental housing", "mortgages", "home financing", "rent-to-own housing",
        "market-rate housing", "private housing market", "real estate market",
        "homeownership programs", "property development", "private sector housing",
        "residential real estate", "housing investment", "landlord-tenant rental"
    ]
}


DEFINITIONS = {
    "Housing general": "explicitly describes provision of housing/shelter, construction of housing/shelter (including eco-friendly building practices), housing policy, technical assistance for housing, or finance for housing",
    "Homelessness": "explicitly describes tents for the homeless, encampments for the homeless, or homeless shelters",
    "Transitional housing": "explicitly describes housing/shelter provided due to crises/disasters/emergencies, refugee shelters, refugee camps, or temporary supportive housing. The project must primarily involve the provision of housing or shelter; vague terms like 'material relief' alone are not sufficient evidence of housing provision. Mentions of refugees alone without additional housing/shelter context also do not qualify as Transitional housing",
    "Incremental housing": "explicitly describes housing sites, housing improvement/repairs, basic services for housing, or housing technical assistance. Basic services refer to water, sanitation, or energy improvements to housing. One important form of Incremental housing to look for is slum upgrading, but this class can be applied to any housing context",
    "Social housing": "explicitly describes Community Land Trusts (CLTs), cooperative housing, public housing, affordable housing, or low-cost housing. CLTs are defined as nonprofit organizations that acquire and hold land for the permanent benefit of the community",
    "Market housing": "explicitly describes home-rental, mortgages, rent-to-own housing, or market-rate housing",
    "Urban": "explicitly describes activities in specific urban locations",
    "Rural": "explicitly describes activities in specific rural locations"
}


SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: when the text {value}' for key, value in DEFINITIONS.items()]),
)


class ReasonedClassification(BaseModel):
    summary: str
    reasoning: str
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]
