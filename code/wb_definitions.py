from pydantic import BaseModel
from typing import Literal
from common import SYSTEM_PROMPT


SUFFIX = "_wb"


RETRIEVAL_TOPICS = {
    "Transitional and Temporary Housing": [
        "homeless shelters",
        "emergency shelters",
        "homeless encampments",
        "tent cities",
        "tents for the homeless",
        "transitional housing for homeless",
        "unsheltered population",
        "temporary housing for homeless",
        "street homelessness",
        "permanent supportive housing",
        "housing first programs",
        "shelter programs for homeless",
        "emergency housing",
        "crisis housing",
        "disaster shelter",
        "disaster housing",
        "temporary housing",
        "refugee shelters",
        "humanitarian shelters",
        "temporary supportive housing",
        "emergency accommodation",
        "shelters for displaced persons",
        "temporary accommodation for refugees"
    ],
    "Resilience and Reconstruction": [
        "post-disaster housing",
        "housing reconstruction",
        "reconstruction loan",
        "reconstruction subsidy",
        "resilient housing"
    ],
    "Incremental and Improved Housing": [
        "housing improvement",
        "home repairs",
        "slum upgrading",
        "basic services for housing",
        "housing technical assistance",
        "incremental housing",
        "informal settlement upgrading",
        "water services for housing",
        "sanitation for housing",
        "energy services for housing",
        "upgrading informal housing",
        "Tenure security"
    ],
    "Social Housing": [
        "cooperative housing",
        "public housing",
        "affordable housing",
        "low-cost housing",
        "social housing",
        "nonprofit housing",
        "subsidized housing",
        "government-funded housing",
        "permanently affordable housing",
        "housing cooperatives",
        "shared-equity housing",
        "municipal housing",
        "state-funded housing",
        "housing program"
    ],
    "Market Enabling": [
        "home rental",
        "rental housing",
        "mortgages",
        "home financing",
        "rent-to-own housing",
        "market-rate housing",
        "private housing market",
        "real estate market",
        "homeownership programs",
        "property development",
        "private sector housing",
        "residential real estate",
        "housing investment",
        "landlord-tenant rental",
        "housing policy",
        "technical assistance for housing",
        "housing finance",
        "affordable home loans",
        "property title",
        "micro-finance for housing",
        "mortgage law"
    ],
    "Housing Supply": [
        "housing provision",
        "shelter provision",
        "housing construction",
        "shelter construction",
        "eco-friendly housing ",
        "green housing ",
        "sustainable housing",
        "construction materials",
        "developers",
        "residential construction",
        "housing development",
        "urban housing",
        "rural housing",
        "inclusive housing",
        "housing access",
        "dwelling construction"
    ]
}


DEFINITIONS = {
    "Transitional and Temporary Housing": "",
    "Resilience and Reconstruction": "",
    "Incremental and Improved Housing": "",
    "Social Housing": "",
    "Market Enabling": "",
    "Housing Supply": ""
}


SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: when the text {value}' for key, value in DEFINITIONS.items()]),
)


class ReasonedClassification(BaseModel):
    summary: str
    reasoning: str
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]
