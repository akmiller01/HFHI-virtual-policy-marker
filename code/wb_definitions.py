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
    "Transitional and Temporary Housing": "explicitly describes the provision of short-term shelter and living arrangements for individuals lacking stable residences. It covers emergency and crisis shelters for the homeless, homeless encampments, tent cities and disaster relief accommodations. It also encompasses refugee and humanitarian shelters, temporary supportive housing and housing first initiatives. Key activities include operation and management of emergency accommodations, transitional housing programs and shelter services for displaced or unsheltered populations.",
    "Resilience and Reconstruction": "explicitly describes efforts to restore and strengthen housing stock following disasters. This subsector covers housing reconstruction and post-disaster shelter programs. It includes the provision of reconstruction loans and subsidies. It also addresses the design and construction of resilient housing to improve durability and disaster resistance.",
    "Incremental and Improved Housing": "explicitly describes interventions to enhance existing dwellings and support incremental housing development. This subsector covers home repairs, slum and informal settlement upgrading, and the provision of basic services for housing, including water, sanitation, and energy. It also encompasses tenure security measures and technical assistance to guide gradual housing improvements in underserved communities.",
    "Social Housing": "explicitly describes housing initiatives designed to provide affordable or low-cost residences through government, municipal or nonprofit programs. It covers public and subsidized housing, cooperative and shared-equity models, and permanently affordable units. This subsector includes nonprofit-managed schemes and state-funded housing programs aimed at expanding access to secure and affordable homes.",
    "Market Enabling": "explicitly describes efforts to strengthen or expand private housing markets through regulatory reform, housing finance, technical assistance for housing, and investment. It includes home rental and rental housing, rent-to-own schemes, market-rate housing, affordable home loans, mortgages, and micro-finance for housing. It involves property development, housing investment, private sector housing activities, and residential real estate transactions. It also covers landlord-tenant rental arrangements, homeownership programs, housing policy, housing finance mechanisms, property title management, mortgage law, and real estate market dynamics.",
    "Housing Supply": "explicitly describes the provision and construction of residential dwellings. It includes planning, financing, and development of urban and rural housing projects by developers. It involves selection of construction materials and application of sustainable or eco-friendly building practices. It encompasses shelter provision, housing development activities, and measures to improve access to adequate and inclusive housing."
}


SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: when the text {value}' for key, value in DEFINITIONS.items()]),
)


class ReasonedClassification(BaseModel):
    summary: str
    reasoning: str
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]
