from pydantic import BaseModel
from typing import Literal
from common import SYSTEM_PROMPT


SUFFIX = "_hfhi"


DEFINITIONS = {
    "Housing in general": "explicitly describes activities related to the provision, development or policy of residential housing or shelter. It covers shelter and dwelling construction, urban and rural housing projects, and design of inclusive or collective living quarters. It includes technical assistance, capacity building, and legal advice on housing, land and property rights. It addresses policy frameworks, public-private partnerships, and smart, green or energy-efficient housing solutions. It also encompasses measures to improve housing access and reduce residential emissions through carbon-neutral design.",
    "Homelessness": "explicitly describes experiences, conditions, services or interventions related to people lacking stable housing. It includes unsheltered populations in tent cities, homeless encampments, street homelessness and hidden homelessness. It covers temporary and supportive accommodations for the unsheltered such as shelters, dormitories, safe houses and Housing First programs. It also encompasses permanent supportive housing, housing vouchers and other shelter and support programs. It addresses statutory, chronic, indigenous, sheltered, unsheltered, absolute and relative homelessness.",
    "Transitional housing": "explicitly describes the direct provision of temporary or emergency living quarters and related support for individuals or families displaced by crises, disasters, or other emergencies. To qualify, the activity must involve providing a physical shelter, not just support services. This includes emergency and crisis shelters, refugee and asylum seeker accommodation (excluding general camp management without a specific shelter provision component), and disaster relief shelters. It covers the distribution of shelter kits, tents, and essential non-food items for the express purpose of establishing immediate, temporary habitation. While it can include short-term rental support or host family placements, mentions of non-specific 'material relief,' 'support services,' or 'disaster preparedness' alone are insufficient.",
    "Incremental housing": "explicitly describes interventions that support gradual and participatory improvements to existing housing and related infrastructure in formal and informal contexts. It includes measures to provide or upgrade basic services, perform structural retrofits, preserve or repurpose dwellings and reduce hazard risks. It covers efforts to regularize, legalize and formalize housing, buildings and land through titling, readjustment and incorporation. It includes in-situ improvements, permanent repairs, enumeration, settlement planning, redevelopment, slum upgrading, relocation and resettlement. It also involves home expansion, core housing upgrades, rehabilitation and community infrastructure such as water, sanitation, drainage and transport networks. The subsector embraces support for construction material suppliers, tool and material vouchers, technical training and legal advice on housing, land and property rights. It addresses resilient and environmentally sound housing through nature-based solutions, ecosystem restoration, adaptation measures and recycling of plastic and agricultural waste. It includes eviction prevention and moratoriums and applies across diverse informal settlement types such as aashwa'i, barracas, bidonvilles, chawls, favelas, kampongs, slums, and villa miseria.",
    "Social housing": "explicitly describes housing programs or systems organized, funded, or managed by governments, nonprofits, or community groups to provide affordable or low-cost dwellings to low-income or vulnerable households. It covers the development, financing, allocation, and management of rental or ownership units with controlled or subsidized costs. It includes supply-side and demand-side subsidies, rent vouchers, permanently affordable models like shared-equity housing and community land trusts, and nonprofit or municipal housing schemes. It also encompasses cooperative housing, co-housing projects, social rental agencies, housing associations, and community-led initiatives. It addresses government-funded construction, rental protection measures, energy service company partnerships, and strategies to improve social acceptance of housing projects.",
    "Market housing": "explicitly describes housing supplied or financed through private market mechanisms. This includes rental housing and landlord-tenant relationships. It covers mortgages and home financing products such as conventional loans, micro-finance schemes, rent-to-own arrangements, mortgage securitization and insurance. It encompasses property development and residential real estate investment, including speculation, home flipping, foreign ownership and mortgage-backed securities. It involves construction and regulatory processes such as land use reforms, zoning, permitting, building technologies and modular construction. It addresses market dynamics such as market-rate rentals, homeownership programs, housing stock, affordability, evictions and tenant protection. It also covers financial instruments and policies such as mortgage bonds, development charges, low-cost and subsidized financing.",
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


if __name__ == '__main__':
    print(SYSTEM_PROMPT)
