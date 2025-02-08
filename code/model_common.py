from pydantic import BaseModel
from typing import Literal

SYSTEM_PROMPT = (
    "You are a highly precise assistant that classifies development and humanitarian activities based on limited text descriptions.\n"
    "You are looking for matches within an expanded definition of the housing sector, which includes the following classes:\n"
    "{}\n"
    "The classes are defined as such:\n"
    "{}\n"
    "Think carefully and do not jump to conclusions: ground your response on the given text.\n"
    "Your response must be in JSON format:\n"
    "{{\n"
    "'summary': 'Summarize the primary activities and objectives of the project in one sentence.',\n"
    "'thoughts': ['List each class and explain your reasoning for why or why not the text is a match.'],\n"
    "'classifications': ['List only the clearly matched classes']\n"
    "}}"
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
    summary: str
    thoughts: list[str]
    classifications: list[Literal[tuple(DEFINITIONS.keys())]]

SECTORS = {
    "11110": "Education policy and administrative management",
    "11120": "Education facilities and training",
    "11130": "Teacher training",
    "11182": "Educational research",
    "11220": "Primary education",
    "11230": "Basic life skills for adults",
    "11231": "Basic life skills for youth",
    "11232": "Primary education equivalent for adults",
    "11240": "Early childhood education",
    "11250": "School feeding",
    "11260": "Lower secondary education",
    "11320": "Upper Secondary Education (modified and includes data from 11322)",
    "11330": "Vocational training",
    "11420": "Higher education",
    "11430": "Advanced technical and managerial training",
    "12110": "Health policy and administrative management",
    "12181": "Medical education/training",
    "12182": "Medical research",
    "12191": "Medical services",
    "12196": "Health statistics and data",
    "12220": "Basic health care",
    "12230": "Basic health infrastructure",
    "12240": "Basic nutrition",
    "12250": "Infectious disease control",
    "12261": "Health education",
    "12262": "Malaria control",
    "12263": "Tuberculosis control",
    "12264": "COVID-19 control",
    "12281": "Health personnel development",
    "12310": "NCDs control, general",
    "12320": "Tobacco use control",
    "12330": "Control of harmful use of alcohol and drugs",
    "12340": "Promotion of mental health and well-being",
    "12350": "Other prevention and treatment of NCDs",
    "12382": "Research for prevention and control of NCDs",
    "13010": "Population policy and administrative management",
    "13020": "Reproductive health care",
    "13030": "Family planning",
    "13040": "STD control including HIV/AIDS",
    "13081": "Personnel development for population and reproductive health",
    "13096": "Population statistics and data",
    "14010": "Water sector policy and administrative management",
    "14015": "Water resources conservation (including data collection)",
    "14020": "Water supply and sanitation - large systems",
    "14021": "Water supply - large systems",
    "14022": "Sanitation - large systems",
    "14030": "Basic drinking water supply and basic sanitation",
    "14031": "Basic drinking water supply",
    "14032": "Basic sanitation",
    "14040": "River basins development",
    "14050": "Waste management/disposal",
    "14081": "Education and training in water supply and sanitation",
    "15110": "Public sector policy and administrative management",
    "15111": "Public finance management (PFM)",
    "15112": "Decentralisation and support to subnational government",
    "15113": "Anti-corruption organisations and institutions",
    "15114": "Domestic revenue mobilisation",
    "15116": "Tax collection",
    "15117": "Budget planning",
    "15118": "National audit",
    "15119": "Debt and aid management",
    "15121": "Foreign affairs",
    "15122": "Diplomatic missions",
    "15123": "Administration of developing countries' foreign aid",
    "15124": "General personnel services",
    "15125": "Public Procurement",
    "15126": "Other general public services",
    "15127": "National monitoring and evaluation",
    "15128": "Local government finance",
    "15129": "Other central transfers to institutions",
    "15130": "Legal and judicial development",
    "15131": "Justice, law and order policy, planning and administration",
    "15132": "Police",
    "15133": "Fire and rescue services",
    "15134": "Judicial affairs",
    "15135": "Ombudsman",
    "15136": "Immigration",
    "15137": "Prisons",
    "15142": "Macroeconomic policy",
    "15143": "Meteorological services",
    "15144": "National standards development",
    "15150": "Democratic participation and civil society",
    "15151": "Elections",
    "15152": "Legislatures and political parties",
    "15153": "Media and free flow of information",
    "15154": "Executive office",
    "15155": "Tax policy and administration support",
    "15156": "Other non-tax revenue mobilisation",
    "15160": "Human rights",
    "15170": "Women's rights organisations and movements, and government institutions",
    "15180": "Ending violence against women and girls",
    "15185": "Local government administration",
    "15190": "Facilitation of orderly, safe, regular and responsible migration and mobility",
    "15196": "Government and civil society statistics and data",
    "15210": "Security system management and reform",
    "15220": "Civilian peace-building, conflict prevention and resolution",
    "15230": "Participation in international peacekeeping operations",
    "15240": "Reintegration and SALW control",
    "15250": "Removal of land mines and explosive remnants of war",
    "15261": "Child soldiers (prevention and demobilisation)",
    "16010": "Social Protection",
    "16011": "Social protection and welfare services policy, planning and administration",
    "16012": "Social security (excl pensions)",
    "16013": "General pensions",
    "16014": "Civil service pensions",
    "16015": "Social services (incl youth development and women+ children)",
    "16020": "Employment creation",
    "16030": "Housing policy and administrative management",
    "16040": "Low-cost housing",
    "16050": "Multisector aid for basic social services",
    "16061": "Culture and cultural diversity",
    "16062": "Statistical capacity building",
    "16063": "Narcotics control",
    "16064": "Social mitigation of HIV/AIDS",
    "16065": "Recreation and sport",
    "16066": "Culture",
    "16070": "Labour rights",
    "16080": "Social dialogue",
    "21010": "Transport policy and administrative management",
    "21011": "Transport policy, planning and administration",
    "21012": "Public transport services",
    "21013": "Transport regulation",
    "21020": "Road transport",
    "21021": "Feeder road construction",
    "21022": "Feeder road maintenance",
    "21023": "National road construction",
    "21024": "National road maintenance",
    "21030": "Rail transport",
    "21040": "Water transport",
    "21050": "Air transport",
    "21061": "Storage",
    "21081": "Education and training in transport and storage",
    "22010": "Communications policy and administrative management",
    "22011": "Communications policy, planning and administration",
    "22012": "Postal services",
    "22013": "Information services",
    "22020": "Telecommunications",
    "22030": "Radio, television, print and online media",
    "22040": "Information and communication technology (ICT)",
    "22081": "Education and training in ICT, telecommunications and media",
    "23110": "Energy policy and administrative management",
    "23111": "Energy sector policy, planning and administration",
    "23112": "Energy regulation",
    "23181": "Energy education/training",
    "23182": "Energy research",
    "23183": "Energy conservation and demand-side efficiency",
    "23210": "Energy generation, renewable sources - multiple technologies",
    "23220": "Hydro-electric power plants",
    "23230": "Solar energy for centralised grids",
    "23231": "Solar energy for isolated grids and standalone systems",
    "23232": "Solar energy - thermal applications",
    "23240": "Wind energy",
    "23250": "Marine energy",
    "23260": "Geothermal energy",
    "23270": "Biofuel-fired power plants",
    "23310": "Energy generation, non-renewable sources, unspecified",
    "23320": "Coal-fired electric power plants",
    "23330": "Oil-fired electric power plants",
    "23340": "Natural gas-fired electric power plants",
    "23350": "Fossil fuel electric power plants with carbon capture and storage (CCS)",
    "23360": "Non-renewable waste-fired electric power plants",
    "23410": "Hybrid energy electric power plants",
    "23510": "Nuclear energy electric power plants and nuclear safety",
    "23610": "Heat plants",
    "23620": "District heating and cooling",
    "23630": "Electric power transmission and distribution (centralised grids)",
    "23631": "Electric power transmission and distribution (isolated mini-grids)",
    "23640": "Retail gas distribution",
    "23641": "Retail distribution of liquid or solid fossil fuels",
    "23642": "Electric mobility infrastructures",
    "24010": "Financial policy and administrative management",
    "24020": "Monetary institutions",
    "24030": "Formal sector financial intermediaries",
    "24040": "Informal/semi-formal financial intermediaries",
    "24050": "Remittance facilitation, promotion and optimisation",
    "24081": "Education/training in banking and financial services",
    "25010": "Business policy and administration",
    "25020": "Privatisation",
    "25030": "Business development services",
    "25040": "Responsible business conduct",
    "31110": "Agricultural policy and administrative management",
    "31120": "Agricultural development",
    "31130": "Agricultural land resources",
    "31140": "Agricultural water resources",
    "31150": "Agricultural inputs",
    "31161": "Food crop production",
    "31162": "Industrial crops/export crops",
    "31163": "Livestock",
    "31164": "Agrarian reform",
    "31165": "Agricultural alternative development",
    "31166": "Agricultural extension",
    "31181": "Agricultural education/training",
    "31182": "Agricultural research",
    "31191": "Agricultural services",
    "31192": "Plant and post-harvest protection and pest control",
    "31193": "Agricultural financial services",
    "31194": "Agricultural co-operatives",
    "31195": "Livestock/veterinary services",
    "31210": "Forestry policy and administrative management",
    "31220": "Forestry development",
    "31261": "Fuelwood/charcoal",
    "31281": "Forestry education/training",
    "31282": "Forestry research",
    "31291": "Forestry services",
    "31310": "Fishing policy and administrative management",
    "31320": "Fishery development",
    "31381": "Fishery education/training",
    "31382": "Fishery research",
    "31391": "Fishery services",
    "32110": "Industrial policy and administrative management",
    "32120": "Industrial development",
    "32130": "Small and medium-sized enterprises (SME) development",
    "32140": "Cottage industries and handicraft",
    "32161": "Agro-industries",
    "32162": "Forest industries",
    "32163": "Textiles, leather and substitutes",
    "32164": "Chemicals",
    "32165": "Fertilizer plants",
    "32166": "Cement/lime/plaster",
    "32167": "Energy manufacturing (fossil fuels)",
    "32168": "Pharmaceutical production",
    "32169": "Basic metal industries",
    "32170": "Non-ferrous metal industries",
    "32171": "Engineering",
    "32172": "Transport equipment industry",
    "32173": "Modern biofuels manufacturing",
    "32174": "Clean cooking appliances manufacturing",
    "32182": "Technological research and development",
    "32210": "Mineral/mining policy and administrative management",
    "32220": "Mineral prospection and exploration",
    "32261": "Coal",
    "32262": "Oil and gas (upstream)",
    "32263": "Ferrous metals",
    "32264": "Nonferrous metals",
    "32265": "Precious metals/materials",
    "32266": "Industrial minerals",
    "32267": "Fertilizer minerals",
    "32268": "Offshore minerals",
    "32310": "Construction policy and administrative management",
    "33110": "Trade policy and administrative management",
    "33120": "Trade facilitation",
    "33130": "Regional trade agreements (RTAs)",
    "33140": "Multilateral trade negotiations",
    "33150": "Trade-related adjustment",
    "33181": "Trade education/training",
    "33210": "Tourism policy and administrative management",
    "41010": "Environmental policy and administrative management",
    "41020": "Biosphere protection",
    "41030": "Biodiversity",
    "41040": "Site preservation",
    "41081": "Environmental education/training",
    "41082": "Environmental research",
    "43010": "Multisector aid",
    "43030": "Urban development and management",
    "43031": "Urban land policy and management",
    "43032": "Urban development",
    "43040": "Rural development",
    "43041": "Rural land policy and management",
    "43042": "Rural development",
    "43050": "Non-agricultural alternative development",
    "43060": "Disaster Risk Reduction",
    "43071": "Food security policy and administrative management",
    "43072": "Household food security programmes",
    "43073": "Food safety and quality",
    "43081": "Multisector education/training",
    "43082": "Research/scientific institutions",
    "51010": "General budget support-related aid",
    "52010": "Food assistance",
    "53030": "Import support (capital goods)",
    "53040": "Import support (commodities)",
    "60010": "Action relating to debt",
    "60020": "Debt forgiveness",
    "60030": "Relief of multilateral debt",
    "60040": "Rescheduling and refinancing",
    "60061": "Debt for development swap",
    "60062": "Other debt swap",
    "60063": "Debt buy-back",
    "72010": "Material relief assistance and services ",
    "72011": "Basic Health Care Services in Emergencies",
    "72012": "Education in emergencies",
    "72040": "Emergency food assistance",
    "72050": "Relief co-ordination and support services",
    "73010": "Immediate post-emergency reconstruction and rehabilitation",
    "74020": "Multi-hazard response preparedness",
    "91010": "Administrative costs (non-sector allocable)",
    "93010": "Refugees/asylum seekers  in donor countries (non-sector allocable)",
    "93011": "Refugees/asylum seekers in donor countries - food and shelter ",
    "93012": "Refugees/asylum seekers in donor countries - training",
    "93013": "Refugees/asylum seekers in donor countries - health",
    "93014": "Refugees/asylum seekers in donor countries - other temporary sustenance",
    "93015": "Refugees/asylum seekers in donor countries - voluntary repatriation",
    "93016": "Refugees/asylum seekers in donor countries - transport",
    "93017": "Refugees/asylum seekers in donor countries - rescue at sea",
    "93018": "Refugees/asylum seekers in donor countries - administrative costs",
    "99810": "Sectors not specified",
    "99820": "Promotion of development awareness (non-sector allocable)",
}
