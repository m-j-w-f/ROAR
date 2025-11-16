import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

ALL_VEHICLES = [
    "eGolf",
    "ID.4",
    "Q8",
    "Taycan",
]

VEHICLE_CLEAN_NAMES = {
    "e-Golf": "eGolf",
    "E-Golf": "eGolf",
    "VW eGolf": "eGolf",
    "ID4": "ID.4",
    "VW ID4": "ID.4",
    "Q8 e-tron": "Q8",
    "AudiQ8": "Q8",
    "Porsche": "Taycan",
}

ALL_TYRES = [
    "EcoContact 6 Q",
    "RainSport 5",
    "PremiumContact 6 AO",
    "P-Zero R",
    "Ventus S1 evo 3 ev",
    "Summer SRTT",
]

TYRE_CLEAN_NAMES = {
    "PremiumContact 6": "PremiumContact 6 AO",
    "Ventus S1 Evo 3": "Ventus S1 evo 3 ev",
    "P-Zero": "P-Zero R",
}
