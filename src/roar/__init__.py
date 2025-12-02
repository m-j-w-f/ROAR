import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data_cleaned"
EXTRAS_DIR = ROOT_DIR / "data_extras"

ALL_VEHICLES = [
    "01 VW ID4",
    "02 Audi Q8",
    "03 Porsche Taycan",
    "04 VW eGolf",
]

VEHICLE_COLORS = {
    "01 VW ID4": "#3498db",
    "02 Audi Q8": "#e74c3c",
    "03 Porsche Taycan": "#2ecc71",
    "04 VW eGolf": "#9b59b6",
}

VEHICLE_CLEAN_NAMES = {
    "e-Golf": "04 VW eGolf",
    "E-Golf": "04 VW eGolf",
    "VW eGolf": "04 VW eGolf",
    "ID4": "01 VW ID4",
    "VW ID4": "01 VW ID4",
    "Q8 e-tron": "02 Audi Q8",
    "AudiQ8": "02 Audi Q8",
    "Porsche": "03 Porsche Taycan",
}

ID_TO_VEHICLE = {i + 1: vehicle for i, vehicle in enumerate(ALL_VEHICLES)}

ALL_TYRES = [
    "EcoContact 6 Q",
    "RainSport 5",
    "PremiumContact 6 AO",
    "P-Zero R",
    "Ventus S1 evo 3 ev",
    "Summer SRTT",
]

TYRE_COLORS = {
    1: "#e63946",
    3: "#f4a261",
    6: "#e9c46a",
    10: "#2a9d8f",
    12: "#264653",
    13: "#7209b7",
}

TYRE_CLEAN_NAMES = {
    "PremiumContact 6": "PremiumContact 6 AO",
    "Ventus S1 Evo 3": "Ventus S1 evo 3 ev",
    "P-Zero": "P-Zero R",
}
