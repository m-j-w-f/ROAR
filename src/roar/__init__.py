import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data_cleaned"
EXTRAS_DIR = ROOT_DIR / "data_extras"

ALL_VEHICLES = [
    "ID.4",
    "Q8 e-tron",
    "Taycan",
    "E-Golf",
]

VEHICLE_COLORS = {
    "ID.4": "#3498db",  # Blue
    "Q8 e-tron": "#e74c3c",  # Red
    "Taycan": "#2ecc71",  # Green
    "E-Golf": "#9b59b6",  # Purple
}

VEHICLE_CLEAN_NAMES = {
    "ID4": "ID.4",
    "VW ID4": "ID.4",
    "Q8 e-tron": "Q8 e-tron",
    "Q8": "Q8 e-tron",
    "AudiQ8": "Q8 e-tron",
    "Porsche": "Taycan",
    "Taycan": "Taycan",
    "e-Golf": "E-Golf",
    "eGolf": "E-Golf",
    "E-Golf": "E-Golf",
    "VW eGolf": "E-Golf",
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

ALL_TRACKS = [150, 211, 259]

TRACK_COLORS = {
    150: "#4A90E2",
    211: "#df4ed8",
    259: "#F5A623",
}

MIC_CHANNELS = [
    "Ch_1_labV12",
    "Ch_2_labV12",
    "Ch_3_labV12",
    "Ch_4_labV12",
    "NAWSSound",
    "mic_iso",
]
