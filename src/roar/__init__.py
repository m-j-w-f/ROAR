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

ID_TO_VEHICLE = {i: vehicle for i, vehicle in enumerate(ALL_VEHICLES)}
VEHICLE_TO_ID = {vehicle: i for i, vehicle in enumerate(ALL_VEHICLES)}

ALL_TYRES = [
    "EcoContact 6 Q",
    "RainSport 5",
    "PremiumContact 6 AO",
    "P-Zero R",
    "Ventus S1 evo 3 ev",
    "Summer SRTT",
]

TYRE_TO_ID = {
    "EcoContact 6 Q": 1,
    "RainSport 5": 3,
    "PremiumContact 6 AO": 6,
    "P-Zero R": 10,
    "Ventus S1 evo 3 ev": 12,
    "Summer SRTT": 13,
}
ID_TO_TYRE = {i: tyre for tyre, i in TYRE_TO_ID.items()}

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

# TODO: track 259 is actually also track 211 (?)
ALL_TRACKS = [150, 211, 259]

TRACK_COLORS = {
    150: "#4A90E2",
    211: "#df4ed8",
    259: "#F5A623",
}

MEASUREMENTS = [f"meas{i}" for i in range(0, 7)]
# A note on measurements:
# meas0: 43-57.6 km/h
# meas1: 40 km/h
# meas2: 75 km/h but also some speeds at 45 km/h
# meas3: 57 km/h - 100 km/h (mostly 100)
# meas4: 45 km/h, accel 1 m/s2
# meas5: 45 km/h, accel 2 m/s2
# meas6: 45 km/h, accel 3-4 m/s2
# vr45: constant 45 km/h
# vr50: same as vr50_b35 (maybe file parsing issue)
# vr80: constant 80 km/h
# vr100: constant 100 km/h
# vr50_b35: accel from 45 km/h to 55 km/h with 1 m/s2
# vr50_b50: accel from 45 km/h to 60 km/h with 2 m/s2
# vr50_b70: accel from 50 km/h to 60 km/h with 3-4 m/s2

MEASUREMENTS_CLEAN_NAMES = {
    "meas0": "meas1",
    "vr45": "meas1",
    "vr50": "meas1",
    "vr80": "meas2",
    "vr100": "meas3",
    "vr50_b35": "meas4",
    "vr50_b50": "meas5",
    "vr50_b70": "meas6",
}

MIC_CHANNELS = [
    "Ch_1_labV12",
    "Ch_2_labV12",
    "Ch_3_labV12",
    "Ch_4_labV12",
    "NAWSSound",
    "mic_iso",
    "mic_2m",
]
