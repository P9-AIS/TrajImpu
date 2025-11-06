from enum import Enum


class VesselType(Enum):
    UNKNOWN = -1
    SPARE1 = 1
    DIVING = 2
    PORTTENDER = 5
    SPARE2 = 6
    RESERVED = 3
    WIG = 4
    SAR = 10
    DREDGING = 11
    NOTPARTYTOCONFLICT = 12
    PLEASURE = 7
    TUG = 17
    UNDEFINED = 13
    TOWING = 14
    MILITARY = 15
    FISHING = 16
    PILOT = 8
    TOWINGLONGWIDE = 9
    OTHER = 18
    ANTIPOLLUTION = 19
    HSC = 20
    SAILING = 21
    PASSENGER = 22
    TANKER = 23
    LAWENFORCEMENT = 24
    CARGO = 25
    MEDICAL = 26
