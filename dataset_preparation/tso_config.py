"""
TSO Configuration Module.

This module defines the relationships between German TSOs and their neighboring
countries/bidding zones, as well as utilities for normalizing TSO names across
different data sources.
"""

from typing import Optional

# TSO neighboring countries/bidding zones
# These represent the primary neighboring bidding zones for each German TSO
TSO_NEIGHBORS: dict[str, list[str]] = {
    "50Hertz": ["DK_2", "DK", "PL", "CZ"],  # DK2 = East Denmark (closer to 50Hertz)
    "TenneT DE": ["DK_1", "DK", "NL", "BE"],  # DK1 = West Denmark (closer to TenneT)
    "Amprion": ["NL", "BE", "FR", "LU"],
    "TransnetBW": ["FR", "CH", "AT"],
}

# Normalize TSO names (handle variations in data sources)
# Maps canonical name to list of known variants
TSO_NAME_MAPPING: dict[str, list[str]] = {
    "50Hertz": ["50Hertz", "50HzT", "DE_50HzT", "50Hertz_DE", "50H"],
    "TenneT DE": ["TenneT", "TenneT DE", "DE_TenneT", "TenneT_DE"],
    "Amprion": ["Amprion", "DE_Amprion", "Amprion_DE"],
    "TransnetBW": ["TransnetBW", "DE_TransnetBW", "TransnetBW_DE", "Transnet BW"],
}

# All German TSOs
GERMAN_TSOS: list[str] = ["50Hertz", "TenneT DE", "Amprion", "TransnetBW"]

# All neighboring countries (union of all TSO neighbors)
ALL_NEIGHBOR_COUNTRIES: list[str] = [
    "DK_1", "DK_2", "PL", "CZ", "NL", "BE", "FR", "LU", "CH", "AT"
]

# Map from German federal state (Bundesland) to responsible TSO
BUNDESLAND_TO_TSO: dict[str, str] = {
    "Schleswig-Holstein": "TenneT DE",
    "Niedersachsen": "TenneT DE",
    "Hessen": "TenneT DE",
    "Bayern": "TenneT DE",
    "Bremen": "TenneT DE",
    "Hamburg": "50Hertz",
    "Mecklenburg-Vorpommern": "50Hertz",
    "Berlin": "50Hertz",
    "Brandenburg": "50Hertz",
    "Sachsen": "50Hertz",
    "Thüringen": "50Hertz",
    "Sachsen-Anhalt": "50Hertz",
    "Nordrhein-Westfalen": "Amprion",
    "Rheinland-Pfalz": "Amprion",
    "Saarland": "Amprion",
    "Baden-Württemberg": "TransnetBW",
}

# Default hourly split method for each TSO
# These were determined empirically by comparing against official 15-min data
TARGET_SPLIT_METHOD_DICT: dict[str, str] = {
    "50Hertz": "equal",
    "Amprion": "split",
    "TenneT DE": "equal",
    "TransnetBW": "equal",
}


def normalize_tso_name(raw_name: str) -> str:
    """
    Normalize TSO name to canonical form.
    
    Parameters
    ----------
    raw_name : str
        The raw TSO name as it appears in various data sources.
        
    Returns
    -------
    str
        The canonical TSO name (e.g., "50Hertz", "TenneT DE", "Amprion", "TransnetBW").
        Returns the input unchanged if no mapping is found.
        
    Examples
    --------
    >>> normalize_tso_name("TenneT")
    'TenneT DE'
    >>> normalize_tso_name("50HzT")
    '50Hertz'
    >>> normalize_tso_name("Unknown TSO")
    'Unknown TSO'
    """
    for canonical, variants in TSO_NAME_MAPPING.items():
        if raw_name in variants:
            return canonical
    return raw_name


def get_neighbors(tso: str) -> list[str]:
    """
    Get neighboring countries/bidding zones for a given TSO.
    
    Parameters
    ----------
    tso : str
        The TSO name (will be normalized internally).
        
    Returns
    -------
    list[str]
        List of neighboring country/bidding zone codes (e.g., ["DK_2", "PL", "CZ"]).
        Returns an empty list if the TSO is not recognized.
        
    Examples
    --------
    >>> get_neighbors("50Hertz")
    ['DK_2', 'PL', 'CZ']
    >>> get_neighbors("TenneT")  # normalized to "TenneT DE"
    ['DK_1', 'NL', 'BE']
    """
    tso_normalized = normalize_tso_name(tso)
    return TSO_NEIGHBORS.get(tso_normalized, [])


def get_tso_for_bundesland(bundesland: str) -> Optional[str]:
    """
    Get the responsible TSO for a German federal state (Bundesland).
    
    Parameters
    ----------
    bundesland : str
        The name of the German federal state in German.
        
    Returns
    -------
    str or None
        The canonical TSO name, or None if the Bundesland is not recognized.
        
    Examples
    --------
    >>> get_tso_for_bundesland("Bayern")
    'TenneT DE'
    >>> get_tso_for_bundesland("Berlin")
    '50Hertz'
    """
    return BUNDESLAND_TO_TSO.get(bundesland)


def get_split_method(tso: str) -> str:
    """
    Get the recommended hourly split method for a given TSO.
    
    The split method determines how redispatch interventions spanning multiple
    hours are allocated to individual hours. Methods were determined empirically
    by comparing processed data against official 15-minute resolution data.
    
    Parameters
    ----------
    tso : str
        The TSO name (will be normalized internally).
        
    Returns
    -------
    str
        The split method: "equal" (overlap-based) or "split" (front+back loaded).
        Defaults to "equal" if the TSO is not recognized.
        
    Examples
    --------
    >>> get_split_method("Amprion")
    'split'
    >>> get_split_method("50Hertz")
    'equal'
    """
    tso_normalized = normalize_tso_name(tso)
    return TARGET_SPLIT_METHOD_DICT.get(tso_normalized, "equal")


def is_german_tso(tso: str) -> bool:
    """
    Check if a TSO name corresponds to a German TSO.
    
    Parameters
    ----------
    tso : str
        The TSO name (will be normalized internally).
        
    Returns
    -------
    bool
        True if this is a German TSO, False otherwise.
        
    Examples
    --------
    >>> is_german_tso("50Hertz")
    True
    >>> is_german_tso("TenneT")  # Dutch spelling, but normalized to TenneT DE
    True
    >>> is_german_tso("APG")  # Austrian TSO
    False
    """
    tso_normalized = normalize_tso_name(tso)
    return tso_normalized in GERMAN_TSOS
