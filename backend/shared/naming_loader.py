"""
NAMING LOADER - Lädt naming.js und macht es für Python verfügbar
Robuster Parser der auch mit JS-Funktionen umgehen kann
"""

import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
NAMING_FILE = ROOT_DIR / 'naming.js'

def load_naming():
    """Parst naming.js und gibt Python dict zurück"""
    with open(NAMING_FILE, 'r') as f:
        content = f.read()

    # Alles zwischen const NAMING = { ... }; extrahieren
    match = re.search(r'const NAMING = ({[\s\S]*});', content)
    if not match:
        raise ValueError("Could not find NAMING object in naming.js")

    js_obj = match.group(1)

    # Funktionen komplett entfernen (mehrzeilige)
    js_obj = re.sub(r'\w+:\s*function\s*\([^)]*\)\s*\{[\s\S]*?\},?', '', js_obj)
    
    # Einzeilige Kommentare entfernen
    js_obj = re.sub(r'//[^\n]*', '', js_obj)
    
    # Mehrzeilige Kommentare entfernen
    js_obj = re.sub(r'/\*[\s\S]*?\*/', '', js_obj)

    # Single quotes zu double quotes
    js_obj = js_obj.replace("'", '"')

    # Unquoted keys quoten: word: -> "word":
    js_obj = re.sub(r'(?<!["\w])(\w+)\s*:', r'"\1":', js_obj)

    # Trailing commas entfernen
    js_obj = re.sub(r',(\s*[}\]])', r'\1', js_obj)
    
    # Leere Objekte/Arrays nach Funktionsentfernung aufräumen
    js_obj = re.sub(r',\s*,', ',', js_obj)
    js_obj = re.sub(r'\{\s*,', '{', js_obj)
    js_obj = re.sub(r',\s*\}', '}', js_obj)

    try:
        return json.loads(js_obj)
    except json.JSONDecodeError as e:
        print(f"[NAMING_LOADER] JSON parse error: {e}")
        print(f"[NAMING_LOADER] Falling back to regex extraction")
        return _extract_fallback(content)


def _extract_fallback(content):
    """Fallback: Wichtige Werte per Regex extrahieren"""
    naming = {}

    # Labels
    labels = {}
    for m in re.finditer(r"(\w+):\s*\{\s*de:\s*'([^']+)'\s*,\s*en:\s*'([^']+)'\s*\}", content):
        labels[m.group(1)] = {'de': m.group(2), 'en': m.group(3)}
    naming['labels'] = labels

    # klineMetricsDurations
    match = re.search(r'klineMetricsDurations:\s*\[([\d,\s]+)\]', content)
    if match:
        naming['klineMetricsDurations'] = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]

    # eventColors
    match = re.search(r'eventColors:\s*\[([\s\S]*?)\]', content)
    if match:
        naming['eventColors'] = re.findall(r'["\']([#\w]+)["\']', match.group(1))

    # overlapEventColors
    match = re.search(r'overlapEventColors:\s*\[([\s\S]*?)\]', content)
    if match:
        naming['overlapEventColors'] = re.findall(r'["\']([#\w]+)["\']', match.group(1))

    # indicatorFields
    naming['indicatorFields'] = []
    for m in re.finditer(r"\{\s*key:\s*'(\w+)'\s*,\s*label:\s*'([^']+)'\s*,\s*color:\s*'([#\w]+)'\s*,\s*type:\s*'(\w+)'", content):
        naming['indicatorFields'].append({'key': m.group(1), 'label': m.group(2), 'color': m.group(3), 'type': m.group(4)})

    # indicatorOperations
    match = re.search(r"indicatorOperations:\s*\[([\s\S]*?)\]", content)
    if match:
        naming['indicatorOperations'] = re.findall(r"'([^']+)'", match.group(1))

    # indicatorAggregators
    match = re.search(r"indicatorAggregators:\s*\[([\s\S]*?)\]", content)
    if match:
        naming['indicatorAggregators'] = re.findall(r"'([^']+)'", match.group(1))

    # candleTimeframes
    naming['candleTimeframes'] = []
    for m in re.finditer(r"\{\s*key:\s*'(\w+)'\s*,\s*label:\s*'([^']+)'\s*,\s*minutes:\s*(\d+)\s*\}", content):
        naming['candleTimeframes'].append({'key': m.group(1), 'label': m.group(2), 'minutes': int(m.group(3))})

    # setColorOptions
    naming['setColorOptions'] = []
    for m in re.finditer(r"\{\s*key:\s*'(\w+)'\s*,\s*label:\s*'([^']+)'\s*,\s*color:\s*'([#\w]+)'\s*\}", content):
        naming['setColorOptions'].append({'key': m.group(1), 'label': m.group(2), 'color': m.group(3)})

    # searchResultColumns
    naming['searchResultColumns'] = []
    for m in re.finditer(r"\{\s*key:\s*'(\w+)'\s*,\s*label:\s*'([^']+)'\s*,\s*default:\s*(true|false)\s*\}", content):
        naming['searchResultColumns'].append({'key': m.group(1), 'label': m.group(2), 'default': m.group(3) == 'true'})

    # avgPeriods
    naming['avgPeriods'] = []
    for m in re.finditer(r"\{\s*key:\s*(\d+)\s*,\s*label:\s*'([^']+)'\s*\}", content):
        naming['avgPeriods'].append({'key': int(m.group(1)), 'label': m.group(2)})

    # timeframes
    match = re.search(r"chartOptions:\s*\[([\s\S]*?)\]", content)
    if match:
        naming['timeframes'] = {'chartOptions': re.findall(r"'([^']+)'", match.group(1))}

    return naming


# Globale Instanz
NAMING = load_naming()
