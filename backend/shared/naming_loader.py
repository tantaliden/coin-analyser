"""
NAMING LOADER - Lädt naming.js und macht es für Python verfügbar
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
    
    # JavaScript Object zu JSON konvertieren
    # Entferne module.exports und const NAMING =
    match = re.search(r'const NAMING = ({[\s\S]*?});', content)
    if not match:
        raise ValueError("Could not parse naming.js")
    
    js_obj = match.group(1)
    
    # JavaScript -> JSON Konvertierung
    # Entferne Funktionen
    js_obj = re.sub(r',\s*\w+:\s*function\s*\([^)]*\)\s*{[^}]*}', '', js_obj)
    js_obj = re.sub(r'\w+:\s*function\s*\([^)]*\)\s*{[\s\S]*?},?', '', js_obj)
    
    # Trailing commas entfernen
    js_obj = re.sub(r',(\s*[}\]])', r'\1', js_obj)
    
    # Single quotes zu double quotes
    js_obj = js_obj.replace("'", '"')
    
    # Unquoted keys quoten
    js_obj = re.sub(r'(\s)(\w+)(\s*:)', r'\1"\2"\3', js_obj)
    
    try:
        return json.loads(js_obj)
    except json.JSONDecodeError as e:
        # Fallback: Wichtige Werte manuell extrahieren
        naming = {}
        
        # klineMetricsDurations
        match = re.search(r'klineMetricsDurations:\s*\[([\d,\s]+)\]', content)
        if match:
            naming['klineMetricsDurations'] = [int(x.strip()) for x in match.group(1).split(',')]
        
        # eventColors
        match = re.search(r'eventColors:\s*\[([\s\S]*?)\]', content)
        if match:
            colors = re.findall(r'["\']([#\w]+)["\']', match.group(1))
            naming['eventColors'] = colors
        
        # indicatorFields
        naming['indicatorFields'] = []
        for m in re.finditer(r'\{\s*key:\s*["\'](\w+)["\'],\s*label:\s*["\']([^"\']+)["\'],\s*color:\s*["\']([#\w]+)["\'],\s*type:\s*["\'](\w+)["\']', content):
            naming['indicatorFields'].append({
                'key': m.group(1),
                'label': m.group(2),
                'color': m.group(3),
                'type': m.group(4)
            })
        
        # databases
        naming['databases'] = {
            'coins': {'tables': {'klines': 'klines', 'klineMetrics': 'kline_metrics'}},
            'app': {'tables': {'users': 'users', 'indicatorSets': 'indicator_sets', 'indicatorItems': 'indicator_items', 'coinGroups': 'coin_groups'}}
        }
        
        return naming

# Globale Instanz
NAMING = load_naming()
