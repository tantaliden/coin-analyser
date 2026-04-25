"""Konvertiert DB-indicator_items zurueck in Criteria fuer den blind_scanner.
Keine Fallbacks: fehlende Pflichtfelder werfen Exception."""

import json


def item_to_criterion(item):
    """item: dict aus indicator_items. Returns: dict im Criterion-Format."""
    if 'fuzzy_config' not in item or item['fuzzy_config'] is None:
        raise ValueError(f"Item {item.get('item_id')} hat keine fuzzy_config")

    fuzzy = item['fuzzy_config']
    if isinstance(fuzzy, str):
        fuzzy = json.loads(fuzzy)

    indicator_type = item['indicator_type']
    op = item['condition_operator']
    pattern_data = item.get('pattern_data')
    if isinstance(pattern_data, str):
        pattern_data = json.loads(pattern_data) if pattern_data else None

    # Pattern / Sequence
    if indicator_type == 'candle_pattern':
        if not pattern_data:
            raise ValueError(f"Item {item.get('item_id')}: candle_pattern ohne pattern_data")
        if 'pattern_id' in pattern_data:
            kind = 'pattern'
            return {
                'kind': kind,
                'field': 'close',
                'pattern_id': pattern_data['pattern_id'],
                'time_offset_from': item['time_start_minutes'],
                'time_offset_to': item['time_end_minutes'],
                'fuzzy': fuzzy,
            }
        if 'sequence' in pattern_data:
            return {
                'kind': 'sequence',
                'field': 'close',
                'sequence': pattern_data['sequence'],
                'time_offset_from': item['time_start_minutes'],
                'time_offset_to': item['time_end_minutes'],
                'fuzzy': fuzzy,
            }
        raise ValueError(f"Item {item.get('item_id')}: pattern_data ohne pattern_id oder sequence")

    # Operator -> kind
    if op == 'between':
        kind = 'range'
    elif op == '=':
        kind = 'value'
    elif op in ('slope', 'ratio'):
        kind = op
    else:
        kind = 'value'  # >,<,>=,<= als value mit Toleranz

    crit = {
        'kind': kind,
        'field': indicator_type,
        'time_offset_from': item['time_start_minutes'],
        'time_offset_to': item['time_end_minutes'],
        'fuzzy': fuzzy,
    }
    if item.get('condition_value') is not None:
        crit['value'] = float(item['condition_value'])
    if item.get('condition_value2') is not None:
        crit['value2'] = float(item['condition_value2'])

    return crit


def initial_item_to_point(item):
    """Konvertiert ein initial_point Item in die Initialpunkt-Struktur."""
    crit = item_to_criterion(item)
    if item.get('initial_fixed_offset') is not None:
        crit['fixed_offset'] = item['initial_fixed_offset']
    return crit
