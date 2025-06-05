
rules = [
    (vocab['ANT'] * vocab['BLUE'] + vocab['RELATION'] * vocab['IMPLIES'] + vocab['CONS'] * vocab['EVEN']).normalized(),
    (vocab['ANT'] * vocab['ODD'] + vocab['RELATION'] * vocab['IMPLIES'] + vocab['CONS'] * vocab['GREEN']).normalized(),
]