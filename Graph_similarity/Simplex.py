def node_simarity(a, b):
    if a["type"] != b["type"]:
        return 100
    diff = 0
    for key in list(a.keys())[1:]:
        par_i = a[key]
        if key not in list(b.keys())[1:]:
            diff += 1
        else:
            par_j = b[key]
            if par_i.isnumeric():
                # delta = abs(float(par_i) - float(par_j))
                diff += 1  # delta
            elif par_i != par_j:
                diff += 1
    return diff