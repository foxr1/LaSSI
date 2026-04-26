

def merge_schema_elements(lhs:dict, rhs:dict)->dict:
    result = ("property", "ell", "xi", "containment")
    values = []
    for key in result:
        values.append(list(set(lhs.get(key, {})).union(set(rhs.get(key, {})))))
    return dict(zip(result, values))