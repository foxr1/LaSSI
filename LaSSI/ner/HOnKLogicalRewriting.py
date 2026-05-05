from LaSSI.structures.kernels.Sentence import get_prepositions

from LaSSI.external_services.Services import Services
from LaSSI.ner.string_functions import lemmatize_verb, is_label_verb

honk = Services.getInstance().getHOnK()
logical_rules = honk.getLogicalRewritingRules()

def is_name_in_honk(name, honk_list, should_lemmatize=False):
    return len(honk_list.intersection({lemmatize_verb(name), name} if should_lemmatize else {name})) > 0

def is_nmod(kernel, node, initial_node, has_nmod, value):
    return has_nmod == value

def match_prepositions(kernel, node, initial_node, has_nmod, value):
    prepositions = get_prepositions(node)
    return value in prepositions

def is_materialised(kernel, node, initial_node, has_nmod, value):
    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getMaterialisationVerbs(), True) == value

def is_causative(kernel, node, initial_node, has_nmod, value):
    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getCausativeVerbs(), True) == value

def has_movement(kernel, node, initial_node, has_nmod, value):
    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getMovementVerbs(), True) == value

def is_in_state(kernel, node, initial_node, has_nmod, value):
    return is_name_in_honk(node.named_entity, honk.getStateVerbs(), True) == value

def has_means(kernel, node, initial_node, has_nmod, value):
    if has_nmod:
        node = initial_node.kernel.source
    return is_name_in_honk(node.named_entity, honk.getMeansVerbs(), False) == value

def is_abstract_entity(kernel, node, initial_node, has_nmod, value):
    return is_name_in_honk(node.named_entity, honk.getAbstractEntities(), True) == value

def has_measurement(kernel, node, initial_node, has_nmod, value):
    # TODO: Aware this is definitely wrong, what is the better way of resolving the measurement?
    if has_nmod:
        possible_measurement = f"{initial_node.kernel.source.named_entity} per {initial_node.kernel.target.named_entity}"
    else:
        possible_measurement = ""
    return is_name_in_honk(possible_measurement, honk.getUnitsOfMeasure(), False) == value

def has_nmod_part_of(kernel, node, initial_node, has_nmod, value):
    if kernel.kernel is None:
        return False
    return (kernel.kernel.source.id == (initial_node.kernel.source.id if initial_node.kernel is not None else node.id)) == value

def is_symmetrical(kernel, node, initial_node, has_nmod, value):
    return (len(
        {x for x in honk.getNounsWithProperties() if lemmatize_verb(node.named_entity) == str(x.label) and (initial_node.kernel is not None and initial_node.kernel.source.named_entity in str(x.hasProperty))}
    ) > 0) == value

def is_a(kernel, node, initial_node, has_nmod, value):
    return (len({x for x in honk.getNounsWithA() if
         node.named_entity == str(x.label) and initial_node.kernel.source.named_entity in str(x.isA)}) > 0) == value

def has_number(kernel, node, initial_node, has_nmod, value):
    # TODO: In our examples, this is the case, will it always be?
    if has_nmod:
        return ("nummod" in dict(initial_node.kernel.source.properties)) == True
    else:
        return ("nummod" in dict(node.properties)) == value

def type_of_node(kernel, node, initial_node, has_nmod, value):
    # TODO: Is "DATE" always "SUTime", could we change the ontology to be matched "DATE"??
    if node.type == "DATE":
        resolved = "SUTime"
    elif node.type == "IN" and bool(set(node.named_entity.lower().split()) & {tn.lower() for tn in honk.getTemporalNouns()}):
        # e.g. "On Saturdays" — preposition-headed span whose content is a day/time noun
        resolved = "SUTime"
    elif node.type in {"GPE", "LOC"}:
        resolved = str(node.type)
    elif node.type == "RB":
        resolved = "RB"
    elif node.type == "IN":
        resolved = "IN"
    elif node.type == "verb":
        resolved = "verb"
    else:
        resolved = "None"
    return resolved == value

def source_is_verb(kernel, node, initial_node, has_nmod, value):
    return is_label_verb(node.named_entity) == value

def is_phrasal_verb(kernel, node, initial_node, has_nmod, value):
    """Check if a SENTENCE node's edge label (stripped of conjunction prefix) is a phrasal verb."""
    if (hasattr(node, 'kernel') and node.kernel is not None and
            node.kernel.edgeLabel is not None):
        edge_name = node.kernel.edgeLabel.named_entity
        _conj_prefixes = {c.lower() for c in honk.getConjunctions()}
        base_parts = [p for p in edge_name.split(' ') if p.lower() not in _conj_prefixes]
        base_verb = ' '.join(base_parts)
        return is_name_in_honk(base_verb, honk.getPhrasalVerbs(), False) == value
    return not value

predicate_interpretation = {
    "MaterialisationVerb": is_materialised,
    "CausativeVerb": is_causative,
    "SingletonHasBeenMatchedBy": type_of_node,
    "AbstractEntity": is_abstract_entity,
    "Number": has_number,
    "UnitOfMeasure": has_measurement,
    "StateVerb": is_in_state,
    "MotionVerb": has_movement,
    "hasNMod": is_nmod,
    "hasNModPartOf": has_nmod_part_of,
    "hasNModIsA": is_a,
    "isSymmetricalIfComparedToNMod": is_symmetrical,
    "Preposition": match_prepositions,
    "sourceIsVerb": source_is_verb,
    "MeansVerb": has_means,
    "PhrasalVerb": is_phrasal_verb
}

def get_matching_logical_rules(kernel, initial_node, has_nmod, _debug_name=None):
    node = initial_node
    if has_nmod:
        node = node.kernel.target

    for rule_key in logical_rules:
        rule = logical_rules[rule_key]
        if all(map(lambda premise: any(map(lambda value: predicate_interpretation[premise.name](kernel, node, initial_node, has_nmod, value), premise.values)), rule.premises)) and all(map(lambda m: any(map(lambda x: not predicate_interpretation[m.name](kernel, node, initial_node, has_nmod, x), m.values)), rule.not_premises)): return node, rule
    return node, None
