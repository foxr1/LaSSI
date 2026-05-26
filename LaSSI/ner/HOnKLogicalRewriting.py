from LaSSI.structures.kernels.Sentence import get_prepositions

from LaSSI.external_services.Services import Services
from LaSSI.ner.SemanticRoleRewriting import normalized_node_type, select_best_matching_rule
from LaSSI.ner.string_functions import lemmatize_verb, is_label_verb, normalise_apostrophes, surface_form_variants

honk = Services.getInstance().getHOnK()

def is_name_in_honk(name, honk_list, should_lemmatize=False):
    candidates = {lemmatize_verb(name), name} if should_lemmatize else {name}
    return bool({str(candidate).lower() for candidate in candidates if candidate} &
                {str(item).lower() for item in honk_list if item})

def _node_name_candidates(node):
    name = node.get_name() if hasattr(node, "get_name") else getattr(node, "named_entity", "")
    candidates = {name, lemmatize_verb(name)}
    for variant in surface_form_variants(name):
        candidates.add(variant)
        candidates.add(lemmatize_verb(variant))
    for part in str(name).split():
        candidates.add(part)
        candidates.add(lemmatize_verb(part))
        for variant in surface_form_variants(part):
            candidates.add(variant)
            candidates.add(lemmatize_verb(variant))
    return {candidate for candidate in candidates if candidate}

def _is_node_in_honk_set(node, honk_list, value):
    candidates = {str(candidate).lower() for candidate in _node_name_candidates(node)}
    ontology_terms = {str(term).lower() for term in honk_list if term}
    return bool(candidates & ontology_terms) == value

def is_nmod(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return has_nmod == value

def _actioned_prepositions(node):
    props = dict(getattr(node, "properties", {}) or {})
    prep_terms = {
        str(p).strip().lower()
        for p in (set(honk.getPrepositions()) | set(honk.getPrototypicalPrepositions()))
        if p
    }
    found = set()
    for key in ("action", "actioned"):
        raw_value = props.get(key)
        if raw_value is None:
            continue
        values = raw_value if isinstance(raw_value, (list, tuple, set, frozenset)) else [raw_value]
        for value in values:
            if not isinstance(value, str):
                continue
            candidate = value.strip().lower()
            if candidate in prep_terms:
                found.add(candidate)
    return found

def match_prepositions(kernel, node, initial_node, has_nmod, value, structural_context=None):
    prepositions = set(get_prepositions(node)) | _actioned_prepositions(node)
    return value in prepositions

def is_materialised(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getMaterialisationVerbs(), True) == value

def is_causative(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if structural_context is not None:
        incoming_edge_labels = structural_context.get("incoming_edge_labels", ())
        if incoming_edge_labels and not structural_context.get("has_case", False):
            has_incoming_causative = any(
                is_name_in_honk(edge_label, honk.getCausativeVerbs(), True)
                for edge_label in incoming_edge_labels
            )
            if has_incoming_causative:
                return has_incoming_causative == value

    if has_nmod and initial_node.kernel is not None:
        source = initial_node.kernel.source
        edge_label = initial_node.kernel.edgeLabel
        candidates = []
        if source is not None:
            candidates.append(source.get_name())
        if edge_label is not None:
            candidates.append(edge_label.named_entity)
        if candidates:
            return any(is_name_in_honk(candidate, honk.getCausativeVerbs(), True) for candidate in candidates) == value

    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getCausativeVerbs(), True) == value

def has_movement(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getMovementVerbs(), True) == value

def is_consumption(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if kernel.kernel is None:
        return False
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_honk(edge_label, honk.getConsumptionVerbs(), True) == value

def is_in_state(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return is_name_in_honk(node.named_entity, honk.getStateVerbs(), True) == value

def has_means(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if has_nmod:
        node = initial_node.kernel.source
    return is_name_in_honk(node.get_name(), honk.getMeansVerbs(), False) == value

def is_abstract_entity(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return is_name_in_honk(node.named_entity, honk.getAbstractEntities(), True) == value

def is_location_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getLocationNouns(), value)

def is_state_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getStateNouns(), value)

def is_facility_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getFacilityNouns(), value)

def is_access_point_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getAccessPointNouns(), value)

def is_route_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getRouteNouns(), value)

def is_service_state_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getServiceStateNouns(), value)

def is_status_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getStatusNouns(), value)

def is_weather_condition_noun(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getWeatherConditionNouns(), value)

def is_weather_condition_adjective(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getWeatherConditionAdjectives(), value)

def is_prediction_verb(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return _is_node_in_honk_set(node, honk.getPredictionVerbs(), value)

def has_measurement(kernel, node, initial_node, has_nmod, value, structural_context=None):
    # TODO: Aware this is definitely wrong, what is the better way of resolving the measurement?
    if has_nmod:
        possible_measurement = f"{initial_node.kernel.source.get_name()} per {initial_node.kernel.target.get_name()}"
    else:
        possible_measurement = ""
    return is_name_in_honk(possible_measurement, honk.getUnitsOfMeasure(), False) == value

def has_nmod_part_of(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if kernel.kernel is None:
        return False
    return (kernel.kernel.source.id == (initial_node.kernel.source.id if initial_node.kernel is not None else node.id)) == value

def is_symmetrical(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return (len(
        {x for x in honk.getNounsWithProperties() if lemmatize_verb(node.named_entity) == str(x.label) and (initial_node.kernel is not None and initial_node.kernel.source.named_entity in str(x.hasProperty))}
    ) > 0) == value

def is_a(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return (len({x for x in honk.getNounsWithA() if
         node.named_entity == str(x.label) and initial_node.kernel.source.named_entity in str(x.isA)}) > 0) == value

def has_number(kernel, node, initial_node, has_nmod, value, structural_context=None):
    node_has_number = "nummod" in dict(getattr(node, "properties", {}) or {})
    if node_has_number:
        return True == value

    if has_nmod:
        # A numeric source should only license the nmod target as numeric when
        # the pair itself is a measurement ("miles per second").  Dates such as
        # "in January 2026 following an incident" also carry nummod on the
        # source, but their nmod target is an event context, not a quantity.
        source = initial_node.kernel.source if getattr(initial_node, "kernel", None) is not None else None
        source_has_number = "nummod" in dict(getattr(source, "properties", {}) or {})
        if source_has_number and has_measurement(kernel, node, initial_node, has_nmod, True, structural_context):
            return True == value
        return False == value

    return False == value

def has_actioned_status(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return bool({'action', 'actioned'} & set(dict(node.properties))) == value

def type_of_node(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return normalized_node_type(node, honk) == value

def source_type_of_node(kernel, node, initial_node, has_nmod, value, structural_context=None):
    source = initial_node.kernel.source if getattr(initial_node, "kernel", None) is not None else None
    return normalized_node_type(source, honk) == value

def dependency_label(kernel, node, initial_node, has_nmod, value, structural_context=None):
    if getattr(initial_node, "kernel", None) is not None and initial_node.kernel.edgeLabel is not None:
        if initial_node.kernel.edgeLabel.named_entity == value:
            return True
    if structural_context is not None:
        return value in structural_context.get("dependency_labels", ())
    return False

def source_is_verb(kernel, node, initial_node, has_nmod, value, structural_context=None):
    return is_label_verb(node.named_entity) == value

def is_phrasal_verb(kernel, node, initial_node, has_nmod, value, structural_context=None):
    """Check if a SENTENCE node's edge label is or contains a phrasal verb.

    Subordinate clauses may preserve their marker in the edge label
    (e.g. "while take place").  HOnK stores the lexical phrasal verb itself
    ("take place"), so strip leading conjunctions/prepositions before matching.
    """
    if (hasattr(node, 'kernel') and node.kernel is not None and
            node.kernel.edgeLabel is not None):
        edge_name = node.kernel.edgeLabel.named_entity
        prefix_terms = (
                {str(c).lower() for c in honk.getConjunctions()} |
                {str(p).lower() for p in honk.getPrepositions()} |
                {str(p).lower() for p in honk.getPrototypicalPrepositions()}
        )
        parts = [p for p in edge_name.split(' ') if p]
        candidates = {edge_name}
        while parts and parts[0].lower() in prefix_terms:
            parts = parts[1:]
            if parts:
                candidates.add(' '.join(parts))
        
        if hasattr(node.kernel, 'target') and node.kernel.target is not None and hasattr(node.kernel.target, 'named_entity'):
            target_name = node.kernel.target.named_entity
            new_candidates = set()
            for c in candidates:
                new_candidates.add(f"{c} {target_name}")
            candidates.update(new_candidates)

        return any(is_name_in_honk(candidate, honk.getPhrasalVerbs(), False) for candidate in candidates) == value
    return not value

predicate_interpretation = {
    "MaterialisationVerb": is_materialised,
    "CausativeVerb": is_causative,
    "ConsumptionVerb": is_consumption,
    "SingletonHasBeenMatchedBy": type_of_node,
    "SourceSingletonHasBeenMatchedBy": source_type_of_node,
    "DependencyLabel": dependency_label,
    "Dependency": dependency_label,
    "AbstractEntity": is_abstract_entity,
    "Number": has_number,
    "UnitOfMeasure": has_measurement,
    "LocationNoun": is_location_noun,
    "StateNoun": is_state_noun,
    "FacilityNoun": is_facility_noun,
    "AccessPointNoun": is_access_point_noun,
    "RouteNoun": is_route_noun,
    "ServiceStateNoun": is_service_state_noun,
    "StatusNoun": is_status_noun,
    "WeatherConditionNoun": is_weather_condition_noun,
    "WeatherConditionAdjective": is_weather_condition_adjective,
    "PredictionVerb": is_prediction_verb,
    "StateVerb": is_in_state,
    "Actioned": has_actioned_status,
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

def get_matching_logical_rules(kernel, initial_node, has_nmod, _debug_name=None, structural_context=None):
    node = initial_node
    if has_nmod:
        node = node.kernel.target

    def predicate_matches(name, value):
        return (
                name in predicate_interpretation and
                predicate_interpretation[name](kernel, node, initial_node, has_nmod, value, structural_context)
        )

    selected_rule = select_best_matching_rule(honk.getLogicalRewritingRules().values(), predicate_matches)
    if selected_rule is not None:
        return node, selected_rule
    return node, None
