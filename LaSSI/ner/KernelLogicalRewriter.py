__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__status__ = "Production"

from collections import defaultdict

from LaSSI.external_services.Services import Services
from LaSSI.ner.HOnKLogicalRewriting import get_matching_logical_rules
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons, Grouping
from LaSSI.structures.kernels.SentenceX import get_prepositions


class KernelLogicalRewriter:
    """Rule-driven property rewriting on top of HOnK's logical-analysis rules.

    Walks a kernel's property bag, asks `get_matching_logical_rules` (the
    HOnK rule matcher) which logical construct each property value belongs to,
    and re-keys it under that construct (SPACE/TIME/CAUSATION/...).
    Construct names and their attach-to/argument metadata come from the
    `LogicalFunction` entries in the ontology — this class never hardcodes a
    construct name."""

    def __init__(self, node_functions, matchers):
        self.services = Services.getInstance()
        self.node_functions = node_functions
        self.matchers = matchers

    def rewrite_properties_logically(self, kernel):
        # TODO: Need to check that properties of properties are encompassed
        properties_to_add = defaultdict(list)
        force_update = False
        if isinstance(kernel, Singleton):
            for key in dict(kernel.properties):
                properties_key_ = dict(kernel.properties)[key]
                if key == 'extra':
                    # As to not continually rewrite previously rewritten nodes
                    properties_to_add[key] = properties_key_
                    continue

                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        properties_to_add = self.rewrite_node_logically(
                            kernel, properties_key_, dict(kernel.properties), type_key=key)
                    else:
                        for prop_node in properties_key_:
                            if isinstance(prop_node, Singleton):
                                if prop_node.kernel is not None:
                                    if prop_node.kernel.edgeLabel.named_entity in DependencyRoles.nominal_modifier_edges():
                                        if isinstance(prop_node.kernel.target, SetOfSingletons):
                                            target_sos = prop_node.kernel.target
                                            inner = target_sos
                                            while isinstance(inner, SetOfSingletons) and inner.entities:
                                                inner = inner.entities[0]
                                            _, key_to_use = self.rewrite_node_logically(
                                                kernel, inner, defaultdict(list), False, True)
                                            if key_to_use is not None:
                                                properties_to_add[key_to_use].append(target_sos)
                                        else:
                                            properties_to_add = self.rewrite_node_logically(
                                                kernel, prop_node, properties_to_add, True)
                                            kernel, properties_to_add, force_update = self.check_property_replacement(kernel, properties_to_add)
                                    elif prop_node.type == "SENTENCE":
                                        prop_node = self.rewrite_properties_logically(prop_node)
                                        properties_to_add = self.rewrite_node_logically(kernel, prop_node, properties_to_add, type_key=key)
                                else:
                                    properties_to_add = self.rewrite_node_logically(kernel, prop_node, properties_to_add, type_key=key)
                            elif isinstance(prop_node, SetOfSingletons):
                                rewritten_entities = []
                                key_to_use = ""
                                for entity in prop_node.entities:
                                    entity, key_to_use = self.rewrite_node_logically(kernel, entity, defaultdict(list), False, True)
                                    rewritten_entities.append(entity)
                                prop_node = prop_node.update_entities(rewritten_entities)

                                if key_to_use is not None:
                                    if isinstance(prop_node, SetOfSingletons) and prop_node.type == Grouping.NOT:
                                        # NOT wrapper: add directly to avoid an extra NONE outer layer.
                                        properties_to_add[key_to_use].append(prop_node)
                                    else:
                                        # Wrap the rewritten entities under the original Grouping if it
                                        # corresponds to one (AND/OR/...); otherwise default to NONE.
                                        grouping_type = Grouping[key] if key in Grouping.__members__ else Grouping.NONE
                                        properties_to_add[key_to_use].append(SetOfSingletons(
                                            id=prop_node.id,
                                            type=grouping_type,
                                            entities=tuple([prop_node]),
                                            min=min(rewritten_entities, key=lambda x: x.min).min,
                                            max=max(rewritten_entities, key=lambda x: x.max).max,
                                            confidence=1
                                        ))
                                    kernel.update_node_props(properties_to_add)
                                else:
                                    properties_to_add[key].append(prop_node)
                else:
                    properties_to_add[key] = properties_key_

        # Rewrite source and target properties
        if hasattr(kernel, "kernel") and kernel.kernel is not None:
            kernel = kernel.update_kernel(self.rewrite_properties_logically(kernel.kernel.source), 'source') if kernel.kernel.source is not None else kernel
            kernel = kernel.update_kernel(self.rewrite_properties_logically(kernel.kernel.target), 'target') if kernel.kernel.target is not None else kernel

        if len(properties_to_add) > 0 or force_update:
            return kernel.update_node_props(properties_to_add)
        else:
            return kernel

    def rewrite_node_logically(self, kernel, initial_node, properties, has_nmod=False, return_key=False, type_key=None):
        honk = self.services.getHOnK()
        initial_node = self._normalize_fixed_expression(initial_node, honk)
        prop_node, selected_rule = get_matching_logical_rules(
            kernel, initial_node, has_nmod, structural_context=self.matchers.logical_rule_context(initial_node))
        prepositions = get_prepositions(prop_node)
        number_value = dict(prop_node.properties)["nummod"] if "nummod" in dict(prop_node.properties) else None

        if selected_rule is not None:
            # Remove prepositions (so long as construct property is not None) as no longer needed
            def _is_position_key(k):
                try:
                    float(k)
                    return True
                except (TypeError, ValueError):
                    return False

            node_props = {k: v for k, v in dict(prop_node.properties).items() if (
                    _is_position_key(k)
            ) or (
                    isinstance(v, str) and
                    v.lower() not in prepositions
            ) or (
                    not isinstance(v, str)
            ) or (
                    k not in {'nummod'} and
                    selected_rule.logicalConstructName in {'quantity', 'measure'}
            )}

            selected_function = honk.get_logical_functions(selected_rule.logicalConstructName, selected_rule.logicalConstructProperty)

            if len(selected_function) == 0 and selected_rule.logicalConstructName:
                # Rule matched but no LogicalFunction entry exists in the ontology yet.
                # Fall back to using the rule's construct name as the type key so the
                # node is still classified (e.g. time_status → TIME_STATUS).
                type_key = selected_rule.logicalConstructName.upper()

            if len(selected_function) > 0:
                selected_function = selected_function[0]  # Use first function in list
                type_key = selected_function.logicalConstructName.upper()
                logical_type = selected_function.logicalConstructProperty

                if selected_function.attachTo == "Singleton":
                    if logical_type is not None:
                        if type_key == "SPECIFICATION":
                            if logical_type == "inverse":
                                if 'extra' in node_props:
                                    if initial_node.kernel.source.id not in [x.id for x in node_props["extra"] if isinstance(x, Singleton)]:
                                        node_props["extra"] = list(node_props["extra"])
                                        node_props["extra"].append(initial_node.kernel.source)
                                else:
                                    node_props["extra"] = [initial_node.kernel.source]
                            else:
                                node_to_add = prop_node
                                prop_node = initial_node.kernel.source
                                node_props = dict(prop_node.get_props())
                                if 'extra' in node_props:
                                    if node_to_add.id not in [x.id for x in node_props["extra"] if isinstance(x, Singleton)]:
                                        node_props["extra"] = list(node_props["extra"])
                                        node_props["extra"].append(node_to_add)
                                else:
                                    node_props["extra"] = [node_to_add]
                        elif logical_type is not None:
                            node_props["type"] = logical_type
                        # When the relation's source is a conjunction
                        # (AND/OR/...), only one conjunct genuinely owns the
                        # nmod target. The conjunct closest in surface order
                        # to `node_to_add` is the original modifier head;
                        # apply the rewrite to that conjunct alone so the
                        # `extra` doesn't bleed across the whole group.
                        if (
                            isinstance(prop_node, SetOfSingletons)
                            and 'extra' in node_props
                            and isinstance(node_to_add, Singleton)
                        ):
                            target_min = node_to_add.min
                            best_idx = None
                            best_gap = None
                            for idx, entity in enumerate(prop_node.entities):
                                if not isinstance(entity, Singleton):
                                    continue
                                if entity.max <= target_min:
                                    gap = target_min - entity.max
                                    if best_gap is None or gap < best_gap:
                                        best_gap = gap
                                        best_idx = idx
                            new_entities = list(prop_node.entities)
                            if best_idx is not None:
                                entity = new_entities[best_idx]
                                entity_props = dict(entity.properties)
                                if 'extra' in entity_props:
                                    existing = entity_props['extra']
                                    existing = list(existing) if isinstance(existing, (list, tuple)) else [existing]
                                    if node_to_add.id not in [x.id for x in existing if isinstance(x, Singleton)]:
                                        existing.append(node_to_add)
                                    entity_props['extra'] = existing
                                else:
                                    entity_props['extra'] = [node_to_add]
                                new_entities[best_idx] = entity.update_node_props(entity_props)
                            prop_node = prop_node.update_entities(new_entities)
                        else:
                            prop_node = prop_node.update_node_props(node_props)

                    val_to_add = prop_node if number_value is None or (number_value is not None and not hasattr(selected_function, "hasNumber")) else number_value
                    if isinstance(val_to_add, Singleton):
                        val_to_add = self.rewrite_properties_logically(val_to_add)
                        if val_to_add.id not in [x.id for x in properties[type_key] if isinstance(x, Singleton)]:
                            properties[type_key].append(val_to_add)
                    else:
                        if val_to_add not in properties[type_key]:
                            properties[type_key].append(val_to_add)

                elif selected_function.attachTo == "Kernel":
                    if logical_type is not None:
                        node_props["type"] = logical_type
                        prop_node = prop_node.update_node_props(node_props)

                    val_to_add = prop_node
                    if selected_function.argument == "subject" and hasattr(prop_node, 'kernel') and prop_node.kernel is not None:
                        # TODO: This might be too hacky, return to this later... (e.g. take[mark:while](?, place)[noun:maintenance work]
                        #  I think really the source should just be maintenance work but it is a property for some reason
                        _props = dict(prop_node.properties)
                        # Only apply the len==1 shortcut for raw (non-classified) properties.
                        # Already-classified keys (e.g. TIME, SPACE) are logical slots, not entity subjects.
                        _non_logical_keys = [k for k in _props if isinstance(k, str) and not k.isupper()]
                        if prop_node.kernel.source.type == 'existential' and len(_props) == 1 and _non_logical_keys:
                            val_to_add = _props[_non_logical_keys[0]][0]
                        elif (
                            selected_rule.logicalConstructName == "temporal_context" and
                            prop_node.kernel.source.type == 'existential' and
                            prop_node.kernel.target is not None and
                            isinstance(prop_node.kernel.target, Singleton)
                        ):
                            # Occurrence clause: the kernel target is the entity that occurred
                            # (e.g. "after storm damage occurred" → target=storm_damage).
                            val_to_add = prop_node.kernel.target
                        else:
                            val_to_add = prop_node.kernel.source

                        if isinstance(val_to_add, Singleton):
                            # Incorporate the verb (edge label) as the 'type' of the subject
                            edge_name = prop_node.kernel.edgeLabel.named_entity
                            prefix_terms = (
                                    {str(c).lower() for c in honk.getConjunctions()} |
                                    {str(p).lower() for p in honk.getPrepositions()} |
                                    {str(p).lower() for p in honk.getPrototypicalPrepositions()}
                            )
                            base_parts = [p for p in edge_name.split(' ') if p]
                            while base_parts and base_parts[0].lower() in prefix_terms:
                                base_parts = base_parts[1:]
                            base_verb = ' '.join(base_parts)

                            if prop_node.kernel.target is not None and hasattr(prop_node.kernel.target, 'named_entity'):
                                combined_verb = f"{base_verb} {prop_node.kernel.target.named_entity}"
                                if any(combined_verb.lower() == str(v).lower() for v in honk.getPhrasalVerbs() if v):
                                    base_verb = combined_verb

                            v_props = dict(val_to_add.properties)
                            v_props['type'] = base_verb
                            val_to_add = val_to_add.update_node_props(v_props)

                        if (
                                selected_rule.logicalConstructName == "temporal_context" and
                                isinstance(prop_node.kernel.target, Singleton)
                        ):
                            properties = self.rewrite_node_logically(
                                kernel,
                                prop_node.kernel.target,
                                properties,
                            )
                        for embedded_key, embedded_value in dict(prop_node.properties).items():
                            if (
                                    isinstance(embedded_key, str) and
                                    embedded_key.isupper() and
                                    isinstance(embedded_value, (list, tuple))
                            ):
                                existing_ids = {
                                    x.id for x in properties[embedded_key]
                                    if isinstance(x, Singleton)
                                }
                                for embedded_node in embedded_value:
                                    if (
                                            not isinstance(embedded_node, Singleton) or
                                            embedded_node.id not in existing_ids
                                    ):
                                        properties[embedded_key].append(embedded_node)
                                        if isinstance(embedded_node, Singleton):
                                            existing_ids.add(embedded_node.id)

                    if isinstance(val_to_add, Singleton):
                        val_to_add = self.rewrite_properties_logically(val_to_add)
                        if val_to_add.id not in [x.id for x in properties[type_key] if isinstance(x, Singleton)]:
                            properties[type_key].append(val_to_add)
                    else:
                        if val_to_add not in properties[type_key]:
                            properties[type_key].append(val_to_add)

                # Apply any additional classifications stored on this rule (e.g. "on or
                # near" which maps to both "stay in place" and "near place").
                # When an additional classification lands in the same property key as the
                # primary, the same-id deduplication check would suppress it, so we wrap
                # the primary and the additional entries together as an OR SetOfSingletons.
                for add_name, add_prop in (selected_rule.additional_classifications or []):
                    add_functions = honk.get_logical_functions(add_name, add_prop)
                    if not add_functions:
                        continue
                    add_fn = add_functions[0]
                    add_type_key = add_fn.logicalConstructName.upper()
                    add_logical_type = add_fn.logicalConstructProperty
                    if add_fn.attachTo in ("Kernel", "Singleton") and add_logical_type is not None:
                        extra_node_props = dict(prop_node.properties)
                        extra_node_props["type"] = add_logical_type
                        extra_prop_node = prop_node.update_node_props(extra_node_props)
                        extra_prop_node = self.rewrite_properties_logically(extra_prop_node)
                        if add_type_key == type_key:
                            existing = [x for x in properties[add_type_key] if isinstance(x, Singleton) and x.id == prop_node.id]
                            if existing:
                                primary_entry = existing[0]
                                primary_type = dict(primary_entry.properties).get('type', '')
                                merged_props = dict(primary_entry.properties)
                                merged_props['type'] = self._merge_disjunctive_property_value(
                                    primary_entry, 'type', primary_type, add_logical_type)
                                merged_entry = primary_entry.update_node_props(merged_props)
                                properties[add_type_key] = [x for x in properties[add_type_key] if not (isinstance(x, Singleton) and x.id == prop_node.id)]
                                properties[add_type_key].append(merged_entry)
                            else:
                                properties[add_type_key].append(extra_prop_node)
                        else:
                            if extra_prop_node.id not in [x.id for x in properties[add_type_key] if isinstance(x, Singleton)]:
                                properties[add_type_key].append(extra_prop_node)

        if return_key:
            return prop_node, type_key
        else:
            # If we don't find a rule, or function, add the initial node to its original key
            if selected_rule is None or (isinstance(selected_function, list) and len(selected_function) == 0):
                type_key = type_key if type_key is not None else initial_node.type
                if not isinstance(properties[type_key], (list, tuple)):
                    properties[type_key] = [initial_node]
                else:
                    if initial_node.id not in [x.id for x in properties[type_key] if isinstance(x, Singleton)]:
                        properties[type_key].append(initial_node)
            return properties

    def _normalize_fixed_expression(self, node, honk):
        """If a node has an amod property and inserting it forms a known HOnK concept, incorporate it.
        Also checks if prepending the preposition (+ amod) forms a known fixed phrase in the ontology
        (e.g. 'until further notice'), and if so uses the full phrase as the entity name."""
        if not isinstance(node, Singleton):
            return node
        node_props = dict(node.properties)
        amod_vals = node_props.get('amod', [])
        if isinstance(amod_vals, str): amod_vals = [amod_vals]

        for amod_val in amod_vals:
            if not isinstance(amod_val, str):
                continue
            # Check: preposition + amod + entity → known fixed phrase (e.g. "until further notice")
            prepositions = get_prepositions(node)
            for prep in sorted(prepositions):
                candidate = f"{prep} {amod_val} {node.named_entity}"
                if len(honk.typeOf(candidate)) > 0:
                    updated_props = {k: v for k, v in node_props.items() if k != 'amod'}
                    remaining_amods = [a for a in amod_vals if a != amod_val]
                    if remaining_amods:
                        updated_props['amod'] = tuple(remaining_amods)
                    return node.update_name(candidate).update_node_props(updated_props)

        # Multi-preposition fixed phrase: e.g. "out of use", "out of service"
        normalized = self._normalize_multi_preposition_phrase(node, node_props, honk)
        if normalized is not None:
            return normalized

        for amod_val in amod_vals:
            if not isinstance(amod_val, str):
                continue
            words = node.named_entity.split(' ')
            if len(words) < 2:
                continue
            candidate = ' '.join(words[:-1] + [amod_val, words[-1]])
            if len(honk.typeOf(candidate)) > 0:
                updated_props = {k: v for k, v in node_props.items() if k != 'amod'}
                remaining_amods = [a for a in amod_vals if a != amod_val]
                if remaining_amods:
                    updated_props['amod'] = tuple(remaining_amods)
                return node.update_name(candidate).update_node_props(updated_props)
        return node

    def _normalize_multi_preposition_phrase(self, node, node_props, honk):
        if not node.named_entity:
            return None
        prepositions = list(get_prepositions(node))
        if len(prepositions) < 2:
            return None

        position_ordered = []
        for k, v in node_props.items():
            if not isinstance(v, str):
                continue
            try:
                pos = float(k)
            except (TypeError, ValueError):
                continue
            position_ordered.append((pos, v.lower()))
        position_ordered.sort(key=lambda x: x[0])
        ordered_preps = [v for _, v in position_ordered if v in {p.lower() for p in prepositions}]

        candidates = []
        if ordered_preps and len(ordered_preps) >= 2:
            candidates.append(" ".join(ordered_preps + [node.named_entity]))
        candidates.append(" ".join(sorted(prepositions) + [node.named_entity]))
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            types = honk.typeOf(candidate)
            if not types:
                continue
            consumed = {p.lower() for p in prepositions}
            stripped_props = {}
            for k, v in node_props.items():
                if isinstance(v, str) and v.lower() in consumed:
                    try:
                        float(k)
                        continue
                    except (TypeError, ValueError):
                        if k in DependencyRoles.preposition_marker_labels():
                            continue
                stripped_props[k] = v
            new_node = node.update_name(candidate).update_node_props(stripped_props)
            if any('JJ' in str(t) or 'Adjective' in str(t) for t in types):
                new_node = new_node.update_type('JJ')
            return new_node
        return None

    def _logical_property_value_node(self, property_name, name, reference_node):
        return Singleton(
            id=self.node_functions.fresh_id(),
            named_entity=name,
            properties=frozenset(),
            min=reference_node.min,
            max=reference_node.max,
            type=f"logical_{property_name}",
            confidence=reference_node.confidence,
            kernel=None,
        )

    def _merge_disjunctive_property_value(self, reference_node, property_name, *property_values):
        entities = []
        seen = set()
        for property_value in property_values:
            if isinstance(property_value, SetOfSingletons):
                candidates = property_value.entities
            elif property_value:
                candidates = (self._logical_property_value_node(property_name, property_value, reference_node),)
            else:
                candidates = ()
            for candidate in candidates:
                name = candidate.named_entity if isinstance(candidate, Singleton) else candidate.get_name()
                if name not in seen:
                    seen.add(name)
                    entities.append(candidate)
        if len(entities) == 1:
            return entities[0].named_entity if isinstance(entities[0], Singleton) else entities[0]
        return SetOfSingletons(
            id=self.node_functions.fresh_id(),
            type=Grouping.OR,
            entities=tuple(entities),
            min=reference_node.min,
            max=reference_node.max,
            confidence=reference_node.confidence,
        )

    def check_property_replacement(self, kernel, properties):
        _be_forms = self.services.getHOnK().getCopulaSurfaceForms()
        _copula_types = DependencyRoles.copula_complement_pos_tags()
        edge_label_name = kernel.kernel.edgeLabel.named_entity.lower().strip() if (kernel.kernel and kernel.kernel.edgeLabel) else ""
        is_copula_kernel = (
            edge_label_name in _be_forms and
            kernel.kernel.target is not None and
            kernel.kernel.target.type in _copula_types
        )
        nodes_to_remove = []
        force_update = False
        for key in properties:
            for prop_node in properties[key]:
                if isinstance(prop_node, Singleton):
                    if kernel.kernel.source is not None and kernel.kernel.source.id == prop_node.id and (prop_node.id != -1 or kernel.kernel.source.named_entity == prop_node.named_entity):
                        kernel = kernel.update_kernel(prop_node, "source")
                        nodes_to_remove.append(prop_node)
                    elif not is_copula_kernel and kernel.kernel.target is not None and (
                        (kernel.kernel.target.id == prop_node.id and (prop_node.id != -1 or kernel.kernel.target.named_entity == prop_node.named_entity)) or
                        ('extra' in dict(prop_node.properties) and len([x for x in list(dict(prop_node.properties)['extra']) if x.id == kernel.kernel.target.id and (x.id != -1 or x.named_entity == kernel.kernel.target.named_entity)]) > 0)
                    ):
                        if key == 'INSTRUMENT':
                            kernel = kernel.update_kernel(None, "target")
                        else:
                            kernel = kernel.update_kernel(prop_node, "target")
                            nodes_to_remove.append(prop_node)
        for remove_node in nodes_to_remove:
            for key, value in dict(properties).items():
                force_update = True
                properties[key] = [x for x in properties[key] if x.id != remove_node.id]
        return kernel, properties, force_update
