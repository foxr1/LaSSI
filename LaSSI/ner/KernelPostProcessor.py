__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__status__ = "Production"

import itertools
import re
import string
from collections import defaultdict

import networkx as nx

from LaSSI.external_services.Services import Services
from LaSSI.ner.HOnKLogicalRewriting import get_matching_logical_rules
from LaSSI.ner.KernelLogicalRewriter import KernelLogicalRewriter
from LaSSI.ner.KernelOntologyMatchers import KernelOntologyMatchers
from LaSSI.ner.MergeSetOfSingletons import merge_properties
from LaSSI.ner.node_functions_X import create_props_for_singleton
from LaSSI.ner.structural_rewrites import RewriteContext, default_registry
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons, Relationship, Grouping
from LaSSI.structures.kernels.SentenceX import is_kernel_in_props, is_node_in_kernel_nodes


# Structural property keys that always survive `_filter_invalid_property_keys`
# even when not present as ontology rule constructs.
_STRUCTURAL_PROPERTY_KEYS = frozenset({'SENTENCE'})

# `Grouping` enum names; valid kernel-property keys when the grouping enum is
# stringified into a property key (rather than as an enum) by upstream code.
_GROUPING_PROPERTY_KEYS = frozenset(g.name for g in Grouping)


class KernelPostProcessor:
    """Post-loop transformation pipeline for the final kernel.

    `run(final_kernel, acl_relcl_map)` applies the explicit `PIPELINE` list of
    transformations in order. Each step is a named method here; adding,
    removing, or reordering steps means editing one list."""

    def __init__(self, G, negations, node_functions):
        self.services = Services.getInstance()
        self.G = G
        self.negations = negations
        self.node_functions = node_functions
        self.matchers = KernelOntologyMatchers(G=G)
        self.rewriter = KernelLogicalRewriter(node_functions, self.matchers)
        self.structural_rewrites = default_registry()
        self._rewrite_ctx = RewriteContext(
            node_functions=node_functions,
            services=self.services,
            matchers=self.matchers,
        )
        self._valid_property_keys_cache = None

    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------

    def run(self, final_kernel):
        """Apply the post-processing pipeline in order. The orchestrator runs
        `acl_replacement` (and then `check_for_action_ed_node`) before calling
        this — those mutate the graph and need access to `create_sentence`,
        so they don't sit cleanly inside this list. Everything *after* that
        point lives here.

        Ordering matters: later steps assume the rewrites done by earlier ones.
        Adding, removing, or reordering a step means editing this list.

        Pattern-driven structural rewrites are applied at named phases via
        `_apply_structural_rules(phase)`. Each rule is a class registered in
        `LaSSI/ner/structural_rewrites/__init__.py` — see that module's
        docstring for the registration recipe."""
        pipeline = [
            ('dedupe_properties',                     self.remove_duplicate_properties),
            ('fold_phrasal_advs',                     self.check_for_adv),
            ('promote_contextual_sentence_kernel',     self.promote_contextual_sentence_kernel),
            ('promote_embedded_logical_props',        self.promote_embedded_logical_properties),
            ('rewrite_properties_logically',          self.rewrite_properties_logically),
            ('rules:post_logical_rewrite',            lambda k: self._apply_structural_rules(k, "post_logical_rewrite")),
            ('lift_participial_logical_context',       self.lift_participial_logical_context),
            ('nest_of_terms_under_causation',         self.nest_oft_terms_under_causation),
            # `rewrite_properties_logically` can introduce duplicates (e.g. SPECIFICATION);
            # second dedupe pass cleans those up before the structural rewrites below.
            ('dedupe_properties (post-rewrite)',      self.remove_duplicate_properties),
            ('rules:post_nest_dedupe',                lambda k: self._apply_structural_rules(k, "post_nest_dedupe")),
            ('cleanup_space_property',                self.cleanup_space_property),
            ('rules:post_cleanup',                    lambda k: self._apply_structural_rules(k, "post_cleanup")),
            # ('promote_extra_of_space_entities',       self.promote_extra_of_space_entities),
            ('filter_invalid_property_keys',          self.filter_invalid_property_keys),
            ('apply_honk_normalisations',             self.apply_honk_normalisations)
        ]
        for _name, step in pipeline:
            final_kernel = step(final_kernel)
        return final_kernel

    def _apply_structural_rules(self, kernel, phase: str):
        return self.structural_rewrites.apply_phase(kernel, phase, self._rewrite_ctx)

    # Public wrappers around the pipeline-internal logical rewrite. Kept as
    # methods so `kernel_post_processing` in the orchestrator can also fire
    # individual steps without going through the whole pipeline.
    def rewrite_properties_logically(self, kernel):
        return self.rewriter.rewrite_properties_logically(kernel)

    def check_property_replacement(self, kernel, properties):
        return self.rewriter.check_property_replacement(kernel, properties)

    # ------------------------------------------------------------------
    # promote_extra_of_space_entities
    # ------------------------------------------------------------------

    def promote_extra_of_space_entities(self, kernel):
        """Promote ``extra`` properties on SPACE entries that represent a
        location attached via an 'of' preposition into separate SPACE entries
        with ``type:stay in place``.

        Example: "Haymarket area of Newcastle" produces
        ``SPACE:Haymarket area[(extra:Newcastle[(9:of)]), (type:near place)]``.
        Promoting Newcastle to its own SPACE slot with ``type:stay in place``
        lets the ex-post comparison recognise that sentence 1
        ("near Haymarket area of Newcastle") implies sentence 0
        ("on or near … Newcastle") via the shared Newcastle entity.
        """
        if isinstance(kernel, SetOfSingletons):
            new_entities = [self.promote_extra_of_space_entities(e) for e in kernel.entities]
            if any(a is not b for a, b in zip(kernel.entities, new_entities)):
                kernel = kernel.update_entities(new_entities)
            return kernel
        if not isinstance(kernel, Singleton):
            return kernel

        # Recurse into sub-kernels
        new_kernel = kernel
        if kernel.kernel is not None:
            if kernel.kernel.source is not None:
                cleaned = self.promote_extra_of_space_entities(kernel.kernel.source)
                if cleaned is not kernel.kernel.source:
                    new_kernel = new_kernel.update_kernel(cleaned, 'source')
            if kernel.kernel.target is not None:
                cleaned = self.promote_extra_of_space_entities(kernel.kernel.target)
                if cleaned is not kernel.kernel.target:
                    new_kernel = new_kernel.update_kernel(cleaned, 'target')

        props = dict(new_kernel.properties)
        if 'SPACE' not in props:
            return new_kernel

        space_value = props['SPACE']
        space_list = list(space_value) if isinstance(space_value, (list, tuple)) else [space_value]

        existing_names = {
            s.named_entity.strip().lower()
            for s in space_list
            if isinstance(s, Singleton)
        }

        promoted = []
        for entry in space_list:
            if not isinstance(entry, Singleton):
                continue
            extra_val = dict(entry.properties).get('extra')
            if extra_val is None:
                continue
            # Normalise to a list of candidate Singletons
            if isinstance(extra_val, Singleton):
                extra_items = [extra_val]
            elif isinstance(extra_val, (list, tuple)):
                extra_items = [x for x in extra_val if isinstance(x, Singleton)]
            else:
                continue

            for extra_item in extra_items:
                # Check that the extra is attached via an 'of' preposition:
                # properties like (9, 'of') where 9 is a numeric position key.
                has_of_prep = any(
                    isinstance(v, str) and v.strip().lower() == 'of'
                    for _, v in dict(extra_item.properties).items()
                )
                if not has_of_prep:
                    continue
                name = extra_item.named_entity.strip()
                if not name or name.lower() in existing_names:
                    continue
                # Build promoted entity: keep extra_item's base properties
                # (positional keys will be cleaned by filter_invalid_property_keys)
                # but override / set type to 'stay in place'.
                new_props = {k: v for k, v in dict(extra_item.properties).items()}
                new_props['type'] = 'stay in place'
                promoted.append(extra_item.update_node_props(new_props))
                existing_names.add(name.lower())

        if not promoted:
            return new_kernel

        props['SPACE'] = space_list + promoted
        return new_kernel.update_node_props(props)

    # ------------------------------------------------------------------
    # filter_invalid_property_keys
    # ------------------------------------------------------------------

    def _valid_kernel_property_keys(self):
        """Set of property keys that may legally sit on a final kernel: the
        uppercased construct names emitted by HOnK's logical-rewriting rules
        (SPACE/TIME/CAUSATION/...), structural keys (SENTENCE), and Grouping
        enum names. Computed once per instance from the live ontology."""
        if self._valid_property_keys_cache is not None:
            return self._valid_property_keys_cache
        keys = set(_STRUCTURAL_PROPERTY_KEYS) | set(_GROUPING_PROPERTY_KEYS)
        try:
            rules = self.services.getHOnK().getLogicalRewritingRules() or {}
        except Exception:
            rules = {}
        for rule in rules.values():
            if getattr(rule, 'logicalConstructName', None):
                keys.add(rule.logicalConstructName.upper())
                # Some rule names contain a space (e.g. "passive agent_cause") that
                # downstream code normalises to underscores; accept both forms.
                keys.add(rule.logicalConstructName.upper().replace(' ', '_'))
            for name, _ in (getattr(rule, 'additional_classifications', None) or []):
                if name:
                    keys.add(name.upper())
                    keys.add(name.upper().replace(' ', '_'))
        self._valid_property_keys_cache = keys
        return keys

    def filter_invalid_property_keys(self, kernel):
        """Strip top-level property keys that aren't valid semantic types.
        Walks recursively into SENTENCE values and into kernel source/target
        when they themselves carry a sub-kernel."""
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            if isinstance(kernel, SetOfSingletons):
                cleaned_entities = tuple(
                    self.filter_invalid_property_keys(e) for e in kernel.entities
                )
                if cleaned_entities != kernel.entities:
                    kernel = kernel.update_entities(list(cleaned_entities))
            return kernel

        valid_keys = self._valid_kernel_property_keys()
        cleaned_props = {}
        for key, value in dict(kernel.properties).items():
            if key not in valid_keys:
                continue
            if isinstance(value, (list, tuple)):
                cleaned_value = []
                for item in value:
                    if isinstance(item, (Singleton, SetOfSingletons)) and getattr(item, 'kernel', None) is not None:
                        cleaned_value.append(self.filter_invalid_property_keys(item))
                    elif isinstance(item, SetOfSingletons):
                        cleaned_value.append(self.filter_invalid_property_keys(item))
                    else:
                        cleaned_value.append(item)
                cleaned_props[key] = cleaned_value
            else:
                cleaned_props[key] = value

        new_kernel = kernel
        if kernel.kernel.source is not None:
            new_source = self.filter_invalid_property_keys(kernel.kernel.source)
            if new_source is not kernel.kernel.source:
                new_kernel = new_kernel.update_kernel(new_source, 'source')
        if kernel.kernel.target is not None:
            new_target = self.filter_invalid_property_keys(kernel.kernel.target)
            if new_target is not kernel.kernel.target:
                new_kernel = new_kernel.update_kernel(new_target, 'target')

        return new_kernel.update_node_props(cleaned_props)

    # ------------------------------------------------------------------
    # cleanup_space_property
    # ------------------------------------------------------------------

    _SPACE_CASE_MARKER_KEYS = DependencyRoles.space_case_marker_keys()

    # ------------------------------------------------------------------
    # apply_honk_normalisations
    # ------------------------------------------------------------------
    #
    # Two HOnK-driven canonicalisations on the kernel before FOL is built:
    #
    # (1) Part/whole SPACE swap.  `rel(?, Whole)[SPACE: Part, ...]` becomes
    #     `rel(?, Part)[SPACE: Whole, ...]` when HOnK asserts `partOf(Part, Whole)`
    #     (type-level meronymy with head-noun fallback). Canonicalises e.g.
    #     "close(?, Haymarket Station)[SPACE: Percy Street entrance]" and
    #     "close(?, Percy Street entrance)[SPACE: Haymarket Metro station]"
    #     to the same patient/location split.
    #
    # (2) Causation-to-subject promotion.  `rel(?existential, X)[CAUSATION: C]`
    #     becomes `rel(C, X)` when `rel` is a change-of-state verb (HOnK
    #     `getChangeOfStateVerbs()`).  Gated on the verb class because the
    #     causative alternation does not generalise to agentive verbs:
    #     "John waited due to the storm" /= "The storm waited John".

    _CAUSE_PROP_KEYS = ("CAUSATION", "CAUSE")

    @staticmethod
    def _is_existential_source(node):
        if not isinstance(node, Singleton):
            return False
        name = node.named_entity
        return (
            isinstance(name, str) and len(name) >= 2 and name[0] == "?" and
            name[1:].isdigit() and node.type == "existential"
        )

    @staticmethod
    def _first_singleton(value):
        if isinstance(value, Singleton):
            return value
        if isinstance(value, (tuple, list)) and value and isinstance(value[0], Singleton):
            return value[0]
        return None

    @staticmethod
    def _replace_kernel_properties(sentence, new_props_dict):
        return Singleton(
            id=sentence.id,
            named_entity=sentence.named_entity,
            properties=frozenset(new_props_dict.items()),
            min=sentence.min,
            max=sentence.max,
            type=sentence.type,
            confidence=sentence.confidence,
            kernel=sentence.kernel,
        )

    @staticmethod
    def _expanded_surface_form(node):
        """Surface form combining `named_entity` with its `extra` property.

        `named_entity` is just the chosen head (e.g. "Percy Street"). The
        access-point / facility noun ("entrance", "station") often sits in the
        `extra` property after compound merging, so neither the head nor
        `isPartOf`'s rightmost-1/2-token fallback can find it. We surface the
        extras by concatenating them after the head; `isPartOf` then catches
        the facility noun via its existing head-noun fallback. `extra` may be a
        bare string, a Singleton, or a tuple/list of either.
        """
        if not isinstance(node, Singleton) or not node.named_entity:
            return None
        props = dict(node.properties) if node.properties else {}
        extra = props.get('extra')
        if extra is None:
            return node.named_entity
        if isinstance(extra, str):
            extras = [extra]
        elif isinstance(extra, Singleton):
            extras = [extra.named_entity]
        elif isinstance(extra, (tuple, list)):
            extras = [
                e if isinstance(e, str)
                else (e.named_entity if isinstance(e, Singleton) and e.named_entity else '')
                for e in extra
            ]
        else:
            extras = []
        tail = ' '.join(s.strip() for s in extras if s and s.strip())
        return f"{node.named_entity} {tail}".strip() if tail else node.named_entity

    def _swap_part_whole_space(self, sentence):
        rel = sentence.kernel
        if rel is None:
            return sentence
        target = rel.target
        if not isinstance(target, Singleton) or not target.named_entity:
            return sentence
        props_dict = dict(sentence.properties) if sentence.properties else {}
        space_val = props_dict.get("SPACE")
        part = self._first_singleton(space_val)
        if part is None or not part.named_entity:
            return sentence
        honk = self.services.getHOnK()
        part_label = self._expanded_surface_form(part)
        whole_label = self._expanded_surface_form(target)
        if not honk.isPartOf(part_label, whole_label):
            return sentence

        # The spatial `type` property (`stay in place`, `motion to place`, ...)
        # is the verb's relationship to the SPACE slot, not an attribute of the
        # entity itself. When we swap target<->part, the type stays on the SPACE
        # side rather than travelling with the entity. Transfer those keys from
        # `part.properties` (SPACE-side) onto the new SPACE singleton (formerly
        # the target), and strip them from the new target (formerly the part).
        SPATIAL_TYPE_KEYS = ('type',)
        SPATIAL_TYPE_VALUES = {'stay in place', 'motion to place', 'motion from place', 'near place'}

        def _split_spatial_type(props):
            kept = []
            spatial = []
            for k, v in props:
                if k in SPATIAL_TYPE_KEYS and isinstance(v, str) and v.strip().lower() in SPATIAL_TYPE_VALUES:
                    spatial.append((k, v))
                else:
                    kept.append((k, v))
            return kept, spatial

        part_kept, spatial_props = _split_spatial_type(part.properties or frozenset())
        new_target = Singleton(
            id=part.id,
            named_entity=part.named_entity,
            properties=frozenset(part_kept),
            min=part.min,
            max=part.max,
            type=part.type,
            confidence=part.confidence,
        )
        if spatial_props:
            target_props = list(target.properties or frozenset())
            existing_type_keys = {k for k, _ in target_props if k in SPATIAL_TYPE_KEYS}
            for k, v in spatial_props:
                if k not in existing_type_keys:
                    target_props.append((k, v))
            new_space = Singleton(
                id=target.id,
                named_entity=target.named_entity,
                properties=frozenset(target_props),
                min=target.min,
                max=target.max,
                type=target.type,
                confidence=target.confidence,
            )
        else:
            new_space = target

        new_rel = rel.update_vertex(new_target, 'target')
        new_props = dict(props_dict)
        new_props["SPACE"] = (new_space,)
        new_sentence = self._replace_kernel_properties(sentence, new_props)
        return Singleton(
            id=new_sentence.id,
            named_entity=new_sentence.named_entity,
            properties=new_sentence.properties,
            min=new_sentence.min,
            max=new_sentence.max,
            type=new_sentence.type,
            confidence=new_sentence.confidence,
            kernel=new_rel,
        )

    def _promote_causation_subject(self, sentence):
        rel = sentence.kernel
        if rel is None:
            return sentence
        if not self._is_existential_source(rel.source):
            return sentence
        edge = rel.edgeLabel
        if not isinstance(edge, Singleton) or not edge.named_entity:
            return sentence
        honk = self.services.getHOnK()
        alternators = honk.getChangeOfStateVerbs() or set()
        if edge.named_entity not in alternators:
            return sentence
        props_dict = dict(sentence.properties) if sentence.properties else {}
        cause_key = next((k for k in self._CAUSE_PROP_KEYS if k in props_dict), None)
        if cause_key is None:
            return sentence
        cause = self._first_singleton(props_dict[cause_key])
        if cause is None:
            return sentence
        cause = self._normalise_causal_compound_head(cause)
        new_rel = rel.update_vertex(cause, 'source')
        new_props = {k: v for k, v in props_dict.items() if k != cause_key}
        new_sentence = self._replace_kernel_properties(sentence, new_props)
        return Singleton(
            id=new_sentence.id,
            named_entity=new_sentence.named_entity,
            properties=new_sentence.properties,
            min=new_sentence.min,
            max=new_sentence.max,
            type=new_sentence.type,
            confidence=new_sentence.confidence,
            kernel=new_rel,
        )

    def apply_honk_normalisations(self, kernel):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel
        kernel = self._swap_part_whole_space(kernel)
        kernel = self._promote_causation_subject(kernel)
        if isinstance(kernel.kernel.source, Singleton):
            kernel = kernel.update_kernel(self._normalise_causal_compound_head(kernel.kernel.source), 'source')
        return kernel


    def cleanup_space_property(self, kernel):
        """Tidy up the SPACE property of `kernel` (and recurse into kernel components)
        without touching legitimate disjunctions/conjunctions:
          - Drop entries whose `named_entity` is a bare preposition (e.g. an orphan
            "on" with type "stay in place" that arose from a duplicate-preposition
            sequence like "recorded on On Or Near LOC").
          - When SPACE contains an AND/OR `SetOfSingletons` of two-or-more
            location-like nouns sharing identical case markers, treat it as an
            apposition (e.g. "Edward Place, Newcastle") rather than a real
            conjunction: keep the first entity's case info intact, strip the case
            markers from subsequent entities, and surface them as separate SPACE
            list items."""
        if isinstance(kernel, SetOfSingletons):
            new_entities = [self.cleanup_space_property(e) for e in kernel.entities]
            if any(a is not b for a, b in zip(kernel.entities, new_entities)):
                kernel = kernel.update_entities(new_entities)
            return kernel
        if not isinstance(kernel, Singleton):
            return kernel

        new_kernel = kernel
        if kernel.kernel is not None:
            if kernel.kernel.source is not None:
                cleaned_source = self.cleanup_space_property(kernel.kernel.source)
                if cleaned_source is not kernel.kernel.source:
                    new_kernel = new_kernel.update_kernel(cleaned_source, 'source')
            if kernel.kernel.target is not None:
                cleaned_target = self.cleanup_space_property(kernel.kernel.target)
                if cleaned_target is not kernel.kernel.target:
                    new_kernel = new_kernel.update_kernel(cleaned_target, 'target')

        props = dict(new_kernel.properties)
        if 'SPACE' not in props:
            return new_kernel

        space_value = props['SPACE']
        space_list = list(space_value) if isinstance(space_value, (list, tuple)) else [space_value]
        cleaned_list = []

        for entry in space_list:
            entry = self.cleanup_space_property(entry)
            if self._is_orphan_preposition(entry):
                continue
            unwrapped = self._unwrap_appositional_and(entry)
            if unwrapped is None:
                # Re-apply the SPACE-typing rule (e.g. at → stay in place) on
                # the entry. Singletons that reached SPACE via paths other than
                # `rewrite_properties_logically` -- e.g. the post-logical
                # `passive_progressive` rewrite that lifts the sub-kernel
                # target into SPACE, or kernels constructed via compound-merge
                # of two non-MEU nodes -- have the preposition in their props
                # but were never classified, so their inner `type` is empty.
                # Running the matcher here makes the inner type consistent
                # regardless of how the entry was attached.
                if isinstance(entry, Singleton):
                    entry = self._reapply_space_type_from_prepositions(entry)
                cleaned_list.append(entry)
            else:
                cleaned_list.extend(unwrapped)

        if len(cleaned_list) == len(space_list) and all(a is b for a, b in zip(cleaned_list, space_list)):
            return new_kernel

        if cleaned_list:
            props['SPACE'] = cleaned_list
        else:
            props.pop('SPACE', None)
        return new_kernel.update_node_props(props)

    def _is_orphan_preposition(self, entry):
        if not isinstance(entry, Singleton):
            return False
        if entry.kernel is not None:
            return False
        return self.matchers.is_preposition_name(entry.named_entity)

    def _unwrap_appositional_and(self, entry):
        """If `entry` is a SetOfSingletons (AND/OR) of location-like nouns sharing
        identical case markers (i.e. an apposition like "Edward Place, Newcastle"),
        return a list with the principal entity preserved (with disjunctive SPACE
        type re-applied where applicable) and subsequent entities stripped to a
        bare type. Returns None when the entry should be left alone."""
        if not isinstance(entry, SetOfSingletons):
            return None
        if entry.type not in (Grouping.AND, Grouping.OR):
            return None
        entities = list(entry.entities)
        if len(entities) < 2:
            return None

        case_signatures = []
        for ent in entities:
            if not isinstance(ent, Singleton):
                return None
            sig = self._space_case_signature(ent)
            if not sig:
                return None
            case_signatures.append(sig)

        first_sig = case_signatures[0]
        if not all(sig == first_sig for sig in case_signatures[1:]):
            return None

        principal = self._reapply_space_type_from_prepositions(entities[0])
        cleaned = [principal]
        for ent in entities[1:]:
            stripped_props = {
                k: v for k, v in dict(ent.properties).items()
                if k not in self._SPACE_CASE_MARKER_KEYS and not self._is_position_key(k)
            }
            stripped_entity = ent.update_node_props(stripped_props)
            cleaned.append(self._reapply_space_type_from_prepositions(
                stripped_entity,
                structural_context={'dependency_labels': ('appos',)},
            ))
        return cleaned

    def _reapply_space_type_from_prepositions(self, entity, structural_context=None):
        """Re-apply the SPACE-typing rule that fires on an entity's prepositions
        or its structural context (e.g. ``on or near LOC`` →
        ``type:OR(stay in place, near place)``; an appositional GPE such as
        Newcastle in ``Edward Place, Newcastle`` → ``type:stay in place`` via the
        ``DependencyLabel: appos`` rule). The original
        `rewrite_properties_logically` pass classified the AND-grouped wrapper
        rather than the individual entities, so the per-entity type was never
        written; this restores it by re-running the ontology rule matcher
        directly on the entity post-unwrapping."""
        if not isinstance(entity, Singleton):
            return entity
        from LaSSI.ner.SemanticRoleRewriting import rule_classifications

        _, selected_rule = get_matching_logical_rules(
            entity, entity, False, structural_context=structural_context,
        )
        if selected_rule is None:
            return entity

        type_properties_per_construct = defaultdict(list)
        for classification in rule_classifications(selected_rule):
            if classification.construct_property is None:
                continue
            type_properties_per_construct[classification.construct_name.lower()].append(
                classification.construct_property
            )

        space_types = type_properties_per_construct.get('space', [])
        if not space_types:
            return entity

        if len(space_types) == 1:
            new_type = space_types[0]
        else:
            new_type = f"OR({', '.join(space_types)})"

        props = dict(entity.properties)
        existing_type = str(props.get('type', '') or '')
        if new_type in existing_type:
            return entity
        props['type'] = new_type
        return entity.update_node_props(props)

    def _space_case_signature(self, ent):
        sig = []
        for k, v in dict(ent.properties).items():
            if k in self._SPACE_CASE_MARKER_KEYS or self._is_position_key(k):
                if isinstance(v, (list, tuple)):
                    sig.append((k, tuple(str(x) for x in v)))
                else:
                    sig.append((k, str(v)))
        return tuple(sorted(sig))

    @staticmethod
    def _is_position_key(key):
        if not isinstance(key, (str, int, float)):
            return False
        try:
            float(key)
        except (TypeError, ValueError):
            return False
        return True

    # ------------------------------------------------------------------
    # check_for_adv  (phrasal-verb folding)
    # ------------------------------------------------------------------

    def check_for_adv(self, kernel):
        if kernel.kernel is None:
            return kernel

        source_props = dict(kernel.kernel.source.properties) if isinstance(kernel.kernel.source, Singleton) else None
        if source_props is not None and 'adv' in source_props and source_props['adv']:
            word_permutations = itertools.permutations(kernel.kernel.edgeLabel.named_entity.split(' ') + source_props['adv'].split(' '))
            combined_permutations = {' '.join(p) for p in word_permutations}

            # If 'adv' name is in edge label, we don't need it in properties
            if source_props['adv'] in kernel.kernel.edgeLabel.named_entity:
                kernel = kernel.update_kernel(kernel.kernel.source.remove_prop('adv'), 'source')
                nx.set_node_attributes(self.G, {kernel.id: kernel}, 'data')

            # If the concatenation is not present in the list of phrasal verbs, strip the spurious adv property
            phrasal_verbs = self.services.getHOnK().getPhrasalVerbs()
            found_phrasal_verbs = phrasal_verbs.intersection(combined_permutations)
            if len(found_phrasal_verbs) == 0:
                kernel = kernel.update_kernel(kernel.kernel.source.remove_prop('adv'), 'source')
                nx.set_node_attributes(self.G, {kernel.id: kernel}, 'data')
                return kernel

            new_edge_label_name = list(found_phrasal_verbs)[0] # TODO: What if more than one element?

            edge_label = kernel.kernel.edgeLabel.update_name(new_edge_label_name)
            edge_source = kernel.kernel.source.remove_prop('adv')

            kernel = kernel.update_kernel(edge_source, "source")
            kernel = kernel.update_kernel(edge_label, "edgeLabel")
        else:
            # Detect dobj-based phrasal verbs in SENTENCE kernels (e.g., "takes place")
            if (isinstance(kernel, Singleton) and
                    getattr(kernel, 'type', None) == 'SENTENCE' and
                    kernel.kernel is not None and
                    kernel.kernel.edgeLabel is not None and
                    kernel.kernel.target is not None and
                    isinstance(kernel.kernel.target, Singleton)):
                parts = kernel.kernel.edgeLabel.named_entity.split(' ')
                base_verb = parts[-1]
                target_name = kernel.kernel.target.named_entity
                candidate = f"{base_verb} {target_name}"
                phrasal_verbs = self.services.getHOnK().getPhrasalVerbs()
                if candidate in phrasal_verbs:
                    prefix = ' '.join(parts[:-1])
                    new_label_name = f"{prefix} {candidate}".strip() if prefix else candidate
                    kernel = kernel.update_kernel(kernel.kernel.edgeLabel.update_name(new_label_name), "edgeLabel")
                    kernel = kernel.update_kernel(None, "target")

            if kernel.kernel.source.type == "SENTENCE":
                kernel = kernel.update_kernel(self.check_for_adv(kernel.kernel.source), "source")
            elif kernel.kernel.target is not None and kernel.kernel.target.type == "SENTENCE":
                kernel = kernel.update_kernel(self.check_for_adv(kernel.kernel.target), "target")

            properties_to_keep = defaultdict(list)
            for key in dict(kernel.properties):
                properties_key_ = dict(kernel.properties)[key]
                if isinstance(properties_key_, str):
                    properties_to_keep[key] = properties_key_
                elif properties_key_ is not None:
                    for node in properties_key_:
                        if key == 'SENTENCE':
                            properties_to_keep[key].append(self.check_for_adv(node))
                        else:
                            properties_to_keep[key].append(node)

            kernel = kernel.update_node_props(properties_to_keep)

        return kernel

    # ------------------------------------------------------------------
    # promote_contextual_sentence_kernel
    # ------------------------------------------------------------------

    def _logical_rule_for_node(self, kernel, node):
        try:
            _, selected_rule = get_matching_logical_rules(
                kernel,
                node,
                False,
                structural_context=self.matchers.logical_rule_context(node),
            )
        except Exception:
            return None
        return selected_rule

    def _is_kernel_level_logical_context(self, kernel, node):
        selected_rule = self._logical_rule_for_node(kernel, node)
        if selected_rule is None or not getattr(selected_rule, 'logicalConstructName', None):
            return False
        functions = self.services.getHOnK().get_logical_functions(
            selected_rule.logicalConstructName,
            selected_rule.logicalConstructProperty,
        )
        return any(getattr(fn, 'attachTo', None) == 'Kernel' for fn in functions)

    @staticmethod
    def _references_any_id(value, ids):
        if not ids:
            return False
        if isinstance(value, Singleton):
            if value.id in ids:
                return True
            if value.kernel is not None:
                return (
                        KernelPostProcessor._references_any_id(value.kernel.source, ids) or
                        KernelPostProcessor._references_any_id(value.kernel.target, ids) or
                        KernelPostProcessor._references_any_id(value.kernel.edgeLabel, ids)
                )
        if isinstance(value, SetOfSingletons):
            return any(KernelPostProcessor._references_any_id(entity, ids) for entity in value.entities)
        return False

    @staticmethod
    def _without_references_to_ids(value, ids):
        if isinstance(value, SetOfSingletons):
            kept = [
                entity for entity in value.entities
                if not KernelPostProcessor._references_any_id(entity, ids)
            ]
            if not kept:
                return None
            if len(kept) != len(value.entities):
                return value.update_entities(kept)
        if KernelPostProcessor._references_any_id(value, ids):
            return None
        return value

    def promote_contextual_sentence_kernel(self, kernel):
        """Promote the main clause when a logical-context clause became primary.

        DatagramDB/Stanza can surface an adverbial context clause as a root,
        yielding shapes such as `while take place(...)[SENTENCE:close(...)]`.
        If that outer kernel is classifiable by HOnK as a kernel-level logical
        property, keep it as a `SENTENCE` property of the inner main clause so
        the normal logical rewriter can classify it (e.g. TEMPORAL_CONTEXT).
        """
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel
        props = dict(kernel.properties)
        sentence_values = props.get('SENTENCE')
        if not sentence_values:
            return kernel
        if not self._is_kernel_level_logical_context(kernel, kernel):
            return kernel

        sentence_list = list(sentence_values) if isinstance(sentence_values, (list, tuple)) else [sentence_values]
        main_clause = next(
            (
                candidate for candidate in sentence_list
                if isinstance(candidate, Singleton)
                and candidate.kernel is not None
                and not self._is_kernel_level_logical_context(kernel, candidate)
            ),
            None,
        )
        if main_clause is None:
            return kernel

        outer_props = defaultdict(list)
        for key, value in props.items():
            if key == 'SENTENCE':
                continue
            if isinstance(value, (list, tuple)):
                outer_props[key] = list(value)
            else:
                outer_props[key] = value
        contextual_clause = kernel.update_node_props(outer_props)
        contextual_ids = {
            getattr(node, 'id', None)
            for node in (
                contextual_clause.kernel.source,
                contextual_clause.kernel.target,
                contextual_clause.kernel.edgeLabel,
            )
            if isinstance(node, Singleton)
        }
        contextual_ids.discard(None)

        promoted_props = defaultdict(list)
        for key, value in dict(main_clause.properties).items():
            if isinstance(value, (list, tuple)):
                kept_values = []
                for item in value:
                    kept = self._without_references_to_ids(item, contextual_ids)
                    if kept is not None:
                        kept_values.append(kept)
                if kept_values:
                    promoted_props[key] = kept_values
            else:
                kept = self._without_references_to_ids(value, contextual_ids)
                if kept is not None:
                    promoted_props[key] = kept
        promoted_props['SENTENCE'].append(contextual_clause)

        for candidate in sentence_list:
            if candidate is main_clause:
                continue
            promoted_props['SENTENCE'].append(candidate)
        return main_clause.update_node_props(promoted_props)

    # ------------------------------------------------------------------
    # promote_embedded_logical_properties
    # ------------------------------------------------------------------

    def promote_embedded_logical_properties(self, kernel):
        from LaSSI.ner.node_functions import create_existential_node as _make_existential
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel

        kernel_props = dict(kernel.properties)
        logical_keys = set()

        def promote_from_node(node):
            nonlocal kernel_props, logical_keys
            if not isinstance(node, Singleton):
                return node
            node_props = dict(node.properties)
            promoted = {
                key: list(value)
                for key, value in node_props.items()
                if isinstance(key, str) and key.isupper() and isinstance(value, (list, tuple))
            }
            if not promoted:
                return node
            for key, values in promoted.items():
                existing = list(kernel_props.get(key, []))
                existing_ids = {x.id for x in existing if isinstance(x, Singleton)}
                for value in values:
                    if not isinstance(value, Singleton) or value.id not in existing_ids:
                        existing.append(value)
                        if isinstance(value, Singleton):
                            existing_ids.add(value.id)
                kernel_props[key] = existing
                logical_keys.add(key)
            remaining_props = {key: value for key, value in node_props.items() if key not in promoted}
            return node.update_node_props(remaining_props)

        source = promote_from_node(kernel.kernel.source)
        target = promote_from_node(kernel.kernel.target)

        if logical_keys and target is None and isinstance(source, Singleton) and source.type != 'existential':
            kernel = kernel.update_kernel(_make_existential(), 'source')
            kernel = kernel.update_kernel(source, 'target')
        else:
            kernel = kernel.update_kernel(source, 'source') if source is not None else kernel
            kernel = kernel.update_kernel(target, 'target') if target is not None else kernel

        return kernel.update_node_props(kernel_props)

    # ------------------------------------------------------------------
    # lift_participial_logical_context
    # ------------------------------------------------------------------

    def _is_change_of_state_kernel(self, kernel):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return False
        edge = kernel.kernel.edgeLabel
        if not isinstance(edge, Singleton) or not edge.named_entity:
            return False
        change_verbs = self.services.getHOnK().getChangeOfStateVerbs() or set()
        edge_lemma = edge.named_entity.lower()
        return edge_lemma in {str(v).lower() for v in change_verbs}

    @staticmethod
    def _property_items(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _node_or_string_name(value):
        if isinstance(value, Singleton):
            return value.get_name()
        if isinstance(value, str):
            return value
        return None

    def _normalise_causal_compound_head(self, value):
        """Prefer the semantic head for cause-marked compounds.

        Some dependency shapes surface compounds like "fire safety concerns" as
        `fire safety[(extra:concerns)]`.  When the node is explicitly marked as
        a cause and the parser put an unresolved compound head under `extra`,
        promote that extra term and keep the original surface as its modifier.
        """
        if not isinstance(value, Singleton):
            return value
        props = dict(value.properties)
        if props.get('case') != 'due' or 'extra' not in props:
            return value
        for extra in self._property_items(props.get('extra')):
            extra_name = self._node_or_string_name(extra)
            if not extra_name:
                continue
            remaining_extras = [
                item for item in self._property_items(props.get('extra'))
                if self._node_or_string_name(item) != extra_name
            ]
            new_extra = [value.get_name()] + remaining_extras
            props['extra'] = new_extra[0] if len(new_extra) == 1 else new_extra
            return value.update_name(extra_name).update_node_props(props)
        return value

    def _append_unique_by_id(self, props, key, value):
        if key == 'CAUSATION':
            value = self._normalise_causal_compound_head(value)
        values = props.setdefault(key, [])
        if not isinstance(values, list):
            values = list(values) if isinstance(values, (list, tuple)) else [values]
            props[key] = values
        value_id = getattr(value, 'id', None)
        for existing in values:
            if value_id is not None and getattr(existing, 'id', None) == value_id:
                return
        values.append(value)

    def _classify_participial_argument(self, kernel, argument):
        if not isinstance(argument, Singleton):
            return defaultdict(list)
        classified = defaultdict(list)
        try:
            return self.rewriter.rewrite_node_logically(kernel, argument, classified)
        except Exception:
            return defaultdict(list)

    def lift_participial_logical_context(self, kernel):
        """Lift logical roles from participial SENTENCE children of state changes.

        In clauses like "staircases are closed due to concerns identified during
        inspection", the `identified` participial child carries CAUSATION/TIME
        context for the parent `close` state.  Keep `close` as the predicate and
        lift those logical roles, allowing the existing causation-subject
        normalisation to produce `close(concerns, staircases)`.
        """
        if not self._is_change_of_state_kernel(kernel):
            return kernel
        props = dict(kernel.properties)
        sentence_values = props.get('SENTENCE')
        if not sentence_values:
            return kernel
        sentence_list = list(sentence_values) if isinstance(sentence_values, (list, tuple)) else [sentence_values]

        new_props = {
            key: list(value) if isinstance(value, (list, tuple)) else value
            for key, value in props.items()
            if key != 'SENTENCE'
        }
        remaining_sentences = []
        lifted = False
        for sentence in sentence_list:
            if not isinstance(sentence, Singleton) or sentence.kernel is None:
                remaining_sentences.append(sentence)
                continue
            sentence_props = dict(sentence.properties)
            logical_items = {
                key: value for key, value in sentence_props.items()
                if isinstance(key, str) and key.isupper() and key != 'SENTENCE'
            }
            classified_items = defaultdict(list)
            for endpoint in (sentence.kernel.source, sentence.kernel.target):
                for key, values in self._classify_participial_argument(kernel, endpoint).items():
                    if isinstance(key, str) and key.isupper():
                        classified_items[key].extend(values)
            if not logical_items and not classified_items:
                remaining_sentences.append(sentence)
                continue
            for key, value in logical_items.items():
                values = value if isinstance(value, (list, tuple)) else [value]
                for item in values:
                    self._append_unique_by_id(new_props, key, item)
            for key, values in classified_items.items():
                for item in values:
                    self._append_unique_by_id(new_props, key, item)
            lifted = True

        if remaining_sentences:
            new_props['SENTENCE'] = remaining_sentences
        if not lifted:
            return kernel
        return kernel.update_node_props(new_props)

    # ------------------------------------------------------------------
    # nest_oft_terms_under_causation
    # ------------------------------------------------------------------

    def nest_oft_terms_under_causation(self, kernel):
        if not isinstance(kernel, Singleton):
            return kernel

        kernel_props = dict(kernel.properties)
        causations = list(kernel_props.get('CAUSATION', []))
        of_terms = list(kernel_props.get('OFTERM', []))
        if len(causations) != 1 or not of_terms or not isinstance(causations[0], Singleton):
            return kernel

        causation = causations[0]
        if any(getattr(term, 'min', causation.max) < causation.max for term in of_terms):
            return kernel

        causation_props = dict(causation.properties)
        nested_terms = list(causation_props.get('OFTERM', []))
        # Dedupe against every Singleton already living anywhere on the causation
        # kernel — an `of_term` may already be represented under a sibling key like
        # CAUSATION (e.g. when an exploded MULTIINDOBJ child gets reached via both
        # the participial sub_properties path and the outer-iteration path).
        existing_ids = set()
        for value in causation_props.values():
            if isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Singleton) and hasattr(v, 'id'):
                        existing_ids.add(v.id)
            elif isinstance(value, Singleton) and hasattr(value, 'id'):
                existing_ids.add(value.id)
        for term in of_terms:
            term_id = getattr(term, 'id', None)
            if term_id is not None and term_id in existing_ids:
                continue
            nested_terms.append(term)
            if term_id is not None:
                existing_ids.add(term_id)

        causation_props['OFTERM'] = nested_terms
        kernel_props['CAUSATION'] = [causation.update_node_props(causation_props)]
        kernel_props.pop('OFTERM', None)
        return kernel.update_node_props(kernel_props)

    # ------------------------------------------------------------------
    # acl_replacement
    # ------------------------------------------------------------------

    def acl_replacement(self, kernel, acl_relcl_map):
        # TODO: This isn't entirely recursive...
        if len(acl_relcl_map.values()) > 0 and kernel.kernel is not None:
            kernel_source = self.get_acl_replacement(acl_relcl_map, kernel.kernel.source)
            kernel_target = self.get_acl_replacement(acl_relcl_map, kernel.kernel.target)

            properties_to_keep = defaultdict(list)

            # Check if properties has any Singleton's and replace those
            for key in dict(kernel.properties):
                properties_key_ = dict(kernel.properties)[key]
                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        new_prop = self.get_acl_replacement(acl_relcl_map, properties_key_)
                        properties_to_keep[key].append(new_prop)
                    else:
                        for prop_node in properties_key_:
                            if prop_node.kernel is not None and prop_node.kernel.target is not None and prop_node.kernel.target.id in acl_relcl_map.keys():
                                continue

                            if prop_node.kernel is None:
                                properties_to_keep[key].append(prop_node)
                            elif prop_node.type == 'SENTENCE':
                                prop_sing_source = self.get_acl_replacement(acl_relcl_map, prop_node.kernel.source)
                                prop_sing_target = self.get_acl_replacement(acl_relcl_map, prop_node.kernel.target)

                                properties_to_keep[key].append(Singleton(
                                    id=prop_node.id,
                                    named_entity='',
                                    type='SENTENCE',
                                    min=prop_node.min,
                                    max=prop_node.max,
                                    confidence=1,
                                    kernel=Relationship(
                                        source=prop_sing_source,
                                        target=prop_sing_target,
                                        edgeLabel=prop_node.kernel.edgeLabel,
                                        isNegated=prop_node.kernel.isNegated,
                                    ),
                                    properties=prop_node.properties,
                                ))
                else:
                    properties_to_keep[key].append(properties_key_)

            return Singleton(
                id=kernel.id,
                named_entity='',
                type='SENTENCE',
                min=kernel.min,
                max=kernel.max,
                confidence=1,
                kernel=Relationship(
                    source=kernel_source,
                    target=kernel_target,
                    edgeLabel=kernel.kernel.edgeLabel,
                    isNegated=kernel.kernel.isNegated,
                ),
                properties=create_props_for_singleton(properties_to_keep),
            )
        else:
            return kernel

    @staticmethod
    def get_acl_replacement(acl_relcl_map, node):
        return acl_relcl_map[node.id] if node is not None and node.id in acl_relcl_map.keys() else node

    # ------------------------------------------------------------------
    # remove_duplicate_properties
    # ------------------------------------------------------------------

    def remove_duplicate_properties(self, kernel, kernel_nodes=None):
        if kernel.kernel is None:
            return kernel

        properties_to_keep = defaultdict(list)

        if kernel_nodes is None:
            kernel_nodes = set()
        kernel_nodes = self.add_to_kernel_nodes(kernel, kernel_nodes)

        # Add 'nmod' source and target to kernel nodes, so duplicate nodes are not added to properties
        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key in DependencyRoles.nominal_modifier_edges():
                        properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                        kernel_nodes = self.add_to_kernel_nodes(node if key not in DependencyRoles.nominal_modifier_edges_no_poss() else node.kernel.target, kernel_nodes)

        # Check if empty kernel is in properties and remove, recursively iterate through kernels to remove duplicate properties
        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key == 'SENTENCE':
                        emptied_node = self._check_if_empty_kernel(node, True)
                        if emptied_node is not None:
                            # If given property is `be(? OR in kernel_nodes, ? OR in kernel_nodes)`, then do not add as property as it is redundant.
                            # NB: node.kernel may itself be None (when SENTENCE property carries
                            # a singleton without an inner kernel), and even when present its
                            # edgeLabel may be None (conjoined clauses with no verb head — e.g.
                            # "… and remains under investigation" reaches here as a SENTENCE
                            # property whose inner kernel has no edgeLabel).  Both cases must
                            # behave as "not 'be'" so the property is preserved rather than
                            # mis-classified as a redundant copula.
                            if (hasattr(node, 'kernel') and node.kernel is not None and not (node.kernel.edgeLabel is not None and node.kernel.edgeLabel.named_entity == "be" and (
                                    (
                                            (
                                                    kernel_nodes is not None and node.kernel.source.type != 'existential' and node.kernel.source in kernel_nodes)
                                            and (
                                                    node.kernel.target is not None and node.kernel.target.type == 'existential')
                                    )
                                    or
                                    (
                                            (
                                                    kernel_nodes is not None and node.kernel.target is not None and node.kernel.target.type != 'existential' and node.kernel.target in kernel_nodes)
                                            and (
                                                    node.kernel.source is not None and node.kernel.source.type == 'existential')
                                    )
                            ))) or kernel_nodes is None or not hasattr(node, 'kernel') or node.kernel is None:
                                if not is_node_in_kernel_nodes(emptied_node, kernel_nodes):
                                    properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                                else:
                                    self.remove_duplicate_properties(node, kernel_nodes)
                    elif not is_node_in_kernel_nodes(node, kernel_nodes):
                        if hasattr(node, 'properties'):
                            inner_properties_to_keep = dict()
                            for inner_key in dict(node.properties):
                                value = dict(node.properties)[inner_key]
                                if ((isinstance(value, str) and value in string.punctuation) or (
                                        kernel.kernel.edgeLabel is not None and
                                        isinstance(value, str) and
                                        not re.search(
                                            r"\b" + re.escape(value) + r"\b",
                                            kernel.kernel.edgeLabel.named_entity
                                        )
                                ) or not isinstance(value, str) or kernel.kernel.edgeLabel is None):
                                    inner_properties_to_keep[inner_key] = value
                            properties_to_keep[key].append(node.update_node_props(inner_properties_to_keep))
                        else:
                            properties_to_keep[key].append(node)

        # Check if we have duplicate kernels now they are all added, and keep most relevant one (i.e. two equal kernels but only one has properties)
        if 'SENTENCE' in properties_to_keep:
            sentence_properties = properties_to_keep['SENTENCE']
            if len(sentence_properties) > 1:  # Check for more than one SENTENCE in props
                equal_kernels = [sentence for sentence in sentence_properties if
                                 sentence.kernel == sentence_properties[0].kernel]
                if len(equal_kernels) > 1:  # Check we have at leasts two "equal" sentences
                    sentences_with_props = [sentence for sentence in equal_kernels if len(sentence.properties) > 0]

                    found_properties = defaultdict(list)
                    for given_sentence in sentences_with_props:
                        found_properties = merge_properties(found_properties, dict(given_sentence.properties))
                    properties_to_keep["SENTENCE"] = [Singleton(
                        id=equal_kernels[0].id,
                        named_entity=equal_kernels[0].named_entity,
                        type=equal_kernels[0].type,
                        min=equal_kernels[0].min,
                        max=equal_kernels[0].max,
                        confidence=equal_kernels[0].confidence,
                        kernel=equal_kernels[0].kernel,
                        properties=create_props_for_singleton(found_properties),
                    )]

        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            type=kernel.type,
            min=kernel.min,
            max=kernel.max,
            confidence=kernel.confidence,
            kernel=kernel.kernel,
            properties=create_props_for_singleton(properties_to_keep),
        )

    def _check_if_empty_kernel(self, kernel, force=False):
        """Local copy of `check_if_empty_kernel` from the orchestrator —
        `remove_duplicate_properties` invokes it recursively when scanning
        SENTENCE child kernels. Kept here to avoid an import cycle."""
        properties_to_keep = dict()
        new_kernel = None
        if (
                isinstance(kernel,
                           Singleton) and kernel.kernel is not None and kernel.kernel.edgeLabel is not None and (
                kernel.kernel.edgeLabel.named_entity == "be" if not force else True)
                and
                (
                        ((kernel.kernel.source is not None and kernel.kernel.source.type == 'existential') and (
                                kernel.kernel.target is not None and kernel.kernel.target.type == 'existential'))
                        or
                        ((kernel.kernel.source is None) and (kernel.kernel.target is None))
                )
        ):
            node_props = dict(kernel.properties)
            if len(node_props) > 0 and 'SENTENCE' in node_props:
                for key in node_props:
                    if key == 'SENTENCE':
                        new_kernel = node_props['SENTENCE'][0]
                        new_kernel = self._check_if_empty_kernel(new_kernel)
                    else:
                        properties_to_keep[key] = node_props[key]

        if new_kernel is not None:
            if len(properties_to_keep) > 0:
                return new_kernel.update_node_props(properties_to_keep)
            else:
                return new_kernel
        else:
            return kernel

    def add_to_kernel_nodes(self, node, kernel_nodes):
        if isinstance(node, SetOfSingletons):
            kernel_nodes.add(node)
            for entity in node.entities:
                self.add_to_kernel_nodes(entity, kernel_nodes)
        else:
            if node.kernel is not None:
                kernel_nodes.add(node)

                # Check if properties has any Singleton's and add to kernel nodes also
                if node.kernel.edgeLabel is not None:
                    self.add_singletons_from_node_properties(node.kernel.edgeLabel, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.edgeLabel, kernel_nodes)
                if node.kernel.source is not None:
                    if node.kernel.edgeLabel is not None and node.kernel.edgeLabel.named_entity not in DependencyRoles.nominal_modifier_edges_no_poss():
                        self.add_singletons_from_node_properties(node.kernel.source, kernel_nodes)
                        self.add_to_kernel_nodes(node.kernel.source, kernel_nodes)
                if node.kernel.target is not None:
                    # If it is nmod or obl, we are rewriting it, but the target (e.g. "station") should be considered "consumed"
                    # so it doesn't appear as a redundant property elsewhere.
                    self.add_singletons_from_node_properties(node.kernel.target, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.target, kernel_nodes)
            else:
                kernel_nodes.add(node)

        return kernel_nodes

    def add_singletons_from_node_properties(self, node, kernel_nodes):
        if isinstance(node, Singleton):
            for key in dict(node.properties):
                properties_key_ = dict(node.properties)[key]
                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        self.add_to_kernel_nodes(properties_key_, kernel_nodes)
                    else:
                        for prop_node in properties_key_:
                            if isinstance(prop_node, Singleton):
                                self.add_to_kernel_nodes(prop_node, kernel_nodes)
                            elif isinstance(prop_node, SetOfSingletons):
                                kernel_nodes.add(node)
                                for prop_entity in prop_node.entities:
                                    self.add_to_kernel_nodes(prop_entity, kernel_nodes)
