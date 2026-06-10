__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__status__ = "Production"

import itertools
import re
from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy

from LaSSI.external_services.Services import Services
from LaSSI.ner.KernelPostProcessor import KernelPostProcessor
from LaSSI.ner.MergeSetOfSingletons import merge_properties
from LaSSI.ner.node_functions import create_existential_node
from LaSSI.ner.node_functions_X import create_props_for_singleton, NodeFunctions
from LaSSI.ner.string_functions import is_label_verb, check_semi_modal, lemmatize_verb
from LaSSI.ner.structural_rewrites.base import is_canonical_copula
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Relationship, SetOfSingletons, Grouping
from LaSSI.structures.kernels.SentenceX import (
    case_in_props,
    create_edge_kernel,
    create_existential,
    create_sentence,
    find_action_ed_node_in_kernel,
    is_kernel_in_props,
    rewrite_action_ed_node,
)


class CreateFinalKernelX:
    """Builds the final kernel for a sentence.

    Two phases:
      1. Per-root construction loop — calls `create_sentence` once per
         topological root, combines results across connected components.
      2. Post-processing pipeline — delegated to `KernelPostProcessor`.

    Heavy logic (logical rewriting, ontology lookups, structural cleanup) lives
    in sibling modules:
      - `KernelPostProcessor`        the post-processing pipeline + cleanups
      - `KernelLogicalRewriter`      ontology-rule-driven property rewriting
      - `KernelOntologyMatchers`     pure HOnK predicates (is_state_edge, ...)
    """

    def __init__(self, G, negations, node_functions):
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.G = G
        self.negations = negations
        self.node_functions = node_functions
        self.post = KernelPostProcessor(G, negations, node_functions)

    def constructSentence(self) -> Singleton:
        # Phase 1: identify prepositional/gerund roots and the "true target" set
        true_targets, found_preposition_labels = self.find_prepositions_and_true_targets()

        # Phase 2: pick the topological root nodes that drive each kernel construction
        filtered_top_node_ids = self.get_topological_root_node_ids(true_targets)

        # Identify connected components so genuinely disconnected kernel clauses
        # (e.g. main passive clause + advcl clause produced by p3pass + p3) can be
        # combined correctly rather than discarding all but the last.
        G_undirected = self.G.to_undirected()
        component_map = {
            nid: cid
            for cid, comp in enumerate(nx.connected_components(G_undirected))
            for nid in comp
        }
        roots_by_component = defaultdict(list)
        for node_id in filtered_top_node_ids:
            roots_by_component[component_map.get(node_id, -1)].append(node_id)
        is_multi_component = len(roots_by_component) > 1

        # For multi-component graphs identify the primary root before the loop modifies
        # node data.  Primary = the root whose Singleton has a non-empty 'adv' property
        # (set by p3pass to record the advcl verb that links the two components).
        primary_root_id = None
        if is_multi_component:
            for node_id in filtered_top_node_ids:
                node_data = self.G.nodes[node_id]['data']
                if (
                    isinstance(node_data, Singleton) and
                    'adv' in dict(node_data.properties) and
                    dict(node_data.properties)['adv']
                ):
                    primary_root_id = node_id
                    break

        # Phase 3: per-root construction loop
        used_edges = set()
        acl_relcl_map = dict()
        position_pairs = self.get_position_pairs()
        for node_id in filtered_top_node_ids:
            loop_settings = SimpleNamespace(shouldLoop=True, edgeForKernel=None, previousKernel=None)
            while loop_settings.shouldLoop:
                descendant_node_ids = list(nx.bfs_tree(self.G, node_id))

                # Ensure the "descendant nodes" are in the edge, and not in used_edges (previous loop), unless we have preposition labels
                filtered_edges = [
                    x for x in self.G.edges(data=True, keys=True) if
                    # Edge source and target are in descendent nodes OR target has preposition label
                    (
                        (x[0] in descendant_node_ids and x[1] in descendant_node_ids) or
                        (x[1] in found_preposition_labels)
                    )
                    # Edge source and target are NOT in used edges from previous loop OR they both are and we have preposition labels
                    and
                    (
                        (x[0], x[1]) not in used_edges or
                        ((x[0], x[1]) in used_edges and len(found_preposition_labels) > 0)
                    )
                    # Edge target is not equal to current root node in loop and target is not a verb
                    and not (x[1] == node_id and self.G.nodes[x[1]]['data'].type.lower() == 'verb')
                ]

                # If we have an edge from the previous iteration use this as our edges
                if loop_settings.edgeForKernel is not None:
                    filtered_edges = [loop_settings.edgeForKernel]

                used_edges = set(map(lambda y: (y[0], y[1]), filtered_edges))
                descendant_nodes = {key: x for key, x in self.G.nodes(data=True) if key in descendant_node_ids}

                self.G, kernel, loop_settings, acl_relcl_map = create_sentence(
                    self.G, filtered_edges, descendant_nodes, self.negations, node_id, found_preposition_labels,
                    self.node_functions, loop_settings, acl_relcl_map
                )
                kernel = self.kernel_post_processing(kernel, position_pairs)

                # Only check for empty kernel if more than one root node, as if there is only 1 root node we need something, even if it is empty...
                # if len(filtered_top_node_ids) > 1:
                #     kernel = self.check_if_empty_kernel(kernel)  # Check we do not have be(?, ?) as a kernel
                #     if kernel is not None: # (not empty)
                #         # if not loop_settings.edgeForKernel:
                #         nx.set_node_attributes(self.G, {node_id: kernel}, 'data')
                # else:
                #     nx.set_node_attributes(self.G, {node_id: kernel}, 'data')

                # Hoist empty copula wrappers to their underlying SENTENCE kernels
                kernel = self.check_if_empty_kernel(kernel)
                nx.set_node_attributes(self.G, {node_id: kernel}, 'data')

                # Strip 'root'/'kernel' so the same node is not re-used as a root next iteration
                attributes_to_update = {
                    nid: node['data'].strip_root_properties()
                    for nid, node in self.G.nodes(data=True)
                    if nid in descendant_node_ids
                }
                nx.set_node_attributes(self.G, attributes_to_update, 'data')

                # If this current "kernel" is none, then remove so the kernel used is the last occurring correct one
                #  (only if we have at least one other kernel available)
                if (
                        ((isinstance(kernel, Singleton) and kernel.kernel is None) or (kernel is None)) and
                        len(filtered_top_node_ids) > 1 and
                        node_id == filtered_top_node_ids[-1]
                ):
                    filtered_top_node_ids.pop()

        # Return the last node ('highest' topological kernel)
        sorted_G = list(NodeFunctions.sort_G(self.G))
        selected_root_id = (
            [n_id for n_id in sorted_G if n_id in filtered_top_node_ids][-1]
            if len(filtered_top_node_ids) > 0 else sorted_G[-1]
        )
        final_kernel = self.G.nodes[selected_root_id]['data']

        # If the highest topological node is a copula ("are"), but we have a semantic verb ("carry out")
        # in our roots, swap them so the final kernel prioritizes the semantic action.
        if len(filtered_top_node_ids) > 1 and isinstance(final_kernel, Singleton) and final_kernel.kernel is not None:
            try:
                copula_forms = self.services.getHOnK().getCopulaSurfaceForms() or set()
            except Exception:
                copula_forms = set()
            copula_lower = {str(f).lower() for f in copula_forms}

            def is_copula_node(node_data):
                if not isinstance(node_data,
                                  Singleton) or node_data.kernel is None or node_data.kernel.edgeLabel is None:
                    return False
                ename = node_data.kernel.edgeLabel.named_entity if isinstance(node_data.kernel.edgeLabel,
                                                                              Singleton) else ""
                parts = [p for p in ename.split() if p]
                if not parts: return False
                return all(p.lower() in copula_lower or lemmatize_verb(p).lower() in copula_lower for p in parts)

            # If the final kernel defaulted to a copula, search the other roots for the semantic verb
            if is_copula_node(final_kernel):
                for nid in reversed([n for n in sorted_G if n in filtered_top_node_ids]):
                    cand = self.G.nodes[nid]['data']
                    if isinstance(cand, Singleton) and cand.kernel is not None and not is_copula_node(cand):
                        final_kernel = cand
                        selected_root_id = nid
                        break

        preferred_root_id = self._preferred_reduced_relative_reporting_root_id(
            selected_root_id, filtered_top_node_ids, sorted_G
        )
        if preferred_root_id is not None:
            selected_root_id = preferred_root_id
            final_kernel = self.G.nodes[selected_root_id]['data']

        # Multi-component reconciliation: hoist subordinate clauses as SENTENCE properties
        # of the primary kernel.
        if (
            is_multi_component and
            primary_root_id is not None and
            primary_root_id in filtered_top_node_ids
        ):
            primary_kernel = self.G.nodes[primary_root_id]['data']
            subordinate_kernels = [
                self.G.nodes[nid]['data']
                for nid in filtered_top_node_ids
                if nid != primary_root_id
                and self.G.nodes[nid]['data'] is not None
                and isinstance(self.G.nodes[nid]['data'], Singleton)
            ]
            if subordinate_kernels and isinstance(primary_kernel, Singleton):
                new_props = defaultdict(list)
                for k, v in dict(primary_kernel.properties).items():
                    if isinstance(v, (list, tuple)):
                        new_props[k] = list(v)
                    else:
                        new_props[k] = v
                for sk in subordinate_kernels:
                    new_props['SENTENCE'].append(sk)
                primary_kernel = primary_kernel.update_node_props(new_props)
            final_kernel = primary_kernel
        elif not is_multi_component and len(filtered_top_node_ids) > 1:
            # Same component, multiple roots: merge accumulated properties from prior roots
            # into the final kernel before the post-processing pipeline runs.
            merged_props = defaultdict(list)
            for k, v in dict(final_kernel.properties).items():
                if isinstance(v, (list, tuple)):
                    merged_props[k] = list(v)
                else:
                    merged_props[k] = v
            for nid in filtered_top_node_ids:
                if nid == selected_root_id:
                    continue
                other_kernel = self.G.nodes[nid]['data']
                if not isinstance(other_kernel, Singleton):
                    continue
                for k, v in dict(other_kernel.properties).items():
                    if isinstance(v, (list, tuple)):
                        existing_ids = {x.id for x in merged_props.get(k, []) if hasattr(x, 'id')}
                        for item in v:
                            if not hasattr(item, 'id') or item.id not in existing_ids:
                                merged_props[k].append(item)
                                if hasattr(item, 'id'):
                                    existing_ids.add(item.id)
                    elif k not in merged_props:
                        merged_props[k] = v
            final_kernel = final_kernel.update_node_props(merged_props)
        
        # Reconciliation logic for existential constructions and disconnected components
        if (
            (is_multi_component or len(filtered_top_node_ids) > 1) and
            primary_root_id is None and
            isinstance(final_kernel, Singleton) and
            final_kernel.kernel is not None
        ):
            # Fallback: a noun-conjunction component sits disconnected from the
            # main verb kernel (e.g. "We're carrying out inspections and roof
            # repairs at Monkseaton station ...", the GSM grammar leaves the
            # inspections/repairs conj as its own component). Promote that
            # nominal component as the kernel target; demote any existing
            # location target to SPACE.
            #
            # This also handles the single-component case where an existential 
            # copula root ("There are...") and a semantic verb root ("...carried out")
            # are both detected. In that case, we "steal" the nominal target from
            # the copula kernel and promote it to the verb kernel.
            verb_root_id = None
            for nid in filtered_top_node_ids:
                cand = self.G.nodes[nid]['data']
                if (isinstance(cand, Singleton) and cand.kernel is not None and
                        cand.kernel.edgeLabel is not None):
                    verb_root_id = nid
                    break

            edge_label = final_kernel.kernel.edgeLabel
            edge_name = (
                edge_label.named_entity if isinstance(edge_label, Singleton) else None
            ) or ""
            try:
                copula_forms = self.services.getHOnK().getCopulaSurfaceForms() or set()
            except Exception:
                copula_forms = set()
            copula_lower = {str(f).lower() for f in copula_forms}
            edge_parts = [p for p in edge_name.split() if p]
            non_copula_part_count = sum(
                1 for part in edge_parts
                if part.lower() not in copula_lower
                and lemmatize_verb(part).lower() not in copula_lower
            )
            edge_is_copula = (non_copula_part_count == 0)

            old_target = final_kernel.kernel.target
            target_pos = None
            if isinstance(old_target, Singleton):
                op = dict(old_target.properties)
                try:
                    target_pos = float(op.get('pos', float('inf')))
                except (TypeError, ValueError):
                    target_pos = float('inf')

            target_is_promotable = (
                old_target is None or
                (isinstance(old_target, Singleton) and old_target.type == 'existential') or
                (isinstance(old_target, Singleton) and self.post.matchers.is_location_like(old_target))
            )

            def _is_purely_nominal(node):
                if isinstance(node, SetOfSingletons):
                    return (
                        node.type in (Grouping.AND, Grouping.OR, Grouping.NOT, Grouping.NEITHER) and
                        all(_is_purely_nominal(e) for e in node.entities)
                    )
                if isinstance(node, Singleton):
                    if node.kernel is not None:
                        # Allow "weak" copula kernels to be treated as nominals (via their target)
                        if (node.kernel.edgeLabel is not None and 
                            is_canonical_copula(node.kernel.edgeLabel.named_entity) and
                            node.kernel.target is not None):
                            return True
                        return False
                    return 'verb' not in (node.type or '').lower()
                return False

            def _min_position(node):
                if isinstance(node, SetOfSingletons):
                    positions = [_min_position(e) for e in node.entities]
                    positions = [p for p in positions if p is not None]
                    return min(positions) if positions else None
                if isinstance(node, Singleton):
                    p = dict(node.properties).get('pos')
                    if p is None and node.kernel is not None:
                        return _min_position(node.kernel.target)
                    try:
                        return float(p)
                    except (TypeError, ValueError):
                        return None
                return None

            chosen_nominal = None
            if (
                not edge_is_copula and target_is_promotable and verb_root_id is not None
            ):
                for nid in filtered_top_node_ids:
                    if nid == verb_root_id:
                        continue
                    cand = self.G.nodes[nid]['data']
                    if not _is_purely_nominal(cand):
                        continue
                    cand_pos = _min_position(cand)
                    if (
                        target_pos is not None and cand_pos is not None and
                        cand_pos >= target_pos
                    ):
                        continue
                    chosen_nominal = cand
                    break

            if chosen_nominal is not None:
                # If the chosen nominal is a weak copula kernel, extract its target
                if (isinstance(chosen_nominal, Singleton) and chosen_nominal.kernel is not None and 
                    chosen_nominal.kernel.edgeLabel is not None and 
                    is_canonical_copula(chosen_nominal.kernel.edgeLabel.named_entity)):
                    chosen_nominal = chosen_nominal.kernel.target

                new_props = defaultdict(list)
                for k, v in dict(final_kernel.properties).items():
                    if isinstance(v, (list, tuple)):
                        new_props[k] = list(v)
                    else:
                        new_props[k] = v
                if isinstance(old_target, Singleton) and self.post.matchers.is_location_like(old_target):
                    existing_space_ids = {
                        x.id for x in new_props.get('SPACE', [])
                        if isinstance(x, Singleton)
                    }
                    if old_target.id not in existing_space_ids:
                        new_props['SPACE'].append(old_target)
                final_kernel = final_kernel.update_node_props(new_props)

                final_kernel = final_kernel.update_kernel(create_existential_node(), "source")
                final_kernel = final_kernel.update_kernel(chosen_nominal, "target")

        # Subordination-aware correction + sibling-clause preservation.
        #
        # The last-topological default frequently (a) roots on a *subordinate*
        # clause (an advcl with a `mark`, or a relative clause whose argument
        # is a relative pronoun) instead of the main clause, and (b) drops
        # sibling verb clauses entirely. When multiple roots exist we therefore
        # choose the main clause and preserve every *other verb-kernel* root as
        # SENTENCE so its predication is not lost. Only verb-kernel siblings are
        # hoisted — nominal conj components are left to the legacy
        # nominal-promotion path below, so existing behaviour is preserved.
        # Repairs e.g. rooting on "before work ends" / "that had developed a
        # leak" instead of "Roadworks are underway" / "roadworks involve
        # replacement", and recovers "delays considered unlikely".
        def _is_verb_kernel(nd):
            return (
                isinstance(nd, Singleton) and nd.kernel is not None
                and isinstance(nd.kernel.edgeLabel, Singleton)
                and (nd.kernel.edgeLabel.type or '').strip().lower() == 'verb'
            )

        if len(filtered_top_node_ids) > 1 and isinstance(final_kernel, Singleton):
            main_id = None
            if self._is_subordinate_kernel(final_kernel):
                main_id = self._earliest_non_subordinate_root(filtered_top_node_ids, sorted_G)
            main_kernel = self.G.nodes[main_id]['data'] if main_id is not None else final_kernel
            if isinstance(main_kernel, Singleton) and main_kernel.kernel is not None:
                main_edge = (
                    main_kernel.kernel.edgeLabel.named_entity
                    if isinstance(main_kernel.kernel.edgeLabel, Singleton) else None
                )
                sibling_kernels = []
                for nid in filtered_top_node_ids:
                    if nid == main_id:
                        continue
                    nd = self.G.nodes[nid]['data']
                    if not _is_verb_kernel(nd):
                        continue
                    # Skip the root the kept main kernel itself derives from.
                    if main_id is None and isinstance(nd.kernel.edgeLabel, Singleton) \
                            and nd.kernel.edgeLabel.named_entity == main_edge:
                        continue
                    sibling_kernels.append(nd)
                if sibling_kernels:
                    new_props = defaultdict(list)
                    for k, v in dict(main_kernel.properties).items():
                        new_props[k] = list(v) if isinstance(v, (list, tuple)) else v
                    existing_ids = {
                        x.id for x in new_props.get('SENTENCE', [])
                        if isinstance(x, Singleton)
                    }
                    for sk in sibling_kernels:
                        if sk.id not in existing_ids:
                            new_props['SENTENCE'].append(sk)
                            existing_ids.add(sk.id)
                    final_kernel = main_kernel.update_node_props(new_props)

        # Phase 4: post-processing pipeline
        # acl_replacement and check_for_action_ed_node run here because they may
        # mutate self.G via create_existential / create_sentence; the rest of the
        # pipeline is pure-functional and lives in KernelPostProcessor.run().
        final_kernel = self.post.acl_replacement(final_kernel, acl_relcl_map)
        final_kernel = self.check_for_action_ed_node(
            acl_relcl_map, final_kernel,
            SimpleNamespace(shouldLoop=True, edgeForKernel=None, previousKernel=None),
            position_pairs,
        )
        final_kernel = self.post.run(final_kernel)
        final_kernel = self._promote_embedded_reduced_relative_reporting_root(final_kernel)

        print(f"{final_kernel.to_string()}\n")
        return final_kernel

    # Defensive fallback only — the live set comes from HOnK's RelativePronoun
    # class (raw_data/pronouns/relative_pronouns.txt) via _relative_pronouns().
    _REL_PRONOUNS = {"that", "which", "who", "whom", "whose"}

    def _relative_pronouns(self):
        try:
            getter = getattr(self.services.getHOnK(), "getRelativePronouns", None)
            if callable(getter):
                pronouns = {str(p).lower() for p in getter() if p}
                if pronouns:
                    return pronouns
        except Exception:
            pass
        return self._REL_PRONOUNS

    def _is_subordinate_kernel(self, kernel) -> bool:
        """True when `kernel` is a subordinate clause: an adverbial clause
        carrying a `mark` (before/after/while/until/because/…) or a relative
        clause whose source/target is a relative pronoun (`var` type, or a
        surface relative pronoun)."""
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return False
        # advcl marker — may sit on the kernel itself or on its edge label
        # (e.g. "before work ends" carries `mark:before` on the `end` edge).
        if dict(kernel.properties).get('mark'):
            return True
        edge = kernel.kernel.edgeLabel
        if isinstance(edge, Singleton) and dict(edge.properties).get('mark'):
            return True
        for side in (kernel.kernel.source, kernel.kernel.target):
            if isinstance(side, Singleton):
                if (side.type or '').strip().lower() == 'var':
                    return True
                if (side.named_entity or '').strip().lower() in self._relative_pronouns():
                    return True
        return False

    def _kernel_min_pos(self, kernel) -> float:
        """Lowest token position across a kernel's source/edge/target — used to
        order candidate main clauses by surface order (the main clause usually
        precedes its subordinates)."""
        best = float('inf')

        def _scan(node):
            nonlocal best
            if isinstance(node, Singleton):
                p = dict(node.properties).get('pos')
                try:
                    best = min(best, float(p))
                except (TypeError, ValueError):
                    pass
                if node.kernel is not None:
                    for child in (node.kernel.source, node.kernel.edgeLabel, node.kernel.target):
                        _scan(child)

        _scan(kernel)
        return best

    def _earliest_non_subordinate_root(self, root_ids, sorted_G):
        """Among the root nodes, return the id of the earliest (lowest token
        position) non-subordinate verb kernel, or None when every root is
        subordinate (defer to the legacy selection)."""
        candidates = [
            nid for nid in root_ids
            if isinstance(self.G.nodes[nid]['data'], Singleton)
            and self.G.nodes[nid]['data'].kernel is not None
            and not self._is_subordinate_kernel(self.G.nodes[nid]['data'])
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda nid: self._kernel_min_pos(self.G.nodes[nid]['data']))

    def _preferred_reduced_relative_reporting_root_id(self, selected_root_id, root_ids, sorted_G):
        """Prefer a passive reduced-relative reporting verb over an outcome verb.

        Crime rows such as "an investigation recorded near X concluded ..."
        produce two same-component roots after the acl normalisation: the matrix
        lifecycle/outcome verb (`conclude`) and the passive reduced-relative
        reporting verb (`record`) sharing the same subject.  The reporting event
        is the stable predicate for cross-row comparison; the outcome clause is
        status context.
        """
        if selected_root_id is None or selected_root_id not in self.G:
            return None
        selected_kernel = self.G.nodes[selected_root_id]['data']
        if not self._kernel_edge_matches_class(selected_kernel, "LifecycleOutcomeVerb"):
            return None

        for nid in reversed([n for n in sorted_G if n in root_ids]):
            if nid == selected_root_id:
                continue
            candidate = self.G.nodes[nid]['data']
            if not self._kernel_edge_matches_class(candidate, "ReportingVerb"):
                continue
            if not self._is_passive_reduced_relative_root(nid):
                continue
            if not self._roots_share_subject(nid, selected_root_id, candidate, selected_kernel):
                continue
            return nid
        return None

    def _kernel_edge_matches_class(self, kernel, class_name):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return False
        edge = kernel.kernel.edgeLabel
        if not isinstance(edge, Singleton):
            return False
        try:
            return self.post.matchers.matches_class(edge, class_name, kernel=kernel)
        except Exception:
            return False

    def _is_passive_reduced_relative_root(self, root_id):
        passive_labels = {"nsubjpass", "subjpass"}
        return bool(self._root_subject_ids(root_id, passive_labels))

    def _roots_share_subject(self, lhs_root_id, rhs_root_id, lhs_kernel, rhs_kernel):
        lhs_subjects = self._root_subject_ids(lhs_root_id, {"nsubj", "nsubjpass", "subj", "subjpass"})
        rhs_subjects = self._root_subject_ids(rhs_root_id, {"nsubj", "nsubjpass", "subj", "subjpass"})
        if lhs_subjects and rhs_subjects:
            return bool(lhs_subjects & rhs_subjects)

        lhs_refs = self._referenced_ids(lhs_kernel.kernel.source) | self._referenced_ids(lhs_kernel.kernel.target)
        rhs_refs = self._referenced_ids(rhs_kernel.kernel.source) | self._referenced_ids(rhs_kernel.kernel.target)
        for value in dict(rhs_kernel.properties).values():
            rhs_refs |= self._referenced_ids(value)
        return bool(lhs_refs and rhs_refs and lhs_refs & rhs_refs)

    def _root_subject_ids(self, root_id, labels):
        if root_id not in self.G:
            return set()
        subject_ids = set()
        for _src, dst, data in self.G.out_edges(root_id, data=True):
            label = data.get("label")
            label_name = str(getattr(label, "named_entity", "") or "")
            if label_name not in labels or dst not in self.G:
                continue
            subject_ids.add(dst)
            subject_ids |= self._referenced_ids(self.G.nodes[dst].get("data"))
        return subject_ids

    @classmethod
    def _referenced_ids(cls, value):
        ids = set()
        if isinstance(value, Singleton):
            ids.add(value.id)
            if value.kernel is not None:
                ids |= cls._referenced_ids(value.kernel.source)
                ids |= cls._referenced_ids(value.kernel.target)
                ids |= cls._referenced_ids(value.kernel.edgeLabel)
            for prop_value in dict(value.properties).values():
                ids |= cls._referenced_ids(prop_value)
        elif isinstance(value, SetOfSingletons):
            ids.add(value.id)
            for entity in value.entities:
                ids |= cls._referenced_ids(entity)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                ids |= cls._referenced_ids(item)
        return ids

    def _promote_embedded_reduced_relative_reporting_root(self, kernel):
        """Recover a reporting root when a lifecycle outcome swallowed it.

        Some parses reduce "an investigation recorded near X concluded..." to
        `conclude(?, AND(recorded, damage))` with the event subject stranded in
        `TIME_STATUS`.  The candidate-root helper cannot see `recorded` at that
        point because it is no longer a root, so this fallback promotes the
        embedded reporting verb under the same narrow ontology guards.
        """
        if not self._kernel_edge_matches_class(kernel, "LifecycleOutcomeVerb"):
            return kernel
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel

        reporting_node, content_nodes = self._embedded_reporting_target_parts(kernel.kernel.target)
        if reporting_node is None or not content_nodes:
            return kernel

        props = defaultdict(list)
        for key, value in dict(kernel.properties).items():
            props[key] = list(value) if isinstance(value, (list, tuple)) else value

        status_nodes = [
            item for item in self._as_list(props.get("TIME_STATUS"))
            if isinstance(item, Singleton)
        ]
        if status_nodes:
            original_status = status_nodes[0]
            event_content = self._status_event_content_node(original_status)
            status = self._status_node_with_lifecycle_completion(original_status)
            props["TIME_STATUS"] = [
                status if getattr(item, "id", None) == status.id else item
                for item in self._as_list(props.get("TIME_STATUS"))
            ]
            if event_content is not None and not any(
                self._same_node_identity(event_content, node) for node in content_nodes
            ):
                content_nodes.append(event_content)
        props = self._drop_lifecycle_outcome_specifications(props)

        new_edge_name = dict(reporting_node.properties).get("lemma") or reporting_node.named_entity
        new_edge = reporting_node.update_name(new_edge_name)
        promoted = kernel.update_kernel(new_edge, "edgeLabel")
        promoted = promoted.update_kernel(self._wrap_promoted_target(kernel.kernel.target, content_nodes), "target")
        return promoted.update_node_props(props)

    def _drop_lifecycle_outcome_specifications(self, props):
        specs = self._as_list(props.get("SPECIFICATION"))
        if not specs:
            return props
        kept = [
            item for item in specs
            if not self._value_matches_class(item, "LifecycleOutcomeVerb")
        ]
        if kept == specs:
            return props
        new_props = defaultdict(list)
        for key, value in props.items():
            new_props[key] = value
        if kept:
            new_props["SPECIFICATION"] = kept
        else:
            new_props.pop("SPECIFICATION", None)
        return new_props

    def _value_matches_class(self, value, class_name):
        if isinstance(value, Singleton):
            try:
                if self.post.matchers.matches_class(value, class_name):
                    return True
            except Exception:
                pass
            if value.kernel is not None:
                return self._kernel_edge_matches_class(value, class_name)
        elif isinstance(value, SetOfSingletons):
            return any(self._value_matches_class(entity, class_name) for entity in value.entities)
        return False

    def _embedded_reporting_target_parts(self, target):
        if isinstance(target, SetOfSingletons):
            entities = list(target.entities)
        elif isinstance(target, Singleton):
            entities = [target]
        else:
            return None, []

        reporting = None
        content = []
        for entity in entities:
            if isinstance(entity, Singleton) and self.post.matchers.matches_class(entity, "ReportingVerb"):
                reporting = entity
                continue
            if isinstance(entity, (Singleton, SetOfSingletons)):
                content.append(entity)
        return reporting, content

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _same_node_identity(left, right):
        if not isinstance(left, Singleton) or not isinstance(right, Singleton):
            return False
        if left.id is not None and right.id is not None and left.id == right.id:
            return True
        return bool(left.named_entity and left.named_entity == right.named_entity)

    def _status_node_with_lifecycle_completion(self, status):
        props = dict(status.properties)
        existing = props.get("type")
        props["type"] = "complete"
        status_name = self._status_head_surface(status)
        if status_name is not None:
            return status.update_name(status_name).update_node_props(props)
        return status.update_node_props(props)

    def _status_head_surface(self, status):
        if not isinstance(status, Singleton) or not isinstance(status.named_entity, str):
            return None
        parts = status.named_entity.split()
        status_parts = []
        while parts and self._surface_matches_class(parts[-1], "StatusNoun"):
            status_parts.insert(0, parts.pop())
        return " ".join(status_parts) if status_parts else None

    def _status_event_content_node(self, status):
        if not isinstance(status, Singleton) or not isinstance(status.named_entity, str):
            return None
        parts = status.named_entity.split()
        while parts and self._surface_matches_class(parts[-1], "StatusNoun"):
            parts.pop()
        if not parts:
            return None
        content_name = " ".join(parts)
        if content_name == status.named_entity:
            return None
        props = {
            key: value
            for key, value in dict(status.properties).items()
            if key != "type"
        }
        return Singleton(
            id=status.id,
            named_entity=content_name,
            properties=frozenset(props.items()),
            min=status.min,
            max=status.max,
            type=status.type,
            confidence=status.confidence,
            kernel=status.kernel,
        )

    def _surface_matches_class(self, surface, class_name):
        node = Singleton(
            id=-1,
            named_entity=surface,
            properties=frozenset(),
            min=-1,
            max=-1,
            type="noun",
            confidence=1.0,
            kernel=None,
        )
        try:
            return self.post.matchers.matches_class(node, class_name)
        except Exception:
            return False

    @staticmethod
    def _wrap_promoted_target(original_target, content_nodes):
        if len(content_nodes) == 1:
            return content_nodes[0]
        return SetOfSingletons(
            id=getattr(original_target, "id", -1),
            type=Grouping.AND,
            entities=content_nodes,
            min=min(getattr(node, "min", 0) for node in content_nodes),
            max=max(getattr(node, "max", 0) for node in content_nodes),
            confidence=min(getattr(node, "confidence", 1.0) for node in content_nodes),
            root=getattr(original_target, "root", False),
        )

    def find_prepositions_and_true_targets(self):
        found_preposition_labels = {}
        true_targets = set()

        # Remove 'dep' edges, given there are other edges. TODO: Could this be done at the preprocessing stage?
        edge_labels = defaultdict(set)  # (source ID, target ID) : edge label name
        for edge in self.G.edges(data=True, keys=True):
            edge_labels[(edge[0], edge[1])].add(edge[3]['label'].named_entity)
        edge_labels = {key: "dep" in value and len(value) > 1 for key, value in edge_labels.items()}  # (source ID, target ID): True/False

        edges_to_remove = []
        for edge in self.G.edges(data=True, keys=True):
            if edge[3]['label'].named_entity == 'dep' and edge_labels[(edge[0], edge[1])]:
                edges_to_remove.append((edge[0], edge[1], edge[2]))

        self.G.remove_edges_from(edges_to_remove)

        if len(self.G.edges(data=True)) > 0:
            # A single word that precedes a noun phrase complement and expresses spatial relations (*in* the house)
            prototypical_prepositions = self.services.getHOnK().getPrototypicalPrepositions()

            if prototypical_prepositions:
                prepositions_pattern = r"\b(" + "|".join(
                    map(re.escape, sorted(prototypical_prepositions, key=len, reverse=True))) + r")\b"
                prepositions_regex = re.compile(prepositions_pattern)
            else:
                prepositions_regex = None

            for edge in self.G.edges(data=True):
                source, target, edge_label = edge
                source = self.G.nodes[source]['data']
                target = self.G.nodes[target]['data']
                edge_label = edge_label['label'].named_entity

                is_prepositional_phrase = False
                if prepositions_regex:
                    match = prepositions_regex.search(edge_label)
                    if (match and
                            match.group(1) != edge_label and  # Check the preposition is part of a larger phrase
                            not is_label_verb(edge_label.split()[0]) and
                            not case_in_props(dict(target.properties))):
                        is_prepositional_phrase = True

                is_gerund_phrase = (
                        edge_label.endswith('ing') and
                        source.type != "existential" and
                        not self.node_functions.check_node_coordinations_for_auxiliary(edge, self.G)
                )

                if is_prepositional_phrase or is_gerund_phrase:
                    nx.set_node_attributes(self.G, {target.id: target.add_property('kernel', 'root')}, 'data')
                    found_preposition_labels[target.id] = lemmatize_verb(edge_label)
                elif (is_label_verb(edge_label) and
                      is_kernel_in_props(source) and
                      target.type != 'existential'):
                    nx.set_node_attributes(self.G, {target.id: target.strip_root_properties()}, 'data')
                elif not is_kernel_in_props(target):
                    true_targets.add(target.id)

        return true_targets, found_preposition_labels

    def get_topological_root_node_ids(self, true_targets):
        filtered_top_node_ids = set()

        # Loop over every source and target for every edge
        for node_id in itertools.chain.from_iterable(map(lambda x: [x[0], x[1]], self.G.edges(data=True))):
            edge_node = self.G.nodes[node_id]['data']

            # Check if edge node is NOT None, NOT in true targets, and IS a root
            if (
                    edge_node is not None and
                    edge_node.id not in true_targets and
                    is_kernel_in_props(edge_node)
            ):
                filtered_top_node_ids.add(node_id)

        # Keep only genuine roots: nodes with no incoming edges
        candidates = {nid for nid in filtered_top_node_ids if len(list(self.G.in_edges(nid))) == 0}

        # candidates = {
        #     nid for nid in filtered_top_node_ids
        #     if len([(u, v, data) for u, v, data in self.G.in_edges(nid, data=True) if
        #             data['label'].named_entity != 'acl']) == 0
        # }

        if candidates:
            filtered_top_node_ids = candidates

        if len(filtered_top_node_ids) == 0:
            # Get either the only node that remains, or all nodes that contain a root
            filtered_top_node_ids = \
                [list(self.G.nodes)[-1]] if len(self.G.nodes) == 1 else \
                    [
                        self.node_functions.get_node_id(x[0]) for x in self.G.nodes(data=True) if
                        is_kernel_in_props(x[1]['data'], False)
                    ]
            # Remove duplicate IDs
            filtered_top_node_ids = list(map(int, numpy.unique(filtered_top_node_ids)))

        return [id for id in list(NodeFunctions.sort_G(self.G)) if id in filtered_top_node_ids]

    def get_position_pairs(self):
        position_pairs = {}
        for edge in self.G.edges(data=True):
            source, target, _ = edge
            source = self.G.nodes[source]['data']
            target = self.G.nodes[target]['data']

            from LaSSI.ner.node_functions import get_min_position
            source_pos = get_min_position(source)
            target_pos = get_min_position(target)
            if target_pos > source_pos:
                if source.id in position_pairs:
                    if target_pos < position_pairs[source.id]:
                        position_pairs[source.id] = target_pos
                else:
                    position_pairs[source.id] = target_pos
        return position_pairs

    def check_for_action_ed_node(self, acl_relcl_map, final_kernel, loop_settings, position_pairs):
        action_ed_node = find_action_ed_node_in_kernel(final_kernel)
        # If no kernel at all OR we have a kernel with edge label "be" or equal to a found action(ed) node
        if (
                not hasattr(final_kernel, 'kernel') or
                final_kernel.kernel is None or
                (
                    action_ed_node and
                    (
                        is_canonical_copula(final_kernel.kernel.edgeLabel.named_entity) or
                        final_kernel.kernel.edgeLabel == action_ed_node
                    )
                )
        ):
            # If a Singleton of type verb, make this an edge with no source or target
            if isinstance(final_kernel, Singleton) and final_kernel.type.lower() == 'verb':
                final_kernel = create_edge_kernel(final_kernel)
                final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
            else:
                nodes = {final_kernel.id: self.G.nodes[final_kernel.id]}
                #
                if not hasattr(final_kernel, 'properties') or not action_ed_node:
                    self.G, _ = create_existential(self.G, nodes, self.node_functions)
                    self.G, final_kernel, loop_settings, acl_relcl_map = (
                        create_sentence(
                            self.G, self.G.edges(data=True, keys=True), nodes, self.negations, final_kernel.id, {},
                            self.node_functions, loop_settings, acl_relcl_map
                        ))
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
                else:
                    final_kernel = rewrite_action_ed_node(final_kernel, action_ed_node, self.negations)
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
        return final_kernel

    # Check if final kernel is "empty" be(?, ?) and use the properties of 'SENTENCE'
    def check_if_empty_kernel(self, kernel, force=False):  # Note: use _check_if_empty_kernel in KernelPostProcessor.py
        properties_to_keep = dict()
        new_kernel = None
        if (
                isinstance(kernel, Singleton) and
                kernel.kernel is not None and
                kernel.kernel.edgeLabel is not None and
                (is_canonical_copula(kernel.kernel.edgeLabel.named_entity) if not force else True)
        ):
            source_empty = kernel.kernel.source is None or kernel.kernel.source.type == 'existential'
            target_empty = kernel.kernel.target is None or kernel.kernel.target.type == 'existential'
            is_be = is_canonical_copula(kernel.kernel.edgeLabel.named_entity)

            if (source_empty or target_empty) if is_be else (source_empty and target_empty):
                node_props = dict(kernel.properties)
                if len(node_props) > 0 and 'SENTENCE' in node_props:
                    for key in node_props:
                        if key == 'SENTENCE':
                            new_kernel = node_props['SENTENCE'][0]
                            new_kernel = self.check_if_empty_kernel(new_kernel, force)

                            if is_be and new_kernel is not None and new_kernel.kernel is not None:
                                outer_nominal = kernel.kernel.source if not source_empty else kernel.kernel.target
                                inner_source = new_kernel.kernel.source
                                inner_target = new_kernel.kernel.target

                                # Check if the inner verb syntax mapped the nominal to its source
                                if outer_nominal is not None and inner_source is not None and getattr(inner_source,
                                                                                                      'id',
                                                                                                      None) == getattr(
                                        outer_nominal, 'id', None):
                                    is_loc = False

                                    # Safe to invert if target is empty, existential, or a location
                                    if inner_target is None or getattr(inner_target, 'type', None) == 'existential':
                                        is_loc = True
                                    elif hasattr(self, 'post') and hasattr(self.post,
                                                                           'matchers') and self.post.matchers.is_location_like(
                                            inner_target):
                                        is_loc = True
                                    elif hasattr(self, 'matchers') and self.matchers.is_location_like(inner_target):
                                        is_loc = True

                                    if is_loc:
                                        from LaSSI.ner.node_functions import create_existential_node
                                        new_kernel = new_kernel.update_kernel(create_existential_node(), "source")
                                        new_kernel = new_kernel.update_kernel(outer_nominal, "target")

                                        # Demote the location target to a SPACE property
                                        if inner_target is not None and getattr(inner_target, 'type',
                                                                                None) != 'existential':
                                            if 'SPACE' not in properties_to_keep:
                                                properties_to_keep['SPACE'] = []
                                            properties_to_keep['SPACE'].append(inner_target)
                        else:
                            properties_to_keep[key] = node_props[key]

        if new_kernel is not None:
            if len(properties_to_keep) > 0:
                from LaSSI.ner.MergeSetOfSingletons import merge_properties
                # Merge the outer wrapper's properties directly onto the inner kernel
                merged_props = merge_properties(dict(new_kernel.properties), properties_to_keep)
                return new_kernel.update_node_props(merged_props)
            else:
                return new_kernel
        else:
            return kernel

    # Rewrite kernel if positions are not correct
    def kernel_post_processing(self, kernel, position_pairs):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel

        # If target is an adjective, and there is a pronoun or entity in the properties, swap these round
        if kernel.kernel.target is not None and isinstance(kernel.kernel.target, Singleton) and kernel.kernel.target.type.startswith("JJ"):
            for key in dict(kernel.properties):
                properties_key = dict(kernel.properties)[key]
                for node in properties_key:
                    if node.type in DependencyRoles.predicative_subject_pos_tags():
                        new_kernel_properties = defaultdict(list)
                        new_kernel_properties[kernel.kernel.target.type].append(kernel.kernel.target)
                        new_kernel_properties = merge_properties(dict(kernel.properties), new_kernel_properties)
                        kernel = kernel.update_node_props(new_kernel_properties)
                        kernel = kernel.update_kernel(properties_key[0], "target")
                        break

        # TODO: Temporary patch, investigate how applicable this is to other scenarios
        # be(ENTITY1, ?)[SENTENCE:verb(?, ENTITY2)[]]
        if (
            kernel.kernel.target is not None and
            kernel.kernel.target.type == 'existential' and
            is_canonical_copula(kernel.kernel.edgeLabel.named_entity) and
            'SENTENCE' in dict(kernel.properties)
        ):
            for key in dict(kernel.properties):
                if key == 'SENTENCE':
                    sentences = dict(kernel.properties)[key]
                    for sentence in sentences:
                        if sentence.kernel.source.type == 'existential':
                            for sentence_key in dict(sentence.properties):
                                sentence_properties = dict(sentence.properties)[sentence_key]
                                for prop_node in sentence_properties:
                                    if prop_node.id == kernel.kernel.source.id:
                                        kernel = sentence.update_kernel(prop_node, "source")

        properties_to_keep = dict(kernel.properties)
        new_target = kernel.kernel.target

        # Semi-modal kernels: pull the appropriate SENTENCE child up as the target
        if (
                kernel.id in position_pairs and
                kernel.kernel.edgeLabel is not None and
                check_semi_modal(kernel.kernel.edgeLabel.named_entity)
        ):
            position_value = position_pairs[kernel.id]
            if 'SENTENCE' in dict(kernel.properties):
                properties_to_keep['SENTENCE'] = []
                for sentence_elm in list(dict(kernel.properties)['SENTENCE']):
                    if int(float(dict(sentence_elm.kernel.edgeLabel.properties)['pos'])) == position_value:
                        new_target = sentence_elm
                    else:
                        properties_to_keep['SENTENCE'].append(sentence_elm)

            return Singleton(
                id=kernel.id,
                named_entity="",
                type="SENTENCE",
                min=kernel.min,
                max=kernel.max,
                confidence=kernel.confidence,
                kernel=Relationship(
                    source=kernel.kernel.source,
                    target=new_target,
                    edgeLabel=kernel.kernel.edgeLabel,
                    isNegated=kernel.kernel.isNegated
                ),
                properties=create_props_for_singleton(properties_to_keep),
            )
        # Adjective + entity kernel without an edge label (e.g. None(clear, vision))
        elif (
                kernel.kernel.edgeLabel is None and
                kernel.kernel.source is not None and
                kernel.kernel.target is not None and
                (
                    (kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY') or
                    (kernel.kernel.target.type.startswith('JJ') and kernel.kernel.source.type == 'ENTITY')
                )
        ):
            new_edges = []
            root_id = None
            # Update source/target with adjective as properties
            if kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY':
                node_props = merge_properties(
                    dict(kernel.kernel.target.properties),
                    {kernel.kernel.source.type: kernel.kernel.source}
                )
                root_id = kernel.kernel.target.id
                nx.set_node_attributes(self.G, {root_id: kernel.kernel.target.update_node_props(node_props)}, 'data')
                create_existential(self.G, {kernel.kernel.target.id: self.G.nodes[kernel.kernel.target.id]}, self.node_functions)
            elif kernel.kernel.target.type.startswith('JJ') and kernel.kernel.source.type == 'ENTITY':
                node_props = merge_properties(
                    dict(kernel.kernel.source.properties),
                    {kernel.kernel.target.type: kernel.kernel.target}
                )
                root_id = kernel.kernel.source.id
                nx.set_node_attributes(self.G, {root_id: kernel.kernel.source.update_node_props(node_props)}, 'data')
                create_existential(self.G, {kernel.kernel.target.id: self.G.nodes[kernel.kernel.target.id]}, self.node_functions)

            self.G, kernel, edge_to_loop, acl_relcl_map = create_sentence(
                self.G, new_edges, {key: x for key, x in self.G.nodes(data=True)}, self.negations, root_id, {}, self.node_functions,
                SimpleNamespace(shouldLoop=False, edgeForKernel=None, previousKernel=None), {}
            )
            kernel = self.kernel_post_processing(kernel, position_pairs)
        return kernel
