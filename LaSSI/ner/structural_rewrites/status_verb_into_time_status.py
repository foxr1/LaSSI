__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons


class StatusVerbIntoTimeStatusRule(StructuralRewriteRule):
    """Fold a status/state verb stranded on the event's own argument onto a
    co-occurring ``TIME_STATUS`` entity.

    Background: "<event> is awaiting a court outcome" parses with ``await`` as a
    finite verb. Its object (the court outcome) is demoted to a ``TIME_STATUS``
    property while its subject becomes the clause argument, so
    ``rewrite_properties_logically`` classifies the state verb ``await`` as a
    kernel-level context and bolts ``type:await`` onto that argument — a
    *duplicate* of the kernel's own source/target. The subsequent dedupe pass
    then deletes the duplicate, losing the ``awaiting`` state entirely, while the
    genuine status entity (the court outcome) sits under ``TIME_STATUS`` with no
    state at all.

    This rule reunites them: when a logical-context property entry is merely the
    kernel's own source/target re-tagged with a HOnK state-verb ``type``, and a
    ``TIME_STATUS`` entity is present, the verb is moved onto that entity's
    ``type`` and the redundant context entry dropped. The result —
    ``TIME_STATUS:court[type:awaiting]`` — matches how the verbless participial
    form ("Awaiting court outcome") is already represented, so the two
    paraphrases converge. General for any "<event> is awaiting / pending /
    remaining <status outcome>".

    Three independent gates keep the rewrite tightly scoped (so genuine temporal
    contexts and unrelated status clauses are untouched):
      1. the context entry must *duplicate* the kernel's own source/target
         (id or surface name) — i.e. it is the event argument re-tagged, not
         independent content;
      2. its ``type`` must be a HOnK-classed state verb;
      3. a ``TIME_STATUS`` entity must already be present to receive it.
    """

    name = "status_verb_into_time_status"
    phase = "post_logical_rewrite"

    @staticmethod
    def _state_verbs(ctx):
        try:
            return {str(v).lower() for v in (ctx.services.getHOnK().getStateVerbs() or set())}
        except Exception:
            return set()

    @staticmethod
    def _argument_ids_and_names(kernel):
        ids, names = set(), set()
        for endpoint in (kernel.kernel.source, kernel.kernel.target):
            if isinstance(endpoint, Singleton):
                ids.add(endpoint.id)
                if endpoint.named_entity:
                    names.add(endpoint.named_entity)
            elif isinstance(endpoint, SetOfSingletons):
                for entity in endpoint.entities:
                    if isinstance(entity, Singleton):
                        ids.add(entity.id)
                        if entity.named_entity:
                            names.add(entity.named_entity)
        return ids, names

    @staticmethod
    def _type_values(node):
        if not isinstance(node, Singleton):
            return []
        type_value = dict(node.properties).get('type')
        if type_value is None:
            return []
        return list(type_value) if isinstance(type_value, (list, tuple)) else [type_value]

    def _is_state_verb(self, value, state_verbs):
        if not isinstance(value, str) or not value.strip():
            return False
        v = value.lower()
        return v in state_verbs or lemmatize_verb(value).lower() in state_verbs

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        props = dict(kernel.properties)

        status_values = props.get('TIME_STATUS')
        if not status_values:
            return None
        status_list = status_values if isinstance(status_values, (list, tuple)) else [status_values]
        status_entity = next((s for s in status_list if isinstance(s, Singleton)), None)
        if status_entity is None:
            return None

        state_verbs = self._state_verbs(ctx)
        if not state_verbs:
            return None
        arg_ids, arg_names = self._argument_ids_and_names(kernel)

        for key, value in props.items():
            if not isinstance(key, str) or not key.isupper() or key == 'TIME_STATUS':
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for idx, entry in enumerate(values):
                if not isinstance(entry, Singleton):
                    continue
                # Gate 1: the entry must be the event's own argument re-tagged.
                if entry.id not in arg_ids and entry.named_entity not in arg_names:
                    continue
                # Gate 2: ...carrying a HOnK state-verb `type`.
                verb_types = [v for v in self._type_values(entry)
                              if self._is_state_verb(v, state_verbs)]
                if not verb_types:
                    continue
                return {
                    "context_key": key,
                    "context_index": idx,
                    "verb_types": verb_types,
                    "status_id": status_entity.id,
                }
        return None

    def apply(self, kernel, bindings, ctx):
        props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        key = bindings["context_key"]
        idx = bindings["context_index"]
        verb_types = bindings["verb_types"]
        status_id = bindings["status_id"]

        # 1. Drop the stranded context entry (and the now-empty key).
        ctx_values = props.get(key)
        ctx_values = list(ctx_values) if isinstance(ctx_values, (list, tuple)) else [ctx_values]
        ctx_values = [v for i, v in enumerate(ctx_values) if i != idx]
        if ctx_values:
            props[key] = ctx_values
        else:
            props.pop(key, None)

        # 2. Merge the state verb onto the TIME_STATUS entity's `type`.
        status_values = props.get('TIME_STATUS')
        status_values = list(status_values) if isinstance(status_values, (list, tuple)) else [status_values]
        new_status = []
        for s in status_values:
            if isinstance(s, Singleton) and s.id == status_id:
                s_props = dict(s.properties)
                existing = s_props.get('type')
                merged = list(existing) if isinstance(existing, (list, tuple)) else (
                    [existing] if existing is not None else [])
                for verb in verb_types:
                    if verb not in merged:
                        merged.append(verb)
                s_props['type'] = tuple(merged) if len(merged) > 1 else merged[0]
                new_status.append(s.update_node_props(s_props))
            else:
                new_status.append(s)
        props['TIME_STATUS'] = new_status

        return kernel.update_node_props(props)
