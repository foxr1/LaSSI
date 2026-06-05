"""Row→chunk→row processing pipeline.

Encapsulates the entire chunk-based notice-processing path that used to
live inline in :class:`LaSSI.LaSSI.LaSSI`. Responsibilities:

1. **Row expansion** (:meth:`expand_rows`) — split each YAML row into
   structural chunks via :func:`LaSSI.phases.StructuredSentenceLoader.split_structured`,
   rejoin verbless bodies onto labels for CoreNLP-friendly parsing,
   split conjunctive ACTION chunks ("Abandon X and replace Y"), and
   profile each chunk's role via :func:`LaSSI.ner.ChunkProfiler.profile`.

2. **Per-row merging** (:meth:`merge_intermediate_representations`) —
   reassemble per-sub-sentence kernels back into one kernel per YAML
   row. When a row contains a contiguous run of ACTION chunks, wrap
   them in ``SetOfSingletons(Grouping.AND, …)``. When a
   ``TOPIC_HEADER`` chunk anchors the row, synthesise an outer
   ``be(topic, AND(actions))`` kernel. Repair the broken
   ``be(also, ?)[NP-properties]`` shape into a bare NP Singleton and
   fold it into the AND group as an additional conjunct.

3. **MeuDB merge / output rewriting** — :meth:`merge_meu_dbs` and
   :meth:`rewrite_string_rep` keep the display-facing outputs aligned
   with the merged per-row representation.

The class owns its own chunk metadata (``chunk_meta``, ``chunk_roles``,
``row_to_sub_indices``, ``row_original_text``) and a counter for
synthetic node ids. The orchestrating :class:`~LaSSI.LaSSI.LaSSI`
instance retains only thin references to these for backward
compatibility (the cache-invalidation path and a few benchmark helpers
read ``row_to_sub_indices`` directly).
"""

from __future__ import annotations

import re as _re
from collections import defaultdict
from typing import Any, Callable, List, Optional

from LaSSI.files.JSONDump import json_dumps
from LaSSI.ner.ChunkProfiler import ChunkRole, profile as profile_chunk
from LaSSI.phases.StructuredSentenceLoader import (
    StructuredChunk,
    split_structured,
)


class RowChunkPipeline:
    """See module docstring."""

    # ------------------------------------------------------------------
    # Class-level constants (formerly on :class:`LaSSI`).
    # ------------------------------------------------------------------

    # Role priority for choosing the primary kernel inside a row. ACTION
    # (imperative roadworks/transport notice verbs) wins over substantive
    # PROSE; REPORT_HEADER (weather forecast frames) wins over generic
    # HEADER when no clause body exists. TOPIC_HEADER sits last because
    # it is only meant to provide a *subject anchor* for the M4 synthetic
    # be(topic, AND(actions)) kernel — it should never be picked as
    # primary over a real verb-led clause.
    _CHUNK_PRIMARY_PRIORITY = (
        "ACTION", "PROSE", "REPORT_HEADER", "HEADER", "TOPIC_HEADER",
    )

    # Roles whose construct-key properties are promoted onto the primary
    # without also retaining the sub-kernel under SENTENCE. These are
    # structurally subsidiary (a label, a status fragment, a metric
    # list, a date range) — their predication is fully captured by the
    # promoted properties.
    _CHUNK_SUBSIDIARY_ROLES = frozenset({
        "HEADER", "REPORT_HEADER", "STATUS", "ATTRIBUTE", "TIME_RANGE",
        "CONTEXT",
    })

    # Roles that contribute meaning *only* via their relationship to the
    # next chunk (the body that the colon introduced). A bare label like
    # "Traffic management" has no clausal content of its own — once any
    # construct-key properties have been promoted onto the primary, the
    # remainder should be dropped, not retained as ``be(label, ?)`` noise
    # under SENTENCE.
    _CHUNK_LABEL_ONLY_ROLES = frozenset({"HEADER", "REPORT_HEADER"})

    # Delimiters that can *open* an ACTION conjunct group — the first
    # conjunct sits at the head of the row or after a colon-introduced
    # header body. Subsequent conjuncts are joined by ``and X`` or by a
    # bare verb-adjacency split (see
    # :data:`_ACTION_CONJUNCT_FOLLOW_DELIMS`). Anything else
    # (a PERIOD, HYPHEN, SEMICOLON) breaks the run — those are
    # sentence-level separators, not within-action conjunctions.
    _ACTION_CONJUNCT_OPEN_DELIMS = frozenset({"ROW_START", "COLON"})
    _ACTION_CONJUNCT_FOLLOW_DELIMS = frozenset({"CONJUNCTION", "VERB_ADJACENCY"})

    # Adverbial heads that mark a verbless coordinate fragment in
    # block-language notices (Quirk et al. §13.51). When CoreNLP roots
    # such a fragment on the adverb itself, the resulting kernel takes
    # the shape ``be(RB[also], existential)[…NP-properties…]`` and the
    # real head noun is dropped from the surface form.
    # :meth:`_repair_adverb_fragment` recovers the NP head from the
    # chunk text.
    _ADVERB_FRAGMENT_TOKENS = frozenset({"also", "too", "additionally"})

    # Light particle/skip set used by :meth:`_extract_chunk_np`
    # to walk past leading determiners, prepositions, and the adverb
    # itself while scanning for the NP head. Kept conservative — the
    # head-noun selection is gated by the HOnK preposition set when
    # available.
    _ADVERB_FRAGMENT_SKIP = frozenset({
        "the", "a", "an", "of", "to", "for", "in", "on", "at",
        "by", "with", "from", "as", "into", "onto", "no", "not",
        "never", "also", "too", "additionally", "and", "or",
    })

    # Chunk roles whose intermediate representation never carries a
    # clausal verb — re-joining them onto a preceding label avoids
    # CoreNLP fragmenting a metric list / date range into a tree it
    # cannot root.
    _VERBLESS_BODY_ROLES = frozenset({"ATTRIBUTE", "TIME_RANGE"})

    # Label roles eligible for the verbless-body rejoin step.
    _LABEL_ROLES_FOR_REJOIN = frozenset({"HEADER", "REPORT_HEADER"})

    def __init__(
        self,
        honk: Any,
        services: Any,
        logger: Callable[[str], None] = print,
    ):
        self.honk = honk
        self.services = services
        self.logger = logger
        # Per-chunk metadata, populated by :meth:`expand_rows`.
        self.chunk_meta: List[StructuredChunk] = []
        self.chunk_roles: List[Any] = []
        self.row_to_sub_indices: List[List[int]] = []
        self.row_original_text: List[str] = []
        # Counter for synthetic ids issued during merging (AND groups,
        # outer be kernels). Starts in a high range so it cannot collide
        # with per-sub-sentence kernel ids coming from CoreNLP.
        self._synthetic_node_counter = 1_000_000_000
        # Lazy cache for the set of construct-key property names emitted
        # by HOnK's logical-rewriting rules.
        self._ontology_construct_keys_cache: Optional[set] = None
        # Cached set of narrow imperative verbs used by
        # :meth:`_split_conjunctive_action`.
        self._action_verbs_cache: Optional[set] = None

    # ------------------------------------------------------------------
    # Phase 1 — Row expansion.
    # ------------------------------------------------------------------

    def has_multi_chunk_rows(self) -> bool:
        return any(len(s) > 1 for s in self.row_to_sub_indices)

    def _verb_classes_for_split(self) -> set:
        """Narrow imperative-verb set used by the conjunctive-action
        splitter. Intentionally limited to CausativeVerb and
        MaterialisationVerb (abandon, demolish, install, rebuild,
        replace, excavate, renew) so that HOnK's fuzzy-loaded entries
        in TransitiveVerb (which include common nouns like "bicycle"
        and particles like "on") do not misfire the splitter on
        ordinary subject-led prose."""
        if self._action_verbs_cache is not None:
            return self._action_verbs_cache
        classes: set = set()
        if self.honk is not None:
            for acc in ("getCausativeVerbs", "getMaterialisationVerbs"):
                fn = getattr(self.honk, acc, None)
                if fn is None:
                    continue
                try:
                    classes.update(fn() or set())
                except Exception:
                    continue
        self._action_verbs_cache = classes
        return classes

    def _profile(self, chunk: StructuredChunk):
        return profile_chunk(chunk, self.honk)

    def _split_conjunctive_action(
        self, chunks: List[StructuredChunk],
    ) -> List[StructuredChunk]:
        """Split ACTION chunks that join two verb-led clauses, either
        via *and X* (``Abandon X and replace Y``) or by a bare
        verb-noun-verb adjacency (``Demolish FW2 rebuild FW4``). Each
        verb gets its own clause so the merger can capture both as
        sibling actions instead of CoreNLP collapsing the second verb
        under the first."""
        action_verbs = self._verb_classes_for_split()
        if not action_verbs:
            return chunks
        and_re = _re.compile(r"\s+and\s+([A-Za-z][A-Za-z'\-]*)")
        out: List[StructuredChunk] = []
        for chunk in chunks:
            if self._profile(chunk) != ChunkRole.ACTION:
                out.append(chunk)
                continue
            text = chunk.text
            m = and_re.search(text)
            if m and m.group(1).lower() in action_verbs:
                head = text[:m.start()].rstrip()
                tail = text[m.end() - len(m.group(1)):]
                out.append(StructuredChunk(
                    text=head, is_label=False,
                    delim_before=chunk.delim_before,
                ))
                out.append(StructuredChunk(
                    text=tail, is_label=False,
                    delim_before="CONJUNCTION",
                ))
                continue
            # Bare verb-noun-verb adjacency.
            token_iter = list(_re.finditer(
                r"[A-Za-z][A-Za-z0-9'\-]*", text,
            ))
            split_at = None
            for i in range(2, len(token_iter)):
                cur_tok = token_iter[i].group(0).lower()
                prev_tok = token_iter[i - 1].group(0).lower()
                if (cur_tok in action_verbs
                        and prev_tok not in action_verbs
                        and token_iter[0].group(0).lower() in action_verbs):
                    split_at = token_iter[i].start()
                    break
            if split_at is not None:
                head = text[:split_at].rstrip()
                tail = text[split_at:].lstrip()
                if head and tail:
                    out.append(StructuredChunk(
                        text=head, is_label=False,
                        delim_before=chunk.delim_before,
                    ))
                    out.append(StructuredChunk(
                        text=tail, is_label=False,
                        delim_before="VERB_ADJACENCY",
                    ))
                    continue
            out.append(chunk)
        return out

    def _rejoin_label_with_verbless_body(
        self, chunks: List[StructuredChunk],
    ) -> List[StructuredChunk]:
        """Glue a verbless body (a bare metric list, a bare date range)
        back onto the label-chunk that introduced it. CoreNLP cannot
        root a parse tree for a verbless fragment in isolation, so the
        rejoined surface ``Met Office forecast for X: 12.76°C, 57% …``
        is what makes it parse cleanly.

        Restricted to genuinely unparseable bodies (``ATTRIBUTE`` /
        ``TIME_RANGE``). A bare-NP body such as ``Some carriageway
        incursion`` is *not* rejoined: it parses fine on its own as
        ``be(carriageway[incursion], ?)``, whereas rejoining it behind a
        colon (``Traffic management: Some carriageway incursion``) makes
        the kernel builder root on the label and drop the body. The
        disruption content is recovered downstream as ``DISRUPTION`` from
        the standalone-parsed chunk instead."""
        if len(chunks) < 2 or self.honk is None:
            return chunks
        roles = [self._profile(c) for c in chunks]
        out: List[StructuredChunk] = []
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            cur_role = roles[i]
            if (i + 1 < len(chunks)
                    and getattr(cur_role, "value", cur_role) in self._LABEL_ROLES_FOR_REJOIN
                    and getattr(roles[i + 1], "value", roles[i + 1]) in self._VERBLESS_BODY_ROLES
                    and chunks[i + 1].delim_before == "COLON"):
                nxt = chunks[i + 1]
                out.append(StructuredChunk(
                    text=f"{cur.text}: {nxt.text}",
                    is_label=False,
                    delim_before=cur.delim_before,
                ))
                i += 2
            else:
                out.append(cur)
                i += 1
        return out

    def _inject_elided_copula(self, text: str) -> str:
        """Restore the passive copula elided in telegraphic notices.

        Block-language notices drop the copula of a passive/raising
        periphrasis — ``Work expected to end 2026-04-27`` rather than
        ``Work is expected to end …``, ``Lane closures scheduled to begin
        …``. CoreNLP mis-roots the copula-less form (it treats the bare
        head as an imperative/root verb), and the kernel builder then
        drops the infinitival complement and its date. Inserting the
        elided copula before a ``ModalAdjective`` that heads a
        ``<ModalAdj> to <verb>`` periphrasis makes the chunk parse like
        ordinary prose, so the *same* downstream path that already
        handles ``X is expected to end on <date>`` applies uniformly.

        Vocabulary is HOnK's ``ModalAdjective`` / copula classes — no
        hardcoded word lists. Only the chunk text sent to CoreNLP is
        affected; the user-facing ``row_original_text`` is untouched."""
        if self.honk is None:
            return text
        try:
            modals = {m.lower() for m in (self.honk.getModalAdjectives() or set())}
            copulas = {c.lower() for c in (self.honk.getCopulaSurfaceForms() or set())}
        except Exception:
            return text
        if not modals:
            return text
        tokens = text.split()
        if len(tokens) < 3:
            return text

        def _bare(tok):
            return _re.sub(r"[^A-Za-z'\-]", "", tok).lower()

        # Only telegraphic (copula-elided) chunks qualify. If the chunk
        # already contains a finite copula, the modal adjective is part of
        # a full clause — e.g. a reduced relative "… with parakeets
        # expected to fly over" inside "Newcastle is forecast to …".
        # Injecting there would corrupt the sentence, so bail out.
        if copulas and any(_bare(t) in copulas for t in tokens):
            return text

        for i in range(1, len(tokens) - 1):
            if (_bare(tokens[i]) in modals
                    and _bare(tokens[i + 1]) == "to"
                    and _bare(tokens[i - 1]) not in copulas):
                prev = _bare(tokens[i - 1])
                cop = "are" if prev.endswith("s") and not prev.endswith("ss") else "is"
                tokens.insert(i, cop)
                return " ".join(tokens)
        return text

    def expand_rows(self, raw_rows) -> List[str]:
        """Process raw YAML rows into a flat list of sub-sentence texts.

        Populates :attr:`chunk_meta`, :attr:`chunk_roles`,
        :attr:`row_to_sub_indices`, :attr:`row_original_text`. Returns
        the flat sub-sentence text list that downstream phases (NER,
        CoreNLP) consume."""
        expanded: List[str] = []
        self.chunk_meta = []
        self.chunk_roles = []
        self.row_to_sub_indices = []
        self.row_original_text = []
        for row_text in raw_rows:
            self.row_original_text.append(str(row_text))
            chunks = split_structured(str(row_text))
            if not chunks:
                chunks = [StructuredChunk(text=str(row_text))]
            chunks = self._rejoin_label_with_verbless_body(chunks)
            chunks = self._split_conjunctive_action(chunks)
            sub_indices: List[int] = []
            for chunk in chunks:
                chunk.text = self._inject_elided_copula(chunk.text)
                sub_indices.append(len(expanded))
                expanded.append(chunk.text)
                self.chunk_meta.append(chunk)
                self.chunk_roles.append(self._profile(chunk))
            self.row_to_sub_indices.append(sub_indices)
        return expanded

    # ------------------------------------------------------------------
    # Phase 2 — Per-row merge of intermediate representations.
    # ------------------------------------------------------------------

    def _ontology_construct_keys(self) -> set:
        """Set of property keys (uppercased construct names) that the
        HOnK logical-rewriting rules emit, e.g. SPACE, TIME,
        TIME_STATUS, CAUSATION. Used to decide which sub-sentence
        properties to promote onto the primary kernel."""
        if self._ontology_construct_keys_cache is not None:
            return self._ontology_construct_keys_cache
        keys: set = set()
        try:
            rules = self.honk.getLogicalRewritingRules() or {} if self.honk else {}
        except Exception:
            rules = {}
        for rule in rules.values():
            if getattr(rule, "logicalConstructName", None):
                keys.add(rule.logicalConstructName.upper())
            for name, _ in (getattr(rule, "additional_classifications", None) or []):
                if name:
                    keys.add(name.upper())
        self._ontology_construct_keys_cache = keys
        return keys

    def role_for_sub(self, sub_idx: int) -> str:
        roles = self.chunk_roles
        if not roles or sub_idx >= len(roles) or roles[sub_idx] is None:
            return "PROSE"
        role = roles[sub_idx]
        return getattr(role, "value", role)

    def _select_primary_idx(self, sub_indices, intermediate_representations):
        """Pick the sub-sentence index whose kernel should anchor the
        merged row when no ACTION conjunct group is present. Walks role
        priorities first, then falls back to any available Singleton,
        then to the first sub-index unconditionally."""
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton
        for pri in self._CHUNK_PRIMARY_PRIORITY:
            for idx in sub_indices:
                if not isinstance(intermediate_representations[idx], Singleton):
                    continue
                if self.role_for_sub(idx) == pri:
                    return idx
        for idx in sub_indices:
            if isinstance(intermediate_representations[idx], Singleton):
                return idx
        return sub_indices[0]

    def _synthetic_node_id(self) -> int:
        nxt = self._synthetic_node_counter
        self._synthetic_node_counter = nxt + 1
        return nxt

    def _detect_action_conjuncts(self, sub_indices, intermediate_representations):
        """Return the list of sub-indices that form a *contiguous*
        ACTION conjunct group: a head chunk introduced by ROW_START /
        COLON followed by ≥1 conjunct introduced by CONJUNCTION /
        VERB_ADJACENCY. A non-ACTION chunk, or a chunk introduced by
        any other delimiter (PERIOD / SEMICOLON / HYPHEN), breaks the
        run — those mark a sentence-level boundary, not a
        within-action coordination."""
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton

        def _is_singleton_action(idx):
            return (self.role_for_sub(idx) == "ACTION"
                    and idx < len(self.chunk_meta)
                    and isinstance(intermediate_representations[idx], Singleton))

        for start_pos, head_idx in enumerate(sub_indices):
            if not _is_singleton_action(head_idx):
                continue
            head_delim = self.chunk_meta[head_idx].delim_before
            if head_delim not in self._ACTION_CONJUNCT_OPEN_DELIMS:
                continue
            group = [head_idx]
            for follow_idx in sub_indices[start_pos + 1:]:
                if not _is_singleton_action(follow_idx):
                    break
                if self.chunk_meta[follow_idx].delim_before not in self._ACTION_CONJUNCT_FOLLOW_DELIMS:
                    break
                group.append(follow_idx)
            if len(group) >= 2:
                return group
        return []

    def _build_and_group(self, action_kernels):
        from LaSSI.structures.internal_graph.EntityRelationship import (
            Grouping,
            SetOfSingletons,
        )
        return SetOfSingletons(
            id=self._synthetic_node_id(),
            type=Grouping.AND,
            entities=tuple(action_kernels),
            min=-1,
            max=-1,
            confidence=1.0,
            root=True,
        )

    def _resolve_topic_header_head(self, sub_indices, intermediate_representations):
        """Locate the topic-substantive head noun (Singleton) inside the
        intermediate representation of a TOPIC_HEADER chunk. Returns
        ``(idx, singleton)`` for the first such chunk, or
        ``(None, None)``.

        The HEADER chunk is parsed by CoreNLP as a bare NP — depending
        on the surface form, the head noun may be the kernel source,
        or nested under the source's properties (when prepositional
        modifiers like ``on Fern Drive`` show up as an inner
        Singleton). We do a shallow walk that covers both shapes."""
        from LaSSI.structures.internal_graph.EntityRelationship import (
            SetOfSingletons,
            Singleton,
        )
        try:
            service_state = (
                self.honk.getServiceStateNouns() if self.honk else set()
            ) or set()
        except Exception:
            service_state = set()
        if not service_state:
            return None, None

        def _match(node):
            if node is None or not isinstance(node, Singleton):
                return None
            name = (node.named_entity or "").strip().lower()
            if not name:
                return None
            if name in service_state:
                return node
            # Stanza/Parmenides may lemmatise the surface plural before
            # building the Singleton; check the lemma property too.
            props = dict(getattr(node, "properties", ()) or ())
            lemma = (props.get("lemma") or "").strip().lower()
            if lemma and lemma in service_state:
                return node
            return None

        def _walk(node, depth=0):
            if node is None or depth > 4:
                return None
            if isinstance(node, Singleton):
                hit = _match(node)
                if hit is not None:
                    return hit
                kernel = getattr(node, "kernel", None)
                if kernel is not None:
                    for child in (kernel.source, kernel.target, kernel.edgeLabel):
                        found = _walk(child, depth + 1)
                        if found is not None:
                            return found
                for _, v in dict(getattr(node, "properties", ()) or ()).items():
                    if isinstance(v, (list, tuple)):
                        for item in v:
                            found = _walk(item, depth + 1)
                            if found is not None:
                                return found
                    else:
                        found = _walk(v, depth + 1)
                        if found is not None:
                            return found
            elif isinstance(node, SetOfSingletons):
                for entity in node.entities or ():
                    found = _walk(entity, depth + 1)
                    if found is not None:
                        return found
            return None

        for idx in sub_indices:
            if self.role_for_sub(idx) != "TOPIC_HEADER":
                continue
            kernel = intermediate_representations[idx]
            head = _walk(kernel)
            if head is not None:
                return idx, head
        return None, None

    def _build_synthetic_be_outer(self, topic_head, and_group_or_target):
        """Build a synthetic outer Singleton kernel of shape
        ``be(topic_head, and_group_or_target)``. When ``topic_head`` is
        ``None``, the source defaults to a fresh existential. Matches
        the shape produced naturally for elided copulae
        (``be(carriageway, ?8)``), so downstream kernel printing / FOL
        rewriting consumes it without special handling."""
        from LaSSI.structures.internal_graph.EntityRelationship import (
            Relationship,
            Singleton,
        )
        if topic_head is None:
            source = Singleton(
                id=self._synthetic_node_id(),
                named_entity=f"?{self._synthetic_node_id()}",
                properties=frozenset(),
                min=-1, max=-1, type="existential", confidence=1.0,
                kernel=None,
            )
        else:
            source = topic_head
        be_edge = Singleton(
            id=self._synthetic_node_id(),
            named_entity="be",
            properties=frozenset(),
            min=-1, max=-1, type="verb", confidence=1.0,
            kernel=None,
        )
        return Singleton(
            id=self._synthetic_node_id(),
            named_entity="",
            properties=frozenset(),
            min=-1, max=-1, type="SENTENCE", confidence=1.0,
            kernel=Relationship(
                source=source,
                target=and_group_or_target,
                edgeLabel=be_edge,
                isNegated=False,
            ),
        )

    def _is_adverb_fragment_kernel(self, sub_kernel):
        """``be(RB[also|too|additionally], existential)[…]`` pattern
        check. Returns the source adverb Singleton when the kernel
        matches, else ``None``."""
        from LaSSI.structures.internal_graph.EntityRelationship import (
            Relationship,
            Singleton,
        )
        if not isinstance(sub_kernel, Singleton) or sub_kernel.kernel is None:
            return None
        kernel = sub_kernel.kernel
        if not isinstance(kernel, Relationship):
            return None
        edge = kernel.edgeLabel
        if not (isinstance(edge, Singleton)
                and (edge.named_entity or "").lower() == "be"):
            return None
        source = kernel.source
        if not isinstance(source, Singleton):
            return None
        if (source.type or "").upper() not in {"RB", "JJ", "ADV"}:
            return None
        if (source.named_entity or "").strip().lower() not in self._ADVERB_FRAGMENT_TOKENS:
            return None
        target = kernel.target
        if not (isinstance(target, Singleton) and (target.type or "").lower() == "existential"):
            return None
        return source

    def _extract_chunk_np(self, chunk_text: str, adverb_token: str):
        """Return ``(head, modifiers)`` for an adverb-fragment chunk by
        scanning ``chunk_text`` left-to-right. ``head`` is the last
        content noun before the first preposition (e.g. ``permit`` in
        ``also a connection permit for works …``); ``modifiers`` are the
        preceding content nouns — the head's compound modifiers
        (``connection``) — which the broken ``be(also, ?)`` kernel drops.
        Returns ``(None, [])`` if no plausible head is found."""
        if not chunk_text:
            return None, []
        try:
            preps = {
                p.lower() for p in
                (self.honk.getPrepositions() if self.honk else set()) or set()
            }
        except Exception:
            preps = set()
        if not preps:
            preps = {
                "for", "on", "at", "in", "by", "with", "from", "to",
                "of", "about", "near", "into", "onto", "between",
                "across", "during", "before", "after", "until",
            }
        tokens = [m.group(0) for m in _re.finditer(r"[A-Za-z][A-Za-z'\-]*", chunk_text)]
        if not tokens:
            return None, []
        adverb_lc = (adverb_token or "").strip().lower()
        content = []
        for tok in tokens:
            tok_lc = tok.lower()
            if tok_lc == adverb_lc:
                continue
            if tok_lc in preps:
                if content:
                    break
                continue
            if tok_lc in self._ADVERB_FRAGMENT_SKIP:
                continue
            content.append(tok)
        if not content:
            return None, []
        return content[-1], content[:-1]

    def _repair_adverb_fragment(self, sub_idx, sub_kernel):
        """Reconstruct a bare-NP Singleton from a ``be(also, ?)[…]``
        kernel by hoisting the kernel's construct properties onto the
        recovered head noun, restoring the head's compound modifiers
        (``connection`` → ``extra``), and recording the adverb itself
        under ``adv`` (it is an adverbial discourse marker, not part of
        the NP). Returns the rebuilt Singleton, or ``None`` when the
        chunk does not match the adverb-fragment shape."""
        from LaSSI.ner.node_functions import create_props_for_singleton
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton
        adverb = self._is_adverb_fragment_kernel(sub_kernel)
        if adverb is None:
            return None
        chunk_text = (
            self.chunk_meta[sub_idx].text
            if sub_idx < len(self.chunk_meta) and self.chunk_meta[sub_idx] is not None
            else ""
        )
        head_name, modifier_names = self._extract_chunk_np(
            chunk_text, getattr(adverb, "named_entity", ""),
        )
        if not head_name:
            return None
        head_props = {}
        for k, v in dict(sub_kernel.properties).items():
            head_props[k] = list(v) if isinstance(v, (list, tuple)) else v
        # Restore the head's compound modifiers (dropped by the broken
        # be(also, ?) kernel) as `extra` Singletons.
        if modifier_names:
            extra_existing = head_props.get("extra")
            extras = []
            if extra_existing is not None:
                extras = (list(extra_existing)
                          if isinstance(extra_existing, (list, tuple))
                          else [extra_existing])
            for name in modifier_names:
                extras.append(Singleton(
                    id=self._synthetic_node_id(),
                    named_entity=name,
                    properties=frozenset(),
                    min=-1, max=-1, type="noun", confidence=1.0, kernel=None,
                ))
            head_props["extra"] = extras
        # The adverb ("also") is a discourse marker on the clause, not an
        # NP constituent — record it under `adv`, never `extra`.
        head_props["adv"] = adverb
        return Singleton(
            id=self._synthetic_node_id(),
            named_entity=head_name,
            properties=create_props_for_singleton(head_props),
            min=getattr(adverb, "min", -1),
            max=getattr(adverb, "max", -1),
            type="noun",
            confidence=1.0,
            kernel=None,
        )

    def merge_intermediate_representations(self, intermediate_representations):
        """Per-YAML-row merge. See module docstring for the role-based
        merging logic. Returns one merged representation per row."""
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton

        construct_keys = self._ontology_construct_keys()
        merged = []
        for sub_indices in self.row_to_sub_indices:
            if not sub_indices:
                continue
            if len(sub_indices) == 1:
                single = intermediate_representations[sub_indices[0]]
                # Single-clause rows skip the chunk-merge projections, but a
                # DisruptionNoun-headed TOGETHERNESS still needs reclassifying to
                # DISRUPTION so it matches the multi-chunk notices' shape (keeps
                # the identity-refining key consistent across paraphrases).
                if isinstance(single, Singleton):
                    new_props = defaultdict(list)
                    for k, v in dict(single.properties).items():
                        new_props[k] = list(v) if isinstance(v, (list, tuple)) else v
                    self._reclassify_togetherness_disruption(new_props)
                    single = single.update_node_props(new_props)
                merged.append(single)
                continue

            # M2 — detect a contiguous ACTION conjunct group. When ≥2
            # such kernels exist, wrap them in SetOfSingletons(AND) and
            # use that group as the row primary (possibly nested under
            # a synthetic be() outer kernel in M4).
            action_indices = self._detect_action_conjuncts(
                sub_indices, intermediate_representations,
            )

            # M5 — when a HYPHEN-introduced ``also …`` fragment follows
            # the last ACTION conjunct, repair its broken
            # ``be(also, ?)`` kernel into a bare NP Singleton and fold
            # it in as an extra conjunct of the AND group.
            adverb_fold = {}
            if action_indices:
                last_action_idx = action_indices[-1]
                for idx in sub_indices:
                    if idx <= last_action_idx:
                        continue
                    if idx >= len(self.chunk_meta):
                        continue
                    if self.chunk_meta[idx].delim_before != "HYPHEN":
                        break
                    repaired = self._repair_adverb_fragment(
                        idx, intermediate_representations[idx],
                    )
                    if repaired is None:
                        break
                    adverb_fold[idx] = repaired

            action_index_set = set(action_indices)
            and_group = None
            if action_indices:
                action_kernels = [
                    intermediate_representations[i] for i in action_indices
                ]
                for idx in sorted(adverb_fold):
                    action_kernels.append(adverb_fold[idx])
                and_group = self._build_and_group(action_kernels)

            # M1/M4 — locate the topic-substantive HEADER head noun so
            # the outer kernel can anchor on it.
            topic_header_idx, topic_head = (None, None)
            if and_group is not None:
                topic_header_idx, topic_head = self._resolve_topic_header_head(
                    sub_indices, intermediate_representations,
                )

            # Build the row primary.
            primary_consumed_indices = set()
            if and_group is not None:
                primary = self._build_synthetic_be_outer(topic_head, and_group)
                primary_consumed_indices |= action_index_set
                primary_consumed_indices |= set(adverb_fold)
                if topic_header_idx is not None:
                    primary_consumed_indices.add(topic_header_idx)
            else:
                primary_idx = self._select_primary_idx(
                    sub_indices, intermediate_representations,
                )
                primary = intermediate_representations[primary_idx]
                if not isinstance(primary, Singleton):
                    merged.append(primary)
                    continue
                primary_consumed_indices.add(primary_idx)

            # Seed new_props from the primary.
            new_props = defaultdict(list)
            for k, v in dict(primary.properties).items():
                if isinstance(v, (list, tuple)):
                    new_props[k] = list(v)
                else:
                    new_props[k] = v

            # When M4 is active, promote the topic-head Singleton's own
            # construct properties (e.g. SPACE:Fern Drive[on]) onto the
            # outer kernel so the row's location is not lost.
            if and_group is not None and topic_header_idx is not None:
                header_kernel = intermediate_representations[topic_header_idx]
                if isinstance(header_kernel, Singleton):
                    for k, v in dict(header_kernel.properties).items():
                        if k not in construct_keys:
                            continue
                        items = list(v) if isinstance(v, (list, tuple)) else [v]
                        existing = new_props.get(k)
                        if existing is None:
                            new_props[k] = items
                        elif isinstance(existing, list):
                            new_props[k] = existing + items
                        else:
                            new_props[k] = [existing] + items

            for sub_idx in sub_indices:
                if sub_idx in primary_consumed_indices:
                    continue
                sub_kernel = intermediate_representations[sub_idx]
                if sub_kernel is None:
                    continue
                sub_role = self.role_for_sub(sub_idx)
                promoted_any = False
                if isinstance(sub_kernel, Singleton):
                    for k, v in dict(sub_kernel.properties).items():
                        if k not in construct_keys:
                            continue
                        # M3 — when an AND group is the primary, an
                        # ACTION sub-kernel that is *not* part of the
                        # group is still treated as a secondary, but
                        # the conjuncts themselves are already absorbed
                        # into the group and must not flatten their
                        # SPECIFICATION / TOGETHERNESS onto siblings.
                        if and_group is not None and sub_role == "ACTION":
                            continue
                        items = list(v) if isinstance(v, (list, tuple)) else [v]
                        existing = new_props.get(k)
                        if existing is None:
                            new_props[k] = items
                        elif isinstance(existing, list):
                            new_props[k] = existing + items
                        else:
                            new_props[k] = [existing] + items
                        promoted_any = True
                sub_is_subsidiary = sub_role in self._CHUNK_SUBSIDIARY_ROLES
                sub_is_label_only = sub_role in self._CHUNK_LABEL_ONLY_ROLES
                # Pure label chunks contribute nothing if there are no
                # construct keys to lift onto the primary — they exist
                # only to mark the colon-introduced body chunk that
                # follows.
                if sub_is_label_only and not promoted_any:
                    continue
                if (not sub_is_subsidiary) or (not promoted_any):
                    new_props["SENTENCE"].append(sub_kernel)
            self._project_lifecycle_from_sentence(new_props)
            self._project_disruption_from_sentence(new_props)
            self._reclassify_togetherness_disruption(new_props)
            self._project_action_object_nominal(new_props, primary)
            merged.append(primary.update_node_props(new_props))
        return merged

    # ------------------------------------------------------------------
    # DISRUPTION projection.
    # ------------------------------------------------------------------

    def _reclassify_togetherness_disruption(self, new_props):
        """A ``TOGETHERNESS`` ("with X") whose head is a DisruptionNoun is really
        a ``DISRUPTION`` facet — e.g. "with some carriageway incursion" lands as
        ``TOGETHERNESS:carriageway[extra:incursion]`` from a single-clause parse,
        but the multi-chunk notice surfaces the same fact as ``DISRUPTION``.
        Reclassify so paraphrases carry the SAME identity-refining key (otherwise
        the directional similarity is capped — TOGETHERNESS vs DISRUPTION read as
        distinguishing content). Genuine accompaniment TOGETHERNESS is untouched."""
        items = new_props.get("TOGETHERNESS")
        if not items:
            return
        disruption_nouns, _ = self._disruption_lookup()
        if not disruption_nouns:
            return
        items = list(items) if isinstance(items, (list, tuple)) else [items]
        kept, moved = [], []
        for it in items:
            if self._node_disruption_head(it, disruption_nouns) is not None:
                moved.append(it)
            else:
                kept.append(it)
        if not moved:
            return
        if kept:
            new_props["TOGETHERNESS"] = kept
        else:
            new_props.pop("TOGETHERNESS", None)
        new_props.setdefault("DISRUPTION", []).extend(moved)

    def _disruption_lookup(self):
        """``(disruption_nouns, modal_adjectives)`` lowercased lookup sets,
        or ``(set(), set())`` when HOnK is unavailable."""
        try:
            dis = {d.lower() for d in (self.honk.getDisruptionNouns() or set())} if self.honk else set()
        except Exception:
            dis = set()
        try:
            modals = {m.lower() for m in (self.honk.getModalAdjectives() or set())} if self.honk else set()
        except Exception:
            modals = set()
        return dis, modals

    @staticmethod
    def _name_candidates(node):
        """Lowercased name candidates for a node: surface form, its
        ``lemma`` property, and a naive de-pluralised form. Enough to match
        ``delays``/``delay`` and ``incursion`` against the singular-form
        DisruptionNoun list without pulling in the full matcher."""
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton
        if not isinstance(node, Singleton):
            return set()
        out = set()
        name = (node.named_entity or "").strip().lower()
        if name:
            out.add(name)
            if name.endswith("s") and not name.endswith("ss"):
                out.add(name[:-1])
        lemma = dict(node.properties).get("lemma")
        if isinstance(lemma, str) and lemma.strip():
            out.add(lemma.strip().lower())
        return out

    def _node_disruption_head(self, node, disruption_nouns):
        """Return the Singleton that carries the disruption — ``node``
        itself when it is a DisruptionNoun, else its ``extra`` child that
        is one (``carriageway[extra:incursion]`` → the carriageway head is
        kept, since *incursion* identifies it as a disruption)."""
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton
        if not isinstance(node, Singleton):
            return None
        if self._name_candidates(node) & disruption_nouns:
            return node
        extras = dict(node.properties).get("extra")
        extras = extras if isinstance(extras, (list, tuple)) else ([extras] if extras is not None else [])
        for extra in extras:
            if isinstance(extra, Singleton) and (self._name_candidates(extra) & disruption_nouns):
                return node
        return None

    def _disruption_node(self, item, disruption_nouns, modals):
        """Project a SENTENCE clause into a DISRUPTION Singleton, or
        ``None`` when it is not a disruption assertion. Two shapes:

        * ``delay(?, unlikely)`` — the edge verb is a DisruptionNoun; emit
          ``delay[(MODALITY:unlikely)]`` (modality from a modal-adjective
          target).
        * ``be(carriageway[extra:incursion], ?)`` — the source NP is/holds
          a DisruptionNoun; emit that NP unchanged."""
        from LaSSI.ner.node_functions import create_props_for_singleton
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton
        if not isinstance(item, Singleton) or item.kernel is None:
            return None
        edge = item.kernel.edgeLabel
        source = item.kernel.source
        target = item.kernel.target

        # Shape A: disruption noun is the edge predicate (delay(?, modal)).
        if isinstance(edge, Singleton) and (self._name_candidates(edge) & disruption_nouns):
            props = {}
            if isinstance(target, Singleton) and (self._name_candidates(target) & modals):
                props["MODALITY"] = [target]
            return Singleton(
                id=self._synthetic_node_id(),
                named_entity=edge.named_entity,
                properties=create_props_for_singleton(props),
                min=getattr(edge, "min", -1), max=getattr(edge, "max", -1),
                type="noun", confidence=1.0, kernel=None,
            )

        # Shape B: disruption noun heads the source NP (be(carriageway[incursion], ?)).
        head = self._node_disruption_head(source, disruption_nouns)
        if head is not None:
            return head
        return None

    def _project_disruption_from_sentence(self, new_props):
        """Lift predicted-disruption facts (``Delays unlikely``, the
        ``Traffic management: Some carriageway incursion`` value) out of
        merged SENTENCE entries onto the row primary as ``DISRUPTION``.

        Runs after :meth:`_project_lifecycle_from_sentence`: these clauses
        survive the merge as SENTENCE sub-kernels, and only become
        meaningful disruption facets once attached to the row kernel."""
        sentence_items = new_props.get("SENTENCE")
        if not sentence_items:
            return
        disruption_nouns, modals = self._disruption_lookup()
        if not disruption_nouns:
            return
        kept = []
        for item in sentence_items:
            projected = self._disruption_node(item, disruption_nouns, modals)
            if projected is not None:
                new_props.setdefault("DISRUPTION", []).append(projected)
            else:
                kept.append(item)
        if kept:
            new_props["SENTENCE"] = kept
        else:
            new_props.pop("SENTENCE", None)

    def _project_lifecycle_from_sentence(self, new_props):
        """Lift lifecycle facts out of merged SENTENCE entries.

        Sub-sentences like *Investigation complete* and *no suspect
        identified* survive :meth:`merge_intermediate_representations`
        as top-level kernels attached under SENTENCE on the primary
        clause. At that point they are not visible to the
        structural-rewrite pipeline (which has already run per
        sub-sentence before the merge). Re-apply the lifecycle
        projections here so a copula
        ``be(StatusNoun, ?[cop:adj])`` becomes
        ``TIME_STATUS:StatusNoun[type:adj]`` and a
        ``verb(?, NOT(lifecycle))`` clause becomes
        ``SPECIFICATION:NOT(lifecycle)``."""
        sentence_items = new_props.get("SENTENCE")
        if not sentence_items:
            return
        try:
            from LaSSI.ner.KernelOntologyMatchers import KernelOntologyMatchers
            from LaSSI.ner.structural_rewrites import RewriteContext
            from LaSSI.ner.structural_rewrites.lifecycle_property_promotion import (
                LifecyclePropertyPromotionRule,
            )
        except Exception:
            return

        rule = LifecyclePropertyPromotionRule()
        ctx = RewriteContext(
            node_functions=None,
            services=self.services,
            matchers=KernelOntologyMatchers(G=None),
        )

        kept = []
        for item in sentence_items:
            promoted = False
            try:
                status_node = rule._sentence_status_projection(item, ctx)
            except Exception:
                status_node = None
            if status_node is not None:
                new_props.setdefault("TIME_STATUS", []).append(status_node)
                promoted = True
            else:
                try:
                    negated = rule._negated_lifecycle_projection(item, ctx)
                except Exception:
                    negated = None
                if negated is not None:
                    new_props.setdefault("SPECIFICATION", []).append(negated)
                    promoted = True
            if not promoted:
                kept.append(item)

        if kept:
            new_props["SENTENCE"] = kept
        else:
            new_props.pop("SENTENCE", None)

    def _object_subtypes(self, supertype: str) -> set:
        """Lowercased set of ``supertype`` subtypes from the ontology
        (e.g. ``pipe`` -> SI, LP PE, …). The isA adjacency is keyed by
        supertype, so subtypes live under ``_isA_supers[supertype]``."""
        try:
            supers = getattr(self.honk, "_isA_supers", None) if self.honk else None
            if not supers:
                return set()
            return {str(x).lower() for x in (supers.get(supertype) or set())}
        except Exception:
            return set()

    @staticmethod
    def _action_object_nominal_projections() -> list:
        """Load the data-driven action->nominal projection table
        (``raw_data/action_object_nominal_projection.json``). Cached on the
        class so the file is read once per process."""
        cached = getattr(RowChunkPipeline, "_aonp_cache", None)
        if cached is not None:
            return cached
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "raw_data", "action_object_nominal_projection.json")
        try:
            with open(os.path.abspath(path)) as fh:
                data = json.load(fh)
            projections = list(data.get("projections", []))
        except Exception:
            projections = []
        RowChunkPipeline._aonp_cache = projections
        return projections

    def _project_action_object_nominal(self, new_props, primary):
        """Surface a canonical nominal ``SPECIFICATION`` when the row's action
        verbs act on an object of a configured supertype.

        Notices describe work imperatively — e.g. ``AND(Abandon(SI),
        replace(LP PE))`` where ``SI``/``LP PE`` are pipes (ontology ``isA
        pipe``) — while paraphrases use the nominal (``pipe replacement``).
        Emitting the nominal here aligns the action-based description with the
        nominal ones for the eFOL comparison. The (supertype -> nominal) table
        is data, in ``raw_data/action_object_nominal_projection.json`` — not
        hardcoded here — because the noun<->verb relation cannot be bridged at
        comparison time, so the alignment must be projected upstream."""
        from LaSSI.structures.internal_graph.EntityRelationship import (
            SetOfSingletons,
            Singleton,
        )
        projections = self._action_object_nominal_projections()
        if not projections or not isinstance(primary, Singleton) or primary.kernel is None:
            return
        action_verbs = self._verb_classes_for_split()
        target = primary.kernel.target
        actions = (target.entities if isinstance(target, SetOfSingletons)
                   else (target,) if target is not None else ())

        def _acts_on(object_terms):
            for act in actions:
                if not (isinstance(act, Singleton) and act.kernel is not None):
                    continue
                edge = act.kernel.edgeLabel
                if not (isinstance(edge, Singleton)
                        and (edge.named_entity or "").strip().lower() in action_verbs):
                    continue
                obj = act.kernel.target
                obj_candidates = (obj.entities if isinstance(obj, SetOfSingletons)
                                  else (obj,) if obj is not None else ())
                if any(isinstance(o, Singleton)
                       and (o.named_entity or "").strip().lower() in object_terms
                       for o in obj_candidates):
                    return True
            return False

        for projection in projections:
            supertype = str(projection.get("object_supertype", "")).strip().lower()
            nominal = str(projection.get("projected_nominal", "")).strip()
            if not supertype or not nominal:
                continue
            object_terms = self._object_subtypes(supertype)
            if not object_terms or not _acts_on(object_terms):
                continue
            existing = new_props.get("SPECIFICATION")
            existing = (list(existing) if isinstance(existing, (list, tuple))
                        else ([existing] if existing is not None else []))
            if any(isinstance(x, Singleton)
                   and (x.named_entity or "").strip().lower() == nominal.lower()
                   for x in existing):
                continue
            existing.append(Singleton(
                id=self._synthetic_node_id(),
                named_entity=nominal,
                properties=frozenset(),
                min=-1, max=-1, type="noun", confidence=1.0, kernel=None,
            ))
            new_props["SPECIFICATION"] = existing

    # ------------------------------------------------------------------
    # Phase 3 — MeuDB merge + output rewriting.
    # ------------------------------------------------------------------

    def merge_meu_dbs(self, meu_dbs):
        """Concatenate per-sub-sentence MeuDB entries into one MeuDB
        per YAML row, using the original (un-chunked) row text as the
        display ``first_sentence`` so the merged
        :file:`string_rep.txt` keeps the user's punctuation."""
        from LaSSI.structures.meuDB.meuDB import MeuDB
        merged = []
        originals = self.row_original_text or []
        for row_idx, sub_indices in enumerate(self.row_to_sub_indices):
            if not sub_indices:
                continue
            display_text = (originals[row_idx]
                            if row_idx < len(originals)
                            else meu_dbs[sub_indices[0]].first_sentence)
            if len(sub_indices) == 1:
                row = meu_dbs[sub_indices[0]]
                if display_text != row.first_sentence:
                    row = MeuDB(first_sentence=display_text,
                                multi_entity_unit=row.multi_entity_unit)
                merged.append(row)
                continue
            combined_meu = []
            for i in sub_indices:
                combined_meu.extend(meu_dbs[i].multi_entity_unit)
            merged.append(MeuDB(
                first_sentence=display_text,
                multi_entity_unit=combined_meu,
            ))
        return merged

    def persist_merged_internals(self, intermediate_representations, internals_path):
        """Overwrite the internals cache with the merged-per-row
        representation. Does *not* overwrite ``meuDBs.json`` — graph
        rewriting caches are per-sub-sentence and future reruns need a
        matching per-sub-sentence MeuDB cache while reconstructing
        internals."""
        try:
            with open(internals_path, "w") as f:
                f.write(json_dumps(intermediate_representations))
        except Exception as exc:
            self.logger(f"Could not persist merged internals.json: {exc}")

    def collapse_per_row_benchmark(self, sentences_benchmark):
        """Aggregate per-sub-sentence rows in the sentences benchmark
        down to one row per YAML input row (sum the timings, max the
        sentence length). Otherwise the benchmark CSV reports more
        sentences than the user wrote."""
        if sentences_benchmark is None:
            return
        data = sentences_benchmark.data
        if not data:
            return
        sub_to_row = {}
        for row_idx, sub_indices in enumerate(self.row_to_sub_indices):
            for sub_idx in sub_indices:
                sub_to_row[sub_idx] = row_idx
        merged_rows = {}
        for entry in data:
            sub_idx = entry.get("id")
            row_idx = sub_to_row.get(sub_idx, sub_idx)
            target = merged_rows.setdefault(row_idx, {"id": row_idx})
            for key, value in entry.items():
                if key == "id":
                    continue
                if not isinstance(value, (int, float)):
                    target[key] = value
                    continue
                if key == "Sentence length":
                    target[key] = max(target.get(key, 0), value)
                else:
                    target[key] = target.get(key, 0.0) + value
        data.clear()
        for row_idx in sorted(merged_rows):
            data.append(merged_rows[row_idx])

    def rewrite_string_rep(
        self,
        intermediate_representations,
        meu_dbs,
        string_rep_dir,
        is_logical: bool,
    ):
        """Rewrite :file:`string_rep.txt` against the merged
        per-row kernels, pairing each kernel with its merged MeuDB's
        ``first_sentence`` for display."""
        if string_rep_dir is None:
            return
        with open(string_rep_dir, "w") as f:
            for idx, intermediate in enumerate(intermediate_representations):
                first_sentence = (meu_dbs[idx].first_sentence
                                  if idx < len(meu_dbs) else "")
                try:
                    if intermediate is None:
                        f.write(f"{first_sentence} ⇒ ERROR\n")
                    elif is_logical:
                        f.write(f"{first_sentence} ⇒ {intermediate.to_string()}\n")
                    else:
                        f.write(f"{first_sentence} ⇒ {intermediate}\n")
                except Exception:
                    f.write(f"{first_sentence} ⇒ ERROR\n")
