"""Lossless, source-preserving evidence delivery for independent memory review.

IDs identify exact scalar observations (or exact string chunks), never inferred
claims. Identical observations within one evidence root share the first real
source pointer. The document retains every occurrence and its original order.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any


_CITABLE_ROOTS = {"task", "public_trace", "official_feedback"}
_MARKERS = {"$e", "$join", "$s", "$object"}


def _json(value: Any) -> str:
    # Default separators make the bounds conservative for compact transports.
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


class EvidencePack:
    """A frozen snapshot with deduplicated wire values and bounded reads.

    ``full_payload`` delivers all citable IDs. Overview previews and catalog
    metadata are not deliveries: only their explicit ``delivered_ids`` count.
    ``read`` is atomic and raises ValueError when a requested page cannot fit.
    """

    def __init__(self, evidence: dict, chunk_chars: int = 4000):
        _positive(chunk_chars, "chunk_chars")
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be an object")
        _json(evidence)  # Reject non-JSON values/nonfinite numbers up front.
        self.original = deepcopy(evidence)
        self.chunk_chars = chunk_chars
        self.entries: dict[str, dict] = {}
        self._values: dict[str, Any] = {}
        self._value_ids: dict[tuple, str] = {}
        self._entry_ids: dict[tuple, str] = {}
        self._entry_values: dict[str, str] = {}
        self._subtree_ids: dict[tuple, str] = {}
        self._subtrees: dict[str, Any] = {}
        self._structure_nodes: list[dict] | None = None
        self._container_counts: Counter = Counter()
        self._count(self.original, "")
        root = self._encode(self.original, "", "")
        self.document = {"root": root, "subtrees": self._subtrees}

    @staticmethod
    def _key(value: Any) -> tuple:
        # bool, int, float remain distinct even when Python compares them equal.
        return type(value).__name__, _json(value)

    def _count(self, value: Any, root: str) -> None:
        if isinstance(value, (dict, list)):
            encoded = _json(value)
            if len(encoded) >= 128:
                self._container_counts[(root, encoded)] += 1
            children = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in children:
                self._count(child, root or str(key))

    def _entry(self, value: Any, pointer: str, root: str,
               start: int | None = None, end: int | None = None) -> str:
        key = self._key(value)
        entry_key = (root, *key)
        if entry_key in self._entry_ids:
            return self._entry_ids[entry_key]
        eid = f"E{len(self.entries) + 1:06d}"
        vid = self._value_ids.get(key)
        if vid is None:
            vid = f"V{len(self._values) + 1:06d}"
            self._value_ids[key] = vid
            self._values[vid] = value
        entry = {"pointer": pointer, "root": root, "value": value}
        if start is not None:
            entry.update(start=start, end=end)
        self.entries[eid] = entry
        self._entry_ids[entry_key] = eid
        self._entry_values[eid] = vid
        return eid

    def _encode(self, value: Any, pointer: str, root: str) -> Any:
        if isinstance(value, str) and len(value) > self.chunk_chars:
            return {"$join": [self._entry(value[start:start + self.chunk_chars], pointer, root,
                                         start, min(start + self.chunk_chars, len(value)))
                              for start in range(0, len(value), self.chunk_chars)]}
        if not isinstance(value, (dict, list)):
            return {"$e": self._entry(value, pointer, root)}
        key = (root, _json(value))
        shared = self._container_counts[key] > 1 and len(key[1]) >= 128
        if shared and key in self._subtree_ids:
            return {"$s": self._subtree_ids[key]}
        if shared:
            sid = f"S{len(self._subtree_ids) + 1:06d}"
            self._subtree_ids[key] = sid
        if isinstance(value, list):
            node = [self._encode(child, f"{pointer}/{index}", root)
                    for index, child in enumerate(value)]
        else:
            node = {name: self._encode(child, f"{pointer}/{_token(name)}", root or name)
                    for name, child in value.items()}
            if _MARKERS.intersection(node):
                node = {"$object": node}
        if shared:
            self._subtrees[sid] = node
            return {"$s": sid}
        return node

    def restore(self) -> dict:
        """Decode the public document and value references, not the snapshot."""
        def decode(node):
            if isinstance(node, list):
                return [decode(child) for child in node]
            if not isinstance(node, dict):
                raise ValueError("Invalid evidence node")
            if "$e" in node:
                return self.entries[node["$e"]]["value"]
            if "$join" in node:
                return "".join(self.entries[eid]["value"] for eid in node["$join"])
            if "$s" in node:
                return decode(self.document["subtrees"][node["$s"]])
            if "$object" in node:
                node = node["$object"]
            return {key: decode(child) for key, child in node.items()}
        return decode(self.document["root"])

    def _metadata(self, eid: str) -> dict:
        return {"id": eid, **{key: value for key, value in self.entries[eid].items()
                              if key != "value"}, "value_id": self._entry_values[eid]}

    def _citable(self, eid: str) -> bool:
        entry = self.entries[eid]
        return entry["root"] in _CITABLE_ROOTS and entry["value"] not in (None, "")

    def full_payload(self) -> dict:
        return deepcopy({
            "schema_version": 2, "delivery": "lossless_deduplicated_evidence",
            "document": self.document,
            "entries": {eid: {key: value for key, value in self._metadata(eid).items()
                              if key != "id"} for eid in self.entries},
            "values": self._values,
            "delivered_ids": [eid for eid in self.entries if self._citable(eid)],
            "complete": True,
        })

    def read(self, ids: list[str], *, max_chars: int = 16000,
             max_entries: int = 32) -> list[dict]:
        _positive(max_chars, "max_chars")
        _positive(max_entries, "max_entries")
        if not isinstance(ids, list) or len(ids) > max_entries:
            raise ValueError("Evidence read exceeds the entry limit")
        if any(not isinstance(eid, str) or eid not in self.entries for eid in ids):
            raise ValueError("Unknown evidence ID")
        result = [{"id": eid, **self.entries[eid]} for eid in dict.fromkeys(ids)]
        if len(_json(result)) > max_chars:
            raise ValueError("Exact evidence read exceeds max_chars; request fewer IDs")
        return deepcopy(result)

    def citation(self, eid: str) -> dict:
        if not isinstance(eid, str) or eid not in self.entries or not self._citable(eid):
            raise ValueError("Evidence ID is unknown or not a citable observation")
        entry = self.entries[eid]
        value = entry["value"]
        quote = value if isinstance(value, str) else _json(value)
        return {"pointer": entry["pointer"], "quote": quote}

    def catalog(self, offset: int = 0, limit: int = 64, *, max_chars: int = 16000,
                root: str | None = None, pointer_prefix: str | None = None) -> dict:
        """Page metadata only; offsets refer to the optional filtered catalog."""
        _positive(max_chars, "max_chars")
        _positive(limit, "limit")
        if type(offset) is not int or offset < 0:
            raise ValueError("Catalog offset must be a nonnegative integer")
        ids = [eid for eid, entry in self.entries.items()
               if (root is None or entry["root"] == root)
               and (pointer_prefix is None or entry["pointer"].startswith(pointer_prefix))]
        if offset > len(ids):
            raise ValueError("Catalog offset exceeds total entries")
        result = {"entries": [], "offset": offset, "total": len(ids),
                  "next_offset": offset if offset < len(ids) else None,
                  "omitted": len(ids) - offset, "delivered_ids": []}
        if len(_json(result)) > max_chars:
            raise ValueError("Catalog metadata cannot fit max_chars")
        for eid in ids[offset:offset + limit]:
            candidate = deepcopy(result)
            candidate["entries"].append(self._metadata(eid))
            end = offset + len(candidate["entries"])
            candidate.update(next_offset=end if end < len(ids) else None, omitted=len(ids) - end)
            if len(_json(candidate)) > max_chars:
                break
            result = candidate
        if offset < len(ids) and not result["entries"]:
            raise ValueError("One catalog entry cannot fit max_chars")
        return result

    def _ids_for(self, value: Any, root: str) -> list[str]:
        if isinstance(value, dict):
            ids = [eid for child in value.values() for eid in self._ids_for(child, root)]
        elif isinstance(value, list):
            ids = [eid for child in value for eid in self._ids_for(child, root)]
        else:
            values = ([value[start:start + self.chunk_chars]
                       for start in range(0, len(value), self.chunk_chars)]
                      if isinstance(value, str) and len(value) > self.chunk_chars else [value])
            ids = [self._entry_ids[(root, *self._key(chunk))] for chunk in values]
        return list(dict.fromkeys(ids))

    def _event_index(self) -> list[dict]:
        events = []
        trace = self.original.get("public_trace", [])
        for trace_index, record in enumerate(trace if isinstance(trace, list) else []):
            if not isinstance(record, dict):
                continue
            log = record.get("event_log", [record])
            for event_index, event in enumerate(log if isinstance(log, list) else []):
                if not isinstance(event, dict):
                    continue
                pointer = f"/public_trace/{trace_index}"
                if "event_log" in record:
                    pointer += f"/event_log/{event_index}"
                item = {"pointer": pointer}
                for field in ("sequence", "hook", "execution_status"):
                    value = event.get(field)
                    if isinstance(value, (str, int, float, bool)):
                        item[field] = value[:80] if isinstance(value, str) else value
                        if isinstance(value, str) and len(value) > 80:
                            item[field + "_truncated"] = True
                # Exact IDs allow lookup even when the source occurrence was deduplicated.
                item["evidence_ids"] = list(dict.fromkeys(
                    self._entry_ids[("public_trace", *self._key(event[field]))]
                    for field in ("sequence", "hook", "execution_status")
                    if isinstance(event.get(field), (str, int, float, bool))
                    and ("public_trace", *self._key(event[field])) in self._entry_ids))
                item["evidence_groups"] = {}
                for field in ("proposed_action", "effective_action",
                              "intercepted_by", "guard_verdicts", "tool_result",
                              "synthetic_tool_result", "processor_observations"):
                    if field in event:
                        ids = self._ids_for(event[field], "public_trace")
                        item["evidence_groups"][field] = {"ids": ids[:4], "count": len(ids),
                                                          "omitted_ids": max(0, len(ids) - 4)}
                events.append(item)
        return events

    def event_catalog(self, offset: int = 0, limit: int = 32, *, max_chars: int = 16000) -> dict:
        """Page chronological metadata, including links to deduplicated values."""
        _positive(limit, "limit")
        _positive(max_chars, "max_chars")
        events = self._event_index()
        if type(offset) is not int or not 0 <= offset <= len(events):
            raise ValueError("Event offset must lie within the event catalog")
        result = {"events": [], "offset": offset, "total": len(events),
                  "next_offset": offset if offset < len(events) else None,
                  "omitted": len(events) - offset, "delivered_ids": []}
        if len(_json(result)) > max_chars:
            raise ValueError("Event catalog metadata cannot fit max_chars")
        for event in events[offset:offset + limit]:
            candidate = deepcopy(result)
            candidate["events"].append(event)
            end = offset + len(candidate["events"])
            candidate.update(next_offset=end if end < len(events) else None,
                             omitted=len(events) - end)
            if len(_json(candidate)) > max_chars:
                break
            result = candidate
        if offset < len(events) and not result["events"]:
            raise ValueError("One event catalog entry cannot fit max_chars")
        return result

    def _structure_index(self) -> list[dict]:
        """Source preorder, retaining every occurrence even when values repeat.

        Container lengths and ordered child pointers preserve keys, array order,
        and empty containers without an unbounded keys array. Long strings have
        one row per chunk occurrence, including repetitions of the same ID.
        """
        if self._structure_nodes is not None:
            return self._structure_nodes
        nodes = []

        def visit(value, pointer, root):
            if isinstance(value, (dict, list)):
                nodes.append({"pointer": pointer, "kind": "object" if isinstance(value, dict) else "list",
                              "length": len(value)})
                children = value.items() if isinstance(value, dict) else enumerate(value)
                for key, child in children:
                    visit(child, f"{pointer}/{_token(str(key))}", root or str(key))
                return
            scalar_type = {str: "string", bool: "boolean", int: "integer", float: "number",
                           type(None): "null"}[type(value)]
            chunks = ([value[start:start + self.chunk_chars]
                       for start in range(0, len(value), self.chunk_chars)]
                      if isinstance(value, str) and len(value) > self.chunk_chars else [value])
            for index, chunk in enumerate(chunks):
                node = {"pointer": pointer, "kind": "scalar", "scalar_type": scalar_type,
                        "evidence_ids": [self._entry_ids[(root, *self._key(chunk))]]}
                if isinstance(value, str):
                    node.update(chunk_index=index, chunk_count=len(chunks))
                nodes.append(node)

        visit(self.original, "", "")
        self._structure_nodes = nodes
        return nodes

    def structure_catalog(self, offset: int = 0, limit: int = 64, *, max_chars: int = 16000) -> dict:
        """Page exact source occurrences; metadata never delivers scalar values.

        Rows follow source preorder. Direct child pointers supply object keys in
        their original order. Repeated string chunks retain their chunk_index;
        concatenate exact values in that order, without deduplicating IDs again.
        """
        _positive(limit, "limit")
        _positive(max_chars, "max_chars")
        nodes = self._structure_index()
        if type(offset) is not int or not 0 <= offset <= len(nodes):
            raise ValueError("Structure offset must lie within the structure catalog")
        result = {"entries": [], "offset": offset, "total": len(nodes),
                  "next_offset": offset if offset < len(nodes) else None,
                  "omitted": len(nodes) - offset, "delivered_ids": []}
        if len(_json(result)) > max_chars:
            raise ValueError("Structure catalog metadata cannot fit max_chars")
        for node in nodes[offset:offset + limit]:
            candidate = deepcopy(result)
            candidate["entries"].append(node)
            end = offset + len(candidate["entries"])
            candidate.update(next_offset=end if end < len(nodes) else None,
                             omitted=len(nodes) - end)
            if len(_json(candidate)) > max_chars:
                break
            result = candidate
        if offset < len(nodes) and not result["entries"]:
            raise ValueError("One structure catalog entry cannot fit max_chars")
        return deepcopy(result)

    def overview_payload(self, max_chars: int = 16000) -> dict:
        """Bounded current-episode context, chronology, and first catalog page.

        Previews disclose omissions and cannot be cited until an exact read.
        Only exact short global-context values enter ``delivered_ids``.
        """
        _positive(max_chars, "max_chars")
        events = self._event_index()
        structure = self._structure_index()
        roots = {root: [eid for eid, entry in self.entries.items() if entry["root"] == root]
                 for root in ("task", "official_feedback")}
        result = {"schema_version": 2, "delivery": "bounded_evidence_overview", "complete": False,
                  "global_context": {root: [] for root in roots}, "events": [],
                  "event_count": len(events), "evidence_count": len(self.entries),
                  "event_next_offset": 0 if events else None,
                  "catalog": {"entries": [], "offset": 0, "total": len(self.entries),
                              "next_offset": 0 if self.entries else None},
                  "structure_catalog": {"entries": [], "offset": 0, "total": len(structure),
                                        "next_offset": 0 if structure else None,
                                        "omitted": len(structure), "delivered_ids": []},
                  "omissions": {"events": len(events), "task": len(roots["task"]),
                                "official_feedback": len(roots["official_feedback"]),
                                "evidence_values": len(self.entries)},
                  "delivered_ids": [],
                  "preview_contract": "Previews and catalog IDs are not delivered evidence; read IDs for exact values."}
        if len(_json(result)) > max_chars:
            raise ValueError("Base evidence overview cannot fit max_chars")

        # Reserve room for chronology and a pageable catalog. Alternate roots so
        # a large task/tool catalog cannot crowd out released official feedback.
        context_limit = max_chars * 0.68
        for index in range(max((len(ids) for ids in roots.values()), default=0)):
            for root, ids in roots.items():
                if index >= len(ids):
                    continue
                eid = ids[index]
                value = self.entries[eid]["value"]
                item = self._metadata(eid)
                truncated = isinstance(value, str) and len(value) > 160
                if truncated:
                    item.update(preview=value[:160], preview_truncated=True)
                else:
                    item.update(value=value, preview_truncated=False)
                candidate = deepcopy(result)
                candidate["global_context"][root].append(item)
                candidate["omissions"][root] -= 1
                if not truncated:
                    candidate["omissions"]["evidence_values"] -= 1
                    if self._citable(eid):
                        candidate["delivered_ids"].append(eid)
                if len(_json(candidate)) <= context_limit:
                    result = candidate
        for event in events:
            candidate = deepcopy(result)
            candidate["events"].append(event)
            candidate["omissions"]["events"] -= 1
            end = len(candidate["events"])
            candidate["event_next_offset"] = end if end < len(events) else None
            if len(_json(candidate)) > max_chars * 0.85:
                break
            result = candidate
        for node in structure[:64]:
            candidate = deepcopy(result)
            candidate["structure_catalog"]["entries"].append(node)
            end = len(candidate["structure_catalog"]["entries"])
            candidate["structure_catalog"].update(
                next_offset=end if end < len(structure) else None,
                omitted=len(structure) - end)
            if len(_json(candidate)) > max_chars * 0.95:
                break
            result = candidate
        for eid in self.entries:
            candidate = deepcopy(result)
            candidate["catalog"]["entries"].append(self._metadata(eid))
            end = len(candidate["catalog"]["entries"])
            candidate["catalog"]["next_offset"] = end if end < len(self.entries) else None
            if len(_json(candidate)) > max_chars:
                break
            result = candidate
        return deepcopy(result)
