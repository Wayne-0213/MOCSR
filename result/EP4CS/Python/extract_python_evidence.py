from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tree_sitter import Language, Parser
import tree_sitter_python as tspython


PY_LANGUAGE = Language(tspython.language())
DEFAULT_INPUT = Path(__file__).resolve().parent / "test_new_pred_EP4CS.jsonl"
DEFAULT_EVIDENCE_FIELD = "code_evidence"
DEFAULT_ERROR_FIELD = "code_evidence_error"


MUTATING_CALL_KEYWORDS = {
    "add", "append", "extend", "insert", "remove", "pop", "clear",
    "update", "setdefault", "put", "set", "write", "writelines",
    "send", "emit", "save", "delete", "discard", "push", "enqueue",
    "dequeue", "drain", "close", "flush", "commit", "rollback",
    "log", "print",
}


def make_parser(language: Language) -> Parser:
    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
        return parser


def split_identifier(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = name.replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def join_conditions(conds: List[str]) -> str:
    if not conds:
        return "always"
    if len(conds) == 1:
        return conds[0]
    return " and ".join(f"({c})" for c in conds)


def invert_condition(cond: str, lang: str) -> str:
    return f"!({cond})" if lang == "java" else f"not ({cond})"


def is_mutating_call_name(name: Optional[str]) -> bool:
    if not name:
        return False
    lname = name.lower()
    return any(lname == k or lname.endswith(k) for k in MUTATING_CALL_KEYWORDS)


@dataclass
class FlowNode:
    id: str
    type: str
    text: str


@dataclass
class Edge:
    from_id: str
    to_id: str
    label: str


@dataclass
class ReturnFact:
    when: str
    expr: str


@dataclass
class UpdateFact:
    when: str
    action: str


@dataclass
class Signature:
    name: str
    name_split: str
    parameters: List[str]
    return_type: Optional[str]


class BaseTreeSitterExtractor:
    def __init__(self, language_name: str, language: Language):
        self.language_name = language_name
        self.language = language
        self.parser = make_parser(language)
        self.node_counter = 0
        self.flow_nodes: List[FlowNode] = []
        self.edges: List[Edge] = []
        self.main_path: List[str] = []
        self.alternate_path: List[str] = []
        self.returns: List[ReturnFact] = []
        self.updates: List[UpdateFact] = []
        self.throws_facts: List[ReturnFact] = []
        self.source_code = ""
        self.source_bytes = b""

    def extract(self, code: str) -> Dict[str, Any]:
        self._reset_state()
        prepared_code = self.prepare_code(code)
        self.source_code = prepared_code
        self.source_bytes = prepared_code.encode("utf-8")
        tree = self.parser.parse(self.source_bytes)
        root = tree.root_node
        func = self.find_first_callable(root)
        if func is None:
            raise ValueError(f"No callable found for language={self.language_name}")
        signature = self.extract_signature(func)
        body_stmts = self.get_callable_body_statements(func)
        self._build_flow(body_stmts, next_after=None)
        self._collect_semantic_facts(body_stmts, conditions=[])
        relevance = self._build_relevance(signature)
        return {
            "coherence": {
                "flow": [asdict(x) for x in self.flow_nodes],
                "edges": [asdict(x) for x in self.edges],
                "main_path": self.main_path,
                "alternate_path": self.alternate_path,
            },
            "consistency": {
                "signature": asdict(signature),
                "returns": [asdict(x) for x in self.returns],
                "updates": [asdict(x) for x in self.updates],
            },
            "relevance": relevance,
        }

    def prepare_code(self, code: str) -> str:
        return code

    def find_first_callable(self, root) -> Optional[Any]:
        raise NotImplementedError

    def extract_signature(self, func) -> Signature:
        raise NotImplementedError

    def get_callable_body_statements(self, func) -> List[Any]:
        raise NotImplementedError

    def is_if(self, node) -> bool:
        raise NotImplementedError

    def is_for(self, node) -> bool:
        raise NotImplementedError

    def is_foreach(self, node) -> bool:
        return False

    def is_while(self, node) -> bool:
        raise NotImplementedError

    def is_try(self, node) -> bool:
        raise NotImplementedError

    def is_return(self, node) -> bool:
        raise NotImplementedError

    def is_throw(self, node) -> bool:
        raise NotImplementedError

    def is_block(self, node) -> bool:
        raise NotImplementedError

    def get_block_statements(self, node) -> List[Any]:
        raise NotImplementedError

    def get_if_parts(self, node) -> Tuple[str, List[Any], List[Any]]:
        raise NotImplementedError

    def get_loop_body_statements(self, node) -> List[Any]:
        raise NotImplementedError

    def get_loop_text(self, node) -> str:
        raise NotImplementedError

    def get_try_parts(self, node) -> Tuple[List[Any], List[List[Any]], List[Any]]:
        raise NotImplementedError

    def get_return_expr_text(self, node) -> str:
        raise NotImplementedError

    def get_throw_expr_text(self, node) -> str:
        raise NotImplementedError

    def collect_updates_from_stmt(self, stmt, conditions: List[str]) -> None:
        raise NotImplementedError

    def _reset_state(self) -> None:
        self.node_counter = 0
        self.flow_nodes = []
        self.edges = []
        self.main_path = []
        self.alternate_path = []
        self.returns = []
        self.updates = []
        self.throws_facts = []
        self.source_code = ""
        self.source_bytes = b""

    def next_id(self) -> str:
        self.node_counter += 1
        return f"B{self.node_counter}"

    def node_text(self, node) -> str:
        return self.source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def named_children(self, node) -> List[Any]:
        return list(node.named_children)

    def first_named_child_of_type(self, node, type_name: str):
        for ch in self.named_children(node):
            if ch.type == type_name:
                return ch
        return None

    def named_descendants(self, node) -> Iterable[Any]:
        stack = [node]
        while stack:
            cur = stack.pop()
            for ch in reversed(self.named_children(cur)):
                stack.append(ch)
            yield cur

    def add_flow_node(self, node_type: str, text: str) -> str:
        nid = self.next_id()
        self.flow_nodes.append(FlowNode(id=nid, type=node_type, text=" ".join(text.split())))
        return nid

    class FlowResult:
        def __init__(self, entry: Optional[str], exits: List[str]):
            self.entry = entry
            self.exits = exits

    def _build_flow(self, stmts: List[Any], next_after: Optional[str]) -> Tuple[Optional[str], List[str]]:
        first_entry = None
        open_exits: List[str] = []
        for idx, stmt in enumerate(stmts):
            succ_next = next_after if idx == len(stmts) - 1 else None
            fr = self._build_stmt_flow(stmt, succ_next)
            if first_entry is None:
                first_entry = fr.entry
            if open_exits and fr.entry:
                for ex in open_exits:
                    self.edges.append(Edge(from_id=ex, to_id=fr.entry, label="next"))
            open_exits = fr.exits
        return first_entry, open_exits

    def _build_stmt_flow(self, stmt, next_after: Optional[str]) -> "BaseTreeSitterExtractor.FlowResult":
        if self.is_if(stmt):
            cond_text, then_stmts, else_stmts = self.get_if_parts(stmt)
            cond_id = self.add_flow_node("if", cond_text)
            then_entry, then_exits = self._build_flow(then_stmts, next_after)
            if then_entry:
                self.edges.append(Edge(from_id=cond_id, to_id=then_entry, label="true"))
            exits = list(then_exits)
            if else_stmts:
                else_entry, else_exits = self._build_flow(else_stmts, next_after)
                if else_entry:
                    self.edges.append(Edge(from_id=cond_id, to_id=else_entry, label="false"))
                exits.extend(else_exits)
            else:
                exits.append(cond_id)
            if cond_id not in self.main_path:
                self.main_path.append(cond_id)
            if cond_id not in self.alternate_path:
                self.alternate_path.append(cond_id)
            if then_entry and then_entry not in self.main_path:
                self.main_path.append(then_entry)
            if else_stmts:
                else_entry = None
                for edge in self.edges:
                    if edge.from_id == cond_id and edge.label == "false":
                        else_entry = edge.to_id
                        break
                if else_entry and else_entry not in self.alternate_path:
                    self.alternate_path.append(else_entry)
            return self.FlowResult(cond_id, exits)

        if self.is_for(stmt):
            loop_id = self.add_flow_node("for", self.get_loop_text(stmt))
            body_entry, _ = self._build_flow(self.get_loop_body_statements(stmt), loop_id)
            if body_entry:
                self.edges.append(Edge(from_id=loop_id, to_id=body_entry, label="loop"))
            if next_after:
                self.edges.append(Edge(from_id=loop_id, to_id=next_after, label="exit"))
            if loop_id not in self.main_path:
                self.main_path.append(loop_id)
            return self.FlowResult(loop_id, [loop_id] if next_after is None else [])

        if self.is_foreach(stmt):
            loop_id = self.add_flow_node("foreach", self.get_loop_text(stmt))
            body_entry, _ = self._build_flow(self.get_loop_body_statements(stmt), loop_id)
            if body_entry:
                self.edges.append(Edge(from_id=loop_id, to_id=body_entry, label="loop"))
            if next_after:
                self.edges.append(Edge(from_id=loop_id, to_id=next_after, label="exit"))
            if loop_id not in self.main_path:
                self.main_path.append(loop_id)
            return self.FlowResult(loop_id, [loop_id] if next_after is None else [])

        if self.is_while(stmt):
            loop_id = self.add_flow_node("while", self.get_loop_text(stmt))
            body_entry, _ = self._build_flow(self.get_loop_body_statements(stmt), loop_id)
            if body_entry:
                self.edges.append(Edge(from_id=loop_id, to_id=body_entry, label="loop"))
            if next_after:
                self.edges.append(Edge(from_id=loop_id, to_id=next_after, label="exit"))
            if loop_id not in self.main_path:
                self.main_path.append(loop_id)
            return self.FlowResult(loop_id, [loop_id] if next_after is None else [])

        if self.is_try(stmt):
            try_body, catches, final_body = self.get_try_parts(stmt)
            try_id = self.add_flow_node("try", "try")
            body_entry, body_exits = self._build_flow(try_body, next_after)
            if body_entry:
                self.edges.append(Edge(from_id=try_id, to_id=body_entry, label="try"))
            exits = list(body_exits)
            for i, catch_stmts in enumerate(catches):
                if not catch_stmts:
                    continue
                catch_id = self.add_flow_node("catch", f"catch_{i}")
                self.edges.append(Edge(from_id=try_id, to_id=catch_id, label=f"catch_{i}"))
                catch_entry, catch_exits = self._build_flow(catch_stmts, next_after)
                if catch_entry:
                    self.edges.append(Edge(from_id=catch_id, to_id=catch_entry, label="handler"))
                exits.extend(catch_exits)
            if final_body:
                final_id = self.add_flow_node("finally", "finally")
                for ex in exits:
                    self.edges.append(Edge(from_id=ex, to_id=final_id, label="finally"))
                final_entry, final_exits = self._build_flow(final_body, next_after)
                if final_entry:
                    self.edges.append(Edge(from_id=final_id, to_id=final_entry, label="body"))
                exits = final_exits
            if try_id not in self.main_path:
                self.main_path.append(try_id)
            return self.FlowResult(try_id, exits)

        if self.is_return(stmt):
            nid = self.add_flow_node("return", "return " + self.get_return_expr_text(stmt))
            return self.FlowResult(nid, [])

        if self.is_throw(stmt):
            nid = self.add_flow_node("throw", "throw " + self.get_throw_expr_text(stmt))
            return self.FlowResult(nid, [])

        if self.is_block(stmt):
            return self.FlowResult(*self._build_flow(self.get_block_statements(stmt), next_after))

        text = self.node_text(stmt).strip().replace("\n", " ")
        nid = self.add_flow_node(stmt.type, text)
        return self.FlowResult(nid, [nid])

    def _collect_semantic_facts(self, stmts: List[Any], conditions: List[str]) -> None:
        for stmt in stmts:
            if self.is_if(stmt):
                cond, then_stmts, else_stmts = self.get_if_parts(stmt)
                self._collect_semantic_facts(then_stmts, conditions + [cond])
                if else_stmts:
                    self._collect_semantic_facts(
                        else_stmts,
                        conditions + [invert_condition(cond, self.language_name)],
                    )
            elif self.is_for(stmt):
                loop_cond = self.get_loop_text(stmt)
                self._collect_semantic_facts(self.get_loop_body_statements(stmt), conditions + [loop_cond])
            elif self.is_foreach(stmt):
                loop_cond = self.get_loop_text(stmt)
                self._collect_semantic_facts(self.get_loop_body_statements(stmt), conditions + [loop_cond])
            elif self.is_while(stmt):
                loop_cond = self.get_loop_text(stmt)
                self._collect_semantic_facts(self.get_loop_body_statements(stmt), conditions + [loop_cond])
            elif self.is_try(stmt):
                try_body, catches, final_body = self.get_try_parts(stmt)
                self._collect_semantic_facts(try_body, conditions + ["try"])
                for i, catch_stmts in enumerate(catches):
                    self._collect_semantic_facts(catch_stmts, conditions + [f"catch_{i}"])
                if final_body:
                    self._collect_semantic_facts(final_body, conditions + ["finally"])
            elif self.is_return(stmt):
                self.returns.append(ReturnFact(when=join_conditions(conditions), expr=self.get_return_expr_text(stmt)))
            elif self.is_throw(stmt):
                self.throws_facts.append(ReturnFact(when=join_conditions(conditions), expr=self.get_throw_expr_text(stmt)))
            else:
                self.collect_updates_from_stmt(stmt, conditions)

    def _build_relevance(self, signature: Signature) -> Dict[str, Any]:
        essential_facts = []
        supporting_facts = []
        for ret in self.returns:
            essential_facts.append({"when": ret.when, "outcome": f"return {ret.expr}"})
        for thrown in self.throws_facts:
            essential_facts.append({"when": thrown.when, "outcome": f"throw {thrown.expr}"})
        for update in self.updates:
            action_lower = update.action.lower()
            if any(k in action_lower for k in [
                "append(", ".append(", "add(", ".add(", "update(", ".update(",
                "write(", ".write(", "save(", "delete(", "remove(", "pop(",
                "set(", ".set(", "print(", ".print(", "log(", ".log(",
                "send(", ".send(", "emit(", ".emit(", "drain(", ".drain(",
                "put(", ".put(",
            ]):
                essential_facts.append({"when": update.when, "outcome": update.action})
            else:
                supporting_facts.append({"when": update.when, "detail": update.action})
        return {
            "signature": asdict(signature),
            "essential_facts": essential_facts,
            "supporting_facts": supporting_facts[:2],
        }


class PythonTreeSitterExtractor(BaseTreeSitterExtractor):
    FUNCTION_TYPES = {"function_definition", "async_function_definition"}

    def __init__(self):
        super().__init__("python", PY_LANGUAGE)

    def find_first_callable(self, root):
        for node in self.named_descendants(root):
            if node.type in self.FUNCTION_TYPES:
                return node
        return None

    def extract_signature(self, func) -> Signature:
        name_node = func.child_by_field_name("name")
        params_node = func.child_by_field_name("parameters")
        ret_node = func.child_by_field_name("return_type")
        name = self.node_text(name_node) if name_node else "unknown"
        return Signature(
            name=name,
            name_split=split_identifier(name),
            parameters=self._extract_params(params_node),
            return_type=self.node_text(ret_node) if ret_node else None,
        )

    def get_callable_body_statements(self, func) -> List[Any]:
        body = func.child_by_field_name("body")
        return self.get_block_statements(body) if body else []

    def is_if(self, node) -> bool:
        return node.type in {"if_statement", "elif_clause"}

    def is_for(self, node) -> bool:
        return node.type == "for_statement"

    def is_while(self, node) -> bool:
        return node.type == "while_statement"

    def is_try(self, node) -> bool:
        return node.type == "try_statement"

    def is_return(self, node) -> bool:
        return node.type == "return_statement"

    def is_throw(self, node) -> bool:
        return node.type == "raise_statement"

    def is_block(self, node) -> bool:
        return node is not None and node.type == "block"

    def get_block_statements(self, node) -> List[Any]:
        if node is None:
            return []
        if node.type == "block":
            return self.named_children(node)
        if node.type == "else_clause":
            kids = self.named_children(node)
            if len(kids) == 1 and kids[0].type == "block":
                return self.named_children(kids[0])
            return kids
        if node.type == "elif_clause":
            return [node]
        return [node]

    def get_if_parts(self, node) -> Tuple[str, List[Any], List[Any]]:
        cond_node = node.child_by_field_name("condition")
        then_node = node.child_by_field_name("consequence") or node.child_by_field_name("body")
        else_node = node.child_by_field_name("alternative")
        named = self.named_children(node)

        if cond_node is None:
            for ch in named:
                if ch.type not in {"block", "else_clause", "elif_clause"}:
                    cond_node = ch
                    break
        if then_node is None:
            for ch in named:
                if ch.type == "block":
                    then_node = ch
                    break
        if else_node is None:
            for ch in named:
                if ch.type in {"else_clause", "elif_clause"}:
                    else_node = ch
                    break

        cond = self.node_text(cond_node) if cond_node else "<if>"
        then_stmts = self.get_block_statements(then_node) if then_node else []
        else_stmts = self.get_block_statements(else_node) if else_node else []
        return cond, then_stmts, else_stmts

    def get_loop_body_statements(self, node) -> List[Any]:
        body = node.child_by_field_name("body")
        if body:
            return self.get_block_statements(body)
        for ch in self.named_children(node):
            if ch.type == "block":
                return self.get_block_statements(ch)
        return []

    def get_loop_text(self, node) -> str:
        return " ".join(self.node_text(node).split())

    def get_try_parts(self, node) -> Tuple[List[Any], List[List[Any]], List[Any]]:
        body_node = node.child_by_field_name("body")
        try_body = self.get_block_statements(body_node) if body_node else []
        catches = []
        final_body = []
        for ch in self.named_children(node):
            if ch.type == "except_clause":
                catch_body = ch.child_by_field_name("body") or self.first_named_child_of_type(ch, "block")
                catches.append(self.get_block_statements(catch_body))
            elif ch.type == "finally_clause":
                body = ch.child_by_field_name("body") or self.first_named_child_of_type(ch, "block")
                final_body = self.get_block_statements(body)
        return try_body, catches, final_body

    def get_return_expr_text(self, node) -> str:
        named = self.named_children(node)
        if not named:
            return "None"
        return self.node_text(named[0])

    def get_throw_expr_text(self, node) -> str:
        named = self.named_children(node)
        if not named:
            return "raise"
        return self.node_text(named[0])

    def collect_updates_from_stmt(self, stmt, conditions: List[str]) -> None:
        when = join_conditions(conditions)
        stype = stmt.type
        if stype in {"assignment", "augmented_assignment"}:
            self.updates.append(UpdateFact(when=when, action=" ".join(self.node_text(stmt).split())))
            return

        if stype == "expression_statement":
            children = self.named_children(stmt)
            expr_child = children[0] if children else None
            if expr_child and expr_child.type == "call":
                call_name = self._get_call_name(expr_child)
                if is_mutating_call_name(call_name):
                    self.updates.append(UpdateFact(when=when, action=" ".join(self.node_text(expr_child).split())))
            return

        for node in self.named_descendants(stmt):
            if node is stmt:
                continue
            if node.type in {"assignment", "augmented_assignment"}:
                self.updates.append(UpdateFact(when=when, action=" ".join(self.node_text(node).split())))
            elif node.type == "call":
                call_name = self._get_call_name(node)
                if is_mutating_call_name(call_name):
                    self.updates.append(UpdateFact(when=when, action=" ".join(self.node_text(node).split())))

    def _extract_params(self, params_node) -> List[str]:
        if params_node is None:
            return []
        return [" ".join(self.node_text(ch).split()) for ch in self.named_children(params_node)]

    def _get_call_name(self, call_node) -> Optional[str]:
        func_node = call_node.child_by_field_name("function")
        if func_node is None:
            named = self.named_children(call_node)
            if named:
                func_node = named[0]
        if func_node is None:
            return None
        if func_node.type == "identifier":
            return self.node_text(func_node)
        if func_node.type == "attribute":
            attr = func_node.child_by_field_name("attribute")
            if attr:
                return self.node_text(attr)
        return self.node_text(func_node).split(".")[-1]


class TreeSitterEvidenceExtractor:
    def __init__(self):
        self.impl = PythonTreeSitterExtractor()

    def extract(self, code: str) -> Dict[str, Any]:
        return self.impl.extract(code)


def process_jsonl(
    input_path: Path,
    output_path: Optional[Path],
    evidence_field: str,
    error_field: str,
    backup: bool,
) -> Dict[str, int]:
    input_path = input_path.resolve()
    output_path = output_path.resolve() if output_path else input_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    extractor = TreeSitterEvidenceExtractor()
    stats = {"total": 0, "ok": 0, "error": 0, "blank": 0}

    with input_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                dst.write(line)
                stats["blank"] += 1
                continue
            stats["total"] += 1
            item: Any = None
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL item must be an object")
                code = item.get("code")
                if not isinstance(code, str):
                    raise ValueError("missing string field: code")
                item[evidence_field] = extractor.extract(code)
                item[error_field] = None
                stats["ok"] += 1
            except Exception as exc:
                stats["error"] += 1
                if not isinstance(item, dict):
                    item = {"_raw_line": line.rstrip("\n")}
                item[evidence_field] = None
                item[error_field] = f"line {line_no}: {type(exc).__name__}: {exc}"
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    if output_path == input_path:
        if backup:
            shutil.copy2(input_path, input_path.with_suffix(input_path.suffix + ".bak"))
        tmp_path.replace(input_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Python code evidence from test_python.jsonl.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path. Defaults to in-place update.")
    parser.add_argument("--evidence-field", default=DEFAULT_EVIDENCE_FIELD, help="Field used for extracted evidence.")
    parser.add_argument("--error-field", default=DEFAULT_ERROR_FIELD, help="Field used for extraction errors.")
    parser.add_argument("--backup", action="store_true", help="Create a .bak file before in-place replacement.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = process_jsonl(
        input_path=args.input,
        output_path=args.output,
        evidence_field=args.evidence_field,
        error_field=args.error_field,
        backup=args.backup,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
