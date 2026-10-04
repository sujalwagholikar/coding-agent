"""Regression tests for making tree-sitter PRIMARY (not fallback) for
JS/TS/TSX symbol extraction.

Context: repo_intel.py already loaded real tree-sitter grammars for
Python/JS/TS/TSX/CSS/HTML, but `_FALLBACK_LANGS` forced ALL FIVE to always
use hand-rolled regex/ast parsers instead, with tree-sitter never actually
reached for them. Direct comparison showed the regex JS/TS parser
(`_fallback_js`) misses real-world patterns a grammar catches for free -
most importantly object-literal methods (`{ onClick: () => {} }`), which is
extremely common in React/config/test-mock code. Python/CSS/HTML fallbacks
are themselves real parsers (ast, a brace-matching scanner, html.parser) and
were verified to already be at least as accurate as tree-sitter for their
languages, so only JS/TS/TSX were switched to tree-sitter-primary.

These tests also cover a real bug caught during manual end-to-end
verification of the switch: naively using a `variable_declarator`'s own
node boundaries (or a `function_declaration`'s) as the patchable symbol span
loses the enclosing `export` keyword from the AVAILABLE range but NOT from
what a model naturally writes back - patching 'plainHelper' with
replacement text starting 'export function plainHelper...' produced
'export export function plainHelper...' in the file. Fixed by widening the
symbol's span to the enclosing export_statement/lexical_declaration.
"""
import sys, importlib.abc, importlib.machinery, tempfile, pathlib
from unittest.mock import MagicMock


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIX = ("rich", "google", "openai", "prompt_toolkit", "pygments", "httpx", "dotenv")
    _busy = False
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top not in self.PREFIX or self._busy:
            return None
        import importlib.util
        _Stub._busy = True
        try:
            missing = importlib.util.find_spec(top) is None
        except Exception:
            missing = True
        finally:
            _Stub._busy = False
        if missing:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name; return m
    def exec_module(self, m): pass


sys.meta_path.insert(0, _Stub())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coder3
import repo_intel as ri

ok = True
def check(label, cond, detail=""):
    global ok; ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"  -> {str(detail)[:300]}" if not cond else ""))


def names(entry):
    return {(s["kind"], s.get("qualname", s["name"])) for s in entry.symbols}


# --------------------------------------------------------------------------- #
# 1. Tree-sitter is actually being used for the 5 target languages
# --------------------------------------------------------------------------- #
check("tree-sitter loaded successfully in this environment", ri.HAVE_TREE_SITTER,
      ri._language_load_errors)

TS_SAMPLE = b"""
import { Injectable } from "@angular/core";

@Injectable()
export class UserService {
  private cache = new Map();

  async getUser(id: string) {
    return this.cache.get(id);
  }
}

const config = {
  onLoad: () => console.log("loaded"),
  handlers: {
    click: function handleClick(e) { return e; }
  }
};

export const useCounter = () => {
  return 1;
};

function plainHelper(x) { return x * 2; }
"""
entry = ri._parse_file(pathlib.Path("x.ts"), TS_SAMPLE)
found = names(entry)
check("class + its method are found", ("class", "UserService") in found and ("method", "UserService.getUser") in found, found)
check("object-literal arrow method 'onLoad' is found (regex fallback misses this entirely)",
      ("method", "onLoad") in found, found)
check("nested named function expression 'handlers.click' is found (regex fallback misses this entirely)",
      ("method", "click") in found, found)
check("exported const-arrow function is found", ("func", "useCounter") in found, found)
check("plain function declaration still found (no regression)", ("func", "plainHelper") in found, found)


# --------------------------------------------------------------------------- #
# 2. Python/CSS/HTML are UNCHANGED (still their own real parsers, not tree-sitter)
#    - switching them would have been a downgrade (ast/html.parser/brace
#    scanner are already exact; tree-sitter would just reconstruct the same
#    info heuristically), so this locks in that they were deliberately left alone.
# --------------------------------------------------------------------------- #
py_entry = ri._parse_file(pathlib.Path("x.py"), b"class Foo:\n    def bar(self):\n        pass\n")
check("python still resolves class/method with exact qualnames",
      ("method", "Foo.bar") in names(py_entry), names(py_entry))

css_entry = ri._parse_file(pathlib.Path("x.css"), b".card { color: red; }\n@media (min-width: 600px) { .card { color: blue; } }")
check("css rules still resolve (brace-scanner untouched)",
      any(s["kind"] == "rule" and s["name"] == ".card" for s in css_entry.symbols), css_entry.symbols)

html_entry = ri._parse_file(pathlib.Path("x.html"), b'<div id="app"><script>function init(){}</script></div>')
check("html elements + inline script symbols still resolve (html.parser untouched)",
      ("element", "#app") in names(html_entry) and ("func", "init") in names(html_entry), names(html_entry))


# --------------------------------------------------------------------------- #
# 3. JSX/TSX
# --------------------------------------------------------------------------- #
jsx_entry = ri._parse_file(pathlib.Path("x.jsx"), b"""
const Button = ({ onClick, label }) => {
  return null;
};
export default function App() { return null; }
""")
check("jsx: arrow-function component found", ("func", "Button") in names(jsx_entry), names(jsx_entry))
check("jsx: default-export function component found", ("func", "App") in names(jsx_entry), names(jsx_entry))

tsx_entry = ri._parse_file(pathlib.Path("x.tsx"), b"""
interface Props { label: string; }
export const Card: React.FC<Props> = ({ label }) => { return null; };
""")
check("tsx: interface found", ("interface", "Props") in names(tsx_entry), names(tsx_entry))
check("tsx: typed const-arrow component found", ("func", "Card") in names(tsx_entry), names(tsx_entry))


# --------------------------------------------------------------------------- #
# 4. THE BUG: symbol span must include 'export' so a patch doesn't duplicate it
# --------------------------------------------------------------------------- #
def mkws(src: str, fname: str):
    root = pathlib.Path(tempfile.mkdtemp())
    (root / fname).write_text(src)
    return root, coder3.Workspace(root, auto_approve=True)

root, ws = mkws(
    "export function plainHelper(x) {\n  return x * 2;\n}\n\n"
    "export const useCounter = () => {\n  return 1;\n};\n",
    "a.ts",
)
ws.edit_file("a.ts", new="export function plainHelper(x) {\n  return x * 3;\n}", symbol="plainHelper")
out1 = (root / "a.ts").read_text()
check("patching an exported function does not duplicate 'export'",
      "export export" not in out1 and "export function plainHelper" in out1 and "x * 3" in out1, out1)

ws.edit_file("a.ts", new="export const useCounter = () => {\n  return 2;\n};", symbol="useCounter")
out2 = (root / "a.ts").read_text()
check("patching an exported const-arrow does not duplicate 'export'",
      "export export" not in out2 and "export const useCounter" in out2 and "return 2" in out2, out2)

root2, ws2 = mkws(
    "@Injectable()\nexport class Svc {\n  async run() {\n    return 1;\n  }\n}\n",
    "b.ts",
)
ws2.edit_file("b.ts", new="  async run() {\n    return 2;\n  }", symbol="Svc.run")
out3 = (root2 / "b.ts").read_text()
check("decorator stays intact and un-duplicated when patching a method of a decorated exported class",
      out3.count("@Injectable()") == 1 and out3.count("export class Svc") == 1 and "return 2" in out3, out3)


print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
