"""Write immutable executable provenance; environment values are never evaluated as shell code."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(os.environ['FML_BENCH_ROOT']).resolve().parent
local = Path(__file__).resolve().parent
pending = [root / 'FML/Global/Global.cpp'] + list(local.glob('*.cpp'))
files = {p: 'FML/Performance/' + p.name for p in local.iterdir() if p.suffix in {'.py', '.h'} or p.name == 'Makefile'}
files = {p: name for p, name in files.items() if p.name != 'BuildInfo.h'}
seen = set()
while pending:
    path = pending.pop().resolve()
    if path in seen or path.name == 'BuildInfo.h':
        continue
    seen.add(path)
    files[path] = str(path.relative_to(root)) if path.is_relative_to(root) else 'FML/Performance/' + path.name
    for kind, name in re.findall(r'^\s*#\s*include\s*([<"])([^>"\n]+)[>"]', path.read_text(), re.M):
        target = root / name if name.startswith('FML/') else path.parent / name
        if (name.startswith('FML/') or kind == '"') and target.is_file():
            pending.append(target)
digest = hashlib.sha256()
for name, path in sorted({name: path for path, name in files.items()}.items()):
    digest.update(name.encode() + b'\0' + path.read_bytes() + b'\0')

def git(*args):
    result = subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'

info = {key: os.environ.get('FML_BENCH_' + key.upper(), '') for key in ['cxx', 'options', 'includes', 'libraries', 'links']}
info.update(revision=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain', '--untracked-files=no'), source_sha256=digest.hexdigest())
text = '#ifndef FML_BENCH_BUILD_INFO_H\n#define FML_BENCH_BUILD_INFO_H\nnamespace build_info {\n'
for key, value in info.items():
    text += 'inline constexpr const char *' + key + ' = ' + json.dumps(value) + ';\n'
text += '}\n#endif\n'
output = Path(sys.argv[1])
if not output.exists() or output.read_text() != text:
    output.write_text(text)
