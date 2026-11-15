#!/bin/bash
# Fail if a tracked file is larger than the limit (default 1 MB) and not in large_files_allowlist.txt. Big files stay
# in the git history for good (every clone downloads them), so add one only if it is needed: compress images (JPEG for
# photos and renders), generate data instead of storing it, or keep it outside the repository.
# Usage: check_large_files.sh [limit in bytes]
cd "$(dirname "$0")/../.." || exit 1
limit=${1:-1048576}
allow=.github/ci/large_files_allowlist.txt
failed=0
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  size=$(wc -c < "$f" | tr -d ' ')
  if [ "$size" -gt "$limit" ] && ! grep -qxF "$f" "$allow"; then
    echo "Too large: $f ($(( size / 1024 )) kB > $(( limit / 1024 )) kB)"
    failed=1
  fi
done < <(git ls-files -z)
if [ $failed = 1 ]; then
  echo "Make these files smaller or, if they are needed, add them to $allow"
  exit 1
fi
echo "No tracked file over $(( limit / 1024 )) kB outside the allow-list"
