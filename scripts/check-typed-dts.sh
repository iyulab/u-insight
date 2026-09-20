#!/usr/bin/env bash
# Fails when a generated TypeScript declaration types an exported function's
# return as `any`.
#
# `as` is the only thing a consumer can write against `any`, and `as` is
# exactly the construct that silences a wrong assumption about a result's
# shape: a renderer that read a per-point boolean mask as an array of indices
# compiled, shipped, and printed the same subgroup twelve times. A declared
# return type makes that a compile error on the first line instead.
#
# This runs on the publish path, not only in CI: the two run side by side on
# the same push, so a check that lives only in CI cannot stop a publish.
#
# Usage: check-typed-dts.sh <file.d.ts> [more.d.ts ...]
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <file.d.ts> [more.d.ts ...]" >&2
    exit 2
fi

status=0
for dts in "$@"; do
    if [ ! -f "$dts" ]; then
        echo "FAIL: $dts does not exist" >&2
        status=1
        continue
    fi

    total=$(grep -c '^export function' "$dts" || true)
    if [ "$total" -eq 0 ]; then
        echo "FAIL: $dts declares no exported functions -- generated from the wrong build?" >&2
        status=1
        continue
    fi

    untyped=$(grep '^export function' "$dts" | grep -E '\):\s*any;' || true)
    if [ -n "$untyped" ]; then
        count=$(printf '%s\n' "$untyped" | wc -l | tr -d ' ')
        echo "FAIL: $dts returns \`any\` from $count of $total exported function(s):" >&2
        printf '%s\n' "$untyped" | sed 's/^/  /' >&2
        echo "  Derive the declaration from the result struct (tsify) and name it in" >&2
        echo "  #[wasm_bindgen(unchecked_return_type = \"...\")]." >&2
        status=1
    else
        echo "OK: $dts -- all $total exported function(s) declare a return type"
    fi
done

exit "$status"
