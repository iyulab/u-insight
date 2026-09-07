#!/usr/bin/env bash
#
# Single source of truth for the release version is `version` in Cargo.toml.
# Every release artifact must agree with it:
#
#   - the C# binding (bindings/csharp/UInsight/UInsight.csproj <Version>)
#
# nuget-release.yml triggers only on a commit that changes UInsight.csproj's
# <Version> tag — the main release routine (crates.io + npm bump) does not
# touch this file, so the NuGet channel silently drifts behind unless this is
# checked (real incident: 4 releases — 0.10.1/0.11.0/0.12.x/0.13.0 — were never
# published to NuGet because nothing bumped this file in lockstep; see
# u-insight's issue tracker / the umbrella's issue draft that reported it).
#
# The npm package is produced by wasm-pack directly from this Cargo.toml, so it
# is always consistent and needs no check here.
#
# Run locally before pushing a version bump:
#   bash scripts/check-version-consistency.sh
#
# Enforced in CI (.github/workflows/ci.yml, "Version Consistency" job).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

crate_version="$(grep -m1 -E '^version = ' Cargo.toml | sed -E 's/version = "(.*)"/\1/')"
if [[ -z "${crate_version}" ]]; then
  echo "::error file=Cargo.toml::could not read [package] version"
  exit 1
fi
echo "crate version: ${crate_version}"

status=0

csproj="bindings/csharp/UInsight/UInsight.csproj"
cs_version="$(grep -oE '<Version>[^<]+</Version>' "${csproj}" | sed -E 's#</?Version>##g')"
if [[ "${cs_version}" != "${crate_version}" ]]; then
  echo "::error file=${csproj}::<Version>${cs_version}</Version>, expected ${crate_version} (bump this file's <Version> tag in the same commit to trigger nuget-release.yml)"
  status=1
else
  echo "NuGet binding version consistent at ${crate_version}"
fi

# The changelog must have gained a heading for the version being released. A bump
# whose entry is still sitting under `## [Unreleased]` publishes a version whose
# consumers have no record of what they upgraded into, and nothing else here
# would notice: every version string can agree while the changelog says nothing.
changelog="CHANGELOG.md"
if [[ -f "${changelog}" ]]; then
  if grep -qE "^## \[${crate_version//./\\.}\]" "${changelog}"; then
    echo "Changelog has an entry for ${crate_version}"
  else
    echo "::error file=${changelog}::no '## [${crate_version}]' heading — add the entry for this release in the same commit as the version bump (move it out of '## [Unreleased]')"
    status=1
  fi
fi

exit "${status}"
