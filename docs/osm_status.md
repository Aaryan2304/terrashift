# OSM and Overpass status

**State: blocked, deliberately not retried.** D2.4 stays deferred until the
manual labels and the local patch index exist.

The operational record produced at the time is
`outputs/audit/osm_query_status.md`. That path is inside `outputs/`, which is
gitignored, so the facts that matter are repeated here in a tracked file.

## What was attempted

On 2026-11-13 the candidate-context script issued 20 small Overpass queries: the
10 highest-ranked candidate areas for Pair A and the same for Pair B, each
looking for buildings and roads within 150 m. The saved result is
`outputs/audit/osm_candidate_context.csv`.

| Outcome | Count |
|---|---|
| Successful queries | 0 |
| HTTP 406 responses | 2 |
| `Network is unreachable` responses | 18 |

No building or road count was obtained. Absence of an OSM result is not evidence
that an area is unchanged, and it is not evidence that no building exists there.

## Why it failed, as far as can be verified

Two separate causes were observed, in this order:

1. **HTTP 406.** The client sent the default Python `requests` User-Agent.
   Overpass rejected it as unidentifiable. Current Overpass usage guidance
   requires an identifying User-Agent, and asks clients to pause at least 30
   seconds after a 406 or 429 before trying again.
2. **`Network is unreachable`.** After the 406 responses, the host could not
   reach the endpoint at all. Ordinary HTTPS to other hosts works from this WSL
   environment, so the failure is specific to the endpoint reachable from here,
   not a total loss of networking.

The first run continued issuing queries after the 406 responses. That is a
policy violation of the Overpass guidance, and it is the reason the client was
changed before it was stopped.

## What was changed

- `data/osm.py` sends `skyserve-assignment/0.1 (+https://github.com/Aaryan2304)`
  as its User-Agent.
- It does not retry HTTP 406 or 429, and does not switch endpoints
  automatically.
- `scripts/query_candidate_osm.py` stops the batch at the first service or
  network error instead of working through the remaining candidates.

No second batch has been issued since that correction. It has not been retried
during this work, by instruction.

## Why OSM was wanted, and what replaces it

OSM was intended as independent corroboration for candidate areas: a building or
road in OSM near a candidate would support a human reviewer's decision. It is
not needed for detection, it cannot supply historical state for the 2022-2023 or
2023-2026 acquisition dates, and it is current-state data only.

If it becomes available, the correct path is a bulk extract, not the Overpass
API: `https://download.geofabrik.de/asia/india-latest.osm.pbf`, filtered
locally with `pyosmium`, which avoids the quota and politeness problems that
caused the 406. That download is roughly 1.4 GB and has not been attempted.

## Rules that stay in force

- OSM records never become ground truth. A nearby building is context for a
  reviewer, not a label.
- The detection register stays human-authored. No script may populate it from
  OSM.
- Attribution if OSM data is ever used: © OpenStreetMap contributors, ODbL.

## Source

OpenStreetMap Wiki, Overpass API: <https://wiki.openstreetmap.org/wiki/Overpass_API>
