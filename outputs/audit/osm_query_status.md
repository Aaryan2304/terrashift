# OSM lookup status

Date: 2026-11-13

## Result

The candidate-context script tried 20 small Overpass queries: 10 ranked Pair A areas and 10 ranked Pair B areas. The saved output is `outputs/audit/osm_candidate_context.csv`.

- Successful queries: 0.
- HTTP 406 responses: 2.
- Network-unreachable responses: 18.
- No OSM building or road counts were obtained. Do not infer zero buildings or roads from these errors.

The initial client used the default Python requests User-Agent. Current Overpass guidance requires an identifiable client User-Agent. The OpenStreetMap Overpass API wiki also says that users should pause at least 30 seconds after HTTP errors such as 406 or 429. The first run continued after the two 406 responses and then received network errors. The corrected script now stops the batch after the first HTTP 406/429 or network-unreachable error. It does not switch endpoints automatically.

## Correction

`data/osm.py` now sends `skyserve-assignment/0.1 (+https://github.com/Aaryan2304)` as its User-Agent. It does not retry HTTP 406 or 429. The candidate script stops the batch after a service or network error. It does not switch to a different endpoint automatically.

No second request batch has been made after that correction. The environment reported `Network is unreachable`, so this source remains blocked from WSL. We can continue the analysis without OSM. We must not treat OSM absence as evidence that an area is unchanged.

## Limits

Even if a later query succeeds, OSM is current context. It is not historical ground truth for the Sentinel-2 acquisition dates. Use it only to support human review. Keep the licence attribution: © OpenStreetMap contributors, ODbL.

## Source

OpenStreetMap Wiki, Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
