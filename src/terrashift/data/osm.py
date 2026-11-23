"""Fetch OpenStreetMap buildings and roads for a bounding box.

Uses the Overpass API directly with ``requests``. No extra dependency.

The Overpass query requests ways tagged ``building`` and ways tagged
``highway`` inside the bounding box. Relations are not requested, so large
buildings mapped as relations may be incomplete. That is a known limit of
this helper.

OpenStreetMap data is ODbL. Attribution: "Copyright OpenStreetMap contributors,
ODbL, https://www.openstreetmap.org/copyright".
"""

import time

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "skyserve-assignment/0.1 (+https://github.com/Aaryan2304)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

QUERY = """
[out:json][timeout:60];
(
  way["building"]({south},{west},{north},{east});
  way["highway"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""


def fetch_osm(west, south, east, north, retries=3, timeout=90):
    """Return a list of OSM ways with tags and node coordinates.

    Each element of the result is a dict with ``type``, ``id``, ``tags`` and
    ``nodes`` (a list of ``[lon, lat]`` pairs).
    """
    query = QUERY.format(south=south, west=west, north=north, east=east)
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query},
                                headers=HEADERS, timeout=timeout)
            if resp.status_code in (406, 429):
                raise requests.HTTPError(
                    "Overpass returned HTTP %d; stop this batch and wait at least "
                    "30 seconds before any new request" % resp.status_code,
                    response=resp)
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("elements", [])
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (406, 429) or attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    return []


def split_bbox(west, south, east, north, n=2):
    """Split a bounding box into an n x n grid of smaller boxes.

    Overpass can time out on large boxes. Splitting keeps each query small.
    """
    boxes = []
    xs = [west + (east - west) * i / n for i in range(n + 1)]
    ys = [south + (north - south) * i / n for i in range(n + 1)]
    for i in range(n):
        for j in range(n):
            boxes.append((xs[j], ys[i], xs[j + 1], ys[i + 1]))
    return boxes


def fetch_osm_tiled(west, south, east, north, n=2, pause=1.0):
    """Fetch OSM data for a bounding box, split into n x n tiles."""
    elements = []
    for box in split_bbox(west, south, east, north, n):
        elements.extend(fetch_osm(*box))
        time.sleep(pause)
    return elements


if __name__ == "__main__":
    import sys
    w, s, e, n = (float(x) for x in sys.argv[1:5])
    els = fetch_osm_tiled(w, s, e, n)
    ways = [x for x in els if x.get("type") == "way"]
    buildings = [x for x in ways if "building" in x.get("tags", {})]
    highways = [x for x in ways if "highway" in x.get("tags", {})]
    print("elements:", len(els), "ways:", len(ways),
          "buildings:", len(buildings), "highways:", len(highways))
