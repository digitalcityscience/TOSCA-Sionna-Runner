# TOSCA-Sionna Runner

Runs [Sionna RT](https://nvlabs.github.io/sionna/rt/) on your own GPU for the
TOSCA "Radio planning" panel (`sionna` feature in TOSCA-2). It is a small
FastAPI service on the user's machine (not in Docker) with its own Python
environment; it needs only this folder, not the rest of TOSCA.

How it works: the TOSCA site serves the building data (its Sionna CityJSON
endpoint), the browser sends the area and that endpoint to this runner on
port 8770, the runner downloads the CityJSON tiles, builds the Sionna scene
and computes on the local GPU. Results go straight back to the browser.

Install [uv](https://docs.astral.sh/uv/), then:

```
cd TOSCA-Sionna-Runner
cp .env.example .env   # and edit it
uv run --env-file .env uvicorn app:app --host 127.0.0.1 --port 8770
uv run --env-file .env pytest   # no GPU needed; without .env: uv run pytest
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Mitsuba variant, GPU yes/no, working files on disk |
| `GET /models` | Parameter schema and defaults (TOSCA runner contract) |
| `GET /scenes` | Scenes already on this machine: the sample data folder and Sionna's built-in scenes |
| `POST /scenes/local` | Load one of them by its `id` (sample areas also take `tile_source`) |
| `POST /scenes` | Upload a CityJSON file, or a Mitsuba scene as one `.zip` |
| `GET /scenes/{id}/local` | Map position → scene metres, with the roof or ground height there |
| `POST /scenes/{id}/run` | Parameters + transmitters/receivers → radio map, paths (rendered PNG only for scenes without georeference) |
| `POST /scenes/area` | `{"bbox": [west, south, east, north], "buffer_m": 300, "tile_source": "https://…"}` (degrees) → a scene of the Hamburg LoD3 buildings in the area plus the buffer, from the Sionna CityJSON endpoint `tile_source` (default `SIONNA_TILE_SOURCE`); the radio map covers only the area |
| `DELETE /scenes` | Close the open scene and delete all working files |

The scene list starts with four Hamburg sample areas (`SAMPLE_AREAS` in `app.py`:
centre, west, east, rural), opened like a drawn area from the LoD3 tiles.
Further sample data is read from `data/` next to `app.py` (set `SIONNA_DATA_DIR` for
another folder): every `*.city.json(l)` is a CityJSON tile, every `*.xml` a
Mitsuba scene.

A CityJSON scene has no terrain: every building is set on flat ground at z = 0,
so heights are metres above ground, in the scene and on the map.

Parameters mirror the Sionna RT desktop GUI (`sionna-rt-gui`): TX/RX array,
radio map, paths. `frequency_ghz` is added; the GUI is fixed at 3.5 GHz.

Settings (environment, or a `.env` file as in `.env.example`):

- `SIONNA_RUNNER_ORIGINS`: comma-separated web origins allowed to call the
  runner. Default `http://localhost:5173,http://127.0.0.1:5173`.
- `SIONNA_RUNNER_TOKEN`: if set, every request needs `Authorization: Bearer <token>`.
  The web panel does not send one, so leave it unset for the panel.
- `SIONNA_WORK_DIR`: where uploaded and converted scenes go (default: the
  system temp folder). The runner keeps only the open scene in
  `<dir>/sionna-runner`, empties it on start and stop, and removes the
  `sionna-runner-*` folders older versions left behind. Results are never
  written to disk; they go to the browser.
- `SIONNA_TILE_SOURCE`: the Sionna CityJSON endpoint used when the TOSCA site
  sends none (its `VITE_SIONNA_CITYJSON_URL`): a LoD3 CityJSONSeq collection
  with `metadata.json` and `tiles/<name>.city.jsonl`. No default.
- `SIONNA_TILE_CACHE`: where downloaded tiles stay, one folder per endpoint
  (default `~/.cache/tosca-sionna-runner/tiles`). A tile is downloaded once
  (up to ~90 MB); delete the folder to fetch a newer release.

## Hosted TOSCA site, your own GPU

The panel calls the runner from the browser, so a TOSCA site on a server
(for example `https://tosca.example.org`) computes on the GPU of the computer
the visitor sits at. Nothing runs on the server. On that computer:

```
cd TOSCA-Sionna-Runner
$env:SIONNA_RUNNER_ORIGINS = "https://tosca.example.org"   # PowerShell
export SIONNA_RUNNER_ORIGINS="https://tosca.example.org"   # bash
uv run uvicorn app:app --host 127.0.0.1 --port 8770
```

When the runner cannot be reached, the panel shows this command with the
site's own address filled in.

- The runner checks the `Origin` of every request itself, so no other web
  site can use the GPU. Bind it to `127.0.0.1` only.
- Chrome and Edge treat an https site calling `http://127.0.0.1` as Local
  Network Access: the panel's requests carry `targetAddressSpace`
  (`loopback`, or `local` for a LAN address), the browser asks the user once,
  and the runner answers the preflight with
  `Access-Control-Allow-Private-Network: true`. A blocked prompt is undone in
  the site settings.
- A runner on another machine in the LAN (`VITE_SIONNA_RUNNER_URL` with a
  private address) works the same way, but then the runner is open to that
  network for clients without an `Origin` (scripts). Use it only in a trusted
  network; the panel cannot send `SIONNA_RUNNER_TOKEN`.

Limits: one loaded scene at a time; planar radio map only (no terrain drape);
one result per run (no progressive accumulation). ITU materials are undefined
below 1 GHz for some types, so e.g. 600 MHz fails with a clear error; walls
(brick) end at 40 GHz. The ground is concrete (1 to 100 GHz), as ITU's ground
types only cover 1 to 10 GHz.

## License and attribution

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Built on
NVIDIA's [Sionna RT](https://github.com/NVlabs/sionna-rt) (library) and
[Sionna RT GUI](https://github.com/NVlabs/sionna-rt-gui) (desktop application,
whose parameter names and defaults the runner follows), both Apache-2.0,
Copyright (c) NVIDIA CORPORATION & AFFILIATES.
