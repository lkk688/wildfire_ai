# Fire-spread scenarios and treatment ranking

`backend/firesim/` simulates fires on the layers of the [California data cache](DATA_CACHE.md) and uses them to rank
where vegetation treatment would protect the most buildings. Everything is numpy / scipy; no external fire model has
to be installed.

## Fire behaviour

| Part | Method |
|---|---|
| Fuels | LANDFIRE LF2025 Scott & Burgan 40 fuel models, canopy cover, height, base height and bulk density; LANDFIRE slope and aspect (30 m) |
| Surface fire | Rothermel (1972) with the Albini (1976) changes as in Andrews (2018), dynamic herbaceous curing, the effective-wind limit of Andrews et al. (2013); wind adjusted to midflame height by fuel depth or canopy (Albini & Baughman 1979) |
| Crown fire | Van Wagner (1977) initiation, Rothermel (1991) active spread (3.34 × fuel model 10), Scott & Reinhardt (2001) criteria |
| Shape | Wind and slope combined as vectors; elliptical spread with Anderson (1983) length-to-breadth ratio |
| Spread | Minimum travel time: Dijkstra on a 16-neighbour graph of 30 m cells, so one ignition gives the arrival time everywhere |
| Weather | Daily gridMET 2000–2025 of the area's 4 km cell, fire-weather days only (ERC at or above the cell's 90th percentile): wind speed and direction, minimum humidity (to 1-h moisture), 100-h moisture |

Checks: the spread rates and flame lengths of the 40 fuel models at 5 mi/h midflame wind match the Scott & Burgan
(2005) charts, and fuel model 10 matches Anderson (1982).

Not modelled: spotting, terrain-steered wind (wind is uniform over the area), wind changing during a fire, suppression,
fire inside built-up areas other than the house-to-house sensitivity below. Results are planning scenarios, not
forecasts.

## Buildings

A building counts as **exposed** when burning fuel with flames of 1.2 m (4 ft) or more comes within 30 m of its
footprint. Fuel treatment near a house therefore counts when it lowers the flames there, not only when it stops the
fire. A sensitivity metric lets exposed buildings ignite neighbours whose footprints are within 10 m, one hop every
30 minutes, until the burning period ends.

## Treatment ranking

1. **Units**: 90 m × 90 m blocks with burnable fuel within 500 m of a building.
2. **Planning fires**: many fires, each pairing a historical fire-weather day with an ignition point drawn from the
   density of past ignitions (NIFC InFORM). Every fire records the path it took from its ignition to each cell that
   exposed a building, so each cell gets credit for the building exposures that passed through it.
3. **Strategies**, all at the same treated area:

   | Strategy | How units are ranked |
   |---|---|
   | simulation | building exposures carried through the unit, per hectare |
   | simulation, iterative | the same, picked in batches (the smallest budget each); the planning fires are re-run on the treated landscape before each batch |
   | hybrid | simulation and exposure scores, each normalised, averaged |
   | exposure | burn probability × buildings within 40 m |
   | proximity | buildings within 40 m |
   | hazard | burn probability × flame length |
   | random | random order, three draws |

4. **Treatment**: surface fuels become the lightest model of their family (grass → GR1, grass-shrub and shrub → GS1,
   timber understory → TU1, litter → TL1, slash → TL3); in forest the canopy base height is raised to at least 3 m and
   the bulk density is cut by 40 %.
5. **Evaluation**: a separate set of fires (other ignitions, other days) is run before and after treatment. The result
   is the reduction in buildings exposed, paired by fire, with a bootstrap 90 % interval. A dry offshore-wind stress
   test is run as well.

## Running it

```bash
python -m backend.firesim.run --aoi san_bruno            # about 10 minutes on 28 cores
python scripts/firesim/figure_priorities.py --aoi san_bruno --out priorities.png
```

Outputs go to `<GEODATA_ROOT>/derived/firesim/<aoi>/`: `burn_probability.tif`, `mean_flame_length_m.tif`,
`path_credit.tif`, `units.tif`, `treatment_units_ranked.geojson` and `results.json`. To add an area, add its bounding
box to `AOIS` in `backend/firesim/landscape.py`.

## One fire, from the map or the agent

`GET /api/firesim/spread?lat=&lng=[&hours=6&wind_ms=&wind_from_deg=&rh_min=&fm100=]` and the agent tool
`simulate_fire_spread` run one fire from an ignition point (about 1 s). They return the burned area, hourly perimeters,
flame lengths, crown-fire share and buildings exposed. Weather that is not given defaults to the cell's fire-weather-day
conditions: 90th-percentile wind from the most common direction, 10th-percentile humidity and 100-h moisture.
