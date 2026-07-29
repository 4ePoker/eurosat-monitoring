"""Named drift scenarios.

A scenario is a triple, and each slot maps to one drift type:

    class_weights  which images arrive        -> label drift
    transform      how the pixels change      -> data drift
    relabel        what they are called       -> concept drift

Keeping them separable is what lets the monitor say which kind it found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from monitoring.eurosat import EUROSAT_CLASSES
from monitoring.drift import transforms as T

Transform = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class Scenario:
    """One reproducible way for production to stop looking like training."""

    name: str
    description: str
    #: The drift types this scenario actually exercises, for the report header.
    drift_types: tuple[str, ...]
    #: Pixel-level shift. Identity means "the images themselves are unchanged".
    transform: Transform = T.identity
    #: Relative sampling weight per class. None means "same mix as the source
    #: split", i.e. no prior shift.
    class_weights: dict[str, float] | None = None
    #: Maps a true class name to the class the *new* labelling rule calls it.
    #: None means the labelling rule is unchanged.
    relabel: dict[str, str] | None = None
    #: Free-form notes that end up in the batch metadata.
    notes: str = ""

    def weights_vector(self) -> np.ndarray | None:
        if self.class_weights is None:
            return None
        w = np.array([self.class_weights.get(c, 0.0) for c in EUROSAT_CLASSES], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError(f"Scenario {self.name}: class weights sum to zero.")
        return w / w.sum()

    def observed_label(self, true_class: str) -> str:
        return self.relabel.get(true_class, true_class) if self.relabel else true_class


# --------------------------------------------------------------------------
# NO DRIFT -- the control
# --------------------------------------------------------------------------

CONTROL = Scenario(
    name="none",
    description="Production looks exactly like training. Nothing changed.",
    drift_types=(),
    notes=(
        "The most important scenario in the file. Any detector can find drift "
        "when you shout at it; the useful question is how often it cries wolf "
        "on a quiet day. This batch calibrates the false-positive rate."
    ),
)


# --------------------------------------------------------------------------
# DATA DRIFT -- P(X) moves, the labelling rule does not
# --------------------------------------------------------------------------

AUTUMN = Scenario(
    name="autumn",
    description="Same region, months later: senescent vegetation, harvested fields.",
    drift_types=("data",),
    transform=T.seasonal_shift,
    notes=(
        "The model was trained on whatever seasonal mix EuroSAT happened to "
        "contain. Deploy it in October and the vegetated classes change colour "
        "while their correct labels stay put. Expect confusion between "
        "AnnualCrop, Pasture and HerbaceousVegetation."
    ),
)

NEW_SENSOR = Scenario(
    name="new_sensor",
    description="The imagery provider changed: different calibration, coarser pixels, more noise.",
    drift_types=("data",),
    transform=T.compose(T.radiometric_shift, T.resolution_loss, T.sensor_noise),
    notes=(
        "Migrating Sentinel-2 -> Landsat-9, or L1C -> L2A, moves several axes "
        "at once. This is the most common hard drift in Earth observation and "
        "the one with a clear business trigger: a procurement decision, not a "
        "natural process. It is therefore predictable *in advance*, which is "
        "an argument for a pre-deployment shadow test rather than waiting for "
        "the monitor to notice."
    ),
)

HAZE = Scenario(
    name="haze",
    description="Thin cloud and aerosol: a transient, weather-driven shift.",
    drift_types=("data",),
    transform=T.atmospheric_haze,
    notes=(
        "Comes and goes. The reason a trigger needs persistence (N consecutive "
        "windows) rather than a single threshold crossing -- otherwise every "
        "hazy week starts a retraining job."
    ),
)

PIPELINE_CHANGE = Scenario(
    name="recompress",
    description="Upstream ETL lowered JPEG quality. The world did not change; your data did.",
    drift_types=("data",),
    transform=T.jpeg_recompress,
    notes=(
        "The response here is to fix the pipeline, not to retrain. Retraining "
        "on corrupted inputs teaches the model to be good at corruption and "
        "bakes the bug into the weights. This scenario exists to make the case "
        "that an automated retrain trigger needs a human-readable diagnosis "
        "attached, not just a number."
    ),
)


# --------------------------------------------------------------------------
# LABEL DRIFT -- P(y) moves, pixels untouched
# --------------------------------------------------------------------------

COASTAL = Scenario(
    name="label_shift_coastal",
    description="Deployment moves to a coastal AOI: mostly water and forest, no industry.",
    drift_types=("label",),
    class_weights={
        "SeaLake": 6.0,
        "River": 2.5,
        "Forest": 2.0,
        "HerbaceousVegetation": 1.0,
        "Pasture": 0.8,
        "AnnualCrop": 0.5,
        "PermanentCrop": 0.3,
        "Residential": 0.4,
        "Highway": 0.2,
        "Industrial": 0.05,
    },
    notes=(
        "Not one pixel is altered -- only which images arrive. A softmax head "
        "trained under a near-uniform prior is miscalibrated under this one, "
        "so confidences degrade even where the features are perfect. The fix "
        "is prior correction (adjust the logits by log of the new/old prior), "
        "which costs nothing and needs no retraining. Worth knowing before "
        "spending a GPU-week."
    ),
)

MONOCULTURE = Scenario(
    name="label_shift_extreme",
    description="An almost single-class stream: 80% water, everything else a rounding error.",
    drift_types=("label",),
    class_weights={
        "SeaLake": 60.0,
        "River": 20.0,
        "Forest": 12.0,
        "HerbaceousVegetation": 3.0,
        "Pasture": 2.0,
        "AnnualCrop": 1.0,
        "PermanentCrop": 0.7,
        "Residential": 0.7,
        "Highway": 0.4,
        "Industrial": 0.2,
    },
    notes=(
        "The coastal scenario did no measurable damage (-0.8 points) while "
        "moving input statistics a long way -- a cheap alarm for an expensive "
        "non-problem. This one exists to find where that stops being true: how "
        "extreme must a harmless prior shift get before it crosses whatever "
        "threshold the detector uses? A threshold calibrated only against the "
        "mild case is calibrated by luck."
    ),
)

URBANISATION = Scenario(
    name="label_shift_urban",
    description="A growing metro area: built-up classes expand at the expense of cropland.",
    drift_types=("label",),
    class_weights={
        "Residential": 4.0,
        "Industrial": 3.0,
        "Highway": 2.5,
        "River": 1.0,
        "AnnualCrop": 0.6,
        "PermanentCrop": 0.5,
        "Pasture": 0.4,
        "HerbaceousVegetation": 0.8,
        "Forest": 0.7,
        "SeaLake": 0.5,
    },
    notes=(
        "A slow, genuine change in the world rather than a sampling accident. "
        "Distinguishing the two matters: sampling changes are reversible and "
        "call for reweighting; real change accumulates and eventually calls "
        "for new training data."
    ),
)


# --------------------------------------------------------------------------
# CONCEPT DRIFT -- the labelling rule itself moves
# --------------------------------------------------------------------------

REDEFINE_CROPS = Scenario(
    name="concept_crop_merge",
    description="The client redefines the taxonomy: PermanentCrop is now reported as AnnualCrop.",
    drift_types=("concept",),
    relabel={"PermanentCrop": "AnnualCrop"},
    notes=(
        "Identical pixels, different correct answer. No input-distribution "
        "monitor can see this -- P(X) is untouched -- which is the whole "
        "lesson: concept drift arrives as a ticket from the business, not as "
        "an alarm from the monitor. The only automated defence is measuring "
        "accuracy on freshly labelled data."
    ),
)

SOLAR_FARMS = Scenario(
    name="concept_solar_farms",
    description="Solar farms are built on pasture; the correct label becomes Industrial.",
    drift_types=("concept", "data"),
    relabel={"Pasture": "Industrial"},
    transform=T.radiometric_shift,
    notes=(
        "The realistic version: the land use genuinely changed, so the pixels "
        "changed *and* the label changed. Here the input monitor does fire, "
        "but if you retrain on the old labels you actively make things worse. "
        "Detection is necessary and nowhere near sufficient."
    ),
)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        CONTROL,
        AUTUMN,
        NEW_SENSOR,
        HAZE,
        PIPELINE_CHANGE,
        COASTAL,
        MONOCULTURE,
        URBANISATION,
        REDEFINE_CROPS,
        SOLAR_FARMS,
    )
}


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Available: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]
