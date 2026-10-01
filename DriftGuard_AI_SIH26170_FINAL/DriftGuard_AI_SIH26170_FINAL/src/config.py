from dataclasses import dataclass

@dataclass(frozen=True)
class ParameterConfig:
    key: str
    label: str
    unit: str
    absolute_max: float
    safety_slope_per_hour: float
    contamination: float = 0.08

# IMPORTANT: These are PROTOTYPE DEMO THRESHOLDS ONLY.
# They are not official ISRO/device datasheet limits.
PARAMETERS = {
    "leakage": ParameterConfig(
        key="leakage",
        label="Leakage Current",
        unit="µA",
        absolute_max=50.0,
        safety_slope_per_hour=0.12,
    ),
    "iddq": ParameterConfig(
        key="iddq",
        label="Standby Current (IDDQ)",
        unit="mA",
        absolute_max=18.0,
        safety_slope_per_hour=0.035,
    ),
    "delay": ParameterConfig(
        key="delay",
        label="Propagation Delay",
        unit="ns",
        absolute_max=28.0,
        safety_slope_per_hour=0.045,
    ),
}

TIME_POINTS = [0, 24, 96, 168]
RANDOM_STATE = 42
