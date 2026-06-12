"""Configuration constants and env-var loading for the Manpower Budget app."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Contribution rates keyed on "Κωδικός Κράτησης" ─────────────────────────
# If a code is present in this map, the matching rate is used for that employee.
# Otherwise NaN (employee flagged in Data Quality tab).
CONTRIB_RATE_MAP: dict[str, float] = {
    "40010": 0.2494,
    "40011": 0.2494,
    "40012": 0.2279,
    "40050": 0.2179,
    "40060": 0.2179,
    "40061": 0.2279,
    "40070": 0.2394,
    "40084": 0.1879,   # fixed +€30 add-on
    "40090": 0.0140,
    "40380": 0.1879,   # fixed +€30 add-on
    "40510": 0.1738,   # fixed +€25 add-on
    "40602": 0.1879,   # fixed +€30 add-on
    "40603": 0.1879,   # fixed +€30 add-on
}

# Fixed monthly add-ons per rate
FIXED_ADDON_30 = {0.1879}
FIXED_ADDON_25 = {0.1738}

# Under-25 discount (N.4583/2018 / e-ΕΦΚΑ Circular 28/2019):
# employer's main-pension contribution is subsidized by 6.66 pp.
# Fixed add-ons (€25 / €30) are NOT affected by this discount.
YOUTH_DISCOUNT_PP: float = 0.0666
YOUTH_AGE_THRESHOLD: int = 25

# Annual training cost by grade band
TRAINING_COST_BANDS = [
    (8,  25),     # grade < 8
    (9,  150),    # grade <= 9
    (13, 250),    # grade <= 13
    (18, 450),    # grade <= 18
    (23, 500),    # grade <= 23
]

# Meal allowance by ΚΑΡΤΑ ΣΙΤΙΣΗΣ flag
MEAL_ALLOWANCE_3 = 1488.0
MEAL_ALLOWANCE_4 = 744.0

# Bonus-month divisor (days → months)
MONTH_DIVISOR = 30.42
BONUS_FACTOR  = 0.04166   # 1/24 for pro-rating bonus allowances

# Columns that should be parsed as dates on load
DATE_CANDIDATES = [
    "Ημ/νία γέννησης", "Ημ/νία αποχώρησης", "Ημ/νία πρόσληψης",
    "Hiring Date", "Retire Date", "Date", "Date of Birth", "Hire Date",
]

# Dimension columns used for filtering + group-by
DIM_COLS = ["Company", "Division", "Department", "Job Property", "Cost Center"]


@dataclass(frozen=True)
class GraphCredentials:
    """Credentials for Microsoft Graph access."""
    tenant_id:     str
    client_id:     str
    client_secret: str
    file_url:      str

    def missing(self) -> list[str]:
        """Return the list of field names that are unset."""
        return [n for n, v in [
            ("SP_TENANT_ID",     self.tenant_id),
            ("SP_CLIENT_ID",     self.client_id),
            ("SP_CLIENT_SECRET", self.client_secret),
            ("SP_FILE_URL",      self.file_url),
        ] if not v]


def load_graph_creds_from_env(env_file: Optional[Path] = None) -> GraphCredentials:
    """Load Graph credentials from env vars (optionally reading a .env file first)."""
    if env_file and env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
        except ImportError:
            pass   # dotenv is optional in test environments
    return GraphCredentials(
        tenant_id     = os.environ.get("SP_TENANT_ID", ""),
        client_id     = os.environ.get("SP_CLIENT_ID", ""),
        client_secret = os.environ.get("SP_CLIENT_SECRET", ""),
        file_url      = os.environ.get(
            "SP_FILE_URL",
            "https://alumilcom-my.sharepoint.com/personal/hrservices_alumil_com"
            "/Documents/ManpowerBudget.xlsx"),
    )
