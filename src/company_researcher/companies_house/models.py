from datetime import date as Date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompaniesHouseModel(BaseModel):
    """Base model that retains new fields added by Companies House."""

    model_config = ConfigDict(extra="allow")


class CompanyProfile(CompaniesHouseModel):
    company_name: str
    company_number: str
    type: str
    company_status: str | None = None
    date_of_creation: Date | None = None
    date_of_cessation: Date | None = None
    sic_codes: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(default_factory=dict)
    registered_office_address: dict[str, Any] | None = None


class FilingHistoryItem(CompaniesHouseModel):
    transaction_id: str
    category: str
    date: Date
    description: str
    type: str
    action_date: Date | None = None
    barcode: str | None = None
    pages: int | None = None
    paper_filed: bool | None = None
    links: dict[str, str] = Field(default_factory=dict)


class FilingHistoryPage(CompaniesHouseModel):
    items: list[FilingHistoryItem]
    items_per_page: int
    start_index: int
    total_count: int


class FilingHistory(BaseModel):
    """Complete filing history assembled from one or more API pages."""

    items: list[FilingHistoryItem]
    total_count: int
