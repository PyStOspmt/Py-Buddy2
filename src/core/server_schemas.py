"""Pydantic request models for the HTTP API."""

from pydantic import BaseModel
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------
class ChatQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    domain: Optional[str] = None
    model: Optional[str] = None

class SetModelRequest(BaseModel):
    model: str

class BundleValidateRequest(BaseModel):
    product_ids: List[str]
    domain: Optional[str] = None

class ParetoRequest(BaseModel):
    budget_max: float = 12000.0
    min_performance: float = 80.0
    domain: Optional[str] = None
    category: Optional[str] = None

class SwitchDomainRequest(BaseModel):
    domain: str
    # R6: a domain preference belongs to a session, not to the process.
    # With session_id set, only that session's subsequent chats resolve to
    # this domain; without it (admin-only) the global default changes.
    session_id: Optional[str] = None

class RegisterDomainRequest(BaseModel):
    domain_id: str
    db_path: str
    name: Optional[str] = None
    description: Optional[str] = None
    async_init: bool = False
    llm_assist: bool = False  # opt-in: model may propose, data verifies

class InitDomainRequest(BaseModel):
    domain: str


class ConstraintStatusRequest(BaseModel):
    rule_id: str
    status: Literal["verified", "candidate"]
