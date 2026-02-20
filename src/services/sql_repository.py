from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sqlalchemy import Column, Float, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base

from src.config import DATABASE_URL
from src.models.schemas import ExtractedRule, RuleComplexity
from src.utils.logging_config import get_logger


logger = get_logger(__name__)

Base = declarative_base()


class ComplianceRule(Base):
    """SQLAlchemy model for simple compliance rules."""

    __tablename__ = "compliance_rules"

    rule_id = Column(String(255), primary_key=True)
    condition = Column(Text, nullable=False)
    threshold = Column(Float, nullable=True)
    required_action = Column(Text, nullable=False)
    rule_text = Column(Text, nullable=False)
    source_pdf = Column(String(512), nullable=True)
    category = Column(String(255), nullable=True)
    rule_complexity = Column(String(50), nullable=True)


def get_rules_engine():
    """
    Create and return a SQLAlchemy engine using the configured DATABASE_URL.

    Supports SQLite, PostgreSQL, MySQL, and any SQLAlchemy-compatible backend.
    The DATABASE_URL is read from the environment variable or defaults to a local SQLite file.
    """
    engine = create_engine(DATABASE_URL, echo=False, future=True)
    return engine


def init_db(db_path: Path | None = None) -> None:
    """
    Initialize the database schema.

    Args:
        db_path: Optional path for backwards compatibility. If None, uses DATABASE_URL from config.
    """
    if db_path:
        engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    else:
        engine = get_rules_engine()
    Base.metadata.create_all(engine)
    logger.info("Initialized database: %s", engine.url)


def persist_simple_rules(db_path: Path | None, rules: Iterable[ExtractedRule]) -> int:
    """
    Persist simple, deterministic rules into the SQL database.

    Args:
        db_path: Optional path for backwards compatibility. If None, uses DATABASE_URL from config.
        rules: Iterable of ExtractedRule objects to persist.

    Returns:
        Number of rules successfully persisted.
    """
    if db_path:
        engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    else:
        engine = get_rules_engine()
    
    inserted = 0
    with Session(engine) as session:
        for rule in rules:
            if rule.rule_complexity != RuleComplexity.SIMPLE:
                continue

            obj = ComplianceRule(
                rule_id=rule.rule_id,
                condition=rule.condition,
                threshold=rule.threshold,
                required_action=rule.required_action,
                rule_text=rule.rule_text,
                source_pdf=rule.source_pdf,
                category=rule.category,
            )
            session.merge(obj)
            inserted += 1

        session.commit()

    logger.info("Persisted %d simple rules into SQL database.", inserted)
    return inserted
