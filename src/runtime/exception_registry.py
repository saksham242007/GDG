from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class ExceptionRegistry:
    """
    Registry for approved exceptions and overrides.

    Allows storing exceptions that have been approved by compliance officers,
    so violations can be checked against this registry before flagging.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.exceptions: List[dict] = []
        self.load()

    def load(self) -> None:
        """Load exceptions from file."""
        if self.registry_path.exists():
            try:
                with self.registry_path.open("r", encoding="utf-8") as f:
                    self.exceptions = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load exception registry: %s", e)
                self.exceptions = []

    def save(self) -> None:
        """Save exceptions to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(self.exceptions, f, indent=2, ensure_ascii=False, default=str)

    def check_exception(
        self,
        rule_id: str,
        record_id: str,
        violation_type: str,
    ) -> Optional[dict]:
        """
        Check if an exception exists for a specific violation.

        Args:
            rule_id: Policy rule ID
            record_id: Record ID with violation
            violation_type: Type of violation

        Returns:
            Exception dict if found, None otherwise
        """
        for exc in self.exceptions:
            if (
                exc.get("rule_id") == rule_id
                and exc.get("record_id") == record_id
                and exc.get("violation_type") == violation_type
                and exc.get("status") == "approved"
            ):
                # Check if exception is still valid (not expired)
                expiry = exc.get("expires_at")
                if expiry:
                    expiry_date = datetime.fromisoformat(expiry)
                    if datetime.now() > expiry_date:
                        logger.info("Exception expired for %s/%s", rule_id, record_id)
                        return None
                return exc

        return None

    def add_exception(
        self,
        rule_id: str,
        record_id: str,
        violation_type: str,
        approved_by: str,
        reason: str,
        expires_at: Optional[str] = None,
    ) -> None:
        """
        Add a new approved exception.

        Args:
            rule_id: Policy rule ID
            record_id: Record ID
            violation_type: Type of violation
            approved_by: Approver ID/name
            reason: Reason for exception
            expires_at: Optional expiry date (ISO format)
        """
        exception = {
            "rule_id": rule_id,
            "record_id": record_id,
            "violation_type": violation_type,
            "status": "approved",
            "approved_by": approved_by,
            "reason": reason,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at,
        }

        self.exceptions.append(exception)
        self.save()

        logger.info("Exception added: %s/%s approved by %s", rule_id, record_id, approved_by)

    def filter_violations_with_exceptions(
        self,
        violations: List[dict],
    ) -> List[dict]:
        """
        Filter out violations that have approved exceptions.

        Args:
            violations: List of violation dicts with rule_id, record_id, etc.

        Returns:
            Filtered list of violations (exceptions removed)
        """
        filtered = []

        for violation in violations:
            rule_id = violation.get("rule_id") or violation.get("rule_id")
            record_id = violation.get("record_id") or violation.get("affected_records", [None])[0]
            violation_type = violation.get("violation_description", "")

            exception = self.check_exception(rule_id, str(record_id), violation_type)

            if exception:
                logger.info(
                    "Violation filtered (exception exists): %s/%s",
                    rule_id,
                    record_id,
                )
                continue

            filtered.append(violation)

        logger.info(
            "Filtered violations: %d original, %d after exception check",
            len(violations),
            len(filtered),
        )

        return filtered


def get_exception_registry(registry_path: Path) -> ExceptionRegistry:
    """
    Get or create exception registry instance.
    """
    return ExceptionRegistry(registry_path)
