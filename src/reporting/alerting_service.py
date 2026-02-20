from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models.review_schemas import ViolationRecord
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


class AlertLevel:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertingService:
    """
    Service for sending alerts on critical violations.

    Supports multiple channels: email, Slack, webhook, etc.
    """

    def __init__(self, alert_log_path: Path):
        self.alert_log_path = alert_log_path
        self.alert_log: List[dict] = []
        self.load_log()

    def load_log(self) -> None:
        """Load alert log from file."""
        if self.alert_log_path.exists():
            try:
                with self.alert_log_path.open("r", encoding="utf-8") as f:
                    self.alert_log = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.alert_log = []

    def save_log(self) -> None:
        """Save alert log to file."""
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alert_log_path.open("w", encoding="utf-8") as f:
            json.dump(self.alert_log, f, indent=2, ensure_ascii=False, default=str)

    def determine_alert_level(self, violation: ViolationRecord) -> str:
        """
        Determine alert level based on violation characteristics.

        Args:
            violation: Violation record

        Returns:
            Alert level: critical, high, medium, low
        """
        # Check severity in violation metadata
        severity = getattr(violation, "severity", None) or "medium"

        # High confidence violations are more urgent
        if violation.confidence >= 0.9:
            if severity == "critical":
                return AlertLevel.CRITICAL
            elif severity == "high":
                return AlertLevel.HIGH

        # Low confidence but critical severity
        if severity == "critical":
            return AlertLevel.HIGH

        # Default based on confidence
        if violation.confidence >= 0.8:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW

    def send_alert(
        self,
        violation: ViolationRecord,
        alert_level: str,
        channels: Optional[List[str]] = None,
    ) -> None:
        """
        Send alert for a violation.

        Args:
            violation: Violation record
            alert_level: Alert level
            channels: List of channels (email, slack, webhook) - None = log only
        """
        if channels is None:
            channels = ["log"]  # Default: log only

        alert = {
            "violation_id": violation.record_id,
            "alert_level": alert_level,
            "violation_details": {
                "rule_id": violation.cited_rule_ids[0] if violation.cited_rule_ids else None,
                "explanation": violation.explanation,
                "confidence": violation.confidence,
            },
            "channels": channels,
            "sent_at": datetime.now().isoformat(),
            "status": "sent",
        }

        # Log alert
        self.alert_log.append(alert)
        self.save_log()

        # In production, would send via actual channels
        if "email" in channels:
            logger.info("Email alert sent for violation %s", violation.record_id)
        if "slack" in channels:
            logger.info("Slack alert sent for violation %s", violation.record_id)
        if "webhook" in channels:
            logger.info("Webhook alert sent for violation %s", violation.record_id)

        logger.info(
            "Alert sent: %s violation %s (level: %s)",
            alert_level.upper(),
            violation.record_id,
            alert_level,
        )

    def check_and_alert_critical_violations(
        self,
        violations: List[ViolationRecord],
        alert_threshold: str = AlertLevel.HIGH,
    ) -> int:
        """
        Check violations and send alerts for critical ones.

        Args:
            violations: List of violations
            alert_threshold: Minimum alert level to send (critical, high, medium, low)

        Returns:
            Number of alerts sent
        """
        alert_levels = [AlertLevel.CRITICAL, AlertLevel.HIGH, AlertLevel.MEDIUM, AlertLevel.LOW]
        threshold_idx = alert_levels.index(alert_threshold)

        alerts_sent = 0

        for violation in violations:
            alert_level = self.determine_alert_level(violation)
            level_idx = alert_levels.index(alert_level)

            if level_idx <= threshold_idx:
                self.send_alert(violation, alert_level)
                alerts_sent += 1

        logger.info("Sent %d alerts for violations (threshold: %s)", alerts_sent, alert_threshold)
        return alerts_sent


def get_alerting_service(alert_log_path: Path) -> AlertingService:
    """
    Get or create alerting service instance.
    """
    return AlertingService(alert_log_path)
