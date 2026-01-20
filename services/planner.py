from __future__ import annotations

from datetime import date, timedelta
from typing import List, Dict, Any

from config import settings


def parse_topics(raw: str) -> List[Dict[str, Any]]:
    topics: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        name, difficulty, priority = parts[0], parts[1], parts[2]
        topics.append(
            {
                "name": name,
                "difficulty": float(difficulty or 1),
                "priority": float(priority or 1),
            }
        )
    return topics


def build_plan(exam_date: date, daily_hours: float, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    today = date.today()
    days_left = (exam_date - today).days
    days_left = max(days_left, 1)

    weighted = sorted(topics, key=lambda t: (t["priority"], t["difficulty"]), reverse=True)
    total_weight = sum(t["priority"] + t["difficulty"] for t in weighted) or 1

    schedule: List[Dict[str, Any]] = []
    current_day = today
    idx = 0
    for day_offset in range(days_left):
        current_day = today + timedelta(days=day_offset)
        tasks: List[str] = []

        if day_offset % settings.MOCK_TEST_EVERY_N_DAYS == 0 and day_offset != 0:
            tasks.append("Mock test (1h)")
        if day_offset % settings.REVISION_EVERY_N_DAYS == 0 and day_offset != 0:
            tasks.append("Revision (1h)")

        hours_remaining = daily_hours
        while hours_remaining > 0 and weighted:
            topic = weighted[idx % len(weighted)]
            share = (topic["priority"] + topic["difficulty"]) / total_weight
            allocated = max(round(share * daily_hours, 1), 0.5)
            allocated = min(allocated, hours_remaining)
            tasks.append(f"{topic['name']} ({allocated}h)")
            hours_remaining -= allocated
            idx += 1
        schedule.append({"day": current_day, "tasks": tasks})
    return schedule
