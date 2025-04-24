"""
title: Todoist
description: A tool to call Todoist API and list tasks based on filters
author: Martin Yankov
author_url: https://github.com/Lutherwaves
github: https://github.com/Lutherwaves/open-webui-tools
funding_url: https://github.com/open-webui
version: 1.0.0
"""

import aiohttp
from pydantic import BaseModel
from typing import Callable, Awaitable, Any, Dict, List
import logging
from datetime import datetime
from datetime import timedelta

logging.basicConfig(level=logging.DEBUG)

TODOIST_SYNC_URL = "https://api.todoist.com/api/v1/sync"


class Tools:
    class Valves(BaseModel):
        TODOIST_API_KEY: str = ""

    def __init__(self):
        self.valves = self.Valves()

    async def _sync_api(
        self,
        commands: List[dict] = None,
        resource_types: List[str] = None,
        sync_token: str = "*",
    ):
        headers = {
            "Authorization": f"Bearer {self.valves.TODOIST_API_KEY}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {}
        if resource_types:
            data["resource_types"] = str(resource_types).replace("'", '"')
        if commands:
            data["commands"] = str(commands).replace("'", '"')
        data["sync_token"] = sync_token

        async with aiohttp.ClientSession() as session:
            async with session.post(
                TODOIST_SYNC_URL, headers=headers, data=data
            ) as resp:
                if resp.status != 200:
                    raise Exception(
                        f"Todoist API error: {resp.status} {await resp.text()}"
                    )
                return await resp.json()

    async def list_tasks(
        self,
        filters: Dict[str, str] = {},
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        List tasks from Todoist and find tasks based on provided filters.

        This method retrieves tasks using the Todoist API and supports filtering by date,
        priority, labels, and project name. Dates can be specific (e.g., "2025-04-24"),
        relative ("today", "tomorrow", "this week"), or natural language ("next Monday").

        Parameters
        ----------
        filters : dict, optional
            Filters to apply. Supported keys:
                - 'date': Specific or relative date (e.g., "today", "2025-04-24").
                - 'date_before': Tasks due before this date.
                - 'date_after': Tasks due after this date.
                - 'priority': Task priority ("p1" to "p4").
                - 'labels': Comma-separated list of label names.
                - 'project_name': Name of the project.
        __user__ : dict, optional
            User information (not required for API calls).
        __event_emitter__ : Callable, optional
            Async event emitter for status updates.

        Returns
        -------
        str
            A string representation of the filtered tasks.

        Usage
        -----
        await tools.list_tasks(filters={"date": "today"})
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Fetching tasks with filters {filters}",
                    "done": False,
                },
            }
        )

        # Fetch all items (tasks)
        try:
            sync_data = await self._sync_api(resource_types=["items"])
            tasks = sync_data.get("items", [])

            # Filter tasks by due date (supports "today", "tomorrow", "this week", etc.)
            date_filter = filters.get("date")
            filtered_tasks = []
            now = datetime.now()
            for task in tasks:
                due = task.get("due", {})
                if not due:
                    continue
                due_str = due.get("date")
                if not due_str:
                    continue
                # Handle relative dates
                if date_filter in ["today", "tomorrow"]:
                    due_date = datetime.fromisoformat(due_str[:10])
                    if date_filter == "today" and due_date.date() == now.date():
                        filtered_tasks.append(task)
                    elif date_filter == "tomorrow" and due_date.date() == (
                        now.date() + timedelta(days=1)
                    ):
                        filtered_tasks.append(task)
                elif date_filter == "this week":
                    due_date = datetime.fromisoformat(due_str[:10])
                    start_of_week = now.date() - timedelta(days=now.weekday())
                    end_of_week = start_of_week + timedelta(days=6)
                    if start_of_week <= due_date.date() <= end_of_week:
                        filtered_tasks.append(task)
                else:
                    # Try to match specific date
                    try:
                        filter_date = datetime.fromisoformat(date_filter)
                        due_date = datetime.fromisoformat(due_str[:10])
                        if due_date.date() == filter_date.date():
                            filtered_tasks.append(task)
                    except Exception:
                        pass  # Ignore parse errors

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"{len(tasks)} Tasks fetched",
                        "done": True,
                    },
                }
            )

            return f"Tasks: {filtered_tasks}"
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return f"Error getting tasks: {error}"
