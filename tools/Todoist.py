"""
title: Todoist
description: A tool to call Todoist API and interact with tasks
author: Martin Yankov
author_url: https://github.com/Lutherwaves
github: https://github.com/Lutherwaves/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.0.3
license: Apache 2.0
requirements: todoist-api-python
"""

from pydantic import BaseModel
from todoist_api_python.api_async import TodoistAPIAsync
from typing import Callable, Awaitable, Any, Dict
import logging

logging.basicConfig(level=logging.DEBUG)


class Tools:
    class Valves(BaseModel):
        """
        Configuration model for the tool.

        Attributes:
            TODOIST_API_KEY (str): The API key for Todoist.
            PROJECT_ID (str): The default project ID to use.
        """

        TODOIST_API_KEY: str = ""
        PROJECT_ID: str = ""

    def __init__(self):
        """
        Initializes the tool with default valves.
        """
        self.valves = self.Valves()

    async def list_tasks(
        self,
        filters: Dict[str, str] = {},
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Lists tasks based on the provided filters.
        Note, all dates can be in any of:
            - Specific date: 10/5/2022, Oct 5th 2022
            - Specific date and time: 10/5/2022 5pm, Oct 5th 5pm
            - Relative date: today, tomorrow, yesterday, 3 days (dated in the next 3 days), -3 days (dated in the past 3 days)
            - Days of the week: Monday, Tuesday, Sunday
        Args:
            filters (Dict[str, str]): Filters to apply. Supported filters include:
                - date: Specific date such as Jan 3, next week
                - date_before: Before specific date such as Jan 3, next week
                - date_after: After specific date such as Jan 3, next week, this sunday
                - priority: Tasks with a specific priority (p1-p4) p1 is highest.
                - labels: Tasks with specific labels (comma-separated).
                - project_name: Tasks in a specific project name.
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            str: A string representation of the filtered tasks.
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
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            # Construct filter string
            filter_str = ""
            if "date" in filters:
                filter_str += f"date: {filters['date']}"

            if "date_before" in filters:
                if filter_str:
                    filter_str += " & "
                filter_str += f"date before: {filters['date']}"

            if "date_after" in filters:
                if filter_str:
                    filter_str += " & "

                filter_str += f"date after: {filters['date']}"

            if "priority" in filters:
                if filter_str:
                    filter_str += " & "
                filter_str += f"{filters['priority']}"

            if "labels" in filters:
                if filter_str:
                    filter_str += " & "
                labels = filters["labels"].split(",")
                filter_str += " | ".join([f"@{label.strip()}" for label in labels])

            if "project_name" in filters:
                if filter_str:
                    filter_str += " & "
                project_name = filters["project_name"]
                filter_str += f"#{project_name}"

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Fetching tasks with filter: {filter_str}",
                        "done": False,
                    },
                }
            )
            if not filter_str:
                filter_str = "date before: next week"
            tasks = await api.get_tasks(filter=filter_str)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Tasks fetched", "done": True},
                }
            )
            return f"Tasks: {tasks}"
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return [f"Error getting tasks: {error}"]
