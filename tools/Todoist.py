"""
title: Todoist
description: A tool to call Todoist API and interact with tasks
author: Martin Yankov
author_url: https://github.com/Lutherwaves
github: https://github.com/Lutherwaves/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.0.2
license: Apache 2.0
requirements: todoist-api-python
"""

from pydantic import BaseModel
from todoist_api_python.api_async import TodoistAPIAsync
from typing import Callable, Awaitable, Any, Dict
import logging
from datetime import datetime

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

        Args:
            filters (Dict[str, str]): Filters to apply. Supported filters include:
                - date: Tasks added on a specific date (YYYY-MM-DD).
                - due_date: Tasks due on a specific date (YYYY-MM-DD).
                - priority: Tasks with a specific priority (1-4).
                - labels: Tasks with specific labels (comma-separated).
                - project_id: Tasks in a specific project (ID).
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            str: A string representation of the filtered tasks.

        Notes:
            For more information on filters, see https://www.todoist.com/help/articles/introduction-to-filters-V98wIH#h_01J0J8ZDRBB65D2JTJ0RP0YCKB
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Fetching tasks", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            # Construct filter string
            filter_str = ""
            if "date" in filters:
                filter_str += (
                    f"added before {filters['date']} or added after {filters['date']}"
                )
            if "due_date" in filters:
                if filter_str:
                    filter_str += " and "
                filter_str += f"due before {filters['due_date']} or due after {filters['due_date']}"
            if "priority" in filters:
                if filter_str:
                    filter_str += " and "
                filter_str += f"priority {filters['priority']}"
            if "labels" in filters:
                if filter_str:
                    filter_str += " and "
                filter_str += f"@{filters['labels'].replace(',', ' @')}"
            if "project_id" in filters or self.valves.PROJECT_ID:
                if filter_str:
                    filter_str += " and "
                project_id = filters.get("project_id", self.valves.PROJECT_ID)
                filter_str += f"project_id({project_id})"

            tasks = await api.get_tasks(filter_str)
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

    async def get_task(
        self,
        task_id: str,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> dict:
        """
        Retrieves a specific task by its ID.

        Args:
            task_id (str): The ID of the task to retrieve.
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            dict: The task object if found, otherwise an error message.

        Raises:
            Exception: If there is an error fetching the task.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": f"Fetching task {task_id}", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            task = await api.get_task(task_id)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Task {task_id} fetched", "done": True},
                }
            )
            return task
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return {"error": f"Failed to get task {task_id}: {error}"}

    async def complete_task(
        self,
        task_id: str,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Marks a task as completed.

        Args:
            task_id (str): The ID of the task to complete.
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            bool: True if the task was completed successfully, otherwise False.

        Raises:
            Exception: If there is an error completing the task.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": f"Completing task {task_id}", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            result = await api.close_task(task_id)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Task {task_id} completed", "done": True},
                }
            )
            return result
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return False

    async def delete_task(
        self,
        task_id: str,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Deletes a task.

        Args:
            task_id (str): The ID of the task to delete.
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            bool: True if the task was deleted successfully, otherwise False.

        Raises:
            Exception: If there is an error deleting the task.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": f"Deleting task {task_id}", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            result = await api.delete_task(task_id)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Task {task_id} deleted", "done": True},
                }
            )
            return result
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return False

    async def edit_task(
        self,
        task_id: str,
        content: str = None,
        due_string: str = None,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Edits a task's content or due date.

        Args:
            task_id (str): The ID of the task to edit.
            content (str): New content for the task (optional).
            due_string (str): New due date string for the task (optional).
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            bool: True if the task was edited successfully, otherwise False.

        Raises:
            Exception: If there is an error editing the task.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": f"Editing task {task_id}", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            data = {}
            if content:
                data["content"] = content
            if due_string:
                data["due_string"] = due_string
            result = await api.update_task(task_id, data)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Task {task_id} updated", "done": True},
                }
            )
            return result
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return False

    async def get_free_time_today(
        self,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Calculates the free time available today.

        Args:
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            str: A string describing the free time today.

        Raises:
            Exception: If there is an error calculating the free time.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Calculating free time today", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            tasks = await api.get_tasks()
            today = datetime.now().date()
            busy_time = 0

            for task in tasks:
                if task.get("due", {}).get(
                        "date") == today.strftime("%Y-%m-%d"):
                    if task.get("due", {}).get("time"):
                        start_time = datetime.strptime(
                            task["due"]["time"], "%H:%M"
                        ).time()
                        end_time = datetime.strptime(
                            task["due"]["time"], "%H:%M"
                        ).time()
                        duration = (
                            datetime.combine(today, end_time)
                            - datetime.combine(today, start_time)
                        ).total_seconds() / 3600
                        busy_time += duration

            free_time = 24 - busy_time
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Free time calculated", "done": True},
                }
            )
            return f"Free time today: {free_time} hours"
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return f"Error calculating free time: {error}"

    async def get_busy_time_today(
        self,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Calculates the busy time allocated today and the ratio between projects.

        Args:
            __user__ (dict): User information (optional).
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Event emitter function (optional).

        Returns:
            str: A string describing the busy time today and project ratios.

        Raises:
            Exception: If there is an error calculating the busy time.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Calculating busy time today", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            tasks = await api.get_tasks()
            today = datetime.now().date()
            busy_time = 0
            project_ratios = {}

            for task in tasks:
                if task.get("due", {}).get(
                        "date") == today.strftime("%Y-%m-%d"):
                    if task.get("due", {}).get("time"):
                        start_time = datetime.strptime(
                            task["due"]["time"], "%H:%M"
                        ).time()
                        end_time = datetime.strptime(
                            task["due"]["time"], "%H:%M"
                        ).time()
                        duration = (
                            datetime.combine(today, end_time)
                            - datetime.combine(today, start_time)
                        ).total_seconds() / 3600
                        busy_time += duration

                        project_id = task["project_id"]
                        if project_id in project_ratios:
                            project_ratios[project_id] += duration
                        else:
                            project_ratios[project_id] = duration

            project_ratios_str = ", ".join(
                f"Project {project_id}: {ratio} hours"
                for project_id, ratio in project_ratios.items()
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Busy time calculated", "done": True},
                }
            )
            return f"Busy time today: {busy_time} hours. Project ratios: {project_ratios_str}"
        except Exception as error:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Error: {error}", "done": True},
                }
            )
            return f"Error calculating busy time: {error}"
