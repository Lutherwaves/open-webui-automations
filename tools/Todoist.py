"""
title: Todoist API Tool
description: A comprehensive tool to call todoist API
author: Martin Yankov
author_url: https://github.com/Lutherwaves
github: https://github.com/Lutherwaves/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.0.1
license: Apache 2.0
requirements: todoist-api-python
"""

from pydantic import BaseModel
from todoist_api_python.api_async import TodoistAPIAsync
from typing import Callable, Awaitable, Any
import logging

logging.basicConfig(level=logging.DEBUG)


class Tools:
    class Valves(BaseModel):
        TODOIST_API_KEY: str = ""

    def __init__(self):
        self.valves = self.Valves()

    async def list_tasks(
        self,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Lists all tasks in your Todoist account.

        :return: A list of Task objects.
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Fetching tasks", "done": False},
            }
        )
        api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)
        try:
            tasks = await api.get_tasks()
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

        :param task_id: The ID of the task to retrieve.
        :return: The task object if found, otherwise an error message.
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

        :param task_id: The ID of the task to complete.
        :return: True if successful, otherwise False.
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

        :param task_id: The ID of the task to delete.
        :return: True if successful, otherwise False.
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

        :param task_id: The ID of the task to edit.
        :param content: New content for the task.
        :param due_string: New due date string for the task.
        :return: True if successful, otherwise False.
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
