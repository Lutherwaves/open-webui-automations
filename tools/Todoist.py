"""
title: Todoist API Tool
description: A comprehensive tool to call todoist API
author: Martin Yankov
author_url: https://github.com/Lutherwaves
github: https://github.com/Lutherwaves/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.0.2
license: Apache 2.0
requirements: todoist-api-python
"""

from pydantic import BaseModel, Field
from todoist_api_python.api_async import TodoistAPIAsync
from todoist_api_python.models import Task
from typing import Optional, Callable, Awaitable, Any
import logging

logging.basicConfig(level=logging.DEBUG)


class Tools:
    """
    A class to provide various tools for Todoist API interactions.
    """

    class Valves(BaseModel):
        TODOIST_API_KEY: str = Field(
            ..., description="The user's Todoist API key"
        )

    def __init__(self, api_key: str):
        self.citation = True
        self.valves = self.Valves(TODOIST_API_KEY=api_key)
        self.api = TodoistAPIAsync(self.valves.TODOIST_API_KEY)

    async def list_tasks(
        self,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> list[Task]:
        """
        Lists all tasks in your Todoist account.

        :return: A list of Task objects.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Fetching tasks", "done": False},
                }
            )
        try:
            tasks = await self.api.get_tasks()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Tasks fetched", "done": True},
                    }
                )
            return tasks
        except Exception as error:
            error_message = f"Error getting tasks: {error}"
            logging.error(error_message)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_message, "done": True},
                    }
                )
            raise

    async def get_task(
        self,
        task_id: int,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> Task:
        """
        Retrieves a specific task by its ID.

        :param task_id: The ID of the task to retrieve.
        :return: A Task object.
        """
        try:
            task = await self.api.get_task(task_id)
            return task
        except Exception as error:
            error_message = f"Error getting task: {error}"
            logging.error(error_message)
            raise

    async def add_task(
        self,
        content: str,
        project_id: Optional[int] = None,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> Task:
        """
        Adds a new task to Todoist.

        :param content: The content of the task.
        :param project_id: (Optional) The ID of the project to add the task to.
        :return: The newly created Task object.
        """
        try:
            task = await self.api.add_task(content, project_id=project_id)
            return task
        except Exception as error:
            error_message = f"Error adding task: {error}"
            logging.error(error_message)
            raise

    async def update_task(
        self,
        task_id: int,
        content: Optional[str] = None,
        due_string: Optional[str] = None,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Updates an existing task.

        :param task_id: The ID of the task to update.
        :param content: (Optional) The new content for the task.
        :param due_string: (Optional) The new due date for the task (e.g., "tomorrow").
        :return: True if the update was successful.
        """
        try:
            success = await self.api.update_task(
                task_id, content=content, due_string=due_string
            )
            return success
        except Exception as error:
            error_message = f"Error updating task: {error}"
            logging.error(error_message)
            raise

    async def close_task(
        self,
        task_id: int,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Closes a task (marks it as completed).

        :param task_id: The ID of the task to close.
        :return: True if the task was closed successfully.
        """
        try:
            success = await self.api.close_task(task_id)
            return success
        except Exception as error:
            error_message = f"Error closing task: {error}"
            logging.error(error_message)
            raise

    async def reopen_task(
        self,
        task_id: int,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Reopens a closed task.

        :param task_id: The ID of the task to reopen.
        :return: True if the task was reopened successfully.
        """
        try:
            success = await self.api.reopen_task(task_id)
            return success
        except Exception as error:
            error_message = f"Error reopening task: {error}"
            logging.error(error_message)
            raise

    async def delete_task(
        self,
        task_id: int,
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> bool:
        """
        Deletes a task.

        :param task_id: The ID of the task to delete.
        :return: True if the task was deleted successfully.
        """
        try:
            success = await self.api.delete_task(task_id)
            return success
        except Exception as error:
            error_message = f"Error deleting task: {error}"
            logging.error(error_message)
            raise
