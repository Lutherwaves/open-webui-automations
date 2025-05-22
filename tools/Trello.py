"""
title: Trello
description: A tool to call Todoist API and list tasks based on filters
author: Martin Yankov
author_url: https://github.com/Lutherwaves
github: https://github.com/Lutherwaves/open-webui-tools
funding_url: https://github.com/open-webui
version: 1.0.0
"""

import aiohttp
from pydantic import BaseModel
from typing import Callable, Awaitable, Any, Dict
import logging

logging.basicConfig(level=logging.DEBUG)

TRELLO_BASE_URL = "https://api.trello.com/1"


class Tools:
    class Valves(BaseModel):
        TRELLO_API_KEY: str = ""
        TRELLO_TOKEN: str = ""

    def __init__(self):
        self.valves = self.Valves()

    async def _make_request(self, method: str, endpoint: str, params: Dict = None):
        """
        Make a request to the Trello API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.)
        endpoint : str
            API endpoint
        params : dict, optional
            Query parameters

        Returns
        -------
        dict
            API response
        """
        url = f"{TRELLO_BASE_URL}/{endpoint}"

        # Always include authentication
        if params is None:
            params = {}
        params["key"] = self.valves.TRELLO_API_KEY
        params["token"] = self.valves.TRELLO_TOKEN

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, params=params) as resp:
                if resp.status != 200:
                    raise Exception(
                        f"Trello API error: {resp.status} {await resp.text()}"
                    )
                return await resp.json()

    async def search_boards(
        self,
        query: str = "",
        filters: Dict[str, str] = {},
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Search for Trello boards based on the provided query and filters.

        This method allows searching for Trello boards using the Trello API's search endpoint.
        It supports filtering by board properties and can return boards matching the search criteria.

        Parameters
        ----------
        query : str, optional
            Search query string
        filters : dict, optional
            Filters to apply. Supported keys:
                - 'is_starred': Filter by starred boards (true/false)
                - 'is_open': Filter by open boards (true/false)
                - 'member_id': Filter by member ID
                - 'organization_id': Filter by organization ID
        __user__ : dict, optional
            User information (not required for API calls)
        __event_emitter__ : Callable, optional
            Async event emitter for status updates

        Returns
        -------
        str
            A string representation of the found boards

        Usage
        -----
        await tools.search_boards(query="Project")
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching for boards with query: {query}",
                        "done": False,
                    },
                }
            )

        try:
            # Prepare search parameters
            search_query = query

            # Add filters to the search query based on Trello's search syntax
            if "is_starred" in filters and filters["is_starred"].lower() == "true":
                search_query += " is:starred"
            if "is_open" in filters:
                if filters["is_open"].lower() == "true":
                    search_query += " is:open"
                else:
                    search_query += " is:closed"
            if "member_id" in filters:
                search_query += f" member:{filters['member_id']}"
            if "organization_id" in filters:
                search_query += f" organization:{filters['organization_id']}"

            # Prepare the parameters for the API request
            params = {
                "query": search_query,
                "modelTypes": "boards",
            }

            # Make the API request
            search_results = await self._make_request("GET", "search", params)
            boards = search_results.get("boards", [])

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(boards)} boards",
                            "done": True,
                        },
                    }
                )

            # Format the results
            if not boards:
                return "No boards found matching your criteria."

            result = "Boards found:\n\n"
            for board in boards:
                result += f"- {board.get('name', 'Unnamed')} (ID: {board.get('id')})\n"
                result += f"  URL: {board.get('url', 'N/A')}\n"
                result += f"  Description: {board.get('desc', 'No description')}\n\n"

            return result

        except Exception as error:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {error}", "done": True},
                    }
                )
            return f"Error searching boards: {error}"

    async def search_cards(
        self,
        query: str = "",
        filters: Dict[str, str] = {},
        __user__: dict = {},
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Search for Trello cards (tasks) based on the provided query and filters.

        This method allows searching for Trello cards using the Trello API's search endpoint.
        It supports filtering by various card properties including due date, labels, and more.

        Parameters
        ----------
        query : str, optional
            Search query string
        filters : dict, optional
            Filters to apply. Supported keys:
                - 'board_id': Filter by specific board ID
                - 'list_name': Filter by list name
                - 'due': Filter by due date ("day", "week", "month", "overdue")
                - 'created': Filter by creation date ("day", "week", "month")
                - 'edited': Filter by last edited date ("day", "week", "month")
                - 'label': Filter by label name or color
                - 'is_archived': Filter by archived status (true/false)
        __user__ : dict, optional
            User information (not required for API calls)
        __event_emitter__ : Callable, optional
            Async event emitter for status updates

        Returns
        -------
        str
            A string representation of the found cards

        Usage
        -----
        await tools.search_cards(query="Task", filters={"due": "day"})
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching for cards with query: {query}",
                        "done": False,
                    },
                }
            )

        try:
            # Prepare search parameters
            search_query = query

            # Add filters to the search query based on Trello's search syntax
            if "board_id" in filters:
                search_query += f" board:{filters['board_id']}"
            if "list_name" in filters:
                search_query += f" list:\"{filters['list_name']}\""
            if "due" in filters:
                search_query += f" due:{filters['due']}"
            if "created" in filters:
                search_query += f" created:{filters['created']}"
            if "edited" in filters:
                search_query += f" edited:{filters['edited']}"
            if "label" in filters:
                search_query += f" label:\"{filters['label']}\""
            if "is_archived" in filters:
                if filters["is_archived"].lower() == "true":
                    search_query += " is:archived"
                else:
                    search_query += " is:open"
            else:
                search_query += " is:open"  # default to open cards if not specified

            # Prepare the parameters for the API request
            params = {
                "query": search_query,
                "modelTypes": "cards",
            }

            # Make the API request
            search_results = await self._make_request("GET", "search", params)
            cards = search_results.get("cards", [])

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(cards)} cards",
                            "done": True,
                        },
                    }
                )

            # Format the results
            if not cards:
                return "No cards found matching your criteria."

            result = "Cards found:\n\n"
            for card in cards:
                result += f"- {card.get('name', 'Unnamed')} (ID: {card.get('id')})\n"
                result += f"  URL: {card.get('url', 'N/A')}\n"
                result += f"  Description: {card.get('desc', 'No description')}\n"

                # Add due date if available
                due = card.get("due")
                if due:
                    result += f"  Due: {due}\n"

                # Add labels if available
                labels = card.get("labels", [])
                if labels:
                    label_str = ", ".join(
                        [
                            f"{label.get('name', '')} ({label.get('color', 'unknown')})"
                            for label in labels
                        ]
                    )
                    result += f"  Labels: {label_str}\n"

                result += "\n"

            return result

        except Exception as error:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {error}", "done": True},
                    }
                )
            return f"Error searching cards: {error}"
