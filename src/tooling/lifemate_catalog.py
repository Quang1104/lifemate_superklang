"""Evidence paths verified in the released LifeMate v2.7 app implementations."""

from copy import deepcopy
from typing import Any

from .evidence import EvidencePath
from .registry import ToolRegistry, ToolSpec


_LIST_COLLECTIONS = {
    "alarm.list": ("alarms",),
    "appliances.list": ("appliances",),
    "banking.get_balance": ("accounts",),
    "banking.list_beneficiaries": ("beneficiaries",),
    "bike_rental.find_bikes": ("bikes",),
    "bill_payment.list_bills": ("bills",),
    "calendar.list_events": ("events",),
    "chat_ott.list_chats": ("chats",),
    "contacts.get": ("contacts",),
    "contacts.search": ("contacts",),
    "door_lock.list_locks": ("locks",),
    "ecommerce.order_status": ("orders",),
    "ecommerce.search": ("products",),
    "email.list": ("emails",),
    "event_tickets.search": ("events",),
    "flights.my_bookings": ("bookings",),
    "flights.search": ("flights",),
    "food_delivery.list_restaurants": ("restaurants",),
    "food_delivery.order_history": ("orders",),
    "grocery.search": ("products",),
    "habit_tracker.list_habits": ("habits",),
    "home_automation.list_rules": ("rules",),
    "home_security.list_alerts": ("alerts",),
    "household_chores.list_chores": ("chores",),
    "learning.list_courses": ("courses",),
    "library.search": ("books",),
    "maps.search_place": ("places",),
    "medication.list": ("schedules",),
    "music.list_tracks": ("tracks", "playlists"),
    "news.headlines": ("articles",),
    "news.search": ("articles",),
    "notes.list": ("notes",),
    "pantry.list_items": ("items",),
    "parking.find": ("lots",),
    "parking.list_sessions": ("sessions",),
    "podcast_radio.search": ("episodes", "shows"),
    "recipes.search": ("recipes",),
    "reminders.list": ("reminders",),
    "shopping_list.get": ("items",),
    "smart_home.get_status": ("devices", "scenes"),
    "subscriptions.list": ("subscriptions",),
    "tasks.list": ("tasks",),
    "train_bus_tickets.my_bookings": ("bookings",),
    "train_bus_tickets.search": ("trips",),
    "tv.guide": ("programs",),
    "video_meeting.list_meetings": ("meetings",),
}

_ITEM_COLLECTIONS = {
    "alarm.create": ("alarms", "item"),
    "calendar.create_event": ("events", "item"),
    "flights.book": ("bookings", "booking"),
    "food_delivery.place_order": ("orders", "order"),
    "home_automation.create_rule": ("rules", "rule"),
    "medication.add": ("schedules", "item"),
    "notes.create": ("notes", "item"),
    "parking.reserve": ("sessions", "session"),
    "reminders.create": ("reminders", "item"),
    "ride_hailing.book": ("rides", "item"),
    "ride_hailing.get_ride": ("rides", "ride"),
    "shopping_list.add": ("items", "item"),
    "tasks.add": ("tasks", "item"),
    "train_bus_tickets.book": ("bookings", "booking"),
    "video_meeting.create": ("meetings", "item"),
}

_EXTRA_PATHS = {
    "food_delivery.get_menu": (EvidencePath("food_delivery.menu", ("menu", "*", "id"),
                                          "restaurant_id"),),
    "food_delivery.list_restaurants": (EvidencePath("food_delivery.merchants",
                                                    ("restaurants", "*", "merchant_id")),),
    "loyalty.get_points": (EvidencePath("loyalty.rewards", ("programs", "*", "rewards", "*", "id")),),
    "package_tracking.track": (EvidencePath("package_tracking.shipments",
                                             ("shipments", "*", "tracking_id")),),
    "robot_vacuum.get_status": (EvidencePath("robot_vacuum.rooms", ("rooms", "*")),),
}


def _paths(name: str, app: str) -> tuple[EvidencePath, ...]:
    paths = [EvidencePath(f"{app}.{collection}", (collection, "*", "id"))
             for collection in _LIST_COLLECTIONS.get(name, ())]
    if name in _ITEM_COLLECTIONS:
        collection, key = _ITEM_COLLECTIONS[name]
        paths.append(EvidencePath(f"{app}.{collection}", (key, "id")))
    paths.extend(_EXTRA_PATHS.get(name, ()))
    return tuple(paths)


def _parameters(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = deepcopy(tool["parameters"])
    parameters["additionalProperties"] = False
    properties = parameters["properties"]
    name = tool["name"]
    if name in {"calendar.create_event", "video_meeting.create"}:
        properties["attendees"]["items"]["x-ref"] = "contacts.contacts"
    if name == "food_delivery.place_order":
        item = properties["items"]["items"]
        item["additionalProperties"] = False
        item["properties"]["item_id"]["x-ref"] = "food_delivery.menu"
    if name == "ewallet.pay_qr":
        properties["merchant_id"]["x-ref"] = "food_delivery.merchants"
    if name == "loyalty.redeem":
        properties["reward_id"]["x-ref"] = "loyalty.rewards"
    if name == "package_tracking.reschedule_delivery":
        properties["tracking_id"]["x-ref"] = "package_tracking.shipments"
    if name == "robot_vacuum.start":
        properties["rooms"]["items"]["x-ref"] = "robot_vacuum.rooms"
    return parameters


def build_registry(catalog: dict[str, Any]) -> ToolRegistry:
    specs = []
    if not isinstance(catalog.get("apps"), list) or not catalog["apps"]:
        raise ValueError("Expected an official apps catalog")
    for app in catalog["apps"]:
        for tool in app["tools"]:
            name = tool["name"]
            if tool["kind"] not in {"read", "write", "meta"} or name.partition(".")[0] != app["id"]:
                raise ValueError("Catalog tool lacks a recognized kind or app")
            if tool.get("app") != app["id"]:
                raise ValueError("Catalog tool app does not match its enclosing app")
            scopes = (("food_delivery.menu", "restaurant_id"),) if name == "food_delivery.place_order" else ()
            specs.append(ToolSpec(name, app["id"], tool["description"], _parameters(tool),
                                  tool["kind"] == "write", _paths(name, app["id"]), scopes))
    # Date and time formats use the official validate method's dialect.
    return ToolRegistry(specs, check_formats=False)
