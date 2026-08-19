"""One-time migration for OceanIQ access-control fields."""
from backend.auth import _auth_collections, utc_now


def main():
    users, _ = _auth_collections()
    now = utc_now()

    admin_result = users.update_many(
        {"role": "admin"},
        {
            "$set": {
                "approval_status": "approved",
                "access_level": "read_write",
                "is_active": True,
                "updated_at": now,
            }
        },
    )

    user_result = users.update_many(
        {
            "role": {"$ne": "admin"},
            "$or": [
                {"approval_status": {"$exists": False}},
                {"access_level": {"$exists": False}},
            ],
        },
        {
            "$set": {
                "approval_status": "pending",
                "access_level": "none",
                "updated_at": now,
            }
        },
    )

    print("OceanIQ access migration complete")
    print("Admin profiles updated:", admin_result.modified_count)
    print("Legacy user profiles moved to pending:", user_result.modified_count)


if __name__ == "__main__":
    main()
