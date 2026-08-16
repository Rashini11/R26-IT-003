from getpass import getpass

from backend.auth import create_user_account


def main():
    print("OceanIQ - Create administrator")
    username = input("Username: ").strip()
    email = input("Email (optional): ").strip() or None
    password = getpass("Password (minimum 10 characters): ")
    confirm = getpass("Confirm password: ")

    if password != confirm:
        raise SystemExit("Passwords do not match")

    try:
        user = create_user_account(
            username=username,
            password=password,
            email=email,
            role="admin",
        )
    except Exception as exc:
        raise SystemExit(f"Could not create admin: {exc}") from exc

    print("Administrator created successfully")
    print("Username:", user["username"])
    print("Role:", user["role"])


if __name__ == "__main__":
    main()
