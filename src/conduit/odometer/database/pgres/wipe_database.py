from dbclients import get_postgres_client
import logging

logger = logging.getLogger(__name__)

get_db_connection = get_postgres_client("context_db", dbname="chain")


def wipe_token_events_database(confirm: bool = False) -> bool:
    """
    Delete all token events from the database.

    Args:
        confirm: Must be True to actually perform the deletion (safety check)

    Returns:
        bool: True if successful, False otherwise
    """
    if not confirm:
        logger.warning(
            "wipe_token_events_database called without confirm=True. No action taken."
        )
        return False

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get count before deletion for logging
                cursor.execute("SELECT COUNT(*) FROM token_events")
                count_before = cursor.fetchone()[0]

                # Delete all records
                cursor.execute("DELETE FROM token_events")
                deleted_count = cursor.rowcount

                # Reset the sequence for the ID column
                cursor.execute("ALTER SEQUENCE token_events_id_seq RESTART WITH 1")

                conn.commit()
                logger.warning(
                    f"Database wiped: deleted {deleted_count} token events (was {count_before} total)"
                )
                return True

    except Exception as e:
        logger.error(f"Failed to wipe database: {e}")
        return False


def main():
    """
    Main function to execute the wipe operation.
    """
    confirm = (
        input("Are you sure you want to wipe the token events database? (yes/no): ")
        .strip()
        .lower()
    )
    if confirm == "yes":
        success = wipe_token_events_database(confirm=True)
        if success:
            print("Database wiped successfully.")
        else:
            print("Failed to wipe the database.")
    else:
        print("Operation cancelled. No changes made.")


if __name__ == "__main__":
    main()
