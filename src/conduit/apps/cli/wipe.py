from conduit.storage.repository.postgres_repository import (
    AsyncPostgresSessionRepository,
)
import asyncio


repo = AsyncPostgresSessionRepository(project_name="conduit_cli")


async def main():
    await repo.wipe()
    print("Wiped the database")


if __name__ == "__main__":
    asyncio.run(main())
