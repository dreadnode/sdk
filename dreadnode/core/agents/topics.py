from aiokafka.admin import AIOKafkaAdminClient, NewTopic  # type: ignore


class KafkaManager:
    def __init__(self, bootstrap_servers: str):
        self.admin_client = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)

    async def start(self):
        await self.admin_client.start()

    async def close(self):
        await self.admin_client.close()

    async def check_topics(self, topics: list[str]) -> list[str] | None:
        try:
            existing_topics = await self.admin_client.list_topics()
        except Exception as e:
            print(f"Error listing topics: {e}")

        # Filter to only create missing topics
        topics_to_create = [t for t in topics if t not in existing_topics]

        if not topics_to_create:
            print("All topics already exist")
            return None

        return topics_to_create

    async def create_topics(self, topics: list[str]):
        await self.admin_client.start()

        try:
            new_topics = await self.check_topics(topics)
            if not new_topics:
                return

            print(f"Creating topics: {new_topics}")

            kafka_topics = [
                NewTopic(name=i, num_partitions=1, replication_factor=1) for i in new_topics
            ]

            await self.admin_client.create_topics(kafka_topics)

            # Create topics
            print("Topics created successfully")

        except Exception as e:
            print(f"Error creating topics: {e}")
        finally:
            await self.admin_client.close()
