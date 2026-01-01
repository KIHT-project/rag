import logging
from config.logging_config import setup_logging, mdc_context_scope

logger = logging.getLogger(__name__)
setup_logging()

class MarketAgent:
    def operation(self):
        logger.error("Something went wrong")

def main():
    logger.info("Operation started")
    logger.debug("Fetching something")

if __name__ == "__main__":
    with mdc_context_scope(
        business_id="test-bid-main",
        user_id="user-123",
        service="service-x",
        operation="main"
    ):
        main()
        agent = MarketAgent()
        agent.operation()
