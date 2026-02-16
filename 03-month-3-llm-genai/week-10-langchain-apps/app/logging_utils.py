
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LangChainQA")

def log_query(query, response, source_docs, start_time):
    """Log details of the Q&A process."""
    elapsed = time.time() - start_time
    logger.info(f"Query: {query}")
    logger.info(f"Response time: {elapsed:.2f}s")
    logger.info(f"Sources used: {len(source_docs)}")
    for i, doc in enumerate(source_docs):
        logger.info(f"Source {i+1} snippet: {doc.page_content[:50]}...")
