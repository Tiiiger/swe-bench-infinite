import logging
import os
import random
import time

import anthropic
from anthropic.types import Message

# Configure module logger (non-propagating)
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only add handler if it doesn't have one
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
# Prevent propagation to avoid duplicate logs
logger.propagate = False


class AnthropicClient:
    def __init__(
        self, max_retries: int = 20, initial_backoff: float = 1.0, max_backoff: float = 60.0
    ):
        """
        Initialize Anthropic client with retry capabilities.

        Args:
            max_retries: Maximum number of retries for rate-limited requests
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
        """
        # Get API key from environment variables
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def create_message_with_retry(
        self, messages, model: str = "claude-3-7-sonnet-latest", max_tokens: int = 16384, **kwargs
    ) -> Message:
        """
        Send a message to the Anthropic API with retry logic for rate limits.

        Args:
            messages: List of message parameters to send
            model: Model name to use
            max_tokens: Maximum number of tokens in the response
            **kwargs: Additional arguments to pass to the create method

        Returns:
            Message response from the API
        """
        current_retry = 0
        backoff_time = self.initial_backoff

        while True:
            try:
                return self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    thinking={"type": "enabled", "budget_tokens": 8192},
                    **kwargs,
                )
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                current_retry += 1
                if current_retry > self.max_retries:
                    if isinstance(e, anthropic.RateLimitError):
                        logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                    elif getattr(e, "status_code", 0) >= 500:
                        logger.error(
                            f"Error calling Anthropic API: Error code: 500 - {e.response.json()}"
                        )
                    else:
                        logger.error(f"Error calling Anthropic API: {e}")
                    raise

                # Calculate sleep time
                if (
                    isinstance(e, anthropic.RateLimitError)
                    and getattr(e, "retry_after", None) is not None
                ):
                    sleep_time = float(e.retry_after)  # type: ignore
                else:
                    # Exponential backoff with jitter
                    sleep_time = min(
                        backoff_time * (1.5 + 0.5 * (2 * random.random() - 1)), self.max_backoff
                    )
                    backoff_time = sleep_time

                # Log appropriate message
                if isinstance(e, anthropic.RateLimitError):
                    error_type = "Rate limit hit"
                elif getattr(e, "status_code", 0) >= 500:
                    error_type = "Internal server error (500)"
                # No need to retry for 4XX errors
                # usually, they are inputs that exceed the input token limit
                elif getattr(e, "status_code", 0) >= 400:
                    logger.error(f"Error calling Anthropic API: {e}")
                    raise
                else:
                    error_type = f"API error: {getattr(e, 'status_code', 'unknown')}"

                logger.warning(
                    f"{error_type}. Retrying in {sleep_time:.2f} seconds (attempt {current_retry}/{self.max_retries})"
                )
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error calling Anthropic API: {e}")
                raise


# Example usage
if __name__ == "__main__":
    # Make sure to set ANTHROPIC_API_KEY environment variable before running
    client = AnthropicClient()

    response = client.create_message_with_retry(
        messages=[{"role": "user", "content": "Hello, Claude! How are you today?"}]
    )
    print("Response content:")
    for content_block in response.content:
        if content_block.type == "text":
            print(content_block.text)
