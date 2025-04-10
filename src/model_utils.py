import json
import os
import re
from typing import Any

from anthropic_client import AnthropicClient
from exceptions import AnthropicResponseError, RequirementsError

from exp_utils import dump_anthropic_response
from logger import CustomLogger


def anthropic_generate_json(
    prompt: str, logger: CustomLogger, output_filename="output.json"
) -> Any:
    """
    Abstract function to handle Anthropic API calls, JSON extraction, and response logging.

    Args:
        prompt (str): The prompt to send to the Anthropic API
        client: The Anthropic client to use for API calls
        logger (logging.Logger, optional): Logger for tracking operations
        output_filename (str, optional): The name of the output file to save the JSON to

    Returns:
        Dict[str, Any]: The parsed JSON data from the response

    Raises:
        AnthropicResponseError: If there's an issue with the Anthropic API response
        RequirementsError: If there's an issue parsing the JSON in the response
    """
    client = AnthropicClient()

    logger.info("Sending prompt to Anthropic API...")
    response = client.create_message_with_retry(messages=[{"role": "user", "content": prompt}])
    logger.info("Received response from Anthropic API")

    # Extract text content from response
    text_block = None
    for content_block in response.content:
        if content_block.type == "text" and content_block.text is not None:
            text_block = content_block.text

    # Dump response for logging
    response_dump = dump_anthropic_response(response, logger)
    # Save response to log file
    with open(logger.get_logdir() + "/anthropic_response.json", "w") as f:
        f.write(response_dump)

    if text_block is None:
        logger.error("Expected text content block, got %s", content_block.type)
        raise AnthropicResponseError("Expected text content block, got different content type")

    # Extract JSON block using regex
    logger.info("Extracting JSON block from response...")
    json_match = re.search(r"```json(.*)```", text_block, re.DOTALL)
    if not json_match:
        logger.error("Could not find JSON block in response")
        raise RequirementsError("Could not find JSON block in response")

    json_block = json_match.group(1)

    # Parse JSON
    logger.info("Parsing JSON block...")
    try:
        parsed_data = json.loads(json_block)
        logger.info("Successfully parsed JSON data")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON block: {e}")
        raise RequirementsError(f"Error parsing JSON block: {e}") from e

    # Save to output file if a logger with handlers is provided
    output_path = os.path.join(logger.get_logdir(), output_filename)
    logger.info(f"Saving JSON results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(parsed_data, f, indent=4)

    return parsed_data
