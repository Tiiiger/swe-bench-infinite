import json
import logging
import os
import re
from typing import Any, Dict

from exceptions import AnthropicResponseError, RequirementsError
from exp_utils import dump_anthropic_response


def anthropic_generate_json(
    prompt: str, client, logger=None, output_filename="output.json"
) -> Dict[str, Any]:
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
    # Create a default logger if none is provided
    if logger is None:
        logger = logging.getLogger("anthropic_generate_json")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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
    if text_block is None:
        logger.error("Expected text content block, got %s", content_block.type)
        raise AnthropicResponseError("Expected text content block, got different content type")

    # Save response to log file
    if logger and hasattr(logger, "handlers") and len(logger.handlers) > 1:
        log_file = logger.handlers[1].baseFilename
        with open(log_file, "w") as f:
            f.write(response_dump)

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
    if logger and hasattr(logger, "handlers") and len(logger.handlers) > 1:
        exp_dir = os.path.dirname(logger.handlers[1].baseFilename)
        output_path = os.path.join(exp_dir, output_filename)
        logger.info(f"Saving JSON results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(parsed_data, f, indent=4)

    return parsed_data
