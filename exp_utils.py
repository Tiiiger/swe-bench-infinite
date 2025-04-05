import os
import json
import logging


def dump_anthropic_response(response, logger):
    """
    Dump Anthropic API response to a JSON format.
    
    Args:
        response: The response object from Anthropic API
        logger (logging.Logger): Logger for tracking operations
        
    Returns:
        str: JSON string representation of the response
    """
    all_blocks = []
    for content_block in response.content:
        if content_block.type == "text":
            all_blocks.append({
                "type": "text",
                "text": content_block.text,
            })
        elif content_block.type == "thinking":
            all_blocks.append({
                "type": "thinking",
                "thinking": content_block.thinking
            })
    return json.dumps(all_blocks, indent=4)


def get_latest_exp_dir():
    """
    Get the latest directory in the exps folder.
    
    Returns:
        str: Path to the latest directory, or None if no directories exist
    """
    if not os.path.exists("exps"):
        return None
    
    # Get all directories that have a main.log file (indicating a proper experiment directory)
    dirs = [d for d in os.listdir("exps") 
            if os.path.isdir(os.path.join("exps", d)) 
            and os.path.exists(os.path.join("exps", d, "main.log"))]
    
    if not dirs:
        return None
    
    # Sort directories by name (timestamp format)
    dirs.sort(reverse=True)
    return os.path.join("exps", dirs[0]) 