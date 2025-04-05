from pipeline.utils import make_file_contents_str

def collect_requirements(file_contents: dict[str, str], commit_hash: str, commit_date: str):
    # load prompt from prompts/requirements_collection.txt
    with open("prompts/requirements_collection.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{file_contents_str}", make_file_contents_str(file_contents))
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    return prompt
