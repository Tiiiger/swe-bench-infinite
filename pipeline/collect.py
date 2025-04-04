def collect_requirements(file_contents, commit_hash, commit_date):
    # load prompt from prompts/requirements_collection.txt
    with open("prompts/requirements_collection.txt", "r") as f:
        prompt = f.read()

    # format each file content with the file path
    file_contents_str = ""
    for file_path, content in file_contents.items():
        file_contents_str += f"==== {file_path} ====\n{content}\n\n"

    prompt = prompt.replace("{file_contents_str}", file_contents_str)
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    return prompt
