def localize_requirements(tree_result, commit_hash, commit_date):
    # load prompt from prompts/requirements_localization.txt
    with open("prompts/requirements_localization.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.format(tree_result=tree_result, commit_hash=commit_hash, commit_date=commit_date, top_k=10)
    print(prompt)
    return prompt

if __name__ == "__main__":
    pass