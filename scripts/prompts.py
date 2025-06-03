import pandas as pd


def extract_prompts(df_gt):
    all_prompts = []
    for index, row in df_gt.iterrows():
        # Create message
        messages = [{"role": "system", "content": "You are a helpful assisstant."}] + \
            [{"role": "user", "content": row["prompts"]}]

        metadata = row.to_dict()
        for key in ["prompts"]:
            metadata.pop(key, None)  # `None` prevents errors if the key isn't found

        all_prompts.append(
            [messages, metadata])
    return all_prompts

def set_prompts(language, gt_file, mode, pred_average):
    if "sanity" in mode:
        language = language + "_sanity"    

    # load gt file
    df_gt = pd.read_csv(gt_file)

    if "prompts" in df_gt.keys():
        all_prompts = extract_prompts(df_gt)
        return all_prompts
