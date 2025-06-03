from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import argparse
from prompts import set_prompts
from tqdm import tqdm
import os

DEVICE = "cuda"



def load_model(model_name, gt_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        # Option 1: Use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16)
    model.config.pad_token_id = model.config.eos_token_id
    # model, tokenizer = None, None
    return model, tokenizer


def tokenize_data(gt_file, tokenizer, language, mode, pred_average):

    all_prompts = set_prompts(language, gt_file, mode, pred_average)

    prompts_template = []
    prompt_metadata = []
    for _, prompt in enumerate(all_prompts):
        messages = tokenizer.apply_chat_template(
            prompt[0], add_generation_prompt=True, tokenize=False)
        prompts_template.append(messages)
        prompt_metadata.append(prompt[1])
    # print(prompts_template[:2])
    return prompts_template, prompt_metadata


def batch_inference(input_texts, model, tokenizer, mode, prompt_metadata, output_file, batch_size=64, num_return_sequences=1):
    """
    Perform batch inference on a list of input texts.

    Parameters:
    - input_texts: List of strings, the texts to run inference on.
    - batch_size: int, the number of texts to process per batch.
    - max_length: int, maximum length of generated response.

    Returns:
    - List of generated responses.
    """
    global LOGITS_PROCESSOR

    results = []
    max_new_tokens = 20
    if "free_generation" in mode:
        max_new_tokens = 500

    if "reduce_compute" in mode:
        batch_size=64
        num_return_sequences=1

    # Process each batch
    for i in tqdm(range(0, len(input_texts), batch_size)):
        batch_texts = input_texts[i:i + batch_size]

        # Tokenize and pad inputs for batch processing
        inputs = tokenizer(batch_texts, return_tensors="pt",
                           padding=True, truncation=True).to(DEVICE)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,  # Ensure padding if needed,
                do_sample=False,
                #logits_processor=LOGITS_PROCESSOR,
                temperature=None,
                top_k=None,
                top_p=None
            )

        # Decode the predictions and append to results
        decoded_outputs = [tokenizer.decode(
            output, skip_special_tokens=True) for output in outputs]
        decoded_outputs = [decoded_outputs[i:i+num_return_sequences]
                           for i in range(0, len(decoded_outputs), num_return_sequences)]
        results += decoded_outputs
        if i % 10 == 0:
            save_data(results, mode, prompt_metadata, output_file)
    return results


def save_prompts(prompts, prompt_metadata, output_file):
    prompt_df = pd.DataFrame(prompt_metadata)
    prompt_df["prompts"] = prompts
    # Define the output directory and file
    output_folder = os.path.dirname(output_file)
    output_dir = os.path.join(output_folder, "prompts")
    output_prompt = os.path.join(output_dir, "prompt.csv")
    os.makedirs(output_dir, exist_ok=True)
    prompt_df.to_csv(output_prompt, index=False)



def save_data(results, mode, prompt_metadata, output_path):
    """
    Save the processed data to a CSV file.

    Parameters:
        data (pd.DataFrame): Data with inference results to be saved.
        output_path (str): Path to the output file.
    """

    split_texts = ["assistant\\n\\n", "assistant\\n",
                  "assistant\n\n", "assistant\n", "assistant", "<|CHATBOT_TOKEN|>"]

    salaries = results
    for split_text in split_texts:
        salaries = [[salary.split(split_text)[-1]
                    for salary in n_seq] for n_seq in salaries]
    processed_data = pd.DataFrame(prompt_metadata[:len(results)])
    processed_data["salary"] = salaries
    processed_data["mode"] = mode
    processed_data.to_csv(output_path, index=False)
    pickle_output = output_path.replace(".csv", ".pkl")
    processed_data.to_pickle(pickle_output)



# Main workflow
def main(output_folder, gt_file, model_name, language, modes, prediction_values_folder):

    # Step 0: Load the Model
    model, tokenizer = load_model(model_name, gt_file)

    for index, mode in enumerate(modes):
        model_name_country = model_name.split("/")[-1]
        if mode != "":
            output_file_country = os.path.join(output_folder, language + "_" + model_name_country + "_" + mode + ".csv")
        else:
            output_file_country = os.path.join(output_folder, language + "_" + model_name_country + ".csv")

        if "sanity" in mode:
            df_preds = pd.read_csv(prediction_values_folder)
            pred_average = df_preds.set_index('Country Name')['GDP_pc_USD'].to_dict()
            output_file_country = os.path.join(output_folder + "_sanity", language + "_" + model_name_country + ".csv")
        else:
            pred_average = None

        #if os.path.exists(output_file_country):
        #    continue

        os.makedirs(os.path.dirname(output_file_country), exist_ok=True)

        # Step 2: Tokenize
        prompts, prompt_metadata = tokenize_data(
            gt_file, tokenizer, language, mode, pred_average)
        save_prompts(prompts, prompt_metadata, output_file_country)

        # Step 3: Perform inference
        results = batch_inference(
            prompts, model, tokenizer, mode, prompt_metadata, output_file_country)

        # Step 4: Save Data
        save_data(results, mode, prompt_metadata, output_file_country)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a dataset and save the results.")
    parser.add_argument("--model_name", type=str, default="",
                        help="Name of the model to use for inference.")
    parser.add_argument("--language", type=str, default="en",
                        help="Language.")
    parser.add_argument("--modes", nargs='+', default=[""],
                        help="Toggle between CoT.")
    parser.add_argument("--prediction_values_folder", type=str, default="",
                        help="For sequential hops.")
    parser.add_argument("--output_folder", type=str,
                        default="output_test/", help="Path to the output folder.")
    parser.add_argument("--gt_file", type=str,
                        default="data/gdp2021.csv", help="Path to the input GT file.")
    args = parser.parse_args()

    # input_folder = os.path.join(args.input_folder, args.language)
    main(args.output_folder, args.gt_file, args.model_name, args.language, args.modes, args.prediction_values_folder)
