import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedModel
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import gc
from itertools import permutations
import numpy as np
import torch
import kagglehub
from huggingface_hub import login
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import random
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from huggingface_hub import login
import seaborn as sns

device = "cpu"  # Default fallback
if torch.cuda.is_available():
    device = "cuda"

# Define the model we are using
def define_model():
    login(token = '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')
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model, pipe

def define_data():
    #sst2 
    sst2 = load_dataset("glue", "sst2")
    data_sst2 = np.array([sst2['train']['sentence'], sst2['train']['label']]).T

    #ag news - replaceing the data with only the titles
    ag_news = load_dataset("ag_news")
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    data_agnews = np.array([pd.read_csv(path + "/train.csv")['Title'].to_list(), ag_news['train']['label']]).T

    setfit = load_dataset("SetFit/CR")
    data_setfit = np.array([setfit['train']['text'], setfit['train']['label']]).T

    trec = load_dataset("CogComp/trec", trust_remote_code=True)
    data_trec = np.array([trec['train']['text'], trec['train']['coarse_label']]).T

    return data_sst2, data_agnews, data_setfit, data_trec

# Dimenison 0 is data, dimension 1 is data! Simple formatting!
sst2, ag_news, setfit, trec = define_data()
tokenizer, model, pipe = define_model()
ag_news_label_map = {"World": 0, "Sports": 1, "Business": 2, "Tech": 3}
sst_label_map = {"Negative": 0, "Positive": 1}
setfit_label_map = {"Negative": 0, "Positive": 1}
trec_label_map = {"Abbreviation": 0, "Entity": 1, "Description": 2, "Human": 3, "Location": 4, "Number": 5}

# Function to create prompts for SST-2
def create_prompt_sst2(examples, test_example, perm_type="none", perm_prop=1):
    prompt = "Classify sentiment as Positive or Negative.\n\n"
    for i, ex in enumerate(examples):
        text = ex[0]
        inverted_map = {value: key for key, value in sst_label_map.items()}
        label = inverted_map[int(ex[1])]
        if perm_type == "rand":
            if (random.random() > (1 - perm_prop)):
                label = permute_label(label, sst_label_map)
        if perm_type == "pos" and perm_prop == i: 
            label = permute_label(label, sst_label_map)
        # prompt += f"Text: {text}\nSentiment: {label}\n\n"
        prompt += f"Text {text} Sentiment {label} "
    # Now the test example
    # prompt += f"Text: {test_example[0]}\nSentiment: "
    prompt += f"Text {test_example[0]} Sentiment "
    return prompt

def create_prompt_agnews(examples, test_example, perm_type="none", perm_prop=1):
    prompt = "Classify into World, Sports, Business, or Tech.\n\n"
    # prompt = "Classify the titles into the following categories World Sports Business Tech\n"
    for i, ex in enumerate(examples):
        text = ex[0]
        inverted_map = {value: key for key, value in ag_news_label_map.items()}
        label = inverted_map[int(ex[1])]
        if perm_type == "rand":
            if (random.random() > (1 - perm_prop)):
                label = permute_label(label, ag_news_label_map)
        if perm_type == "pos" and perm_prop == i:
            label = permute_label(label, ag_news_label_map)
        # prompt += f"Text: {text}\nCategory: {label}\n\n"
        prompt += f"Text {text} Category {label} "
    # Now the test example
    # prompt += f"Text: {test_example[0]}\nCategory: "
    prompt += f"Text {test_example[0]} Category "
    return prompt

# Function to create prompts for AG News
def create_prompt_trec(examples, test_example, perm_type="none", perm_prop=1):
    prompt = "Classify into Abbreviation, Entity, Description, Human, Location, or Number.\n\n"
    for i, ex in enumerate(examples):
        text = ex[0]
        inverted_map = {value: key for key, value in trec_label_map.items()}
        label = inverted_map[int(ex[1])]
        if perm_type == "rand":
            if (random.random() > (1 - perm_prop)):
                label = permute_label(label, ag_news_label_map)
        if perm_type == "pos" and perm_prop == i:
            label = permute_label(label, ag_news_label_map)
        prompt += f"Text {text} Category {label} "
    # Now the test example
    prompt += f"Text {test_example[0]} Category "
    return prompt

def create_prompt_setfit(examples, test_example, perm_type="none", perm_prop=1):
    prompt = "Classify sentiment as Positive or Negative.\n\n"
    for i, ex in enumerate(examples):
        text = ex[0]
        inverted_map = {value: key for key, value in setfit_label_map.items()}
        label = inverted_map[int(ex[1])]
        if perm_type == "rand":
            if (random.random() > (1 - perm_prop)):
                label = permute_label(label, ag_news_label_map)
        if perm_type == "pos" and perm_prop == i:
            label = permute_label(label, ag_news_label_map)
        prompt += f"Text {text} Sentiment {label} "
    # Now the test example
    prompt += f"Text {test_example[0]} Sentiment "
    return prompt

def create_prompt(dataset, data_fun, N, label_map, perm_type="none", perm_prop=1):
    idxs = random.sample(range(0, len(dataset)), N + 1)
    in_context_examples = [dataset[i] for i in idxs[:-1]]
    test_example = dataset[idxs[-1]]

    # Get the label
    label_map_inv = {v: k for k, v in label_map.items()}
    label = label_map_inv[int(test_example[1])]

    # check if we are permuting the prompt
    prompts = []
    if perm_type == "pos":
        for i in range(N):
            prompts.append(data_fun(in_context_examples, test_example, perm_type="pos", perm_prop=i))
    elif perm_type == "perm":
        # perms = list(permutations(range(N)))
        perms = sample_random_permutations(N, 25)
        for perm in perms:
            perm_examples = [in_context_examples[i] for i in perm]
            prompts.append(data_fun(perm_examples, test_example, perm_type=perm_type, perm_prop=perm_prop))
    else:
        prompt = data_fun(in_context_examples, test_example, perm_type=perm_type, perm_prop=perm_prop)
        prompts.append(prompt)
    
    return prompts, label

def sample_random_permutations(N, k):
    """
    Returns k distinct random permutations (as tuples) of range(N),
    without building the entire list(permutations(range(N))).
    """
    seen = set()
    results = []
    
    while len(results) < k:
        # Generate one random permutation by shuffling a list of range(N)
        perm = list(range(N))
        random.shuffle(perm)  # in-place shuffle
        perm_tuple = tuple(perm)
        
        # Ensure distinct permutations if you really need them all distinct:
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            results.append(perm_tuple)
    
    return results

def weighted_pipe(prompt, do_weights, max_new_tokens=2):
    # 1) Convert prompt to input_ids
    input_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    tokens = tokenizer.tokenize(prompt)
    boundaries = find_boundaries(tokens)

    # We'll do a simple loop for exactly max_new_tokens tokens
    for step in range(max_new_tokens):
        # 2) Get embeddings from the entire current sequence
        embeddings = model.get_input_embeddings()(input_ids)

        # 3) Apply a uniform weight of 1.0 to all embeddings (replace with your own logic)
        if do_weights:
            values = np.array([0.04762811753303145, 0.09871496973380767, 0.08490977916546841, 0.07527471968622904, 0.06953211926401466, 0.06826235561315386, 0.06858425399848522, 0.06917140754024442, 0.0881469743541701, 0.13514000723734348])
            weights = (np.mean(values) / values).tolist()
        else:
            weights = [1,1,1,1,1,1,1,1,1,1]
        weights = create_example_weights(embeddings, boundaries, weights=weights) # 1, 89, 2048
        weighted_embeddings = embeddings * weights

        # 4) Forward pass with `inputs_embeds` instead of `input_ids`
        outputs = model(inputs_embeds=weighted_embeddings)

        # 5) Take the *last* token's logits (the one we just predicted)
        next_token_logits = outputs.logits[:, -1, :]

        # 6) Greedy pick the next token
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # shape [batch_size, 1]

        # 7) Append that token to our input_ids
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return output_text

# From weights for each example position, create a weights object that can be applied to the 
def create_example_weights(embeddings, intervals, weights):
    batch_size, seq_len, hidden_dim = embeddings.shape
    weights_out = torch.ones_like(embeddings)

    # print(seq_len)
    # print(intervals)

    for i, (start_idx, end_idx) in enumerate(intervals[1:len(weights)]):
        # print(i, len(weights))
        # weights_out[0, start_idx:end_idx, :] = weights[i]
        continue

    return weights_out

    # output = pipe(prompt, max_length=(len(pipe.tokenizer.tokenize(prompt)) + 2), do_sample=False)[0]['generated_text']
    # generated = output[len(prompt):].strip()

def test_accuracy(pipe, dataset, data_fun, label_map, perm_type="none", do_weight=False, perm_prop=1, N=4, K=2):
    predictions = [[] for i in range(N)]
    labels = []
    for _ in tqdm(range(K)):
        prompts, label = create_prompt(dataset, data_fun, N, label_map, perm_type=perm_type, perm_prop=perm_prop)

        for i, prompt in enumerate(prompts):
            output = pipe(prompt, max_length=(len(pipe.tokenizer.tokenize(prompt)) + 2), do_sample=False)[0]['generated_text']
            # output = weighted_pipe(prompt, do_weight)
            # print(output)
            # print(prompt)
            # print(output)
            generated = output[len(prompt):].strip()
            # print(generated)
            pred_label = -1

            for label_name in label_map.keys():
                if label_name in generated:
                    pred_label = label_name

            # print(i, pred_label, N, K)
            predictions[i].append(pred_label)
        labels.append(label)

    # Compute accuracy
    if perm_type == "pos": #if it is permuting the prompt labels across each position
        acc = []
        # print(predictions)
        # print(len(predictions))
        for i in range(N):
            correct = sum((p == l) for p, l in zip(predictions[i], labels) if p != -1)
            total = sum((p != -1) for p in predictions[i])
            accuracy = correct / total if total > 0 else 0
            acc.append(accuracy)
        return acc
    else: #otherwise, return a single accuracy score
        # print(predictions)
        correct = sum((p == l) for p, l in zip(predictions[0], labels) if p != -1)
        total = sum((p != -1) for p in predictions[0])
        accuracy = correct / total if total > 0 else 0
        return accuracy


def model_wrapper(inputs_embeds):
    """
    Wrapper function for the model to work with Integrated Gradients
    """
    outputs = model(inputs_embeds=inputs_embeds)
    # Return logits for the last token in the sequence
    return outputs.logits[:, -1, :]

def compute_average_embedding(model, tokenizer, device):
    vocab_size = tokenizer.vocab_size
    input_ids = torch.arange(vocab_size).unsqueeze(0).to(device)  # Shape: (1, vocab_size)
    embeddings = model.get_input_embeddings()(input_ids)         # Shape: (1, vocab_size, embed_dim)
    average_embedding = embeddings.mean(dim=1)                  # Shape: (1, embed_dim)
    return average_embedding

def analyze_attributions(input_text, target_text, visualize=True, check_importance=False, n_steps=25):
    # Tokenize input
    tokens = tokenizer.tokenize(input_text)
    input_encoding = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = input_encoding.input_ids.to(device)
    baseline_ids = torch.tensor( [[128000] + [tokenizer.pad_token_id] * len(tokens)]).to(device) # [ 59 171 118 132], [ 24 202  93 161], AOPC .26, .33
    # baseline_ids = torch.tensor( [[128000] + [44918] * len(tokens)]).to(device) # [ 59 148 133 140], [125 162  79 114], AOPC .18, .21
    # baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id) # APOC .25, .25
    # baseline_ids = torch.full_like(input_ids, tokenizer.bos_token_id) # AOPC 0.07, 0.03
    # baseline_ids = torch.full_like(input_ids, tokenizer.eos_token_id) # [ 89 140  95 156], [ 58 191 120 111], AOPC 0.30, 0.17

    # Get embeddings
    inputs_embeds = model.get_input_embeddings()(input_ids)
    baseline_embeds = model.get_input_embeddings()(baseline_ids)
    # baseline_embeds = compute_average_embedding(model, tokenizer, device).expand(inputs_embeds.shape) # APOC .1, 0
    # baseline_embeds = torch.zeros_like(baseline_embeds) # [180 124  79  97], [167  88  83 142], AOPC .18, .13

    # Get target token ID
    target_token_id = tokenizer.encode(target_text, add_special_tokens=False)[0]
    target_token_id = torch.tensor(target_token_id).to(device)

    # Run integrated gradients
    ig = IntegratedGradients(model_wrapper)
    attributions, delta = ig.attribute(
        inputs_embeds,
        target=target_token_id,
        baselines=baseline_embeds,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    prompt_stats = {'input text': input_text, 'target_text': target_text, 'content': []} #first boundary is prompt, last is question
    if visualize or check_importance:
        # Normalize and log-transform the attributions for better scaling in the heatmap
        attributions = torch.tensor(attributions).sum(dim=-1).squeeze(0)  # Sum across embedding dimension
        boundaries = find_boundaries(tokens)
        for i, (start, end) in enumerate(boundaries):
            section_tokens = tokens[start: end]
            section_token_ids = input_ids[:, start: end]
            section_tokens = [[str(tok) for tok in section_tokens]]
            section_attr = attributions[start: end]

            if check_importance:
                boundary_stats = {'pos': i, 'tokens': section_token_ids, 'attr': section_attr, 'max': torch.max(section_attr).cpu(), 'mean': torch.mean(section_attr).cpu(), 'last': section_attr[-1].cpu()}
                prompt_stats['content'].append(boundary_stats)
            if visualize:
                attributions = torch.clamp(attributions, min=1e-4)
                global_min, global_max = attributions.min().item(), attributions.max().item()
                log_norm = LogNorm(vmin=global_min, vmax=global_max)
                # Plot heatmap
                plt.figure(figsize=(max(len(section_tokens) // 2, 10), 2))
                sns.heatmap(
                    section_attr.cpu().reshape(1,-1), annot=section_tokens, fmt="",
                    cmap='viridis', cbar_kws={'label': 'Integrated Gradient Attribution'},
                    annot_kws={'rotation': 90}, vmin=global_min, vmax=global_max, norm=log_norm
                ) 

                plt.title(f"Integrated Gradients Attribution Heatmap - {i}")
                plt.xlabel('Token Position')
                plt.savefig(str(i) + '.png')
                print("saved")

    return attributions, input_ids, delta, prompt_stats

def find_boundaries(tokens):
    # List of boundary words
    example_boundaries = []
    for i, token in enumerate(tokens):
        if 'Text' in token:
            example_boundaries.append(i)
    
    # Get all of the boundaries. idx 0 should be the question, idx -1 should be the answer prompt
    example_intervals = []
    example_intervals.append((0, example_boundaries[0]))
    for j in range(len(example_boundaries) - 1):
        example_intervals.append((example_boundaries[j], example_boundaries[j + 1]))
    example_intervals.append((example_boundaries[-1], len(tokens)))
    # print(tokens)
    # print(example_intervals)
    # for i, (start, end) in enumerate(example_intervals):
    #     print(tokens[start:end])

    return example_intervals

def permute_label(label, label_map):
    labels_alternative = [l for l in label_map.keys() if l != label]
    newlabel = labels_alternative[random.randint(0,len(labels_alternative) - 1)]
    return newlabel

# Experiment looking at the accuracy as we flip different proportions of the labels in the ICL examples
        # This is meant to respond to the the "gold labels don't matter" result
def graph_accuracy_flipping(max_len, dataset, data_fun, label_map, data_name="SST-2"):
    data_acc = []
    data_acc_50p_rand = []
    data_acc_100p_rand = []

    for i in range(max_len - 1):
        data_acc.append(test_accuracy(pipe, dataset, data_fun, label_map, N=i + 1, K=500))
        data_acc_50p_rand.append(test_accuracy(pipe, dataset, data_fun, label_map, perm_type='rand', perm_prop=.5, N=i + 1, K=500))
        data_acc_100p_rand.append(test_accuracy(pipe, dataset, data_fun, label_map, N=i + 1, perm_type='rand', perm_prop=1, K=500))

    # Plotting the accuracies
    plt.plot(range(max_len - 1) + 1, data_acc, marker='o', linestyle='-', label="Correct Labels")
    plt.plot(range(max_len - 1) + 1, data_acc_50p_rand, marker='s', linestyle='-', label=".5 Flipped")
    plt.plot(range(max_len - 1) + 1, data_acc_100p_rand, marker='x', linestyle='-', label="All Flipped")

    # Adding titles and labels
    plt.title("Model Accuracy with Label Flipping for " + data_name)
    plt.xlabel("Number of In-Context Examples (N)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("flipping_" + data_name + ".png")
    plt.clf()

# Experiment looking at the accuracy as we flip flip one example OR all examples but one!
def graph_accuracy_flipping_pos(max_len, dataset, data_fun, label_map, data_name="SST-2"):
    accuracies = test_accuracy(pipe, dataset, data_fun, label_map, perm_type='pos', N=max_len, K=100)
    accuracies2 = test_accuracy(pipe, dataset, data_fun, label_map, do_weight=True, perm_type='pos', N=max_len, K=100)

    # Plotting the accuracies
    plt.plot(range(max_len), accuracies, linestyle='-', label="Unweighted")
    plt.plot(range(max_len), accuracies2, linestyle='-', label="Weighted")

    # Adding titles and labels
    plt.legend()
    plt.title("Model Accuracy with Single Positional Label Flipping for " + data_name)
    plt.xlabel("Incorrect Context Label")
    plt.ylabel("Accuracy")
    plt.savefig("pos_flipping_inv_" + str(max_len) + "_" + data_name + ".png")
    plt.clf()

def analyze_prompt_structure(dataset, data_fun, label_map, N, K, data_name):
    prompt_stats_total = []
    for _ in tqdm(range(K)):
        prompts, label = create_prompt(dataset, data_fun, N, label_map)
        for prompt in prompts:
            attributions, input_ids, delta, prompt_stats = analyze_attributions(prompt, label, visualize=False, check_importance=True)
            prompt_stats_total.append(prompt_stats)

    pos  = {"structure": [], "content": [], "label": []}
    for prompt_stats in prompt_stats_total:
        # structure = prompt_stats['content'][0]['attr'] #entire first example is structure
        structure = torch.tensor([]).to(device)
        label = torch.tensor([]).to(device)
        content = torch.tensor([]).to(device)
        for i in range(N):
            attr = prompt_stats['content'][i + 1]['attr']
            structure = torch.cat((structure, torch.tensor([attr[0]], device=device))) #"Text"
            structure = torch.cat((structure, torch.tensor([attr[-2]], device=device))) #"Category"
            label = torch.cat((label, torch.tensor([attr[-1]], device=device)))
            content = torch.cat((content, attr[1:-2]))
        # attr = prompt_stats['content'][-1]['attr']
        # structure = torch.cat((structure, torch.tensor([attr[0]], device=device)))
        # structure = torch.cat((structure, torch.tensor([attr[-1]], device=device)))
        # content = torch.cat((content, attr[1:-2]))
        
        pos["structure"].append(torch.mean(structure).cpu())
        pos["content"].append(torch.mean(content).cpu())
        pos["label"].append(torch.mean(label).cpu())
    
    # Convert lists in pos to numpy arrays
    data_to_plot = []
    labels = []
    for key in pos.keys():
        # Ensure all data is converted to numpy arrays
        data = np.array(pos[key])
        # print(data)
        data_to_plot.append(data)
        labels.append(key)

    # Create the boxplot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.boxplot(data_to_plot, patch_artist=True)

    # Add labels and title
    plt.xlabel("Context Component")
    plt.ylabel("Attribution Value")
    plt.title("Boxplot of Mean Attribution by Context Component for " + data_name)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add gridlines
    plt.tight_layout()
    plt.savefig("analysis_stucture_" + data_name + ".png",)
    plt.clf()

def analyze_prompt_positions(dataset, data_fun, label_map, N, K, data_name):
    prompt_stats_total = []
    for _ in tqdm(range(K)):
        prompts, label = create_prompt(dataset, data_fun, N, label_map, perm_type="perm")
        for prompt in prompts:
            attributions, input_ids, delta, prompt_stats = analyze_attributions(prompt, label, visualize=False, check_importance=True)
            # print(prompt_stats)
            prompt_stats_total.append(prompt_stats)

    pos  = [[] for _ in range(N)]
    for prompt_stats in prompt_stats_total:
        for i in range(N):
            pos[i].append(prompt_stats['content'][i + 1]['mean']) # i+1 makes sure it takes content not prompt
    
    pos = np.array(pos)

    _, counts = np.unique(pos.argmax(0), return_counts=True)
    print("Prompt Position Analysis, N = ", N)
    print(counts.tolist())
    print(pos.mean(1).tolist())

    # Create the boxplot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.boxplot(pos.T, patch_artist=True)  # Transpose pos for grouping across rows

    # Add labels and title
    plt.xlabel("Position Index")
    plt.ylabel("Label Value")
    plt.title("Boxplot of Label Attribution by Context Position")
    plt.xticks(ticks=range(1, pos.shape[0] + 1), labels=[f"Position {i}" for i in range(pos.shape[0])])

    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add gridlines
    plt.tight_layout()
    plt.savefig("analysis_counts_" + data_name + ".png",)
    plt.clf()

def get_correctness(input_tokens, label):
    # print(input_tokens.shape)
    prompt = str(tokenizer.decode(input_tokens[0].to(torch.long), skip_special_tokens=True))
    output = pipe(prompt, max_length=(len(pipe.tokenizer.tokenize(prompt)) + 2), do_sample=False)[0]['generated_text']
    generated = output[len(prompt):].strip()
    
    if label in generated:
        return 1
    else:
        return 0

def aopc(dataset, data_fun, label_map, N, K, perms, data_name):
    perm_data = [[] for _ in range(perms)]
    perm_data_rand = [[] for _ in range(perms)]
    for _ in tqdm(range(K)):
        prompt, label = create_prompt(dataset, data_fun, N, label_map)
        attributions, input_ids, delta, prompt_stats = analyze_attributions(prompt[0], label, visualize=False, check_importance=True)

        context_tokens = torch.tensor([]).to(device)
        context_attr = torch.tensor([]).to(device)
        for i in range(N):
            context_tokens = torch.cat((context_tokens, prompt_stats['content'][i + 1]['tokens']), dim=1)
            context_attr = torch.cat((context_attr, prompt_stats['content'][i + 1]['attr']))
        _, sorted_idxs = torch.sort(context_attr, descending=True)
        # print(context_tokens)
        # print(sorted_idxs)

        for p in range(perms):
            to_replace = sorted_idxs[:p]
            # to_replace = torch.randint(0, int(torch.max(sorted_idxs)), size=(len(to_replace),)) # do random
            context_tokens_replaced = context_tokens.clone()
            context_tokens_replaced[:, to_replace] = tokenizer.pad_token_id # ID of PAD

            total_tokens = prompt_stats['content'][0]['tokens'][:, 1:] #gets rid of BOS
            total_tokens = torch.cat((total_tokens, context_tokens_replaced), dim=1)
            total_tokens = torch.cat((total_tokens, prompt_stats['content'][-1]['tokens']), dim=1) #gets rid of EOS

            perm_data[p].append(get_correctness(total_tokens.to(torch.uint32), label))

        for p in range(perms):
            to_replace2 = torch.randint(0, int(torch.max(sorted_idxs)), size=(p,)) # do random
            # print(to_replace)
            context_tokens_replaced2 = context_tokens.clone()
            context_tokens_replaced2[:, to_replace2] = tokenizer.pad_token_id # ID of PAD

            total_tokens2 = prompt_stats['content'][0]['tokens'][:, 1:] #gets rid of BOS
            total_tokens2 = torch.cat((total_tokens2, context_tokens_replaced2), dim=1)
            total_tokens2 = torch.cat((total_tokens2, prompt_stats['content'][-1]['tokens']), dim=1) #gets rid of EOS

            perm_data_rand[p].append(get_correctness(total_tokens2.to(torch.uint32), label))

    data = np.array(perm_data).mean(1)
    data_rand = np.array(perm_data_rand).mean(1)

    def aopc_fun(data):
        data = data[0] - data
        data = np.cumsum(data)
        p_range = np.arange(perms)
        p_range[0] = 1
        return data / p_range

    # Plotting the accuracies
    plt.plot(np.arange(perms), data, linestyle='-', label='IG Replacement')
    plt.plot(np.arange(perms), data_rand, linestyle='-', label='Random Replacement')

    print(aopc_fun(data))

    # Adding titles and labels
    plt.legend()
    # plt.title("AOPC Curve for " + data_name, size=14)
    # plt.xlabel("# of Replaced Tokens",size=12)
    # plt.ylabel("AOPC Value", size=12)
    plt.grid()
    # plt.savefig("AOPC_" +  "_" + data_name + ".png")
    plt.title("Replacement Accuracy Curve for " + data_name)
    plt.xlabel("# of Replaced Tokens")
    plt.ylabel("Accuracy")
    plt.savefig("AOPC_acc_" +  "_" + data_name + ".png")
    plt.clf()


"""
Kayla, instructions for computing AOPC. This is super similar to LOO.

1. Start by rank tokens by IG attribution, meaning use the IG scores to rank tokens (we only do this at token-level for now) from most to least important.

2. Iteratively remove top-ranked tokens. So starting with no removals, first measure the model’s accuracy, then mask the top-N tokens according to the attributions with ([PAD] token?) and re-measure the performance. Increase N incrementally (e.g., top-1 token removed, then top-2 tokens removed, top-5, top-10, top-20) and record model accuracy at each step. I think we can do {1, 2, 5, 10, 20} tokens removed to start. 

3. Compute the performance curve by plotting model performance against the number of removed tokens. We should see model accuracy degrade as more high-attribution tokens are removed if the attributions are faithful.

4. Calculate AOPC: for each PAD step k, measure the drop in performance from the original. The AOPC is just the average of these performance drops over all chosen steps, i.e., $AOPC = \frac{1}{L} \sum_{i=1}^{L} \left( M(0) - M(k_i) \right)$
"""


if __name__ == "__main__":
    # print("sst2 acc", test_accuracy(pipe, sst2, create_prompt_sst2, sst_label_map, K=100))
    # print("ag news acc", test_accuracy(pipe, ag_news, create_prompt_agnews, ag_news_label_map, K=100))
    # print("trec", test_accuracy(pipe, trec, create_prompt_trec, trec_label_map, K=100))
    # print("setfit", test_accuracy(pipe, setfit, create_prompt_setfit, setfit_label_map, K=100))
    # graph_accuracy_flipping(11, ag_news, create_prompt_agnews, ag_news_label_map, data_name="AG-News")
    # graph_accuracy_flipping(11, sst2, create_prompt_sst2, sst_label_map, data_name="SST-2")
    # graph_accuracy_flipping_pos(10, ag_news, create_prompt_agnews, ag_news_label_map, data_name="AG-News")
    # graph_accuracy_flipping_pos(10, sst2, create_prompt_sst2, sst_label_map, data_name="SST-2")
    # prompt, label = create_prompt(ag_news, create_prompt_agnews, 3, ag_news_label_map)
    # attributions, input_ids, delta, prompt_stats = analyze_attributions(prompt[0], label, visualize=True, check_importance=True)
    # print(prompt_stats)
    # Ns = [5, 10, 15,20]
    # for n in Ns:
        # analyze_prompt_positions(ag_news, create_prompt_agnews, ag_news_label_map, n, 100, "AG-News")
        # analyze_prompt_positions(sst2, create_prompt_sst2, sst_label_map, n, 100, "SST-2")
        # analyze_prompt_positions(trec, create_prompt_trec, trec_label_map, n, 200, "Trec")
        # analyze_prompt_positions(setfit, create_prompt_setfit, setfit_label_map, n, 200, "Setfit")
    # analyze_prompt_structure(ag_news, create_prompt_agnews, ag_news_label_map, 3, 1000, "AG-News")
    # analyze_prompt_structure(sst2, create_prompt_sst2, sst_label_map, 3, 1000, "SST-2")
    aopc(ag_news, create_prompt_agnews, ag_news_label_map, 4, 1000, 10, "AG-News")
    aopc(sst2, create_prompt_sst2, sst_label_map, 4, 1000, 10, "SST-2")
    # aopc(trec, create_prompt_trec, trec_label_map, 4, 1000, 10, "Trec")
    # aopc(setfit, create_prompt_setfit, setfit_label_map, 4, 1000, 10, "Setfit")


"""
    mean of tokens - baseline_embeds = inputs_embeds.mean(1).expand(inputs_embeds.shape) # AOPC 0.03, 0.04
    same as tokens - baseline_embeds = inputs_embeds # APOC 0,0

    values = np.array([0.10899745, 0.17927496, 0.15147824, 0.16096526])
    weights = np.mean(values) / values

    For normal baseline:
    [286 840 586 688], [0.10899745 0.17927496 0.15147824 0.16096526], weights = [1.33124871, 0.8093864 , 0.95791128, 0.90145361]
    [ 136 1063  460  741], [0.07809052 0.27826612 0.20468691 0.25650586], weights = [2.61731325, 0.73450319, 0.99853651, 0.79681358]


            # tokenized = tokenizer(prompt, return_tensors="pt")
            # output_ids = model.generate(
            #     input_ids=tokenized["input_ids"].to(model.device),
            #     # embedding_weights=torch.tensor(1),
            #     attention_mask=tokenized["attention_mask"].to(model.device),
            #     max_new_tokens=2,         # or however many you need
            #     do_sample=False
            # )
            # output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Define custom model for embedding weighting
class WeightedEmbeddingsModel(PreTrainedModel):
    A wrapper model that intercepts the forward pass
    to apply custom weighting to the input embeddings.
    def __init__(self, base_model):
        Args:
            base_model: An instance of AutoModelForCausalLM.
            weight: A scalar or tensor to multiply with the embeddings.
        super().__init__(base_model.config)
        self._base_model = base_model

    def forward(
        self,
        input_ids=None,
        embedding_weights=None,
        attention_mask=None,
        **kwargs
    ):
        
        # If the pipeline/generation calls with inputs_embeds directly, 
        # you can decide what to do or just pass them through:
        if input_ids is not None:
            # 1. Convert IDs -> embeddings
            embeddings = self._base_model.get_input_embeddings()(input_ids)
            # 2. Apply your custom weighting
            embeddings = embeddings * embedding_weights
            # 3. Call the underlying model with inputs_embeds
            return self._base_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            # If something calls forward with 'inputs_embeds' already, you can:
            return self._base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
    def generate(self, *args, **kwargs):
        return self._base_model.generate(*args, **kwargs)

    model = WeightedEmbeddingsModel(model_init)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model = model_init

RESULTS:
100 example sets, 25 permutations
n = 5
AG-News:
[254, 761, 454, 410, 621]
[246, 742, 468, 422, 622]
SST:
[68, 915, 289, 380, 848]
[55, 1030, 296, 324, 795]
n = 10
AG-News:
[86, 295, 214, 184, 144, 165, 163, 193, 311, 745]
[67, 365, 243, 173, 148, 161, 175, 159, 280, 729]
SST:
[151, 933, 186, 104, 82, 67, 90, 103, 191, 593]
[150, 899, 235, 88, 72, 78, 66, 109, 190, 613]
n = 15
AG-News:
[88, 291, 189, 132, 96, 83, 72, 51, 63, 63, 67, 112, 169, 286, 738]
[62, 263, 182, 124, 92, 78, 74, 70, 65, 66, 70, 123, 197, 329, 705]
SST:
[139, 735, 144, 65, 63, 64, 46, 48, 41, 48, 45, 86, 126, 235, 615]
[129, 704, 139, 94, 72, 69, 62, 41, 41, 64, 72, 90, 142, 256, 525]
n = 20
AG-News:
[36, 226, 134, 94, 72, 49, 44, 32, 33, 34, 38, 38, 59, 40, 58, 90, 116, 197, 348, 762]
[47, 223, 142, 96, 58, 37, 36, 28, 41, 42, 45, 56, 50, 59, 75, 77, 106, 223, 370, 689]
SST:
[134, 583, 100, 57, 37, 44, 38, 39, 37, 23, 35, 35, 44, 33, 56, 63, 99, 132, 278, 633]
[136, 558, 100, 41, 45, 38, 27, 23, 25, 26, 27, 28, 33, 32, 51, 62, 95, 149, 295, 709]


n = 5
AG-News:
[0.08976933848569064, 0.15455372205263665, 0.12238643962177534, 0.11253332255577989, 0.13475652495695067]
[0.09537184107238829, 0.16387564302933177, 0.13313940912674285, 0.12163106392858522, 0.14448854300657332]
SST:
[0.05543741564252116, 0.23752596519765584, 0.15153398728091896, 0.16805095321595662, 0.2329744190019281]
[0.05720136098574211, 0.25874230353369376, 0.1563640325949388, 0.16636000936031886, 0.23576831350145003]
n = 10
AG-News:
[0.04823528754957127, 0.09465735799239683, 0.08247654180216699, 0.07303488907692479, 0.06768973549524483, 0.06694261391897946, 0.06704060584382551, 0.0697414742262889, 0.08627144306499256, 0.13299802094350843]
[0.04702094751649162, 0.10277258147521852, 0.08734301652876983, 0.07751455029553327, 0.07137450303278449, 0.06958209730732824, 0.07012790215314492, 0.06860134085419993, 0.09002250564334763, 0.13728199353117854]
SST:
[0.06470626388110268, 0.18142883045979916, 0.09618868868365235, 0.07286780676583583, 0.06225958456751833, 0.05925058918028445, 0.06009861109915195, 0.0708862616741443, 0.09755606890212871, 0.15654878528405808]
[0.06302502578324067, 0.18222646213230032, 0.09277911938706264, 0.07146190341545605, 0.0615827696199158, 0.05580358179990019, 0.0604410469098151, 0.06988987097060091, 0.09687564347557187, 0.14833976294796108]
n = 15
AG-News:
[0.04686198906140689, 0.08879552723587028, 0.07336835452298851, 0.06340595676922257, 0.05427050805669187, 0.04924954077479924, 0.04694669358306254, 0.04266461089146293, 0.03697548018499714, 0.03688911225301223, 0.03892574245918296, 0.04575650947671366, 0.06283696050444165, 0.08206151217707743, 0.11935737221791515]
[0.03781136467659725, 0.08794703205870963, 0.07526821101183014, 0.06560861932011401, 0.05708050048596974, 0.05051081661123826, 0.048224033049629314, 0.04398311314339909, 0.03598325445040697, 0.03691381589081513, 0.040632489584367534, 0.04874935579090342, 0.06557956429683214, 0.08536047638189312, 0.12172320710329795]
SST:
[0.05775166697220505, 0.14454597750386158, 0.07593844885921897, 0.06268747463511501, 0.05764905911735219, 0.05435521225500561, 0.050527618998164284, 0.05019695473844555, 0.04764977000115116, 0.048023850943918485, 0.05074274846076722, 0.058690418530244336, 0.07023718267491588, 0.09156363303502489, 0.1338774206615708]
[0.057709106881158934, 0.1489702182218405, 0.08082156628446485, 0.06590275077494043, 0.062362210705957544, 0.059787370313941685, 0.057306427908577845, 0.05302357648092725, 0.05225863731384761, 0.056591317614218054, 0.058175226415838846, 0.06365521321680453, 0.07918264964643502, 0.09968140028370523, 0.12954530569994868]
n = 20
AG-News:
[0.03124389755650408, 0.07383895024032461, 0.06533750986393352, 0.05669688652840896, 0.04953305698829476, 0.04240242363670102, 0.03767026036376038, 0.03481399631359284, 0.03347847046077517, 0.03374020077143962, 0.03475954847773007, 0.03595111172240901, 0.04223541649007775, 0.03820421931730975, 0.04258715697276319, 0.0468760962818269, 0.05207184256761229, 0.06772586154146397, 0.08149496166381179, 0.11210355875261088]
[0.03647843981869548, 0.07864530045987474, 0.06705855608774115, 0.05651965054366774, 0.04746104619245237, 0.041967308685691034, 0.034660779564576125, 0.03246030730490629, 0.032453269077666276, 0.03304794487766269, 0.032447882612953824, 0.0361604601888445, 0.04112522362336207, 0.038529819539062905, 0.04135876956115213, 0.04521260151793342, 0.04870572017719096, 0.06671382905313818, 0.0859281876811091, 0.11074106277900755]
SST:
[0.05042831153544713, 0.12472633224915416, 0.06451375828629205, 0.050468021407546006, 0.043603444950505145, 0.04319827324691356, 0.042654084602122985, 0.0453648881691145, 0.045660901522592154, 0.043312242453106595, 0.04432947241764249, 0.043822718074884415, 0.04618566069563697, 0.044099134747689304, 0.04664271368461421, 0.04868398671567611, 0.058316243656826326, 0.06852335955952973, 0.09299585487616603, 0.12739479872828874]
[0.05480570060783724, 0.12026502411140651, 0.06288167019805325, 0.04864715415996678, 0.04618189932750346, 0.043156935262056426, 0.04186758687828176, 0.04396636000916285, 0.04270363697530937, 0.044794614645902175, 0.04516459118121199, 0.04322443920490352, 0.04520011551487363, 0.04452628958772072, 0.04681134689761443, 0.048825538758380686, 0.05815471869490132, 0.0720891992822138, 0.09626497771021553, 0.13592515006876218]
"""
