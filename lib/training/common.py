import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score
from lib.models.networks import get_tokenizer


def test_acc(model, data_loader, metric='acc', return_proba=False):
    model.eval()
    preds = []
    labels = []
    probas = []
    true_zeros = 0
    true_ones = 0
    predicted_true_zeros = 0
    predicted_true_ones = 0
    with torch.no_grad():
        for input_ids, batch_labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[-1] if type(outputs) == list else outputs
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(batch_labels.reshape(-1).cpu().numpy())
            # Count the number of true zeros and correctly predicted zeros
            true_zeros += np.sum(batch_labels.cpu().numpy() == 0)
            predicted_true_zeros += np.sum((batch_preds == 0) & (batch_labels.cpu().numpy() == 0))
            # Count the number of true ones and correctly predicted ones
            true_ones += np.sum(batch_labels.cpu().numpy() == 1)
            predicted_true_ones += np.sum((batch_preds == 1) & (batch_labels.cpu().numpy() == 1))
            if return_proba:
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                probas.extend(probabilities.tolist())
    if metric == 'acc':
        acc = accuracy_score(labels, preds)
        zero_acc = predicted_true_zeros / true_zeros if true_zeros > 0 else 0  # compute accuracy for label 0
        one_acc = predicted_true_ones / true_ones if true_ones > 0 else 0
        f1 = f1_score(labels, preds, average='macro')
        human_recall = recall_score(labels, preds, pos_label=0)
        machine_recall = recall_score(labels, preds, pos_label=1)
        recall = (human_recall+machine_recall)/2.0
    if return_proba:
        return preds, probas, labels, true_zeros, predicted_true_zeros, zero_acc  # return accuracy for label 0
    else:
        return float(acc), float(f1),float(recall)
    

def marginLoss(pooled,labels):
    dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    max_dist = (dist * mask).max()
    cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
    cos_loss = cos_loss.mean()
    return cos_loss

def sclLoss(pooled, labels):
    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
    mask = mask - torch.diag(torch.diag(mask))
    cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
    cos_loss = -torch.log(cos_loss + 1e-5)
    cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
    cos_loss = cos_loss.mean()
    return cos_loss


import yake
def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(lan='en', n=1)
    keywords = kw_extractor.extract_keywords(text)
    return dict(keywords)


def convert_input_ids_to_text(input_ids, tokenizer):
    texts = []
    for ids in input_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        texts.append(text)
    return texts
def marginLoss_yake(pooled, labels, keyword_scores, tokenizer,input_ids):
    weights = torch.ones_like(pooled)

    for token, score in keyword_scores.items():
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        token_positions = (input_ids == token_id).nonzero()
        for pos in token_positions:
            weights[pos[0], pos[1]] *= (2-score)
    weighted_pooled = pooled * weights
    dist = ((weighted_pooled.unsqueeze(1) - weighted_pooled.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    max_dist = (dist * mask).max()
    cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (torch.nn.functional.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
    cos_loss = cos_loss.mean()
    return cos_loss
def train_common_yake(model, optimizer, data_loader, epoch, log_steps=100, pre_model=None, shift_reg=0.1, scl_reg=2.0, loss_type='ce', save_steps=-1, save_dir=None, step_counter=0, max_grad_norm=1.0):
    model.train()
    if pre_model is not None:
        pre_model.eval()
    running_loss = 0
    step_count = 0
    tokenizer = get_tokenizer('beyond/roberta-base')

    for input_ids, labels, attention_masks in data_loader:
        outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if shift_reg > 0 and pre_model is not None:
            dist = torch.sum(torch.abs(torch.cat([p.view(-1) for n, p in model.named_parameters() if 'bert' in n.lower()]) - torch.cat([p.view(-1) for n, p in pre_model.named_parameters() if 'bert' in n.lower()])) ** 2).item()
            loss += shift_reg*dist
        if loss_type == 'margin_weight':
            texts = convert_input_ids_to_text(input_ids, tokenizer)
            for text in texts:
                keyword_scores = extract_keywords(text)
                loss += scl_reg * marginLoss_yake(outputs.hidden_states[-1][:,0,:], labels, keyword_scores, tokenizer, input_ids)

        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if (step_count+1) % log_steps == 0:
            logger.info(f"step {step_count+1}, running loss = {running_loss}")
            running_loss = 0
        step_count += 1
        step_counter += 1

        if save_steps != -1 and save_dir is not None:
            if step_counter % save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), f'{save_dir}/step{step_counter}_model.pt')
                logger.info(f"model saved to {save_dir}/step{step_counter}_model.pt")

    return step_counter