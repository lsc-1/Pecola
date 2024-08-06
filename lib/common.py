import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score

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
        #for input_ids, batch_labels, attention_masks,articles, generated_texts in tqdm(data_loader):
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
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                # Save probabilities for both classes
                probas.extend(probabilities.tolist())

    if metric == 'acc':
        acc = accuracy_score(labels, preds)
        zero_acc = predicted_true_zeros / true_zeros if true_zeros > 0 else 0  # compute accuracy for label 0
        one_acc = predicted_true_ones / true_ones if true_ones > 0 else 0  # compute accuracy for label 1
        print('实际标签为0的数量:', true_zeros)
        print('预测标签为0的数量:', predicted_true_zeros)
        print('实际标签为1的数量:', true_ones)
        print('预测标签为1的数量:', predicted_true_ones)
        print('标签为0的准确率:', zero_acc)
        print('标签为1的准确率:', one_acc)  
    #elif metric == 'f1':
        f1 = f1_score(labels, preds, average='macro')
        #print('标签为1的准确率:', one_acc) 

    if return_proba:
        return preds, probas, labels, true_zeros, predicted_true_zeros, zero_acc  # return accuracy for label 0
    else:
        return float(acc), zero_acc,one_acc,f1  # return accuracy for label 0
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

def combinedLoss(pooled, labels):
    # Distance calculation
    dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
    
    # Mask for positive and negative samples
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    
    # Remove diagonal values
    mask = mask - torch.diag(torch.diag(mask))

    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
    cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
    pos_loss = (dist - cosine_score) * mask
    pos_loss = pos_loss.sum(-1) / (mask.sum(-1) + 1e-3)

    max_dist = (dist * mask).max()
    neg_loss = F.relu(max_dist - dist) * neg_mask
    neg_loss = neg_loss.sum(-1) / (neg_mask.sum(-1) + 1e-3)

    combined_loss = pos_loss.mean() + neg_loss.mean()
    
    return combined_loss

def train_common(model,optimizer,data_loader, epoch, log_steps=100,  pre_model=None, shift_reg=0.1, scl_reg=2.0, loss_type='ce', 
    save_steps=-1, save_dir=None, step_counter=0, max_grad_norm=1.0):
    model.train()
    if pre_model is not None:
        pre_model.eval()
    running_loss = 0
    step_count = 0
    for input_ids, labels, attention_masks in data_loader:
        if hasattr(model, 'centers') or ( hasattr(model,'module') and hasattr(model.module, 'centers')): #LMCL loss version
            outputs = model(input_ids, attention_mask=attention_masks, label=labels)
        else:
            outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        if hasattr(outputs,'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if shift_reg > 0 and pre_model is not None:     
            dist = torch.sum(torch.abs(torch.cat(
                [p.view(-1) for n, p in model.named_parameters() if 'bert' in n.lower()]) - torch.cat(
                [p.view(-1) for n, p in pre_model.named_parameters() if
                    'bert' in n.lower()])) ** 2).item()
            loss += shift_reg*dist
        if loss_type == 'scl':
            loss += scl_reg*sclLoss(outputs.hidden_states[-1][:,0,:], labels)
        elif loss_type == 'margin':
            loss += scl_reg*marginLoss(outputs.hidden_states[-1][:,0,:], labels)
        elif loss_type == 'combine':
            loss += scl_reg*combinedLoss(outputs.hidden_states[-1][:,0,:], labels)
        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if (step_count+1) % log_steps == 0:
            #print(labels)
            logger.info("step {}, running loss = {}".format(step_count+1, running_loss))
            running_loss = 0
        step_count += 1
        step_counter += 1
        
        if save_steps != -1 and save_dir is not None:
            if step_counter % save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(),'{}/step{}_model.pt'.format(save_dir, step_counter))
                logger.info("model saved to {}/step{}_model.pt".format(save_dir, step_counter))
    return step_counter





def train_common2(model, optimizer, data_loader, epoch, log_steps=100, pre_model=None, shift_reg=0, scl_reg=2.0, 
                  loss_type='ce', save_steps=-1, save_dir=None, step_counter=0, max_grad_norm=1.0):
    
    # 初始化跟踪变量
    ce_loss_sum = 0.0
    ce_loss_squared_sum = 0.0
    ce_loss_count = 0

    margin_loss_sum = 0.0
    margin_loss_squared_sum = 0.0
    margin_loss_count = 0

    model.train()
    if pre_model is not None:
        pre_model.eval()
    
    running_loss = 0
    step_count = 0

    for input_ids, labels, attention_masks in data_loader:
        if hasattr(model, 'centers') or ( hasattr(model,'module') and hasattr(model.module, 'centers')): # LMCL loss version
            outputs = model(input_ids, attention_mask=attention_masks, label=labels)
        else:
            outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        loss_fct = CrossEntropyLoss()
        raw_ce_loss = loss_fct(logits, labels)

        # 更新交叉熵损失的跟踪变量
        ce_loss_sum += raw_ce_loss.item()
        ce_loss_squared_sum += raw_ce_loss.item() ** 2
        ce_loss_count += 1
        ce_loss_mean = ce_loss_sum / ce_loss_count
        ce_loss_std = (ce_loss_squared_sum / ce_loss_count - ce_loss_mean ** 2) ** 0.5 + 1e-7

        # 对交叉熵损失进行Z-Score归一化
        normalized_ce_loss = (raw_ce_loss - ce_loss_mean) / ce_loss_std

        if shift_reg > 0 and pre_model is not None:
            dist = torch.sum(torch.abs(torch.cat(
                [p.view(-1) for n, p in model.named_parameters() if 'bert' in n.lower()]) - torch.cat(
                [p.view(-1) for n, p in pre_model.named_parameters() if 'bert' in n.lower()])) ** 2).item()
            normalized_ce_loss += shift_reg * dist
        if loss_type == 'scl':
            loss += scl_reg*sclLoss(outputs.hidden_states[-1][:,0,:], labels)
        elif loss_type == 'margin':
            raw_margin_loss = scl_reg * marginLoss(outputs.hidden_states[-1][:, 0, :], labels)
            
            # 更新marginLoss的跟踪变量
            margin_loss_sum += raw_margin_loss.item()
            margin_loss_squared_sum += raw_margin_loss.item() ** 2
            margin_loss_count += 1
            margin_loss_mean = margin_loss_sum / margin_loss_count
            margin_loss_std = (margin_loss_squared_sum / margin_loss_count - margin_loss_mean ** 2) ** 0.5 + 1e-7

            # 对marginLoss进行Z-Score归一化
            normalized_margin_loss = (raw_margin_loss - margin_loss_mean) / margin_loss_std

            loss = normalized_ce_loss + normalized_margin_loss
        # else:
        #     loss = normalized_ce_loss

        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if (step_count + 1) % log_steps == 0:
            logger.info("step {}, running loss = {}".format(step_count + 1, running_loss))
            running_loss = 0
        step_count += 1
        step_counter += 1

        if save_steps != -1 and save_dir is not None:
            if step_counter % save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), '{}/step{}_model.pt'.format(save_dir, step_counter))
                logger.info("model saved to {}/step{}_model.pt".format(save_dir, step_counter))

    return step_counter


