from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_features(model, data_loaders, pooler_pos=0):
    correct, total = 0, 0
    features = [ [] for i in range(12)]
    labels = []
    for idx,data_loader in enumerate(data_loaders):
        for input_ids, batch_labels, attention_masks in tqdm(data_loader, desc="sample_estimator"):
            total += input_ids.shape[0]
            with torch.no_grad():
                outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                #print(hidden_states[1].shape)
                out_features = [hidden_states[i][:,pooler_pos,:] for i in range(1,13)]
                #print(out_features[0].shape)
                # get hidden features
                for i in range(12):
                    out_features[i] = out_features[i].view(out_features[i].size(0), -1).cpu().numpy()
                #print(out_features[0].shape)
            if idx == 0:
                labels.append(batch_labels.cpu().numpy())
            else:
                labels.append(np.array([-1 for _ in range(input_ids.shape[0])]))
            for i in range(12):
                features[i].append(out_features[i])
        
            assert len(batch_labels) == len(out_features[0]),"{} {} {}".format(len(batch_labels), len(out_features[0]), len(input_ids))
    labels =  np.concatenate(labels)
    for i in range(12):
        features[i] = np.concatenate(features[i])
    print(labels.shape)
    return features,labels

def get_cse_features(model, data_loader):
    features, labels = [], []
    total = 0
    model.eval()
    for input_ids, batch_labels, attention_masks in tqdm(data_loader, desc="extracting features"):
        with torch.no_grad():
            embeddings = model(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True).pooler_output
            features.append(embeddings.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    labels =  np.concatenate(labels)
    features = np.concatenate(features, axis=0)
    print(features.shape)
    return features,labels

from typing import List, Optional
def get_FFN_features(model, 
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    feature_position: Optional[str] = 'self-attn',
):
    assert feature_position in ['self-attn', 'intermediate']
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    if model.config.is_decoder:
        use_cache = use_cache if use_cache is not None else model.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    if token_type_ids is None:
        if hasattr(model.embeddings, "token_type_ids"):
            buffered_token_type_ids = model.base_embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    extended_attention_mask: torch.Tensor = model.get_extended_attention_mask(attention_mask, input_shape, device='cuda')

    head_mask = model.get_head_mask(head_mask, model.config.num_hidden_layers)

    hidden_states = model.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
    )
    
    features = []
    for i, layer_module in enumerate(model.encoder.layer):

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = layer_module.attention(
            hidden_states,
            attention_mask=extended_attention_mask,
            #layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            #past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        if feature_position == 'self-attn':
            features.append(attention_output) # batch_size, sequence length, hidden_size
        intermediate_output = layer_module.intermediate(attention_output)
        if feature_position == 'intermediate':
            features.append(intermediate_output) # batch_size, sequence length, hidden_size
        hidden_states = layer_module.output(intermediate_output, attention_output)
        
    return torch.stack(features) # num_layers, batch_size, sequence_length, hidden_size

def get_full_features(model, data_loader, pooling='avg', mode='eval', pos='after'):
    correct, total = 0, 0
    features = []
    labels = []
    lengths = []
    model.eval()
    if mode == 'train':
        model.train()
    assert pooling in ['cls', 'max', 'avg']
    for input_ids, batch_labels, attention_masks in tqdm(data_loader, desc="sample_estimator"):
        total += input_ids.shape[0]
        with torch.no_grad():
            if pos == 'after': # after every complete Transfomer layer (i.e., between FFN layers and the next self-attn)
                outputs = model.base_model(input_ids, attention_masks, return_dict=True, output_hidden_states=True)                
                hidden_states = outputs.hidden_states  # layers, batch_size, sequence_length, hidden_size
                hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states], dim=0) # layers, batch_size, sequence_length, hidden_size
            elif pos == 'self-attn': # after self-attn, before two FFN layers
                hidden_states = get_FFN_features(model.base_model, input_ids, attention_masks, feature_position='self-attn')
            elif pos == 'inter': # between FFN1 and FFN2 (n_dim=3072 in BERT)
                hidden_states = get_FFN_features(model.base_model, input_ids, attention_masks, feature_position='intermediate')
            else:
                raise NotImplementedError
            if pooling == 'cls':
                hidden_states = hidden_states[:,:, 0, :]
            elif pooling == 'max':
                input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                hidden_states = torch.max(hidden_states, 2)[0]
            else:
                input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 2)
                sum_mask = input_mask_expanded.sum(2)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                hidden_states = sum_embeddings/sum_mask
                #print(hidden_states.shape)

            features.append(hidden_states.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            lengths.append(np.array([sum(mask) for mask in attention_masks.cpu().numpy()]))
    labels =  np.concatenate(labels)
    features = np.concatenate(features, axis=1)
    #features = torch.cat(features, dim=1)
    lengths = np.concatenate(lengths)
    print(features.shape) # layers, total_size, hidden_size
    return features,labels, lengths


def get_features_pooling(model, data_loaders, pooling='first_last_avg'):
    correct, total = 0, 0
    features = []
    labels = []
    model.eval()
    token_pooling = pooling.split('_')[-1]
    layer_pooling = '_'.join(pooling.split('_')[:-1])
    assert token_pooling in ['cls','avg','max']
    for idx,data_loader in enumerate(data_loaders):
        for input_ids, batch_labels, attention_masks in tqdm(data_loader, desc="sample_estimator"):
            total += input_ids.shape[0]
            with torch.no_grad():
                outputs = model.base_model(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states], dim=0) # layers, batch_size, sequence_length, hidden_size
                if token_pooling == 'cls':
                    hidden_states = hidden_states[:,:, 0, :]
                elif token_pooling == 'max':
                    input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
                    hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                    hidden_states = torch.max(hidden_states, 2)[0]
                else:
                    input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 2)
                    sum_mask = input_mask_expanded.sum(2)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    hidden_states = sum_embeddings/sum_mask
                hidden_states = hidden_states.cpu().numpy()
                # layers, batch_size, hidden_size
                if layer_pooling == 'last':
                    out_features = hidden_states[-1]
                elif layer_pooling == 'first_last':
                    out_features = (hidden_states[-1] + hidden_states[1])/2
                elif layer_pooling == 'last':
                    out_features = hidden_states[-1]
                elif layer_pooling == 'last2':
                    out_features = (hidden_states[-1] + hidden_states[-2])/2
                elif layer_pooling == 'avg':
                    out_features = np.mean(hidden_states[1:], axis=0)
                elif ',' in layer_pooling:
                    layers = [int(i) for i in range(layer_pooling.split(','))]
                    out_features = np.mean(hidden_states[layers], axis=0)
                else:
                    raise Exception("unknown pooling way: {}".format(layer_pooling))
                #print(out_features[0].shape)
            if idx == 0:
                labels.append(batch_labels.cpu().numpy())
            else:
                labels.append(np.array([-1 for _ in range(input_ids.shape[0])]))
            #for i in range(12):
            features.append(out_features)
        
            assert len(batch_labels) == len(out_features),"{} {} {}".format(len(batch_labels), len(out_features), len(input_ids))
    labels =  np.concatenate(labels)
    #for i in range(12):
    #    features[i] = np.concatenate(features[i])
    features = np.concatenate(features)
    print(labels.shape)
    return features,labels


def draw(model, dataloaders,pic_path):
    features, labels = get_features_pooling(model, dataloaders)
    tsne = TSNE(n_components=2)
    print(features.shape)
    np.save('{}/labels.npy'.format(pic_path), labels)
    tsne_embedding = tsne.fit_transform(features)
    np.save('{}/emb_lfa.npy'.format(pic_path), tsne_embedding)
    print(tsne_embedding.shape)


import torch
import numpy as np
import torch.nn.functional as F

from loguru import logger
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor


def pca_analysis(model, ind_train_loader, ind_test_loader, ood_test_loader, pooler_pos=0):

    # get features 
    logger.info("Getting Features")
    ind_train_features,_ = get_features(model, [ind_train_loader], pooler_pos=pooler_pos)
    ind_train_features = ind_train_features[-1]
    ind_test_features, ind_test_labels = get_features(model, [ind_test_loader], pooler_pos=pooler_pos)
    ind_test_features = ind_test_features[-1]
    ood_test_features,_ = get_features(model, [ood_test_loader], pooler_pos=pooler_pos)
    ood_test_features = ood_test_features[-1]
    ood_test_labels = [-1 for _ in range(ood_test_features.shape[0])]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(ind_train_features)
    features = np.concatenate([ind_test_features, ood_test_features])
    labels = np.concatenate([ind_test_labels, ood_test_labels])
    pic_path = 'pics'
    np.save('{}/labels.npy'.format(pic_path), labels)
    pca_embedding = pca.transform(features)
    np.save('{}/emb_pca.npy'.format(pic_path), pca_embedding)

