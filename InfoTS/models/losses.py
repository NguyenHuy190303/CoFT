import torch
import random
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d



def subsequence_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    s1 = torch.unsqueeze(torch.max(z1, 1)[0], 1)
    s2 = torch.unsqueeze(torch.max(z2, 1)[0], 1)
    loss = instance_contrastive_loss(s1, s2)
    exit(1)
    return loss


def subsequence_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z2 = z2[:,start:start+crop_leng,:]

    crop_z1 = crop_z1.view(B ,k,crop_size,D)
    crop_z2 = crop_z2.view(B ,k,crop_size,D)

    # debug
    # crop_z1 = crop_z1.reshape(B * k, crop_size, D)
    # crop_z2 = crop_z2.reshape(B * k, crop_size, D)
    # return instance_contrastive_loss(crop_z1, crop_z2)+temporal_contrastive_loss(crop_z1,crop_z2)


    if pooling=='max':
        # crop_z1_pooling = torch.max(crop_z1,2)[0]
        # crop_z2_pooling = torch.max(crop_z2,2)[0]
        # crop_z1_pooling = torch.unsqueeze(crop_z1_pooling.view(B*k, D), 1)
        # crop_z2_pooling = torch.unsqueeze(crop_z2_pooling.view(B*k, D), 1)


        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z2 = crop_z2.reshape(B*k,crop_size,D)

        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)
        crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)


    return InfoNCE(crop_z1_pooling,crop_z2_pooling,temperature)



def local_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z1 = crop_z1.view(B ,k,crop_size,D)


    # crop_z2 = z2[:,start:start+crop_leng,:]
    # crop_z2 = crop_z2.view(B ,k,crop_size,D)


    if pooling=='max':
        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2).reshape(B,k,D)

        # crop_z2 = crop_z2.reshape(B*k,crop_size,D)
        # crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        # crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1,2)

    # B X K * K
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k-1, dtype=torch.float32)
    labels = torch.cat([labels,torch.zeros(1,k-1)],0)
    labels = torch.cat([torch.zeros(k,1),labels],-1)

    pos_labels = labels.cuda()
    pos_labels[k-1,k-2]=1.0


    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0,2]=1.0
    neg_labels[-1,-3]=1.0
    neg_labels = neg_labels.cuda()


    similarity_matrix = similarity_matrices[0]

    # select and combine multiple positives
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss



def global_infoNCE(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature)

def InfoNCE(z1, z2, temperature=1.0):

    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss




def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    # remove self-similarities
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
