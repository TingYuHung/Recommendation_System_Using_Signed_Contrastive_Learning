import torch
import torch.distributed as dist
import diffdist.functional as distops
import math

def get_similarity_matrix(outputs1, outputs2, chunk=2, multi_gpu=False):
            '''
                Compute similarity matrix
                - outputs: (B', d) tensor for B' = B * chunk
                - sim_matrix: (B', B') tensor
            '''

            if multi_gpu:
                outputs1_gathered = []
                for out in outputs1.chunk(chunk):
                    gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
                    gather_t = torch.cat(distops.all_gather(gather_t, out))
                    outputs1_gathered.append(gather_t)
                outputs1 = torch.cat(outputs1_gathered)

                outputs2_gathered = []
                for out in outputs2.chunk(chunk):
                    gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
                    gather_t = torch.cat(distops.all_gather(gather_t, out))
                    outputs2_gathered.append(gather_t)
                outputs2 = torch.cat(outputs2_gathered)

            sim_matrix = torch.mm(outputs1, outputs2.t())  # (B', d), (d, B') -> (B', B')

            return sim_matrix


def invert_perm(perm: torch.Tensor):
            s = torch.zeros_like(perm)
            s[perm] = torch.arange(perm.size(0), device=perm.device)
            return s

def get_permuted_similarity_matrix(outputs1, outputs_p, permute, people_num, chunk=2, multi_gpu=False):
            # reorder the node to the corresponding position 
            # eg. edge of node j are from node i, put j to the i-th position
            inv_perm = invert_perm(permute)

            outputs_p_inv = outputs_p.clone()
            outputs_p_inv[:people_num, :] = outputs_p[inv_perm, :]

            return get_similarity_matrix(outputs1, outputs_p_inv, chunk, multi_gpu)

        
def inter_view(sim_matrix, temperature=8, chunk=2, eps=1e-8):
            '''
                Compute NT_xent loss
                - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
            '''

            device = sim_matrix.device
            B = sim_matrix.size(0) // chunk  # B = B' / chunk
            eye = torch.eye(B * chunk).to(device)  # (B', B')

            sim_matrix = torch.exp(sim_matrix / temperature) 
            sim_of_same_node = sim_matrix.diag()
            sim_matrix = sim_matrix * (1 - eye)  # remove diagonal
            denom = torch.sum(sim_matrix, dim=1, keepdim=True)
            molecular = torch.diag(sim_of_same_node)
            sim_matrix = -torch.log(molecular / (denom + eps) + eps)  # loss matrix

            loss = torch.sum(sim_matrix.diag()  / (B * chunk))

            return loss

def inter_perm_view(sim_matrix, sim_matrix_perm, temperature=8, chunk=2, eps=1e-8):
            device = sim_matrix.device
            B = sim_matrix.size(0) // chunk  # B = B' / chunk
            eye = torch.eye(B * chunk).to(device)  # (B', B')

            sim_matrix = torch.exp(sim_matrix / temperature) 
            sim_of_same_node = sim_matrix.diag()
            sim_matrix = sim_matrix * (1 - eye)  # remove diagonal
            denom = torch.sum(sim_matrix, dim=1, keepdim=True)
            molecular = torch.diag(sim_of_same_node)

            sim_matrix_perm = torch.exp(sim_matrix_perm / temperature) 
            sim_of_same_node_perm = sim_matrix_perm.diag()
            sim_matrix_perm = sim_matrix_perm * (1 - eye)  # remove diagonal
            denom_perm = torch.sum(sim_matrix_perm, dim=1, keepdim=True)
            molecular_perm = torch.diag(sim_of_same_node_perm)

            # sim_matrix = -torch.log((molecular + molecular_perm) / (denom + denom_perm + eps) + eps)  # loss matrix
            sim_matrix = -torch.log((molecular_perm) / (denom_perm + eps) + eps)  # loss matrix

            loss = torch.sum(sim_matrix.diag()  / (B * chunk))

            return loss

def intra_view(pos1_sim_matrix, pos2_sim_matrix, neg1_sim_matrix, neg2_sim_matrix, temperature=8, chunk=2, eps=1e-8):
            device = pos1_sim_matrix.device
            B = pos1_sim_matrix.size(0) // chunk  # B = B' / chunk
            eye = torch.eye(B * chunk).to(device)  # (B', B')

            pos1_sim_matrix = torch.exp(pos1_sim_matrix / temperature) 
            pos2_sim_matrix = torch.exp(pos2_sim_matrix / temperature) 
            molecular = pos1_sim_matrix + pos2_sim_matrix
            molecular = molecular.diag()

            neg1_sim_matrix = torch.exp(neg1_sim_matrix / temperature) 
            neg2_sim_matrix = torch.exp(neg2_sim_matrix / temperature) 
            denom = neg1_sim_matrix + neg2_sim_matrix
            denom = denom.diag()

            loss_matrix = -torch.log(molecular / (denom + eps) + eps)  # loss matrix

            loss = torch.sum(loss_matrix  / (B * chunk))

            return loss

def Contrastive_Loss(config, out_dict, out_dict_perm=None, people_num=None, perm=None):
    # constrasitve loss with preference permutation
    if out_dict_perm is not None and people_num is not None and perm is not None:
        embedding_metrix = out_dict["proj_emb"]
        pos1_embedding = out_dict["pos_emb"][0]
        pos2_embedding = out_dict["pos_emb"][1]
        pos1_embedding_perm = out_dict_perm["pos_emb"][0]

        neg1_embedding = out_dict["neg_emb"][0]
        neg2_embedding = out_dict["neg_emb"][1]
        neg1_embedding_perm = out_dict_perm["neg_emb"][0]

        pos1_sim_matrix = get_similarity_matrix(embedding_metrix, pos1_embedding)
        pos2_sim_matrix = get_similarity_matrix(embedding_metrix, pos2_embedding)
        neg1_sim_matrix = get_similarity_matrix(embedding_metrix, neg1_embedding)
        neg2_sim_matrix = get_similarity_matrix(embedding_metrix, neg2_embedding)

        pos_perm_sim = get_permuted_similarity_matrix(pos1_embedding, pos1_embedding_perm, perm, people_num)
        neg_perm_sim = get_permuted_similarity_matrix(neg1_embedding, neg1_embedding_perm, perm, people_num)
                
        pos_inter_loss = inter_perm_view(get_similarity_matrix(pos1_embedding, pos2_embedding), pos_perm_sim)
        neg_inter_loss = inter_perm_view(get_similarity_matrix(neg1_embedding, neg2_embedding), neg_perm_sim)
        intra_loss = intra_view(pos1_sim_matrix, pos2_sim_matrix, neg1_sim_matrix, neg2_sim_matrix)
        Loss = config.inter_contra_w * (pos_inter_loss + neg_inter_loss) + config.intra_contra_w * intra_loss    
    # original contrastive loss
    else:
        embedding_metrix = out_dict["proj_emb"]
        pos1_embedding = out_dict["pos_emb"][0]
        pos2_embedding = out_dict["pos_emb"][1]

        neg1_embedding = out_dict["neg_emb"][0]
        neg2_embedding = out_dict["neg_emb"][1]

        pos1_sim_matrix = get_similarity_matrix(embedding_metrix, pos1_embedding)
        pos2_sim_matrix = get_similarity_matrix(embedding_metrix, pos2_embedding)
        neg1_sim_matrix = get_similarity_matrix(embedding_metrix, neg1_embedding)
        neg2_sim_matrix = get_similarity_matrix(embedding_metrix, neg2_embedding)
                
        pos_inter_loss = inter_view(get_similarity_matrix(pos1_embedding, pos2_embedding))
        neg_inter_loss = inter_view(get_similarity_matrix(neg1_embedding, neg2_embedding))
        intra_loss = intra_view(pos1_sim_matrix, pos2_sim_matrix, neg1_sim_matrix, neg2_sim_matrix)
        Loss = config.inter_contra_w * (pos_inter_loss + neg_inter_loss) + config.intra_contra_w * intra_loss

    return Loss