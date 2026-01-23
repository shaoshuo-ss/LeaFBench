from fingerprint.fingerprint_interface import LLMFingerprintInterface
import torch



import os
import json
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
import numpy as np
from PIL import Image
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from itertools import repeat
torch.set_grad_enabled(False)
DTYPE = torch.float32


# Noah Amsel, David Persson, Christopher Musco and Robert M. Gower.
# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
# https://arxiv.org/pdf/2505.16932

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]
SAFETY_FACTOR = 1 + 1e-6
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / SAFETY_FACTOR, b / SAFETY_FACTOR**3, c / SAFETY_FACTOR**5)
    for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

def PolarExpress(G: torch.Tensor, steps: int = 15) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.clone()
    if G.size(-2) > G.size(-1): X = X.T
    X = X / (X.norm(dim=(-2, -1),keepdim=True) * SAFETY_FACTOR)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps-len(coeffs_list)))
    for (a,b,c) in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1): X = X.T
    return X

def tokenizers_are_equal(t1, t2):
    return abs(len(t1) - len(t2)) <= 32 and t1.__class__.__name__ == t2.__class__.__name__


def decode_vocab_to_id(tokenizer, vocab):
    decoded_to_id = {}
    for token, token_id in vocab.items():
        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded_token != "":
            decoded_to_id[decoded_token] = token_id
    return decoded_to_id

def polarize(A):
    return PolarExpress(A.to(DTYPE))
    # u, s, vt = torch.linalg.svd(A.to("cuda"), full_matrices=False)
    # return (u@vt)

def plot_matrix(M, out_file, comments="", row="", column="", plot_full=False):
    X_np = M.to('cpu').numpy()
    max_abs = np.max(np.abs(X_np))
    colormap = matplotlib.colormaps.get_cmap('bwr')
    if plot_full:
        normalized = (X_np / max_abs + 1) / 2
        rgb_array = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb_array, mode='RGB')
        img.save(out_file)

    fig, ax = plt.subplots(figsize=(10, 8))
    data_to_plot = X_np[:min(512, M.shape[0]), :min(512, M.shape[1])]
    im = ax.imshow(data_to_plot, cmap=colormap, vmin=-max_abs, vmax=max_abs)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value', rotation=270, labelpad=20)
    ax.set_xlabel(column)
    ax.set_ylabel(row)
    ax.set_title(f'Matrix Visualization {comments}')
    plt.tight_layout()
    plt.savefig(out_file.replace('.png', '_matplotlib.pdf'), bbox_inches='tight')
    plt.close()

def complete_to_square_orthogonal(Q_partial):
    m, n = Q_partial.shape
    if m==n: return Q_partial
    extension = torch.randn(m, m - n, dtype=Q_partial.dtype) / math.sqrt(m)
    full_matrix = torch.cat([Q_partial, extension], dim=1)
    Q_full, _ = torch.linalg.qr(full_matrix)
    for i in range(n):
        if torch.dot(Q_full[:, i], Q_partial[:, i]) < 0:
            Q_full[:, i] = -Q_full[:, i]
    return Q_full

def lognfactorial(n):
    return n * math.log(n) - n + math.log(2*math.pi*n) / 2

def linear_assignment_max_heuristic(matrix: torch.Tensor):
    sel_list = torch.argmax(matrix, dim=1).tolist()
    row_ind = list(range(len(sel_list)))
    sel_mat = torch.zeros_like(matrix)
    sel_mat[row_ind, sel_list] = 1
    return sel_mat, row_ind, sel_list

def linear_assignment_max(matrix: torch.Tensor):
    cost = -(matrix.cpu().numpy())
    row_ind, col_ind = linear_sum_assignment(cost)
    sel_mat = torch.zeros_like(matrix)
    sel_mat[row_ind, col_ind] = 1
    return sel_mat, row_ind, col_ind

# def vocab(model_A_dir, model_B_dir, embed_alias=["model.embed_tokens.weight", "embeddings.weight", "embedding.weight", "emb.weight"]):
#     A_vocab = read_alias(model_A_dir, embed_alias)
#     B_vocab = read_alias(model_B_dir, embed_alias)
#     same_tokenizer = False
#     # try:
#     tokenizer1 = RwkvTokenizer("./rwkv_vocab_v20230424.txt") if 'rwkv' in model_A_dir.lower() else AutoTokenizer.from_pretrained(model_A_dir, trust_remote_code=True)
#     tokenizer2 = RwkvTokenizer("./rwkv_vocab_v20230424.txt") if 'rwkv' in model_B_dir.lower() else AutoTokenizer.from_pretrained(model_B_dir, trust_remote_code=True)
#     # except:
#     #     same_tokenizer = True
#     if same_tokenizer or tokenizers_are_equal(tokenizer1, tokenizer2):
#         print("Assuming tokenizers are equal")
#         length = min(A_vocab.shape[0], B_vocab.shape[0])
#         A_extracted = A_vocab[:length, :].to(DTYPE).to('cuda')
#         B_extracted = B_vocab[:length, :].to(DTYPE).to('cuda')
#     else:
#         vocab1 = tokenizer1.get_vocab()
#         vocab2 = tokenizer2.get_vocab()
#         decoded_to_id1 = decode_vocab_to_id(tokenizer1, vocab1)
#         decoded_to_id2 = decode_vocab_to_id(tokenizer2, vocab2)
#         common_decoded_tokens = set(decoded_to_id1.keys()) & set(decoded_to_id2.keys())
#         ids_in_tokenizer1 = [decoded_to_id1[i] for i in common_decoded_tokens]
#         ids_in_tokenizer2 = [decoded_to_id2[i] for i in common_decoded_tokens]
#         print(f"Vocabulary size of tokenizer1: {len(vocab1)}")
#         print(f"Vocabulary size of tokenizer2: {len(vocab2)}")
#         print(f"Intersection size: {len(common_decoded_tokens)}")
#         A_extracted = A_vocab[ids_in_tokenizer1].to(DTYPE).to('cuda')
#         B_extracted = B_vocab[ids_in_tokenizer2].to(DTYPE).to('cuda')
#     print(A_extracted.shape, B_extracted.shape)
#     C = polarize((B_extracted.T.to(DTYPE) @ A_extracted.to(DTYPE)).to(DTYPE))
#     print("C.shape: ", tuple(C.shape))
#     print("Computing linear sum assignment")
#     P, row_ind, col_ind = linear_assignment_max(C)
#     tr = float(C[row_ind, col_ind].sum())
#     logp = - tr**2 / 2 + lognfactorial(max(C.shape))
#     log10p = logp / math.log(10)
#     return C, tr, row_ind, col_ind, logp, log10p

def tensorprod_permlist(list1, list2):
    l1 = len(list1)
    l2 = len(list2)
    l = list(range(l1*l2))
    for i in range(l1):
        for j in range(l2):
            l[i*l2+j] = list1[i]*l2 + list2[j]
    return l

def reconstruct_permutation(matrix: torch.Tensor, bs: int):
    assert matrix.shape[0] % bs == 0
    nb = matrix.shape[0] // bs
    assert matrix.shape == (nb*bs, nb*bs)
    perm_list_nb = list(range(nb))
    perm_list_bs = list(range(bs))
    block = matrix.abs().reshape(nb, bs, nb, bs).mean(dim=(1,3)).cpu().numpy()
    assert block.shape == (nb, nb)
    row, col = linear_sum_assignment(-block)
    submat = torch.zeros_like(matrix[:bs, :bs])
    for (i, j) in zip(row, col):
        perm_list_nb[i] = int(j)
        submat += matrix[i*bs : (i+1)*bs, j*bs : (j+1)*bs]
    rowsub, colsub = linear_sum_assignment((-submat).cpu().numpy())
    for (i, j) in zip(rowsub, colsub):
        perm_list_bs[i] = int(j)
    perm_mat = torch.zeros_like(matrix)
    perm_list = tensorprod_permlist(perm_list_nb, perm_list_bs)
    perm_mat[list(enumerate(perm_list))] = 1
    matrix /= matrix.max()
    # psnr_value = psnr(matrix, perm_mat)
    return perm_list_nb, perm_list_bs, # float(psnr_value)


class MDIRFingerprint(LLMFingerprintInterface):
    """
    Parameter Distribution Fingerprinting method for LLMs.
    This class implements the fingerprinting logic specific to PDF models.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        # Initialize any specific parameters or configurations for PDF fingerprinting
        

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        pass
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        torch_model, tokenizer = model.load_model()
        return [torch_model.model.embed_tokens.weight, tokenizer]
        
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.
        Calculates correlation coefficients separately for Sq, Sk, Sv, So vectors
        and returns the average as the final similarity score.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Average similarity score between the four parameter vectors.
        """
        # Get fingerprints for both models
        A_vocab, A_tokenizer = base_model.get_fingerprint()
        B_vocab, B_tokenizer = testing_model.get_fingerprint()
        if tokenizers_are_equal(A_tokenizer, B_tokenizer):
            length = min(A_vocab.shape[0], B_vocab.shape[0])
            A_extracted = A_vocab[:length, :].to(DTYPE)
            B_extracted = B_vocab[:length, :].to(DTYPE)
        else:
            vocab1 = A_tokenizer.get_vocab()
            vocab2 = B_tokenizer.get_vocab()
            decoded_to_id1 = decode_vocab_to_id(A_tokenizer, vocab1)
            decoded_to_id2 = decode_vocab_to_id(B_tokenizer, vocab2)
            common_decoded_tokens = set(decoded_to_id1.keys()) & set(decoded_to_id2.keys())
            ids_in_tokenizer1 = [decoded_to_id1[i] for i in common_decoded_tokens]
            ids_in_tokenizer2 = [decoded_to_id2[i] for i in common_decoded_tokens]
            A_extracted = A_vocab[ids_in_tokenizer1].to(DTYPE)
            B_extracted = B_vocab[ids_in_tokenizer2].to(DTYPE)
        print(A_extracted.shape, B_extracted.shape)
        C = polarize((B_extracted.T.to(DTYPE).to(0) @ A_extracted.to(DTYPE).to(0)).to(DTYPE).to("cpu"))
        P, row_ind, col_ind = linear_assignment_max(C)
        tr = float(C[row_ind, col_ind].sum())
        logp = - tr**2 / 2 + lognfactorial(max(C.shape))
        log10p = logp / math.log(10)
        return -log10p