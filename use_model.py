import argparse
import math
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "best_weights.pt"
TOKENIZER_PATH = BASE_DIR / "tokenizer.model"

CONFIG = {
    "vocab_size": 16000,
    "d_model": 768,
    "max_seq_len": 2048,
    "num_heads": 12,       # À confirmer dans le script d'entraînement
    "ffn_hidden_dim": 2176,
    "num_layers": 24,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# NORMALISATION
# ---------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMSNorm standard : normalisation par la moyenne des carrés.
        # Si le code d'entraînement utilise une variante différente,
        # il faut reprendre exactement cette variante.
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


# ---------------------------------------------------------------------
# ROTARY POSITION EMBEDDINGS (RoPE)
# ---------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, seq_len, base=10000.0):
    """
    q, k : [batch, heads, sequence, head_dim]

    Cette implémentation correspond à la variante RoPE qui duplique
    les fréquences et utilise rotate_half en séparant les deux moitiés.
    Elle doit correspondre à celle du script d'entraînement.
    """
    head_dim = q.shape[-1]
    device = q.device
    dtype = q.dtype

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=dtype)[None, None, :, :]

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------
# ATTENTION
# ---------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) doit être divisible par num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, seq_len)

        # Attention causale : chaque token ne voit que les tokens précédents.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(y)


# ---------------------------------------------------------------------
# FEED-FORWARD SwiGLU
# ---------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------
# BLOC TRANSFORMER
# ---------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attention = SelfAttention(d_model, config["num_heads"])
        self.feed_forward = FeedForward(d_model, config["ffn_hidden_dim"])

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


# ---------------------------------------------------------------------
# MODÈLE COMPLET
# ---------------------------------------------------------------------

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.max_seq_len = config["max_seq_len"]

        self.token_embedding = nn.Embedding(
            config["vocab_size"], config["d_model"]
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["num_layers"])]
        )
        self.final_norm = RMSNorm(config["d_model"])
        self.lm_head = nn.Linear(
            config["d_model"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------
# CHARGEMENT
# ---------------------------------------------------------------------

def infer_architecture(state_dict):
    """Vérifie les dimensions déductibles des poids du checkpoint."""
    required = ["token_embedding.weight", "lm_head.weight", "final_norm.weight"]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(
            "Le fichier ne ressemble pas au state_dict attendu. Clés absentes : "
            + ", ".join(missing)
        )

    embedding_shape = tuple(state_dict["token_embedding.weight"].shape)
    head_shape = tuple(state_dict["lm_head.weight"].shape)

    if embedding_shape != (CONFIG["vocab_size"], CONFIG["d_model"]):
        raise RuntimeError(
            f"Dimensions d'embedding trouvées : {embedding_shape}, "
            f"mais la configuration attend {(CONFIG['vocab_size'], CONFIG['d_model'])}."
        )
    if head_shape != (CONFIG["vocab_size"], CONFIG["d_model"]):
        raise RuntimeError(
            f"Dimensions de lm_head trouvées : {head_shape}, "
            f"mais la configuration attend {(CONFIG['vocab_size'], CONFIG['d_model'])}."
        )

    layer_ids = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    if not layer_ids:
        raise RuntimeError("Aucune couche 'blocks.N' trouvée dans le checkpoint.")

    num_layers = max(layer_ids) + 1
    if num_layers != CONFIG["num_layers"]:
        raise RuntimeError(
            f"Le checkpoint contient {num_layers} couches, "
            f"mais CONFIG indique {CONFIG['num_layers']}."
        )

    gate_key = "blocks.0.feed_forward.gate_proj.weight"
    if gate_key in state_dict:
        actual_ffn = state_dict[gate_key].shape[0]
        if actual_ffn != CONFIG["ffn_hidden_dim"]:
            raise RuntimeError(
                f"Dimension FFN trouvée : {actual_ffn}, "
                f"mais CONFIG indique {CONFIG['ffn_hidden_dim']}."
            )


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {CHECKPOINT_PATH}\n"
            "Place best_weights.pt dans le même dossier que use_V2.py."
        )
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_PATH}\n"
            "Place tokenizer.model dans le même dossier que use_V2.py."
        )

    print(f"Appareil : {DEVICE}")
    if DEVICE.type == "cpu":
        print("Attention : avec 24 couches et un contexte de 2048 tokens, "
              "la génération peut être lente sur CPU.")

    print(f"Chargement des poids : {CHECKPOINT_PATH}")
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Le checkpoint ne contient pas un dictionnaire de poids direct."
        )

    # Le fichier observé est un OrderedDict dont les clés sont directement
    # token_embedding.weight, blocks.0..., final_norm.weight, lm_head.weight.
    infer_architecture(state_dict)

    print("Création du modèle...")
    model = TransformerLM(CONFIG)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print(f"Chargement du tokenizer : {TOKENIZER_PATH}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

    if tokenizer.get_piece_size() != CONFIG["vocab_size"]:
        raise RuntimeError(
            f"Le tokenizer contient {tokenizer.get_piece_size()} tokens, "
            f"mais le modèle attend {CONFIG['vocab_size']}. "
            "Il faut le tokenizer exact utilisé pendant l'entraînement."
        )

    print("Modèle et tokenizer chargés.")
    return model, tokenizer


# ---------------------------------------------------------------------
# GÉNÉRATION
# ---------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1500,
    temperature=500,
    top_k=50,
    top_p=0.9,
):
    if temperature <= 0:
        raise ValueError("temperature doit être strictement positive.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p doit être dans l'intervalle ]0, 1].")

    input_tokens = tokenizer.encode(prompt, out_type=int)
    if not input_tokens:
        input_tokens = [tokenizer.bos_id()]
        input_tokens = [t for t in input_tokens if t >= 0]

    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)
    generated = input_ids
    eos_id = tokenizer.eos_id()

    for _ in range(max_new_tokens):
        # Le modèle ne reçoit jamais plus que sa longueur de contexte.
        context = generated[:, -model.max_seq_len:]
        logits = model(context)
        next_logits = logits[:, -1, :].float() / temperature

        if top_k > 0:
            k = min(top_k, next_logits.shape[-1])
            values, _ = torch.topk(next_logits, k)
            cutoff = values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < cutoff, -float("inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_logits, descending=True, dim=-1
            )
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove = cumulative_probs > top_p
            # Garder le premier token qui fait dépasser le seuil.
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            filtered_logits = torch.full_like(next_logits, -float("inf"))
            filtered_logits.scatter_(1, sorted_indices, sorted_logits)
            next_logits = filtered_logits

        probs = torch.softmax(next_logits, dim=-1)
        if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
            raise RuntimeError(
                "Probabilités invalides pendant la génération. "
                "Vérifie les poids et les paramètres de génération."
            )

        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

        if eos_id >= 0 and next_token.item() == eos_id:
            break

    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    completion = tokenizer.decode(new_tokens)
    return completion


# ---------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Utiliser le modèle de langage V2 entraîné."
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Texte de départ (si absent, mode interactif).")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens doit être supérieur ou égal à 1.")
    if args.temperature <= 0:
        parser.error("--temperature doit être strictement positive.")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p doit être dans ]0, 1].")

    model, tokenizer = load_model()

    def run_prompt(prompt):
        print("\nV2 : ", end="", flush=True)
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(result)
        print()

    if args.prompt is not None:
        run_prompt(args.prompt)
        return

    print("\nMode interactif. Tape /quit pour quitter.")
    while True:
        try:
            prompt = input("\nToi : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrêt.")
            break

        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Arrêt.")
            break
        if not prompt:
            continue

        try:
            run_prompt(prompt)
        except Exception as exc:
            print(f"Erreur pendant la génération : {exc}")


if __name__ == "__main__":
    main()