class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_id = None
        self.token = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token, token_id):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id
        node.token = token

    def search(self, text, start_pos):
        match_token, match_token_id = None, None
        pos = start_pos
        node = self.root
        while True:
            char = text[pos]
            if char not in node.children:
                break
            node = node.children[char]
            if node.token:
                match_token = node.token
                match_token_id = node.token_id
            pos += 1
            if pos >= len(text):
                break
        return match_token_id, match_token

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def pack_token(id, space, upper):
    return (id << 2) + (space << 1) + (upper << 0)

def unpack_token(token):
    id = token >> 2
    space = (token >> 1) & 0x01
    upper = (token >> 0) & 0x01
    return id, space, upper

def upper_first(text):
    return text[0].upper() + (text[1:] if len(text) > 1 else "")

def lower_first(text):
    return (text[0].lower() + text[1:]) if len(text) > 1 else text

def expand_vocab(vocab, max_vocab_size):
    updated_vocab = {}
    for i, (token, id) in enumerate(vocab.items()):
        if i >= max_vocab_size:
            return updated_vocab
        updated_vocab[pack_token(id, space=False, upper=True)] = f"{upper_first(token)}"
        updated_vocab[pack_token(id, space=True, upper=True)] = f"Ġ{upper_first(token)}"
        updated_vocab[pack_token(id, space=False, upper=False)] = f"{token}"
        updated_vocab[pack_token(id, space=True, upper=False)] = f"Ġ{token}"
    return updated_vocab


class SpaceTokenizer():
    def __init__(self, vocab_config, vocab_size=None, space_expand_vocab=True):
      self.byte_encoder = bytes_to_unicode()
      self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

      vocab_size = len(vocab_config) if vocab_size is None else vocab_size
      self.vocab_decode = expand_vocab(vocab_config, max_vocab_size=vocab_size) if space_expand_vocab else vocab_config
      self.vocab = {v:k for k,v in self.vocab_decode.items()}

      self.trie = Trie()
      for token, token_id in self.vocab.items():
          self.trie.insert(token, token_id)

    def encode(self, text, return_token_tuple=False):
        text = ''.join(self.byte_encoder[b] for b in text.encode('utf-8'))
        pos = 0
        ids, tokens = [], []
        while True:
            id, token = self.trie.search(text, pos)
            if id is None or token is None:
                raise Exception(f"Error encoding {text[pos:pos+16]}")
            ids.append(id)
            tokens.append(token)
            pos += len(token)
            if pos >= len(text):
                break
        return (ids, tokens) if return_token_tuple else ids

    def decode(self, ids):
        out = ""
        for id in ids:
            if not id in self.vocab_decode:
                raise Exception(f"Error decoding {id}")
            out += self.vocab_decode[id]
        return bytearray([self.byte_decoder[c] for c in out]).decode('utf-8', errors="replace")

class HfTokenizerWrapper():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, return_token_tuple=False):
        output = self.tokenizer.encode(text)
        return (output.ids, output.tokens) if return_token_tuple else output.ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)