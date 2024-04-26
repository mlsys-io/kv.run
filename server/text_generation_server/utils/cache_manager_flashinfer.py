from typing import Set
import math
import torch


START_INDEX_FREE_PAGE = 0 

class RequestKvCache:
    def __init__(self, page_len:int, maxlen: int, seq_init_len: int):
        self.page_len = page_len
        max_num_pages = math.ceil(maxlen / self.page_len)
        self.max_num_pages = max_num_pages

        global START_INDEX_FREE_PAGE # not thread safe and will fix this
        start_page_idx = START_INDEX_FREE_PAGE
        START_INDEX_FREE_PAGE += max_num_pages

        init_num_pages = math.ceil(seq_init_len / self.page_len)
        self.kv_last_page_len = seq_init_len - (init_num_pages - 1) * self.page_len
        self.kv_page_indices = [i for i in range(start_page_idx, start_page_idx + init_num_pages)]
        self.kv_len = seq_init_len

    def increment(self):
        self.kv_len += 1
        self.kv_last_page_len += 1
        if self.kv_last_page_len > self.page_len:
            self.kv_last_page_len -= self.page_len
            self.kv_page_indices.append(self.kv_page_indices[-1] + 1)

class BatchKvCache:
    def __init__(self, cache_data: torch.tensor, page_len, device):
        self.cache_data = cache_data
        self.page_len = page_len
        self.device = device
        self.kvCacheDict: dict[int, RequestKvCache] = {}

    def getOrCreate(self, req_id, maxlen, seq_init_len):
        kvCache = self.kvCacheDict.get(req_id) or RequestKvCache(self.page_len, maxlen, seq_init_len)
        self.kvCacheDict[req_id] = kvCache
        return kvCache
    
    def release(self, req_id):
        del self.kvCacheDict[req_id]

    def increment(self):
        for kvCache in self.kvCacheDict.values():
            kvCache.increment()

    def computeActiveKvData(self):
        kv_page_indices_list = []
        kv_page_indptr_list = []
        kv_last_page_len_list = []
        cum_pages = 0
        for kvCache in self.kvCacheDict.values():
            kv_page_indices_list.extend(kvCache.kv_page_indices)
            kv_page_indptr_list.append(cum_pages)
            kv_last_page_len_list.append(kvCache.kv_last_page_len)
            cum_pages += len(kvCache.kv_page_indices)

        kv_page_indptr_list.append(cum_pages)
        kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=self.device)
        kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=self.device)
        kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=self.device)
        return kv_page_indices, kv_page_indptr, kv_last_page_len

class ModelKvCache:
    def __init__(
        self,
        num_pages: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.cache_data = [ torch.zeros(num_pages, 2, page_len, num_heads, head_dim, dtype=dtype, device=device) ] * num_layers
        self.device = device
        self.page_len = page_len
        self.batchKvCacheDict: dict[int, BatchKvCache] = {}

    def getOrCreate(self, batch_id):
        batchKvCache = self.batchKvCacheDict.get(batch_id) or BatchKvCache(self.cache_data, self.page_len, self.device)
        self.batchKvCacheDict[batch_id] = batchKvCache
        return batchKvCache
