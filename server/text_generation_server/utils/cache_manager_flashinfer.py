from typing import Set, List
import math
import torch


class KvCachePool:
    def __init__(self, max_pages: int, num_layers: int, num_heads: int, head_dim: int, page_len: int, dtype: torch.dtype, device: torch.device):
        self.cache_data = [ torch.zeros(max_pages, 2, page_len, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
        self.device = device
        self.max_pages = max_pages
        self.page_len = page_len
        self.starting_free_page_idx = 0

    def allocate(self, num_pages: int):
        if self.starting_free_page_idx + num_pages > self.max_pages:
            raise Exception("Flashinfer cache pool out of cache pages")
        start_page_idx = self.starting_free_page_idx
        self.starting_free_page_idx += num_pages
        return [i for i in range(start_page_idx, start_page_idx + num_pages)]

class RequestKvCache:
    def __init__(self, kvCachePool: KvCachePool, page_len:int, seq_init_len: int):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        init_num_pages = math.ceil(seq_init_len / self.page_len)
        self.kv_last_page_len = seq_init_len - (init_num_pages - 1) * self.page_len
        self.kv_page_indices = kvCachePool.allocate(init_num_pages)
        self.kv_len = seq_init_len

    def increment(self):
        self.kv_len += 1
        self.kv_last_page_len += 1
        if self.kv_last_page_len > self.page_len:
            self.kv_last_page_len -= self.page_len
            self.kv_page_indices.extend(self.kvCachePool.allocate(1))

class BatchKvCache:
    def __init__(self, kvCachePool: KvCachePool, page_len, device):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        self.device = device
        self.kvCacheDict: dict[int, RequestKvCache] = {}

    def getOrCreate(self, req_id, seq_init_len):
        kvCache = self.kvCacheDict.get(req_id) or RequestKvCache(self.kvCachePool, self.page_len, seq_init_len)
        self.kvCacheDict[req_id] = kvCache
        return kvCache
    
    def release(self, req_id):
        del self.kvCacheDict[req_id]

    def increment(self):
        for kvCache in self.kvCacheDict.values():
            kvCache.increment()
            
    def setRequestOrder(self, requestIds: List[int]):
        self.requestIds = requestIds

    def computeActiveKvData(self):
        kv_page_indices_list = []
        kv_page_indptr_list = []
        kv_last_page_len_list = []
        cum_pages = 0
        for requestId in self.requestIds:
            kvCache = self.kvCacheDict[requestId]        
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
    def __init__(self, kvCachePool: KvCachePool):
        self.kvCachePool = kvCachePool
        self.device = kvCachePool.device
        self.page_len = kvCachePool.page_len
        self.batchKvCacheDict: dict[int, BatchKvCache] = {}

    def getOrCreate(self, batch_id):
        batchKvCache = self.batchKvCacheDict.get(batch_id) or BatchKvCache(self.kvCachePool, self.page_len, self.device)
        self.batchKvCacheDict[batch_id] = batchKvCache
        return batchKvCache
