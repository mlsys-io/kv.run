from typing import Set, List
import math
import torch


class KvCacheBatchPosition:
    def __init__(
        self,
        seq_indptr: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        total_seq_len: int,
    ):
        self.total_seq_len = total_seq_len
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len


class KvCachePool:
    def __init__(
        self,
        max_pages: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.cache_data = [
            torch.zeros(
                max_pages, 2, page_len, num_heads, head_dim, dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        self.device = device
        self.max_pages = max_pages
        self.page_len = page_len
        self.free_page_mask = torch.ones(max_pages, dtype=torch.bool, device=device)

    def allocate(self, num_pages: int):
        free_page_indices = self.free_page_mask.nonzero()
        assert (
            len(free_page_indices) >= num_pages
        ), f"Out of available cache pages: asked {num_pages}, only {len(free_page_indices)} free pages"

        allocated_indices = free_page_indices[:num_pages]
        self.free_page_mask[allocated_indices] = False
        return allocated_indices.squeeze(1).tolist()

    def deallocate(self, kv_page_indices: List[int]):
        self.free_page_mask[kv_page_indices] = True


class RequestKvCache:
    def __init__(self, kvCachePool: KvCachePool, page_len: int, seq_init_len: int):
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
            new_indices = self.kvCachePool.allocate(1)
            self.kv_page_indices.extend(new_indices)

    def release(self):
        self.kvCachePool.deallocate(self.kv_page_indices)


class BatchKvCache:
    def __init__(self, kvCachePool: KvCachePool, page_len, device):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        self.device = device
        self.kvCacheDict: dict[int, RequestKvCache] = {}

    def get(self, req_id):
        return self.kvCacheDict.get(req_id)

    def create(self, req_id, seq_init_len):
        self.kvCacheDict[req_id] = RequestKvCache(
            self.kvCachePool, self.page_len, seq_init_len
        )
        return self.kvCacheDict[req_id]

    def release(self, req_id):
        self.kvCacheDict[req_id].release()
        del self.kvCacheDict[req_id]

    def increment(self):
        for kvCache in self.kvCacheDict.values():
            kvCache.increment()

    def setRequestOrder(self, requestIds: List[int]):
        self.requestIds = requestIds

    def getKvCacheBatchPosition(self, requestIds: List[int], isPrefill: bool):
        kv_page_indices_list = []
        kv_page_indptr_list = []
        seq_indptr_list = []
        kv_last_page_len_list = []
        cum_pages = 0
        cum_seq_len = 0
        for requestId in requestIds:
            kvCache = self.kvCacheDict[requestId]
            kv_page_indices_list.extend(kvCache.kv_page_indices)
            kv_page_indptr_list.append(cum_pages)
            seq_indptr_list.append(cum_seq_len)
            kv_last_page_len_list.append(kvCache.kv_last_page_len)
            cum_pages += len(kvCache.kv_page_indices)
            cum_seq_len += kvCache.kv_len if isPrefill else 1

        kv_page_indptr_list.append(cum_pages)
        seq_indptr_list.append(cum_seq_len)
        kv_page_indices = torch.tensor(
            kv_page_indices_list, dtype=torch.int32, device=self.device
        )
        kv_page_indptr = torch.tensor(
            kv_page_indptr_list, dtype=torch.int32, device=self.device
        )
        kv_last_page_len = torch.tensor(
            kv_last_page_len_list, dtype=torch.int32, device=self.device
        )
        seq_indptr = torch.tensor(
            seq_indptr_list, dtype=torch.int32, device=self.device
        )
        return KvCacheBatchPosition(
            seq_indptr=seq_indptr,
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            total_seq_len=cum_seq_len,
        )


class ModelKvCache:
    def __init__(self, kvCachePool: KvCachePool):
        self.kvCachePool = kvCachePool
        self.device = kvCachePool.device
        self.page_len = kvCachePool.page_len
        self.batchKvCacheDict: dict[int, BatchKvCache] = {}

    def getOrCreate(self, batch_id):
        batchKvCache = self.batchKvCacheDict.get(batch_id) or BatchKvCache(
            self.kvCachePool, self.page_len, self.device
        )
        self.batchKvCacheDict[batch_id] = batchKvCache
        return batchKvCache
