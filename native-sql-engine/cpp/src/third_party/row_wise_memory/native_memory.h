/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 *
 * contributor license agreements.  See the NOTICE file distributed with
 * this
 * work for additional information regarding copyright ownership.
 * The ASF
 * licenses this file to You under the Apache License, Version 2.0
 * (the
 * "License"); you may not use this file except in compliance with
 * the
 * License.  You may obtain a copy of the License at
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by
 * applicable law or agreed to in writing, software
 * distributed under the
 * License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the
 * specific language governing permissions and
 * limitations under the
 * License.
 */
#ifndef __NATIVE_MEMORY_H
#define __NATIVE_MEMORY_H

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define WORD_SIZE 8
#define MAX_METRICS_NUM 200

#define FALSE 0
#define TRUE 1

#define EOF (-1)
#define NOTAVAIL (-2)

/***************************/
//#define MEM_STAT
/***************************/
typedef enum {
  MEMTYPE_ROW = 0,
  MEMTYPE_COLBUF,
  MEMTYPE_COLVECTOR,
  MEMTYPE_COLUMNBATCH,
  MEMTYPE_BUFFER,
  MEMTYPE_HASHMAP,
  MEMTYPE_HASHMAPCOL,
  MEMTYPE_DECIMAL,
  MEMTYPE_MISC,
  MEMTYPE_HSET,
  MEMTYPE_LZ4,
  MEMTYPE_POOL,
  MEMTYPE_PROCESTREAM,
  MEMTYPE_INMEMSORT,
  MEMTYPE_EXTERNALSORT,
  MEMTYPE_RADIXSORT,
  MEMTYPE_LAST = 0x7fffffff
} MemType;

#if defined(MEM_STAT)

typedef struct MemAllocRecord_ {
  uint64_t addr;
  uint32_t size;
  uint32_t id;  // who alloc it
  bool needfree;
} MemAllocRecord_t;

#define MEM_ALLOC_RECORD_BUF_SIZE (1024 * 100)
typedef struct MemStat_ {
  /*--- xxx ---*/
  MemAllocRecord_t allocs[MEM_ALLOC_RECORD_BUF_SIZE];
  uint32_t cursor;
  /*--- xxx ---*/
  uint64_t allocCnt;
  uint64_t reallocCnt;
  uint64_t freeCnt;
  /*--- xxx ---*/
  uint64_t createColumnBatchCnt;
  uint64_t freeColumnBatchCnt;
} MemStat;

extern MemStat* gmemstat;

static uint32_t inline lookupEmptySlot(MemStat* memstat) {
  uint32_t cur = memstat->cursor;
  uint32_t cnt = 0;
  while (memstat->allocs[cur].needfree) {
    cur = (cur + 1) % MEM_ALLOC_RECORD_BUF_SIZE;

    cnt++;
    if (cnt >= MEM_ALLOC_RECORD_BUF_SIZE) {
      fprintf(stdout,
              "alloc item buffer full, too many memory alloc,  resize the allocs "
              "buffer!!!\n");
      fflush(stdout);
      assert(0);
    }
  }
  memstat->cursor = cur;
  return cur;
}

static uint32_t inline lookupAllocedSlot(MemStat* memstat, uint64_t addr) {
  uint32_t cur = 0;
  while (cur < MEM_ALLOC_RECORD_BUF_SIZE) {
    if (memstat->allocs[cur].needfree && memstat->allocs[cur].addr == addr) {
      return cur;
    }
    cur++;
  }

  fprintf(stdout, "the addrs is not record in this buffer or freeed. addr:%ld\n", addr);
  fflush(stdout);
  assert(0);
  return 0;
}

static void inline statMalloc(MemStat* memstat, uint64_t addr, uint32_t size,
                              uint32_t id) {
  if (addr == 0 || size <= 0) return;

  uint32_t cur = lookupEmptySlot(memstat);
  assert(cur < MEM_ALLOC_RECORD_BUF_SIZE);
  memstat->allocs[cur].addr = addr;
  memstat->allocs[cur].size = size;
  memstat->allocs[cur].id = id;
  memstat->allocs[cur].needfree = true;
  // fprintf(stdout, "insert:  cur:%d with %ld \n", cur,addr);
  // fflush(stdout);
}

static void inline statRealloc(MemStat* memstat, uint64_t oldaddr, uint64_t newaddr,
                               uint32_t newsize, uint32_t id) {
  if (newaddr == 0 || newsize <= 0) return;
  if (oldaddr == 0) {  // insert a new
    // fprintf(stdout, "realloc-> alloc: %ld ",oldaddr);
    statMalloc(memstat, newaddr, newsize, id);
  } else {  // update the oldone.

    int cur = lookupAllocedSlot(memstat, oldaddr);

    if (oldaddr == newaddr) {
      memstat->allocs[cur].size = newsize;
      return;
    } else {
      memstat->allocs[cur].size = newsize;
      memstat->allocs[cur].addr = newaddr;
    }
    // fprintf(stdout, "realloc: update: cur:%d %ld \n", cur, oldaddr);
  }
}

static void inline statFree(MemStat* memstat, uint64_t addr) {
  if (addr == 0) return;
  uint32_t cur = lookupAllocedSlot(memstat, addr);
  memstat->allocs[cur].needfree = false;
  // fprintf(stdout, "free:  cur:%d with %ld \n", cur,addr);
}

static void inline dumpAllocs(MemStat* memstat) {
  int i;
  fprintf(stdout,
          "\n--------------------Detail of need free "
          "list-----------------------------------\n");
  for (i = 0; i < 10; i++) {
    if (memstat->allocs[i].needfree)
      fprintf(stdout, "addr:%llx size:%d id:%d flag:%d\n",
              (long long unsigned int)memstat->allocs[i].addr, memstat->allocs[i].size,
              memstat->allocs[i].id, memstat->allocs[i].needfree);
  }
  fprintf(stdout, "--------------------- end ----------------------------------\n");
  fflush(stdout);
}
static uint64_t inline sumNonFreedSize(MemStat* memstat, uint32_t* notfreeCnt) {
  uint32_t i = 0;
  uint32_t cnt = 0;
  uint64_t sum = 0;
  for (i = 0; i < MEM_ALLOC_RECORD_BUF_SIZE; i++) {
    if (memstat->allocs[i].needfree) {
      sum += memstat->allocs[i].size;
      cnt++;
    }
  }
  *notfreeCnt = cnt;

  return sum;
}
static void inline dumpMemstat() {
  fprintf(stdout, "\n--------------------summary-----------------------------------\n");
  fprintf(stdout, "    allocCnt: %ld  freeCnt: %ld reallocCnt: %ld \n",
          gmemstat->allocCnt, gmemstat->freeCnt, gmemstat->reallocCnt);
  fprintf(stdout, "    createBatch: %ld  freeBatch: %ld \n",
          gmemstat->createColumnBatchCnt, gmemstat->freeColumnBatchCnt);
  uint32_t notfreeCnt = 0;
  uint64_t sum;
  sum = sumNonFreedSize(gmemstat, &notfreeCnt);
  fprintf(stdout, "    non-free-memory-size: %ld non-free-cnt: %d\n", sum, notfreeCnt);
  fprintf(stdout, "--------------------- end ----------------------------------\n\n");
  if (notfreeCnt > 0) {
    dumpAllocs(gmemstat);
  }
  fflush(stdout);
}
#endif

static inline void statCreateColumnBatch(void) {
#if defined(MEM_STAT)
  gmemstat->createColumnBatchCnt++;
#endif
}
static inline void statFreeColumnBatch(void) {
#if defined(MEM_STAT)
  gmemstat->freeColumnBatchCnt++;
#endif
}

static inline void* nativeMalloc(size_t size, uint32_t id) {
  void* addr = malloc(size);

#if defined(MEM_STAT)
  assert(id < MEMTYPE_LAST);
  gmemstat->allocCnt++;
  statMalloc(gmemstat, (uint64_t)addr, size, id);
#endif

  return addr;
}

static inline void* nativeRealloc(void* ptr, size_t newsize, uint32_t id) {
  void* addr = realloc(ptr, newsize);

#if defined(MEM_STAT)
  assert(id < MEMTYPE_LAST);
  gmemstat->reallocCnt++;
  statRealloc(gmemstat, (uint64_t)ptr, (uint64_t)addr, newsize, id);
#endif

  return addr;
}
static inline void nativeFree(void* ptr) {
#if defined(MEM_STAT)
  gmemstat->freeCnt++;
  statFree(gmemstat, (uint64_t)ptr);
#endif
  free(ptr);
}

#endif
