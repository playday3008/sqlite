/*
** 2026-03-24
**
** The author disclaims copyright to this source code.  In place of
** a legal notice, here is a blessing:
**
**    May you do good and not evil.
**    May you find forgiveness for yourself and forgive others.
**    May you share freely, never taking more than you give.
**
******************************************************************************
**
** This SQLite extension implements transparent per-column zstd compression
** with dictionary support.
**
** SQL functions:
**
**   zstd_compress(VALUE)
**   zstd_compress(VALUE, DICT_NAME)
**     Compress VALUE using zstd. Returns a BLOB with a header encoding the
**     original SQLite type, uncompressed size, and flags. If DICT_NAME is
**     provided and not NULL, uses the named dictionary from _zstd_dicts.
**     If compression doesn't reduce size, stores data with a passthrough
**     marker (1 byte overhead).
**
**   zstd_decompress(BLOB)
**   zstd_decompress(BLOB, DICT_NAME)
**     Decompress a BLOB previously produced by zstd_compress(). Restores
**     the original SQLite type (TEXT, BLOB, INTEGER, REAL). If the value
**     was compressed with a dictionary, DICT_NAME must match.
**
**   zstd_train_dict(TABLE, COLUMN, DICT_NAME, DICT_SIZE)
**     Train a zstd dictionary from sample data in TABLE.COLUMN. The
**     dictionary is stored in _zstd_dicts with the given name. DICT_SIZE
**     is the target dictionary size in bytes (default ~110KB).
**
**   zstd_enable(TABLE)
**     Enable transparent compression on TABLE. Reads _zstd_config to
**     determine which columns to compress and with what dictionaries.
**     Renames TABLE to _TABLE_zstd, creates a VIEW with decompression,
**     and INSTEAD OF triggers for INSERT/UPDATE/DELETE with compression.
**
**   zstd_disable(TABLE)
**     Reverse of zstd_enable(). Drops view and triggers, renames the
**     storage table back.
**
**   zstd_compress_table(TABLE)
**     Compress existing uncompressed data in the storage table in-place.
**
**   zstd_decompress_table(TABLE)
**     Decompress all data in the storage table in-place.
**
** Compressed blob wire format:
**
**   Bytes 0-3:  Magic bytes 0x5A 0x43 0x01 0x00 ("ZC" + version 1)
**   Byte 4:     Original SQLite type (1=INT, 2=FLOAT, 3=TEXT, 4=BLOB)
**   Byte 5:     Flags (bit 0: dictionary was used)
**   Bytes 6-9:  Uncompressed size, little-endian uint32
**   Bytes 10+:  Zstd compressed data
**
**   Passthrough (compression didn't help):
**   Bytes 0-3:  Magic bytes 0x5A 0x43 0x01 0x00
**   Byte 4:     Original type | 0x80
**   Bytes 5+:   Original data unchanged
*/

/*
** When compiled as part of the testfixture, require SQLITE_HAVE_ZSTD to be
** defined.  When compiled standalone as a loadable extension, include
** unconditionally.
*/
#if !defined(SQLITE_CORE) || defined(SQLITE_HAVE_ZSTD)

#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <zstd.h>
#include <zdict.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

/* 4-byte magic identifying a zstdcol compressed blob: "ZC" + version + 0x00 */
#define ZSTDCOL_MAGIC1     0x5A  /* 'Z' */
#define ZSTDCOL_MAGIC2     0x43  /* 'C' */
#define ZSTDCOL_MAGIC3     0x01  /* format version 1 */
#define ZSTDCOL_MAGIC4     0x00  /* reserved */

/* Header size for full compressed format: magic(4) + type(1) + flags(1) + size(4) */
#define ZSTDCOL_HDR_SIZE   10

/* Header size for passthrough format: magic(4) + type|0x80(1) */
#define ZSTDCOL_PT_HDR     5

/* Flag: dictionary was used for compression */
#define ZSTDCOL_FLAG_DICT  0x01

/* OR'd into type byte to indicate passthrough (no compression) */
#define ZSTDCOL_PASSTHRU   0x80

/* Default compression level */
#define ZSTDCOL_DEFAULT_LEVEL 3

/* Maximum decompressed size (safety limit: 1 GB) */
#define ZSTDCOL_MAX_DECOMP (1024*1024*1024)

/* Maximum number of cached dictionaries */
#define ZSTDCOL_MAX_DICTS 32

/* Maximum number of training samples */
#define ZSTDCOL_MAX_SAMPLES 10000

/* Default dictionary size */
#define ZSTDCOL_DEFAULT_DICT_SIZE 112640

/*
** A single cached dictionary entry.
*/
typedef struct ZstdDictEntry {
  char *zName;           /* Dictionary name (from _zstd_dicts.name) */
  ZSTD_CDict *pCDict;   /* Compiled compression dictionary */
  ZSTD_DDict *pDDict;   /* Compiled decompression dictionary */
  int iLevel;            /* Compression level the CDict was compiled with */
} ZstdDictEntry;

/*
** Per-connection global state. Shared by all zstd_* functions via
** sqlite3_user_data(). Allocated once in sqlite3_zstdcol_init().
*/
typedef struct ZstdGlobal {
  ZstdDictEntry *aDict;  /* Array of cached dictionaries */
  int nDict;             /* Number of entries in aDict[] */
  int nDictAlloc;        /* Allocated slots in aDict[] */
  sqlite3 *db;           /* Database connection */
} ZstdGlobal;

/*
** Column info collected from pragma_table_info.
*/
typedef struct ZstdColInfo {
  char *zName;           /* Column name */
  char *zType;           /* Declared type */
  int iPk;               /* 1 if part of primary key, 0 otherwise */
  int bCompress;         /* 1 if this column should be compressed */
  char *zDict;           /* Dictionary name for this column (or NULL) */
  int iLevel;            /* Compression level for this column */
} ZstdColInfo;

/* Forward declarations */
static void zstdGlobalFree(void *p);
static ZstdDictEntry *zstdFindDict(ZstdGlobal *pGlobal, const char *zName);
static int zstdEnsureTables(sqlite3 *db, char **pzErr);
static void zstdFreeColInfo(ZstdColInfo *, int);

/*
** Free a ZstdGlobal and all cached dictionaries.
*/
static void zstdGlobalFree(void *p){
  ZstdGlobal *pGlobal = (ZstdGlobal*)p;
  int i;
  if( pGlobal==0 ) return;
  for(i=0; i<pGlobal->nDict; i++){
    if( pGlobal->aDict[i].pCDict ) ZSTD_freeCDict(pGlobal->aDict[i].pCDict);
    if( pGlobal->aDict[i].pDDict ) ZSTD_freeDDict(pGlobal->aDict[i].pDDict);
    sqlite3_free(pGlobal->aDict[i].zName);
  }
  sqlite3_free(pGlobal->aDict);
  sqlite3_free(pGlobal);
}

/*
** Look up a dictionary by name. If not cached, try to load it from
** the _zstd_dicts table. Returns NULL if not found.
*/
static ZstdDictEntry *zstdFindDict(ZstdGlobal *pGlobal, const char *zName){
  int i;
  sqlite3_stmt *pStmt = 0;
  int rc;
  const void *pData;
  int nData;
  int iLevel;
  ZstdDictEntry *pEntry;

  if( zName==0 ) return 0;

  /* Check cache first */
  for(i=0; i<pGlobal->nDict; i++){
    if( strcmp(pGlobal->aDict[i].zName, zName)==0 ){
      return &pGlobal->aDict[i];
    }
  }

  /* Try to load from _zstd_dicts table */
  rc = sqlite3_prepare_v2(pGlobal->db,
    "SELECT data, level FROM _zstd_dicts WHERE name=?1", -1, &pStmt, 0);
  if( rc!=SQLITE_OK ){
    sqlite3_finalize(pStmt);
    return 0;
  }
  sqlite3_bind_text(pStmt, 1, zName, -1, SQLITE_STATIC);
  if( sqlite3_step(pStmt)!=SQLITE_ROW ){
    sqlite3_finalize(pStmt);
    return 0;
  }

  pData = sqlite3_column_blob(pStmt, 0);
  nData = sqlite3_column_bytes(pStmt, 0);
  iLevel = sqlite3_column_int(pStmt, 1);
  if( iLevel<=0 ) iLevel = ZSTDCOL_DEFAULT_LEVEL;

  /* Grow cache if needed */
  if( pGlobal->nDict>=pGlobal->nDictAlloc ){
    int nNew = pGlobal->nDictAlloc ? pGlobal->nDictAlloc*2 : 8;
    if( nNew>ZSTDCOL_MAX_DICTS ) nNew = ZSTDCOL_MAX_DICTS;
    if( pGlobal->nDict>=nNew ){
      sqlite3_finalize(pStmt);
      return 0;  /* Cache full */
    }
    pEntry = sqlite3_realloc64(pGlobal->aDict, nNew*sizeof(ZstdDictEntry));
    if( pEntry==0 ){
      sqlite3_finalize(pStmt);
      return 0;
    }
    pGlobal->aDict = pEntry;
    pGlobal->nDictAlloc = nNew;
  }

  /* Create cache entry */
  pEntry = &pGlobal->aDict[pGlobal->nDict];
  memset(pEntry, 0, sizeof(*pEntry));
  pEntry->zName = sqlite3_mprintf("%s", zName);
  if( pEntry->zName==0 ){
    sqlite3_finalize(pStmt);
    return 0;
  }
  pEntry->iLevel = iLevel;
  pEntry->pCDict = ZSTD_createCDict(pData, nData, iLevel);
  pEntry->pDDict = ZSTD_createDDict(pData, nData);

  sqlite3_finalize(pStmt);

  if( pEntry->pCDict==0 || pEntry->pDDict==0 ){
    if( pEntry->pCDict ) ZSTD_freeCDict(pEntry->pCDict);
    if( pEntry->pDDict ) ZSTD_freeDDict(pEntry->pDDict);
    sqlite3_free(pEntry->zName);
    return 0;
  }

  pGlobal->nDict++;
  return pEntry;
}

/*
** Invalidate a cached dictionary entry (e.g. after training a new one).
*/
static void zstdInvalidateDict(ZstdGlobal *pGlobal, const char *zName){
  int i;
  for(i=0; i<pGlobal->nDict; i++){
    if( strcmp(pGlobal->aDict[i].zName, zName)==0 ){
      if( pGlobal->aDict[i].pCDict ) ZSTD_freeCDict(pGlobal->aDict[i].pCDict);
      if( pGlobal->aDict[i].pDDict ) ZSTD_freeDDict(pGlobal->aDict[i].pDDict);
      sqlite3_free(pGlobal->aDict[i].zName);
      /* Shift remaining entries down */
      pGlobal->nDict--;
      if( i<pGlobal->nDict ){
        memmove(&pGlobal->aDict[i], &pGlobal->aDict[i+1],
                (pGlobal->nDict - i)*sizeof(ZstdDictEntry));
      }
      return;
    }
  }
}

/*
** Ensure the _zstd_dicts and _zstd_config tables exist.
*/
static int zstdEnsureTables(sqlite3 *db, char **pzErr){
  return sqlite3_exec(db,
    "CREATE TABLE IF NOT EXISTS _zstd_dicts("
      "name TEXT PRIMARY KEY,"
      "data BLOB NOT NULL,"
      "level INTEGER DEFAULT 3"
    ");"
    "CREATE TABLE IF NOT EXISTS _zstd_config("
      "tbl TEXT NOT NULL,"
      "col TEXT NOT NULL,"
      "dict_name TEXT,"
      "level INTEGER DEFAULT 3,"
      "PRIMARY KEY(tbl, col)"
    ");",
    0, 0, pzErr);
}

/*
** Safe SQL execution helper (like spellfix1DbExec pattern).
** Skips execution if *pRc is already an error.
*/
static void zstdDbExec(int *pRc, sqlite3 *db, const char *zFormat, ...){
  va_list ap;
  char *zSql;
  if( *pRc!=SQLITE_OK ) return;
  va_start(ap, zFormat);
  zSql = sqlite3_vmprintf(zFormat, ap);
  va_end(ap);
  if( zSql==0 ){
    *pRc = SQLITE_NOMEM;
  }else{
    *pRc = sqlite3_exec(db, zSql, 0, 0, 0);
    sqlite3_free(zSql);
  }
}

/*
** Serialize a 64-bit integer into a buffer (little-endian, 8 bytes).
*/
static void zstdPutInt64(unsigned char *p, sqlite3_int64 v){
  int i;
  for(i=0; i<8; i++){
    p[i] = (unsigned char)(v & 0xFF);
    v >>= 8;
  }
}

/*
** Deserialize a 64-bit integer from a buffer (little-endian, 8 bytes).
*/
static sqlite3_int64 zstdGetInt64(const unsigned char *p, int n){
  sqlite3_uint64 v = 0;
  int i;
  if( n>8 ) n = 8;
  for(i=n-1; i>=0; i--){
    v = (v << 8) | p[i];
  }
  return (sqlite3_int64)v;
}

/*
** Serialize a double into a buffer (8 bytes, portable via int64 conversion).
*/
static void zstdPutDouble(unsigned char *p, double v){
  sqlite3_int64 i;
  memcpy(&i, &v, 8);
  zstdPutInt64(p, i);
}

/*
** Deserialize a double from a buffer (8 bytes, portable via int64 conversion).
*/
static double zstdGetDouble(const unsigned char *p, int n){
  sqlite3_int64 i;
  double v = 0.0;
  if( n<8 ) return v;
  i = zstdGetInt64(p, n);
  memcpy(&v, &i, 8);
  return v;
}

/* ======================================================================
** SQL function: zstd_compress(VALUE) or zstd_compress(VALUE, DICT_NAME)
** ====================================================================== */
static void zstdCompressFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  ZstdGlobal *pGlobal = (ZstdGlobal*)sqlite3_user_data(ctx);
  const void *pIn;
  unsigned char *pOut;
  unsigned char *pRaw = 0;
  int nIn;
  int origType;
  size_t nBound, nComp;
  const char *zDict = 0;
  ZstdDictEntry *pDict = 0;
  unsigned char flags = 0;

  /* NULL passthrough */
  if( sqlite3_value_type(argv[0])==SQLITE_NULL ){
    sqlite3_result_null(ctx);
    return;
  }

  origType = sqlite3_value_type(argv[0]);

  /* For INTEGER and REAL, serialize to bytes first */
  if( origType==SQLITE_INTEGER ){
    pRaw = sqlite3_malloc64(8);
    if( pRaw==0 ){ sqlite3_result_error_nomem(ctx); return; }
    zstdPutInt64(pRaw, sqlite3_value_int64(argv[0]));
    pIn = pRaw;
    nIn = 8;
  }else if( origType==SQLITE_FLOAT ){
    pRaw = sqlite3_malloc64(8);
    if( pRaw==0 ){ sqlite3_result_error_nomem(ctx); return; }
    zstdPutDouble(pRaw, sqlite3_value_double(argv[0]));
    pIn = pRaw;
    nIn = 8;
  }else{
    pIn = sqlite3_value_blob(argv[0]);
    nIn = sqlite3_value_bytes(argv[0]);
  }

  /* Get dictionary if provided */
  if( argc>=2 && sqlite3_value_type(argv[1])!=SQLITE_NULL ){
    zDict = (const char*)sqlite3_value_text(argv[1]);
    if( zDict==0 ){
      sqlite3_free(pRaw);
      sqlite3_result_error_nomem(ctx);
      return;
    }
    pDict = zstdFindDict(pGlobal, zDict);
    if( pDict==0 ){
      sqlite3_free(pRaw);
      sqlite3_result_error(ctx, "unknown zstd dictionary", -1);
      return;
    }
    flags |= ZSTDCOL_FLAG_DICT;
  }

  /* Allocate output buffer */
  nBound = ZSTD_compressBound(nIn);
  pOut = sqlite3_malloc64(ZSTDCOL_HDR_SIZE + nBound);
  if( pOut==0 ){
    sqlite3_free(pRaw);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  /* Write header */
  pOut[0] = ZSTDCOL_MAGIC1;
  pOut[1] = ZSTDCOL_MAGIC2;
  pOut[2] = ZSTDCOL_MAGIC3;
  pOut[3] = ZSTDCOL_MAGIC4;
  pOut[4] = (unsigned char)origType;
  pOut[5] = flags;
  pOut[6] = (unsigned char)((nIn) & 0xFF);
  pOut[7] = (unsigned char)((nIn >> 8) & 0xFF);
  pOut[8] = (unsigned char)((nIn >> 16) & 0xFF);
  pOut[9] = (unsigned char)((nIn >> 24) & 0xFF);

  /* Compress */
  if( pDict ){
    ZSTD_CCtx *cctx = ZSTD_createCCtx();
    if( cctx==0 ){
      sqlite3_free(pOut);
      sqlite3_free(pRaw);
      sqlite3_result_error_nomem(ctx);
      return;
    }
    nComp = ZSTD_compress_usingCDict(cctx,
      pOut + ZSTDCOL_HDR_SIZE, nBound, pIn, nIn, pDict->pCDict);
    ZSTD_freeCCtx(cctx);
  }else{
    nComp = ZSTD_compress(
      pOut + ZSTDCOL_HDR_SIZE, nBound, pIn, nIn, ZSTDCOL_DEFAULT_LEVEL);
  }

  if( ZSTD_isError(nComp) ){
    sqlite3_free(pOut);
    sqlite3_free(pRaw);
    sqlite3_result_error(ctx, ZSTD_getErrorName(nComp), -1);
    return;
  }

  /* Check if compression actually helped */
  if( nComp + ZSTDCOL_HDR_SIZE >= (size_t)nIn + ZSTDCOL_PT_HDR ){
    /* Passthrough: magic(4) + type|0x80(1) + original data */
    sqlite3_free(pOut);
    pOut = sqlite3_malloc64(ZSTDCOL_PT_HDR + nIn);
    if( pOut==0 ){
      sqlite3_free(pRaw);
      sqlite3_result_error_nomem(ctx);
      return;
    }
    pOut[0] = ZSTDCOL_MAGIC1;
    pOut[1] = ZSTDCOL_MAGIC2;
    pOut[2] = ZSTDCOL_MAGIC3;
    pOut[3] = ZSTDCOL_MAGIC4;
    pOut[4] = (unsigned char)(origType | ZSTDCOL_PASSTHRU);
    if( nIn>0 ) memcpy(pOut + ZSTDCOL_PT_HDR, pIn, nIn);
    sqlite3_result_blob(ctx, pOut, ZSTDCOL_PT_HDR + nIn, sqlite3_free);
  }else{
    sqlite3_result_blob(ctx, pOut, (int)(ZSTDCOL_HDR_SIZE + nComp), sqlite3_free);
  }
  sqlite3_free(pRaw);
}

/* ======================================================================
** SQL function: zstd_decompress(BLOB) or zstd_decompress(BLOB, DICT_NAME)
** ====================================================================== */
static void zstdDecompressFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  ZstdGlobal *pGlobal = (ZstdGlobal*)sqlite3_user_data(ctx);
  const unsigned char *pIn;
  unsigned char *pOut;
  int nIn;
  unsigned char typeByte;
  int origType;
  unsigned int nOrig;
  unsigned char flags;
  size_t nDecomp;
  const char *zDict = 0;
  ZstdDictEntry *pDict = 0;

  /* NULL passthrough */
  if( sqlite3_value_type(argv[0])==SQLITE_NULL ){
    sqlite3_result_null(ctx);
    return;
  }

  /* Non-BLOB passthrough (not compressed data) */
  if( sqlite3_value_type(argv[0])!=SQLITE_BLOB ){
    sqlite3_result_value(ctx, argv[0]);
    return;
  }

  pIn = (const unsigned char*)sqlite3_value_blob(argv[0]);
  nIn = sqlite3_value_bytes(argv[0]);

  /* Verify minimum size and magic bytes */
  if( nIn<ZSTDCOL_PT_HDR
   || pIn[0]!=ZSTDCOL_MAGIC1
   || pIn[1]!=ZSTDCOL_MAGIC2
   || pIn[2]!=ZSTDCOL_MAGIC3
   || pIn[3]!=ZSTDCOL_MAGIC4
  ){
    /* Not our format - pass through unchanged */
    sqlite3_result_value(ctx, argv[0]);
    return;
  }

  typeByte = pIn[4];

  /* Check for passthrough marker */
  if( typeByte & ZSTDCOL_PASSTHRU ){
    origType = typeByte & 0x7F;
    switch( origType ){
      case SQLITE_TEXT:
        sqlite3_result_text(ctx, (const char*)(pIn+ZSTDCOL_PT_HDR),
          nIn-ZSTDCOL_PT_HDR, SQLITE_TRANSIENT);
        return;
      case SQLITE_BLOB:
        sqlite3_result_blob(ctx, pIn+ZSTDCOL_PT_HDR,
          nIn-ZSTDCOL_PT_HDR, SQLITE_TRANSIENT);
        return;
      case SQLITE_INTEGER:
        sqlite3_result_int64(ctx, zstdGetInt64(pIn+ZSTDCOL_PT_HDR,
          nIn-ZSTDCOL_PT_HDR));
        return;
      case SQLITE_FLOAT:
        sqlite3_result_double(ctx, zstdGetDouble(pIn+ZSTDCOL_PT_HDR,
          nIn-ZSTDCOL_PT_HDR));
        return;
      default:
        sqlite3_result_blob(ctx, pIn+ZSTDCOL_PT_HDR,
          nIn-ZSTDCOL_PT_HDR, SQLITE_TRANSIENT);
        return;
    }
  }

  /* Full compressed format */
  if( nIn<ZSTDCOL_HDR_SIZE ){
    sqlite3_result_error(ctx, "invalid zstd compressed blob (too short)", -1);
    return;
  }

  origType = pIn[4];
  flags = pIn[5];
  nOrig = (unsigned int)pIn[6]
        | ((unsigned int)pIn[7] << 8)
        | ((unsigned int)pIn[8] << 16)
        | ((unsigned int)pIn[9] << 24);

  if( nOrig>(unsigned int)ZSTDCOL_MAX_DECOMP ){
    sqlite3_result_error(ctx, "decompressed size exceeds safety limit", -1);
    return;
  }

  /* Get dictionary if needed */
  if( flags & ZSTDCOL_FLAG_DICT ){
    if( argc>=2 && sqlite3_value_type(argv[1])!=SQLITE_NULL ){
      zDict = (const char*)sqlite3_value_text(argv[1]);
    }
    if( zDict==0 ){
      sqlite3_result_error(ctx,
        "dictionary name required for dict-compressed data", -1);
      return;
    }
    pDict = zstdFindDict(pGlobal, zDict);
    if( pDict==0 ){
      sqlite3_result_error(ctx, "unknown zstd dictionary", -1);
      return;
    }
  }

  /* Allocate output (+1 for null terminator if TEXT) */
  pOut = sqlite3_malloc64(nOrig + 1);
  if( pOut==0 ){
    sqlite3_result_error_nomem(ctx);
    return;
  }

  /* Decompress */
  if( pDict ){
    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    if( dctx==0 ){
      sqlite3_free(pOut);
      sqlite3_result_error_nomem(ctx);
      return;
    }
    nDecomp = ZSTD_decompress_usingDDict(dctx,
      pOut, nOrig, pIn + ZSTDCOL_HDR_SIZE, nIn - ZSTDCOL_HDR_SIZE,
      pDict->pDDict);
    ZSTD_freeDCtx(dctx);
  }else{
    nDecomp = ZSTD_decompress(
      pOut, nOrig, pIn + ZSTDCOL_HDR_SIZE, nIn - ZSTDCOL_HDR_SIZE);
  }

  if( ZSTD_isError(nDecomp) ){
    sqlite3_free(pOut);
    sqlite3_result_error(ctx, ZSTD_getErrorName(nDecomp), -1);
    return;
  }

  /* Restore original type */
  switch( origType ){
    case SQLITE_TEXT:
      pOut[nDecomp] = 0;
      sqlite3_result_text(ctx, (char*)pOut, (int)nDecomp, sqlite3_free);
      break;
    case SQLITE_BLOB:
      sqlite3_result_blob(ctx, pOut, (int)nDecomp, sqlite3_free);
      break;
    case SQLITE_INTEGER:
      sqlite3_result_int64(ctx, zstdGetInt64(pOut, (int)nDecomp));
      sqlite3_free(pOut);
      break;
    case SQLITE_FLOAT:
      sqlite3_result_double(ctx, zstdGetDouble(pOut, (int)nDecomp));
      sqlite3_free(pOut);
      break;
    default:
      sqlite3_result_blob(ctx, pOut, (int)nDecomp, sqlite3_free);
      break;
  }
}

/* ======================================================================
** SQL function: zstd_train_dict(TABLE, COLUMN, DICT_NAME, DICT_SIZE)
** ====================================================================== */
static void zstdTrainDictFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  ZstdGlobal *pGlobal = (ZstdGlobal*)sqlite3_user_data(ctx);
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *zTable;
  const char *zColumn;
  const char *zDictName;
  int dictSize;
  char *zSql = 0;
  char *zErr = 0;
  sqlite3_stmt *pStmt = 0;
  int rc;

  /* Buffers for training samples */
  unsigned char *pSampleBuf = 0;  /* Combined sample data */
  size_t *aSampleSizes = 0;      /* Individual sample sizes */
  int nSamples = 0;
  size_t nSampleBuf = 0;
  size_t nSampleBufAlloc = 0;

  void *pDict = 0;
  size_t dictActual;

  (void)argc;

  zTable = (const char*)sqlite3_value_text(argv[0]);
  zColumn = (const char*)sqlite3_value_text(argv[1]);
  zDictName = (const char*)sqlite3_value_text(argv[2]);
  dictSize = sqlite3_value_int(argv[3]);

  if( zTable==0 || zColumn==0 || zDictName==0 ){
    sqlite3_result_error(ctx, "table, column, and dict_name are required", -1);
    return;
  }
  if( dictSize<=0 || dictSize>1024*1024 ){
    dictSize = ZSTDCOL_DEFAULT_DICT_SIZE;
  }

  /* Ensure metadata tables exist */
  rc = zstdEnsureTables(db, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to create metadata tables", -1);
    sqlite3_free(zErr);
    return;
  }

  /* Gather training samples */
  zSql = sqlite3_mprintf(
    "SELECT \"%w\" FROM \"%w\" WHERE \"%w\" IS NOT NULL LIMIT %d",
    zColumn, zTable, zColumn, ZSTDCOL_MAX_SAMPLES);
  if( zSql==0 ){
    sqlite3_result_error_nomem(ctx);
    return;
  }

  rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0);
  sqlite3_free(zSql);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
    return;
  }

  /* Allocate sample size array */
  aSampleSizes = sqlite3_malloc64(ZSTDCOL_MAX_SAMPLES * sizeof(size_t));
  if( aSampleSizes==0 ){
    sqlite3_finalize(pStmt);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  while( sqlite3_step(pStmt)==SQLITE_ROW && nSamples<ZSTDCOL_MAX_SAMPLES ){
    const void *pBlob = sqlite3_column_blob(pStmt, 0);
    int nBlob = sqlite3_column_bytes(pStmt, 0);
    if( nBlob==0 ) continue;

    /* Grow sample buffer if needed */
    if( nSampleBuf + nBlob > nSampleBufAlloc ){
      size_t nNew = (nSampleBufAlloc ? nSampleBufAlloc*2 : 1024*1024);
      unsigned char *pNew;
      while( nNew < nSampleBuf + nBlob ){
        if( nNew > ((size_t)-1)/2 ) break;  /* overflow guard */
        nNew *= 2;
      }
      pNew = sqlite3_realloc64(pSampleBuf, nNew);
      if( pNew==0 ){
        sqlite3_finalize(pStmt);
        sqlite3_free(pSampleBuf);
        sqlite3_free(aSampleSizes);
        sqlite3_result_error_nomem(ctx);
        return;
      }
      pSampleBuf = pNew;
      nSampleBufAlloc = nNew;
    }

    memcpy(pSampleBuf + nSampleBuf, pBlob, nBlob);
    aSampleSizes[nSamples] = nBlob;
    nSampleBuf += nBlob;
    nSamples++;
  }
  sqlite3_finalize(pStmt);

  if( nSamples<100 ){
    sqlite3_free(pSampleBuf);
    sqlite3_free(aSampleSizes);
    sqlite3_result_error(ctx,
      "need at least 100 non-empty samples for dictionary training", -1);
    return;
  }

  /* Train dictionary */
  pDict = sqlite3_malloc64(dictSize);
  if( pDict==0 ){
    sqlite3_free(pSampleBuf);
    sqlite3_free(aSampleSizes);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  dictActual = ZDICT_trainFromBuffer(pDict, dictSize,
    pSampleBuf, aSampleSizes, nSamples);
  sqlite3_free(pSampleBuf);
  sqlite3_free(aSampleSizes);

  if( ZDICT_isError(dictActual) ){
    sqlite3_free(pDict);
    sqlite3_result_error(ctx, ZDICT_getErrorName(dictActual), -1);
    return;
  }

  /* Store dictionary in _zstd_dicts */
  rc = sqlite3_prepare_v2(db,
    "INSERT OR REPLACE INTO _zstd_dicts(name, data, level) VALUES(?1, ?2, ?3)",
    -1, &pStmt, 0);
  if( rc!=SQLITE_OK ){
    sqlite3_free(pDict);
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
    return;
  }
  sqlite3_bind_text(pStmt, 1, zDictName, -1, SQLITE_STATIC);
  sqlite3_bind_blob(pStmt, 2, pDict, (int)dictActual, SQLITE_STATIC);
  sqlite3_bind_int(pStmt, 3, ZSTDCOL_DEFAULT_LEVEL);
  rc = sqlite3_step(pStmt);
  sqlite3_finalize(pStmt);
  sqlite3_free(pDict);

  if( rc!=SQLITE_DONE ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
    return;
  }

  /* Invalidate cached entry if it exists */
  zstdInvalidateDict(pGlobal, zDictName);

  sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
}

/* ======================================================================
** Helper: Get column info and compression config for a table.
** Caller must free the returned array and its strings with
** zstdFreeColInfo().
** ====================================================================== */
static int zstdGetColInfo(
  sqlite3 *db,
  const char *zTable,
  ZstdColInfo **ppCols,
  int *pnCols,
  char **ppPkCol,        /* OUT: name of INTEGER PRIMARY KEY column, or NULL */
  char **pzErr
){
  sqlite3_stmt *pStmt = 0;
  char *zSql;
  int rc;
  ZstdColInfo *aCols = 0;
  int nCols = 0;
  int nAlloc = 0;
  char *zPkCol = 0;
  int nPkCols = 0;

  *ppCols = 0;
  *pnCols = 0;
  *ppPkCol = 0;

  /* Get column info from pragma_table_info */
  zSql = sqlite3_mprintf("SELECT name, type, pk FROM pragma_table_info(%Q)", zTable);
  if( zSql==0 ) return SQLITE_NOMEM;
  rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0);
  sqlite3_free(zSql);
  if( rc!=SQLITE_OK ){
    if( pzErr ) *pzErr = sqlite3_mprintf("table not found: %s", zTable);
    return rc;
  }

  while( sqlite3_step(pStmt)==SQLITE_ROW ){
    const char *zName = (const char*)sqlite3_column_text(pStmt, 0);
    const char *zType = (const char*)sqlite3_column_text(pStmt, 1);
    int iPk = sqlite3_column_int(pStmt, 2);

    if( nCols>=nAlloc ){
      int nNew = nAlloc ? nAlloc*2 : 16;
      ZstdColInfo *aNew = sqlite3_realloc64(aCols, nNew*sizeof(ZstdColInfo));
      if( aNew==0 ){
        int j;
        sqlite3_finalize(pStmt);
        for(j=0;j<nCols;j++){
          sqlite3_free(aCols[j].zName); sqlite3_free(aCols[j].zType);
          sqlite3_free(aCols[j].zDict);
        }
        sqlite3_free(aCols);
        sqlite3_free(zPkCol);
        return SQLITE_NOMEM;
      }
      aCols = aNew;
      nAlloc = nNew;
    }

    memset(&aCols[nCols], 0, sizeof(ZstdColInfo));
    aCols[nCols].zName = sqlite3_mprintf("%s", zName);
    aCols[nCols].zType = sqlite3_mprintf("%s", zType ? zType : "");
    if( aCols[nCols].zName==0 || aCols[nCols].zType==0 ){
      sqlite3_free(aCols[nCols].zName);
      sqlite3_free(aCols[nCols].zType);
      sqlite3_finalize(pStmt);
      zstdFreeColInfo(aCols, nCols);
      sqlite3_free(zPkCol);
      return SQLITE_NOMEM;
    }
    aCols[nCols].iPk = iPk;
    aCols[nCols].bCompress = 0;
    aCols[nCols].zDict = 0;
    aCols[nCols].iLevel = ZSTDCOL_DEFAULT_LEVEL;

    if( iPk>0 ) nPkCols++;
    /* Track single INTEGER PRIMARY KEY for WHERE clause */
    if( iPk==1 && zType && sqlite3_stricmp(zType, "INTEGER")==0 ){
      sqlite3_free(zPkCol);
      zPkCol = sqlite3_mprintf("%s", zName);
    }
    nCols++;
  }
  sqlite3_finalize(pStmt);

  if( nCols==0 ){
    sqlite3_free(zPkCol);
    if( pzErr ) *pzErr = sqlite3_mprintf("table not found or has no columns: %s", zTable);
    return SQLITE_ERROR;
  }

  /* If there are multiple PK columns, don't use single-column PK optimization */
  if( nPkCols>1 ){
    sqlite3_free(zPkCol);
    zPkCol = 0;
  }

  /* Load compression config from _zstd_config */
  zSql = sqlite3_mprintf(
    "SELECT col, dict_name, level FROM _zstd_config WHERE tbl=%Q", zTable);
  if( zSql==0 ){
    zstdFreeColInfo(aCols, nCols);
    sqlite3_free(zPkCol);
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0);
  sqlite3_free(zSql);
  if( rc==SQLITE_OK ){
    while( sqlite3_step(pStmt)==SQLITE_ROW ){
      const char *zCol = (const char*)sqlite3_column_text(pStmt, 0);
      const char *zDictName = (const char*)sqlite3_column_text(pStmt, 1);
      int iLevel = sqlite3_column_int(pStmt, 2);
      int j;
      for(j=0; j<nCols; j++){
        if( sqlite3_stricmp(aCols[j].zName, zCol)==0 ){
          aCols[j].bCompress = 1;
          if( zDictName ){
            aCols[j].zDict = sqlite3_mprintf("%s", zDictName);
            if( aCols[j].zDict==0 ){
              sqlite3_finalize(pStmt);
              zstdFreeColInfo(aCols, nCols);
              sqlite3_free(zPkCol);
              return SQLITE_NOMEM;
            }
          }
          aCols[j].iLevel = iLevel>0 ? iLevel : ZSTDCOL_DEFAULT_LEVEL;
          break;
        }
      }
    }
  }
  sqlite3_finalize(pStmt);

  *ppCols = aCols;
  *pnCols = nCols;
  *ppPkCol = zPkCol;
  return SQLITE_OK;
}

/*
** Free column info array.
*/
static void zstdFreeColInfo(ZstdColInfo *aCols, int nCols){
  int i;
  for(i=0; i<nCols; i++){
    sqlite3_free(aCols[i].zName);
    sqlite3_free(aCols[i].zType);
    sqlite3_free(aCols[i].zDict);
  }
  sqlite3_free(aCols);
}

/* ======================================================================
** SQL function: zstd_enable(TABLE)
** ====================================================================== */
static void zstdEnableFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *zTable;
  ZstdColInfo *aCols = 0;
  int nCols = 0;
  char *zPkCol = 0;
  char *zErr = 0;
  int rc = SQLITE_OK;
  int i;
  int hasCompressed = 0;

  /* String builders for SQL generation */
  char *zViewCols = 0;
  char *zInsertCols = 0;
  char *zInsertVals = 0;
  char *zUpdateSet = 0;
  char *zWhere = 0;

  (void)argc;

  zTable = (const char*)sqlite3_value_text(argv[0]);
  if( zTable==0 ){
    sqlite3_result_error(ctx, "table name is required", -1);
    return;
  }

  /* Ensure metadata tables exist */
  rc = zstdEnsureTables(db, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to create metadata tables", -1);
    sqlite3_free(zErr);
    return;
  }

  /* Get column info and compression config */
  rc = zstdGetColInfo(db, zTable, &aCols, &nCols, &zPkCol, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to get table info", -1);
    sqlite3_free(zErr);
    return;
  }

  /* Verify at least one column is configured for compression */
  for(i=0; i<nCols; i++){
    if( aCols[i].bCompress ) hasCompressed = 1;
  }
  if( !hasCompressed ){
    zstdFreeColInfo(aCols, nCols);
    sqlite3_free(zPkCol);
    sqlite3_result_error(ctx,
      "no columns configured for compression in _zstd_config", -1);
    return;
  }

  /* Check that storage table doesn't already exist */
  {
    sqlite3_stmt *pChk = 0;
    char *zChk = sqlite3_mprintf(
      "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_%q_zstd'",
      zTable);
    if( zChk==0 ){
      zstdFreeColInfo(aCols, nCols);
      sqlite3_free(zPkCol);
      sqlite3_result_error_nomem(ctx);
      return;
    }
    rc = sqlite3_prepare_v2(db, zChk, -1, &pChk, 0);
    sqlite3_free(zChk);
    if( rc==SQLITE_OK && sqlite3_step(pChk)==SQLITE_ROW ){
      sqlite3_finalize(pChk);
      zstdFreeColInfo(aCols, nCols);
      sqlite3_free(zPkCol);
      sqlite3_result_error(ctx,
        "table appears to already be enabled for compression", -1);
      return;
    }
    sqlite3_finalize(pChk);
  }

  /* Build column expressions for VIEW, INSERT trigger, UPDATE trigger */
  for(i=0; i<nCols; i++){
    const char *sep = (i>0) ? ", " : "";
    char *zOld;

    /* VIEW SELECT columns */
    zOld = zViewCols;
    if( aCols[i].bCompress ){
      if( aCols[i].zDict ){
        zViewCols = sqlite3_mprintf("%s%szstd_decompress(\"%w\", %Q) AS \"%w\"",
          zOld ? zOld : "", sep, aCols[i].zName, aCols[i].zDict, aCols[i].zName);
      }else{
        zViewCols = sqlite3_mprintf("%s%szstd_decompress(\"%w\") AS \"%w\"",
          zOld ? zOld : "", sep, aCols[i].zName, aCols[i].zName);
      }
    }else{
      zViewCols = sqlite3_mprintf("%s%s\"%w\"",
        zOld ? zOld : "", sep, aCols[i].zName);
    }
    sqlite3_free(zOld);

    /* INSERT column names */
    zOld = zInsertCols;
    zInsertCols = sqlite3_mprintf("%s%s\"%w\"",
      zOld ? zOld : "", sep, aCols[i].zName);
    sqlite3_free(zOld);

    /* INSERT values */
    zOld = zInsertVals;
    if( aCols[i].bCompress ){
      if( aCols[i].zDict ){
        zInsertVals = sqlite3_mprintf("%s%szstd_compress(new.\"%w\", %Q)",
          zOld ? zOld : "", sep, aCols[i].zName, aCols[i].zDict);
      }else{
        zInsertVals = sqlite3_mprintf("%s%szstd_compress(new.\"%w\")",
          zOld ? zOld : "", sep, aCols[i].zName);
      }
    }else{
      zInsertVals = sqlite3_mprintf("%s%snew.\"%w\"",
        zOld ? zOld : "", sep, aCols[i].zName);
    }
    sqlite3_free(zOld);

    /* UPDATE SET clauses */
    zOld = zUpdateSet;
    if( aCols[i].bCompress ){
      if( aCols[i].zDict ){
        zUpdateSet = sqlite3_mprintf("%s%s\"%w\"=zstd_compress(new.\"%w\", %Q)",
          zOld ? zOld : "", sep, aCols[i].zName, aCols[i].zName, aCols[i].zDict);
      }else{
        zUpdateSet = sqlite3_mprintf("%s%s\"%w\"=zstd_compress(new.\"%w\")",
          zOld ? zOld : "", sep, aCols[i].zName, aCols[i].zName);
      }
    }else{
      zUpdateSet = sqlite3_mprintf("%s%s\"%w\"=new.\"%w\"",
        zOld ? zOld : "", sep, aCols[i].zName, aCols[i].zName);
    }
    sqlite3_free(zOld);
    if( !zViewCols || !zInsertCols || !zInsertVals || !zUpdateSet ) break;
  }

  /* Build WHERE clause for UPDATE/DELETE triggers */
  if( zPkCol ){
    zWhere = sqlite3_mprintf("\"%w\"=old.\"%w\"", zPkCol, zPkCol);
  }else{
    /* Composite PK or no PK: use all PK columns */
    zWhere = 0;
    for(i=0; i<nCols; i++){
      if( aCols[i].iPk>0 ){
        char *zOld = zWhere;
        zWhere = sqlite3_mprintf("%s%s\"%w\"=old.\"%w\"",
          zOld ? zOld : "", zOld ? " AND " : "",
          aCols[i].zName, aCols[i].zName);
        sqlite3_free(zOld);
      }
    }
    if( zWhere==0 ){
      /* No PK at all - use rowid */
      zWhere = sqlite3_mprintf("rowid=old.rowid");
    }
  }

  /* Check all allocations succeeded */
  if( zViewCols==0 || zInsertCols==0 || zInsertVals==0
   || zUpdateSet==0 || zWhere==0 ){
    sqlite3_free(zViewCols);
    sqlite3_free(zInsertCols);
    sqlite3_free(zInsertVals);
    sqlite3_free(zUpdateSet);
    sqlite3_free(zWhere);
    zstdFreeColInfo(aCols, nCols);
    sqlite3_free(zPkCol);
    sqlite3_result_error_nomem(ctx);
    return;
  }

  /* Execute DDL within a savepoint */
  rc = SQLITE_OK;
  zstdDbExec(&rc, db, "SAVEPOINT zstd_enable");

  /* Rename table to storage table */
  zstdDbExec(&rc, db, "ALTER TABLE \"%w\" RENAME TO \"_%w_zstd\"",
    zTable, zTable);

  /* Create decompression view */
  zstdDbExec(&rc, db,
    "CREATE VIEW \"%w\" AS SELECT %s FROM \"_%w_zstd\"",
    zTable, zViewCols, zTable);

  /* INSTEAD OF INSERT trigger */
  zstdDbExec(&rc, db,
    "CREATE TRIGGER \"_%w_zstd_insert\" INSTEAD OF INSERT ON \"%w\" BEGIN"
    " INSERT INTO \"_%w_zstd\"(%s) VALUES(%s);"
    " END",
    zTable, zTable, zTable, zInsertCols, zInsertVals);

  /* INSTEAD OF UPDATE trigger */
  zstdDbExec(&rc, db,
    "CREATE TRIGGER \"_%w_zstd_update\" INSTEAD OF UPDATE ON \"%w\" BEGIN"
    " UPDATE \"_%w_zstd\" SET %s WHERE %s;"
    " END",
    zTable, zTable, zTable, zUpdateSet, zWhere);

  /* INSTEAD OF DELETE trigger */
  zstdDbExec(&rc, db,
    "CREATE TRIGGER \"_%w_zstd_delete\" INSTEAD OF DELETE ON \"%w\" BEGIN"
    " DELETE FROM \"_%w_zstd\" WHERE %s;"
    " END",
    zTable, zTable, zTable, zWhere);

  if( rc==SQLITE_OK ){
    zstdDbExec(&rc, db, "RELEASE zstd_enable");
  }else{
    sqlite3_exec(db, "ROLLBACK TO zstd_enable; RELEASE zstd_enable", 0, 0, 0);
  }

  /* Cleanup */
  sqlite3_free(zViewCols);
  sqlite3_free(zInsertCols);
  sqlite3_free(zInsertVals);
  sqlite3_free(zUpdateSet);
  sqlite3_free(zWhere);
  zstdFreeColInfo(aCols, nCols);
  sqlite3_free(zPkCol);

  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
  }else{
    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
  }
}

/* ======================================================================
** SQL function: zstd_disable(TABLE)
** ====================================================================== */
static void zstdDisableFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *zTable;
  ZstdColInfo *aCols = 0;
  int nCols = 0;
  char *zPkCol = 0;
  char *zErr = 0;
  int rc = SQLITE_OK;
  int i;

  (void)argc;

  zTable = (const char*)sqlite3_value_text(argv[0]);
  if( zTable==0 ){
    sqlite3_result_error(ctx, "table name is required", -1);
    return;
  }

  /* Verify the storage table exists */
  {
    sqlite3_stmt *pChk = 0;
    char *zChk = sqlite3_mprintf(
      "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_%q_zstd'",
      zTable);
    if( zChk==0 ){
      sqlite3_result_error_nomem(ctx);
      return;
    }
    rc = sqlite3_prepare_v2(db, zChk, -1, &pChk, 0);
    sqlite3_free(zChk);
    if( rc!=SQLITE_OK || sqlite3_step(pChk)!=SQLITE_ROW ){
      sqlite3_finalize(pChk);
      sqlite3_result_error(ctx,
        "compression not enabled for this table (storage table not found)", -1);
      return;
    }
    sqlite3_finalize(pChk);
  }

  /* Get column config so we can decompress data before removing the layer */
  rc = zstdGetColInfo(db, zTable, &aCols, &nCols, &zPkCol, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx,
      zErr ? zErr : "could not read column config for decompression", -1);
    sqlite3_free(zErr);
    return;
  }
  sqlite3_free(zErr);

  rc = SQLITE_OK;
  zstdDbExec(&rc, db, "SAVEPOINT zstd_disable");

  /* Decompress only rows that contain compressed data (magic prefix check) */
  if( aCols ){
    for(i=0; i<nCols && rc==SQLITE_OK; i++){
      if( !aCols[i].bCompress ) continue;
      if( aCols[i].zDict ){
        zstdDbExec(&rc, db,
          "UPDATE \"_%w_zstd\" SET \"%w\"=zstd_decompress(\"%w\", %Q)"
          " WHERE typeof(\"%w\")='blob' AND length(\"%w\")>=5"
          " AND substr(\"%w\",1,4)=x'5A430100'",
          zTable, aCols[i].zName, aCols[i].zName, aCols[i].zDict,
          aCols[i].zName, aCols[i].zName, aCols[i].zName);
      }else{
        zstdDbExec(&rc, db,
          "UPDATE \"_%w_zstd\" SET \"%w\"=zstd_decompress(\"%w\")"
          " WHERE typeof(\"%w\")='blob' AND length(\"%w\")>=5"
          " AND substr(\"%w\",1,4)=x'5A430100'",
          zTable, aCols[i].zName, aCols[i].zName,
          aCols[i].zName, aCols[i].zName, aCols[i].zName);
      }
    }
  }

  /* Drop triggers */
  zstdDbExec(&rc, db, "DROP TRIGGER IF EXISTS \"_%w_zstd_insert\"", zTable);
  zstdDbExec(&rc, db, "DROP TRIGGER IF EXISTS \"_%w_zstd_update\"", zTable);
  zstdDbExec(&rc, db, "DROP TRIGGER IF EXISTS \"_%w_zstd_delete\"", zTable);

  /* Drop view */
  zstdDbExec(&rc, db, "DROP VIEW IF EXISTS \"%w\"", zTable);

  /* Rename storage table back */
  zstdDbExec(&rc, db, "ALTER TABLE \"_%w_zstd\" RENAME TO \"%w\"",
    zTable, zTable);

  if( rc==SQLITE_OK ){
    zstdDbExec(&rc, db, "RELEASE zstd_disable");
  }else{
    sqlite3_exec(db, "ROLLBACK TO zstd_disable; RELEASE zstd_disable", 0, 0, 0);
  }

  zstdFreeColInfo(aCols, nCols);
  sqlite3_free(zPkCol);

  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
  }else{
    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
  }
}

/* ======================================================================
** SQL function: zstd_compress_table(TABLE)
** Compress existing uncompressed data in the storage table.
** ====================================================================== */
static void zstdCompressTableFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *zTable;
  ZstdColInfo *aCols = 0;
  int nCols = 0;
  char *zPkCol = 0;
  char *zErr = 0;
  int rc = SQLITE_OK;
  int i;

  (void)argc;

  zTable = (const char*)sqlite3_value_text(argv[0]);
  if( zTable==0 ){
    sqlite3_result_error(ctx, "table name is required", -1);
    return;
  }

  /* Ensure metadata tables exist */
  rc = zstdEnsureTables(db, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to create metadata tables", -1);
    sqlite3_free(zErr);
    return;
  }

  /* Get column info - use the original table name for config lookup */
  rc = zstdGetColInfo(db, zTable, &aCols, &nCols, &zPkCol, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to get table info", -1);
    sqlite3_free(zErr);
    return;
  }

  rc = SQLITE_OK;
  zstdDbExec(&rc, db, "SAVEPOINT zstd_compress_table");

  for(i=0; i<nCols && rc==SQLITE_OK; i++){
    if( !aCols[i].bCompress ) continue;
    /*
    ** Only compress rows where the column is not already compressed.
    ** We detect already-compressed data by checking for our 4-byte magic
    ** prefix (0x5A 0x43 0x01 0x00). This is reliable because the magic
    ** bytes are unlikely to appear as the first four bytes of user data.
    */
    if( aCols[i].zDict ){
      zstdDbExec(&rc, db,
        "UPDATE \"_%w_zstd\" SET \"%w\"=zstd_compress(\"%w\", %Q)"
        " WHERE \"%w\" IS NOT NULL"
        " AND (typeof(\"%w\")!='blob'"
        " OR length(\"%w\")<5"
        " OR substr(\"%w\",1,4)!=x'5A430100')",
        zTable, aCols[i].zName, aCols[i].zName, aCols[i].zDict,
        aCols[i].zName, aCols[i].zName, aCols[i].zName, aCols[i].zName);
    }else{
      zstdDbExec(&rc, db,
        "UPDATE \"_%w_zstd\" SET \"%w\"=zstd_compress(\"%w\")"
        " WHERE \"%w\" IS NOT NULL"
        " AND (typeof(\"%w\")!='blob'"
        " OR length(\"%w\")<5"
        " OR substr(\"%w\",1,4)!=x'5A430100')",
        zTable, aCols[i].zName, aCols[i].zName,
        aCols[i].zName, aCols[i].zName, aCols[i].zName, aCols[i].zName);
    }
  }

  if( rc==SQLITE_OK ){
    zstdDbExec(&rc, db, "RELEASE zstd_compress_table");
  }else{
    sqlite3_exec(db,
      "ROLLBACK TO zstd_compress_table; RELEASE zstd_compress_table", 0, 0, 0);
  }

  zstdFreeColInfo(aCols, nCols);
  sqlite3_free(zPkCol);

  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
  }else{
    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
  }
}

/* ======================================================================
** SQL function: zstd_decompress_table(TABLE)
** Decompress all data in the storage table.
** ====================================================================== */
static void zstdDecompressTableFunc(
  sqlite3_context *ctx,
  int argc,
  sqlite3_value **argv
){
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  const char *zTable;
  ZstdColInfo *aCols = 0;
  int nCols = 0;
  char *zPkCol = 0;
  char *zErr = 0;
  int rc = SQLITE_OK;
  int i;

  (void)argc;

  zTable = (const char*)sqlite3_value_text(argv[0]);
  if( zTable==0 ){
    sqlite3_result_error(ctx, "table name is required", -1);
    return;
  }

  /* Ensure metadata tables exist */
  rc = zstdEnsureTables(db, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to create metadata tables", -1);
    sqlite3_free(zErr);
    return;
  }

  rc = zstdGetColInfo(db, zTable, &aCols, &nCols, &zPkCol, &zErr);
  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, zErr ? zErr : "failed to get table info", -1);
    sqlite3_free(zErr);
    return;
  }

  rc = SQLITE_OK;
  zstdDbExec(&rc, db, "SAVEPOINT zstd_decompress_table");

  for(i=0; i<nCols && rc==SQLITE_OK; i++){
    if( !aCols[i].bCompress ) continue;
    /* Only decompress rows that actually contain our compressed format */
    if( aCols[i].zDict ){
      zstdDbExec(&rc, db,
        "UPDATE \"_%w_zstd\" SET \"%w\"=zstd_decompress(\"%w\", %Q)"
        " WHERE typeof(\"%w\")='blob' AND length(\"%w\")>=5"
        " AND substr(\"%w\",1,4)=x'5A430100'",
        zTable, aCols[i].zName, aCols[i].zName, aCols[i].zDict,
        aCols[i].zName, aCols[i].zName, aCols[i].zName);
    }else{
      zstdDbExec(&rc, db,
        "UPDATE \"_%w_zstd\" SET \"%w\"=zstd_decompress(\"%w\")"
        " WHERE typeof(\"%w\")='blob' AND length(\"%w\")>=5"
        " AND substr(\"%w\",1,4)=x'5A430100'",
        zTable, aCols[i].zName, aCols[i].zName,
        aCols[i].zName, aCols[i].zName, aCols[i].zName);
    }
  }

  if( rc==SQLITE_OK ){
    zstdDbExec(&rc, db, "RELEASE zstd_decompress_table");
  }else{
    sqlite3_exec(db,
      "ROLLBACK TO zstd_decompress_table; RELEASE zstd_decompress_table",
      0, 0, 0);
  }

  zstdFreeColInfo(aCols, nCols);
  sqlite3_free(zPkCol);

  if( rc!=SQLITE_OK ){
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
  }else{
    sqlite3_result_text(ctx, "ok", -1, SQLITE_STATIC);
  }
}

/* ======================================================================
** Extension entry point
** ====================================================================== */
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_zstdcol_init(
  sqlite3 *db,
  char **pzErrMsg,
  const sqlite3_api_routines *pApi
){
  int rc = SQLITE_OK;
  ZstdGlobal *pGlobal;
  SQLITE_EXTENSION_INIT2(pApi);

  /* Create metadata tables so users can INSERT config before calling
  ** any zstd_* functions. Ignore errors for read-only databases. */
  {
    char *zErr = 0;
    int rc2 = zstdEnsureTables(db, &zErr);
    if( rc2!=SQLITE_OK && rc2!=SQLITE_READONLY ){
      if( pzErrMsg && zErr ) *pzErrMsg = zErr;
      else sqlite3_free(zErr);
      return rc2;
    }
    sqlite3_free(zErr);
  }

  /* Allocate per-connection state */
  pGlobal = sqlite3_malloc64(sizeof(*pGlobal));
  if( pGlobal==0 ) return SQLITE_NOMEM;
  memset(pGlobal, 0, sizeof(*pGlobal));
  pGlobal->db = db;

  /*
  ** Register compress/decompress as INNOCUOUS because they appear inside
  ** VIEWs and TRIGGERs stored in the schema. They are pure data
  ** transformation with no side effects.
  **
  ** Attach the destructor to the first registration so pGlobal is always
  ** freed when the connection closes, even if later registrations fail.
  */
  rc = sqlite3_create_function_v2(db, "zstd_compress", 1,
    SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC,
    pGlobal, zstdCompressFunc, 0, 0, zstdGlobalFree);
  if( rc!=SQLITE_OK ){
    /* Destructor was not called because registration failed; free manually */
    zstdGlobalFree(pGlobal);
    return rc;
  }
  /* From here on, pGlobal is owned by the first registration's destructor.
  ** Do not free it manually on error -- it will be freed when db closes. */
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_compress", 2,
      SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC,
      pGlobal, zstdCompressFunc, 0, 0, 0);
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_decompress", 1,
      SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC,
      pGlobal, zstdDecompressFunc, 0, 0, 0);
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_decompress", 2,
      SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC,
      pGlobal, zstdDecompressFunc, 0, 0, 0);
  }

  /*
  ** Management functions: DIRECTONLY because they modify schema or
  ** execute DDL. Must not be callable from untrusted contexts.
  */
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_train_dict", 4,
      SQLITE_UTF8 | SQLITE_DIRECTONLY,
      pGlobal, zstdTrainDictFunc, 0, 0, 0);
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_enable", 1,
      SQLITE_UTF8 | SQLITE_DIRECTONLY,
      pGlobal, zstdEnableFunc, 0, 0, 0);
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_disable", 1,
      SQLITE_UTF8 | SQLITE_DIRECTONLY,
      pGlobal, zstdDisableFunc, 0, 0, 0);
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_compress_table", 1,
      SQLITE_UTF8 | SQLITE_DIRECTONLY,
      pGlobal, zstdCompressTableFunc, 0, 0, 0);
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "zstd_decompress_table", 1,
      SQLITE_UTF8 | SQLITE_DIRECTONLY,
      pGlobal, zstdDecompressTableFunc, 0, 0, 0);
  }

  return rc;
}

#endif /* !SQLITE_CORE || SQLITE_HAVE_ZSTD */
