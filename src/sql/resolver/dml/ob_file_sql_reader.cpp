/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_RESV

#include "sql/resolver/dml/ob_file_sql_reader.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
#include <glob.h>
#endif

#include "share/ob_errno.h"
#include "lib/utility/utility.h"
#include "common/timezone/ob_time_convert.h"
#include "sql/engine/file_source/file_parquet_reader.h"

namespace oceanbase
{
namespace sql
{
namespace
{

const int64_t FILE_SQL_MAX_FILE_SIZE = 1024L * 1024L * 512L;

// ---------------------------------------------------------------- utilities

int str_dup(common::ObIAllocator &alloc, const char *src, const int64_t len, char *&dst)
{
  int ret = OB_SUCCESS;
  dst = NULL;
  if (OB_UNLIKELY(len < 0) || OB_ISNULL(src)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    char *buf = static_cast<char *>(alloc.alloc(len + 1));
    if (OB_ISNULL(buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      if (len > 0) {
        MEMCPY(buf, src, len);
      }
      buf[len] = '\0';
      dst = buf;
    }
  }
  return ret;
}

bool str_eq_ci(const char *s, const int64_t len, const char *lit)
{
  bool equal = false;
  const int64_t lit_len = static_cast<int64_t>(STRLEN(lit));
  if (NULL != s && lit_len == len) {
    equal = true;
    for (int64_t i = 0; equal && i < len; ++i) {
      char a = s[i];
      char b = lit[i];
      if (a >= 'A' && a <= 'Z') {
        a = static_cast<char>(a - 'A' + 'a');
      }
      if (b >= 'A' && b <= 'Z') {
        b = static_cast<char>(b - 'A' + 'a');
      }
      if (a != b) {
        equal = false;
      }
    }
  }
  return equal;
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool is_int_str(const char *p, const int64_t len)
{
  bool valid = len > 0;
  int64_t i = 0;
  if (valid && (p[0] == '-' || p[0] == '+')) {
    i = 1;
    valid = len > 1;
  }
  for (; valid && i < len; ++i) {
    if (!is_digit(p[i])) {
      valid = false;
    }
  }
  if (valid && len < 20) {
    char tmp[32];
    MEMCPY(tmp, p, len);
    tmp[len] = '\0';
    errno = 0;
    char *end = NULL;
    (void)::strtoll(tmp, &end, 10);
    valid = (0 == errno && NULL != end && tmp + len == end);
  }
  return valid;
}

bool is_double_str(const char *p, const int64_t len)
{
  bool valid = len > 0;
  bool has_dot = false;
  bool has_exp = false;
  int64_t i = 0;
  int64_t digits = 0;
  if (valid && (p[0] == '-' || p[0] == '+')) {
    i = 1;
  }
  for (; valid && i < len; ++i) {
    const char c = p[i];
    if (is_digit(c)) {
      ++digits;
    } else if ('.' == c && !has_dot && !has_exp) {
      has_dot = true;
    } else if (('e' == c || 'E' == c) && !has_exp && digits > 0) {
      has_exp = true;
      if (i + 1 < len && (p[i + 1] == '-' || p[i + 1] == '+')) {
        ++i;
      }
    } else {
      valid = false;
    }
  }
  return valid && digits > 0;
}

bool is_bool_str(const char *p, const int64_t len)
{
  return str_eq_ci(p, len, "true") || str_eq_ci(p, len, "false");
}

bool is_date_str(const char *p, const int64_t len)
{
  bool valid = (10 == len) && is_digit(p[0]) && is_digit(p[1]) && is_digit(p[2]) &&
               is_digit(p[3]) && '-' == p[4] && is_digit(p[5]) && is_digit(p[6]) &&
               '-' == p[7] && is_digit(p[8]) && is_digit(p[9]);
  if (valid) {
    const int month = (p[5] - '0') * 10 + (p[6] - '0');
    const int day = (p[8] - '0') * 10 + (p[9] - '0');
    valid = month >= 1 && month <= 12 && day >= 1 && day <= 31;
  }
  return valid;
}

bool is_datetime_str(const char *p, const int64_t len)
{
  bool valid = len >= 19 && is_date_str(p, 10) && (p[10] == ' ' || p[10] == 'T') &&
               is_digit(p[11]) && is_digit(p[12]) && ':' == p[13] &&
               is_digit(p[14]) && is_digit(p[15]) && ':' == p[16] &&
               is_digit(p[17]) && is_digit(p[18]);
  if (valid) {
    const int hour = (p[11] - '0') * 10 + (p[12] - '0');
    const int minute = (p[14] - '0') * 10 + (p[15] - '0');
    const int second = (p[17] - '0') * 10 + (p[18] - '0');
    valid = hour <= 23 && minute <= 59 && second <= 59;
    if (valid && len > 19) {
      valid = '.' == p[19];
      for (int64_t i = 20; valid && i < len; ++i) {
        valid = is_digit(p[i]);
      }
    }
  }
  return valid;
}

// ---------------------------------------------------------------- raw data

// A text level representation of the file content, produced by the CSV/JSON
// parsers and consumed by materialize().
class ObFileSqlRawData
{
public:
  ObFileSqlRawData() : col_cnt_(0), row_cnt_(0) {}
  int add_name(const common::ObString &name)
  {
    return names_.push_back(name);
  }
  int add_cell(const char *ptr, const int32_t len)
  {
    int ret = OB_SUCCESS;
    if (OB_FAIL(lens_.push_back(len))) {
    } else if (OB_FAIL(cells_.push_back(ptr))) {
    }
    return ret;
  }
  common::ObSEArray<common::ObString, 32> names_;   // column names
  common::ObSEArray<const char *, 64> cells_;       // row-major text, NULL => SQL NULL
  common::ObSEArray<int32_t, 64> lens_;
  int64_t col_cnt_;
  int64_t row_cnt_;
};

int materialize(common::ObIAllocator &alloc,
                ObFileSqlRawData &raw,
                ObFileSqlTableData &table)
{
  int ret = OB_SUCCESS;
  const int64_t col_cnt = raw.names_.count();
  if (OB_UNLIKELY(col_cnt <= 0) || OB_UNLIKELY(raw.cells_.count() % col_cnt != 0)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid raw file data", K(col_cnt), K(raw.cells_.count()), K(ret));
  } else {
    const int64_t row_cnt = raw.cells_.count() / col_cnt;
    for (int64_t j = 0; OB_SUCC(ret) && j < col_cnt; ++j) {
      bool all_int = true;
      bool all_double = true;
      bool all_bool = true;
      bool all_date = true;
      bool all_datetime = true;
      bool all_null = true;
      bool nullable = false;
      for (int64_t i = 0; OB_SUCC(ret) && i < row_cnt; ++i) {
        const int64_t idx = i * col_cnt + j;
        const char *p = raw.cells_.at(idx);
        const int32_t len = raw.lens_.at(idx);
        if (NULL == p) {
          nullable = true;
          continue;
        }
        all_null = false;
        if (!is_int_str(p, len)) {
          all_int = false;
        }
        if (!is_double_str(p, len)) {
          all_double = false;
        }
        if (!is_bool_str(p, len)) {
          all_bool = false;
        }
        if (!is_date_str(p, len)) {
          all_date = false;
        }
        if (!is_datetime_str(p, len)) {
          all_datetime = false;
        }
      }
      ObFileSqlColumnKind kind = FILE_SQL_COL_VARCHAR;
      common::ObObjType obj_type = common::ObVarcharType;
      if (all_null) {
        kind = FILE_SQL_COL_VARCHAR;
        obj_type = common::ObVarcharType;
      } else if (all_int) {
        kind = FILE_SQL_COL_BIGINT;
        obj_type = common::ObIntType;
      } else if (all_double) {
        kind = FILE_SQL_COL_DOUBLE;
        obj_type = common::ObDoubleType;
      } else if (all_bool) {
        kind = FILE_SQL_COL_BOOLEAN;
        obj_type = common::ObTinyIntType;
      } else if (all_date) {
        kind = FILE_SQL_COL_DATE;
        obj_type = common::ObDateType;
      } else if (all_datetime) {
        kind = FILE_SQL_COL_DATETIME;
        obj_type = common::ObDateTimeType;
      }
      ObFileSqlColumnMeta meta;
      meta.name_ = raw.names_.at(j);
      meta.type_ = obj_type;
      meta.kind_ = kind;
      meta.cs_type_ = common::CS_TYPE_UTF8MB4_GENERAL_CI;
      meta.nullable_ = nullable || all_null;
      if (OB_FAIL(table.cols_.push_back(meta))) {
      } else {
        LOG_TRACE("file sql column inferred", K(j), K(meta.name_), K(obj_type), K(kind));
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < row_cnt; ++i) {
      for (int64_t j = 0; OB_SUCC(ret) && j < col_cnt; ++j) {
        const int64_t idx = i * col_cnt + j;
        const char *p = raw.cells_.at(idx);
        const int32_t len = raw.lens_.at(idx);
        const ObFileSqlColumnMeta &meta = table.cols_.at(j);
        common::ObObj obj;
        if (NULL == p) {
          obj.set_null();
        } else if (FILE_SQL_COL_BIGINT == meta.kind_) {
          obj.set_int(::strtoll(p, NULL, 10));
        } else if (FILE_SQL_COL_DOUBLE == meta.kind_) {
          obj.set_double(::strtod(p, NULL));
        } else if (FILE_SQL_COL_BOOLEAN == meta.kind_) {
          obj.set_tinyint(static_cast<int8_t>(str_eq_ci(p, len, "true") ? 1 : 0));
        } else if (FILE_SQL_COL_DATE == meta.kind_) {
          int32_t date_val = 0;
          common::ObString date_str(len, const_cast<char *>(p));
          if (OB_FAIL(common::ObTimeConverter::str_to_date(date_str, date_val))) {
            LOG_WARN("failed to convert date", K(date_str), K(ret));
          } else {
            obj.set_date(date_val);
          }
        } else if (FILE_SQL_COL_DATETIME == meta.kind_) {
          char *dt_buf = NULL;
          if (OB_FAIL(str_dup(alloc, p, len, dt_buf))) {
          } else {
            for (int32_t k = 0; k < len; ++k) {
              if ('T' == dt_buf[k]) {
                dt_buf[k] = ' ';
              }
            }
            int64_t dt_val = 0;
            int16_t scale = 0;
            common::ObTimeConvertCtx cvrt_ctx(NULL, false);
            common::ObString dt_str(len, dt_buf);
            if (OB_FAIL(common::ObTimeConverter::str_to_datetime(dt_str, cvrt_ctx, dt_val, &scale))) {
              LOG_WARN("failed to convert datetime", K(dt_str), K(ret));
            } else {
              obj.set_datetime(dt_val);
            }
          }
        } else {
          char *str_buf = NULL;
          if (OB_FAIL(str_dup(alloc, p, len, str_buf))) {
          } else {
            obj.set_varchar(common::ObString(len, str_buf));
            obj.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
            obj.set_collation_level(common::CS_LEVEL_IMPLICIT);
          }
        }
        if (OB_SUCC(ret) && OB_FAIL(table.cells_.push_back(obj))) {
        }
      }
    }
    if (OB_SUCC(ret)) {
      table.row_cnt_ = row_cnt;
    }
  }
  return ret;
}

int read_whole_file(common::ObIAllocator &alloc,
                    const common::ObString &path,
                    char *&buf,
                    int64_t &len)
{
  int ret = OB_SUCCESS;
  char *path_buf = NULL;
  FILE *fp = NULL;
  buf = NULL;
  len = 0;
  if (OB_FAIL(str_dup(alloc, path.ptr(), path.length(), path_buf))) {
  } else if (NULL == (fp = ::fopen(path_buf, "rb"))) {
    ret = OB_FILE_NOT_EXIST;
    LOG_WARN("failed to open file", K(path), K(errno), K(ret));
  } else if (0 != ::fseek(fp, 0, SEEK_END)) {
    ret = OB_IO_ERROR;
  } else if ((len = static_cast<int64_t>(::ftell(fp))) < 0) {
    ret = OB_IO_ERROR;
  } else if (len > FILE_SQL_MAX_FILE_SIZE) {
    ret = OB_SIZE_OVERFLOW;
    LOG_WARN("file is too large for file sql", K(path), K(len), K(ret));
  } else if (0 != ::fseek(fp, 0, SEEK_SET)) {
    ret = OB_IO_ERROR;
  } else {
    char *tmp = static_cast<char *>(alloc.alloc(len + 1));
    if (OB_ISNULL(tmp)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (len > 0 && static_cast<size_t>(len) != ::fread(tmp, 1, static_cast<size_t>(len), fp)) {
      ret = OB_IO_ERROR;
    } else {
      tmp[len] = '\0';
      buf = tmp;
    }
  }
  if (NULL != fp) {
    (void)::fclose(fp);
  }
  return ret;
}

// ---------------------------------------------------------------- CSV parser

// Parse one CSV field starting at buf[i]. The value is copied into the
// allocator so that quoted content and escapes stay valid.
int parse_csv_field(common::ObIAllocator &alloc,
                    const char *buf,
                    const int64_t len,
                    int64_t &i,
                    const char *&field,
                    int32_t &field_len,
                    bool &is_null)
{
  int ret = OB_SUCCESS;
  field = NULL;
  field_len = 0;
  is_null = false;
  if (i < len && '"' == buf[i]) {
    ++i;
    const int64_t start = i;
    bool has_double_quote = false;
    bool closed = false;
    while (i < len && !closed) {
      if ('"' == buf[i]) {
        if (i + 1 < len && '"' == buf[i + 1]) {
          has_double_quote = true;
          i += 2;
        } else {
          ++i;
          closed = true;
        }
      } else {
        ++i;
      }
    }
    const int64_t end = (closed && i > start) ? (i - 1) : i;
    const int64_t raw_len = end > start ? end - start : 0;
    char *dst = NULL;
    if (OB_FAIL(str_dup(alloc, buf + start, raw_len, dst))) {
    } else if (has_double_quote) {
      int32_t out = 0;
      for (int64_t k = 0; k < raw_len; ++k) {
        dst[out++] = dst[k];
        if ('"' == dst[k] && k + 1 < raw_len && '"' == dst[k + 1]) {
          ++k;
        }
      }
      dst[out] = '\0';
      field_len = out;
      field = dst;
    } else {
      field_len = static_cast<int32_t>(raw_len);
      field = dst;
    }
  } else {
    const int64_t start = i;
    while (i < len && ',' != buf[i] && '\n' != buf[i] && '\r' != buf[i]) {
      ++i;
    }
    const int64_t raw_len = i - start;
    if (0 == raw_len) {
      is_null = true;
    } else {
      char *dst = NULL;
      if (OB_FAIL(str_dup(alloc, buf + start, raw_len, dst))) {
      } else {
        field = dst;
        field_len = static_cast<int32_t>(raw_len);
      }
    }
  }
  return ret;
}

// Duplicate header names are renamed as <name>__<n>, e.g. id, id -> id, id__2
int dedup_names(common::ObIAllocator &alloc, ObFileSqlRawData &raw)
{
  int ret = OB_SUCCESS;
  common::ObSEArray<common::ObString, 64> bases;
  for (int64_t j = 0; OB_SUCC(ret) && j < raw.names_.count(); ++j) {
    if (OB_FAIL(bases.push_back(raw.names_.at(j)))) {
      LOG_WARN("failed to push base name", K(ret));
    }
  }
  for (int64_t j = 0; OB_SUCC(ret) && j < bases.count(); ++j) {
    int64_t occurrences = 0;
    for (int64_t k = 0; k < j; ++k) {
      if (bases.at(j).case_compare_equal(bases.at(k))) {
        ++occurrences;
      }
    }
    if (occurrences > 0) {
      char name_buf[256];
      int64_t name_len = snprintf(name_buf, sizeof(name_buf), "%.*s__%ld",
                                  bases.at(j).length(),
                                  NULL == bases.at(j).ptr() ? "" : bases.at(j).ptr(),
                                  occurrences + 1);
      char *dup_name = NULL;
      if (OB_UNLIKELY(name_len <= 0 || name_len >= static_cast<int64_t>(sizeof(name_buf)))) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(str_dup(alloc, name_buf, name_len, dup_name))) {
        LOG_WARN("failed to copy duplicate column name", K(ret));
      } else {
        raw.names_.at(j).assign_ptr(dup_name, static_cast<int32_t>(name_len));
      }
    }
  }
  return ret;
}

int parse_csv_content(common::ObIAllocator &alloc,
                      const char *buf,
                      const int64_t len,
                      const bool has_header,
                      ObFileSqlRawData &raw)
{
  int ret = OB_SUCCESS;
  int64_t i = 0;
  bool first_record = true;
  bool first_record_was_header = false;
  bool header_done = !has_header;
  while (OB_SUCC(ret) && i < len) {
    int64_t record_cell_cnt = 0;
    bool record_end = false;
    while (OB_SUCC(ret) && !record_end) {
      const char *field = NULL;
      int32_t field_len = 0;
      bool is_null = false;
      if (OB_FAIL(parse_csv_field(alloc, buf, len, i, field, field_len, is_null))) {
      } else if (!header_done) {
        common::ObString name(is_null ? 0 : field_len,
                              is_null ? const_cast<char *>("") : const_cast<char *>(field));
        if (OB_FAIL(raw.add_name(name))) {
        }
      } else if (is_null) {
        if (OB_FAIL(raw.add_cell(NULL, 0))) {
        }
      } else if (OB_FAIL(raw.add_cell(field, field_len))) {
      }
      if (OB_SUCC(ret)) {
        ++record_cell_cnt;
      }
      if (OB_SUCC(ret)) {
        if (i < len && ',' == buf[i]) {
          ++i;
          if (i >= len) {
            // a trailing comma at the end of the file means one more empty field
            if (header_done) {
              if (OB_FAIL(raw.add_cell(NULL, 0))) {
              } else {
                ++record_cell_cnt;
              }
            }
            record_end = true;
          }
        } else {
          record_end = true;
          if (i < len && '\r' == buf[i]) {
            ++i;
          }
          if (i < len && '\n' == buf[i]) {
            ++i;
          }
        }
      }
    }
    if (OB_SUCC(ret) && first_record) {
      first_record = false;
      if (!header_done) {
        raw.col_cnt_ = raw.names_.count();
        if (raw.col_cnt_ <= 0) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          header_done = true;
          first_record_was_header = true;
          record_cell_cnt = 0;
          if (OB_FAIL(dedup_names(alloc, raw))) {
            LOG_WARN("failed to dedup header names", K(ret));
          }
        }
      }
    }
    if (OB_SUCC(ret) && first_record_was_header) {
      // the header record was consumed into raw.names_, do not treat it as data
      first_record_was_header = false;
      continue;
    }
    if (OB_SUCC(ret) && header_done && 0 != raw.col_cnt_) {
      // make every data record have exactly col_cnt_ fields
      if (record_cell_cnt < raw.col_cnt_) {
        for (int64_t k = record_cell_cnt; OB_SUCC(ret) && k < raw.col_cnt_; ++k) {
          if (OB_FAIL(raw.add_cell(NULL, 0))) {
          }
        }
      } else if (record_cell_cnt > raw.col_cnt_) {
        int64_t extra = record_cell_cnt - raw.col_cnt_;
        while (extra > 0) {
          raw.cells_.pop_back();
          raw.lens_.pop_back();
          --extra;
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    if (raw.names_.count() <= 0) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_UNLIKELY(raw.cells_.count() % raw.names_.count() != 0)) {
      ret = OB_ERR_UNEXPECTED;
    }
  }
  return ret;
}

// ---------------------------------------------------------------- JSON parser

void json_skip_space(const char *buf, const int64_t len, int64_t &i)
{
  while (i < len && (' ' == buf[i] || '\t' == buf[i] || '\n' == buf[i] || '\r' == buf[i])) {
    ++i;
  }
}

int json_parse_string(common::ObIAllocator &alloc,
                      const char *buf,
                      const int64_t len,
                      int64_t &i,
                      const char *&str,
                      int32_t &str_len)
{
  int ret = OB_SUCCESS;
  str = NULL;
  str_len = 0;
  if (i >= len || '"' != buf[i]) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ++i;
    char *dst = static_cast<char *>(alloc.alloc(len - i + 1));
    if (OB_ISNULL(dst)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      int32_t out = 0;
      bool closed = false;
      while (i < len && !closed) {
        const char c = buf[i];
        if ('"' == c) {
          ++i;
          closed = true;
        } else if ('\\' == c && i + 1 < len) {
          const char esc = buf[i + 1];
          i += 2;
          switch (esc) {
            case 'n': dst[out++] = '\n'; break;
            case 't': dst[out++] = '\t'; break;
            case 'r': dst[out++] = '\r'; break;
            case 'b': dst[out++] = '\b'; break;
            case 'f': dst[out++] = '\f'; break;
            case 'u': {
              // keep a placeholder, good enough for text comparison
              dst[out++] = '?';
              i += 4;
              break;
            }
            default: dst[out++] = esc; break;
          }
        } else {
          dst[out++] = c;
          ++i;
        }
      }
      if (!closed) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        dst[out] = '\0';
        str = dst;
        str_len = out;
      }
    }
  }
  return ret;
}

int json_parse_scalar(common::ObIAllocator &alloc,
                      const char *buf,
                      const int64_t len,
                      int64_t &i,
                      const char *&val,
                      int32_t &val_len,
                      bool &is_null)
{
  int ret = OB_SUCCESS;
  val = NULL;
  val_len = 0;
  is_null = false;
  if (i >= len) {
    ret = OB_ERR_UNEXPECTED;
  } else if ('"' == buf[i]) {
    if (OB_FAIL(json_parse_string(alloc, buf, len, i, val, val_len))) {
    }
  } else if ('{' == buf[i] || '[' == buf[i]) {
    // nested object/array: keep the raw text
    const int64_t start = i;
    int32_t depth = 0;
    while (i < len) {
      const char c = buf[i];
      if ('"' == c) {
        ++i;
        while (i < len && '"' != buf[i]) {
          if ('\\' == buf[i]) {
            ++i;
          }
          ++i;
        }
        ++i;
        continue;
      }
      if ('{' == c || '[' == c) {
        ++depth;
      } else if ('}' == c || ']' == c) {
        --depth;
        if (0 == depth) {
          ++i;
          break;
        }
      }
      ++i;
    }
    char *dst = NULL;
    if (OB_FAIL(str_dup(alloc, buf + start, i - start, dst))) {
    } else {
      val = dst;
      val_len = static_cast<int32_t>(i - start);
    }
  } else {
    const int64_t start = i;
    while (i < len && ',' != buf[i] && '}' != buf[i] && ']' != buf[i] &&
           '\n' != buf[i] && '\r' != buf[i]) {
      ++i;
    }
    int64_t end = i;
    while (end > start && (' ' == buf[end - 1] || '\t' == buf[end - 1])) {
      --end;
    }
    const int64_t raw_len = end - start;
    if (raw_len > 0 && str_eq_ci(buf + start, raw_len, "null")) {
      is_null = true;
    } else if (raw_len > 0) {
      char *dst = NULL;
      if (OB_FAIL(str_dup(alloc, buf + start, raw_len, dst))) {
      } else {
        val = dst;
        val_len = static_cast<int32_t>(raw_len);
      }
    } else {
      is_null = true;
    }
  }
  return ret;
}

struct ObFileSqlJsonRow
{
  ObFileSqlJsonRow() {}
  common::ObSEArray<common::ObString, 32> keys_;
  common::ObSEArray<common::ObString, 32> vals_;
};

int json_parse_line(common::ObIAllocator &alloc,
                    const char *buf,
                    const int64_t len,
                    ObFileSqlJsonRow &row)
{
  int ret = OB_SUCCESS;
  int64_t i = 0;
  json_skip_space(buf, len, i);
  if (i >= len || '{' != buf[i]) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ++i;
    bool done = false;
    while (OB_SUCC(ret) && !done) {
      json_skip_space(buf, len, i);
      if (i < len && '}' == buf[i]) {
        ++i;
        done = true;
      } else {
        const char *key = NULL;
        int32_t key_len = 0;
        if (OB_FAIL(json_parse_string(alloc, buf, len, i, key, key_len))) {
        } else {
          json_skip_space(buf, len, i);
          if (i >= len || ':' != buf[i]) {
            ret = OB_ERR_UNEXPECTED;
          } else {
            ++i;
            json_skip_space(buf, len, i);
            const char *val = NULL;
            int32_t val_len = 0;
            bool is_null = false;
            if (OB_FAIL(json_parse_scalar(alloc, buf, len, i, val, val_len, is_null))) {
            } else if (OB_FAIL(row.keys_.push_back(common::ObString(key_len, const_cast<char *>(key))))) {
            } else if (OB_FAIL(row.vals_.push_back(is_null ? common::ObString()
                                                           : common::ObString(val_len, const_cast<char *>(val))))) {
            } else {
              json_skip_space(buf, len, i);
              if (i < len && ',' == buf[i]) {
                ++i;
              } else if (i < len && '}' == buf[i]) {
                ++i;
                done = true;
              } else if (i >= len) {
                done = true;
              } else {
                ret = OB_ERR_UNEXPECTED;
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

int parse_json_lines(common::ObIAllocator &alloc,
                     const char *buf,
                     const int64_t len,
                     ObFileSqlRawData &raw)
{
  int ret = OB_SUCCESS;
  common::ObSEArray<ObFileSqlJsonRow *, 8> rows;
  int64_t i = 0;
  while (OB_SUCC(ret) && i < len) {
    const int64_t line_start = i;
    while (i < len && '\n' != buf[i]) {
      ++i;
    }
    int64_t line_end = i;
    if (i < len) {
      ++i;  // skip '\n'
    }
    while (line_end > line_start &&
           ('\r' == buf[line_end - 1] || ' ' == buf[line_end - 1] || '\t' == buf[line_end - 1])) {
      --line_end;
    }
    int64_t k = line_start;
    json_skip_space(buf, line_end, k);
    if (k >= line_end) {
      continue;  // empty line
    }
    ObFileSqlJsonRow *row = static_cast<ObFileSqlJsonRow *>(alloc.alloc(sizeof(ObFileSqlJsonRow)));
    if (OB_ISNULL(row)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      new (row) ObFileSqlJsonRow();
      if (OB_FAIL(json_parse_line(alloc, buf + line_start, line_end - line_start, *row))) {
      } else if (OB_FAIL(rows.push_back(row))) {
      }
    }
  }
  if (OB_SUCC(ret) && rows.count() <= 0) {
    ret = OB_ERR_UNEXPECTED;
  }
  // build the union of the keys, in the order of first appearance
  for (int64_t r = 0; OB_SUCC(ret) && r < rows.count(); ++r) {
    ObFileSqlJsonRow *row = rows.at(r);
    for (int64_t c = 0; OB_SUCC(ret) && c < row->keys_.count(); ++c) {
      const common::ObString &key = row->keys_.at(c);
      bool found = false;
      for (int64_t n = 0; !found && n < raw.names_.count(); ++n) {
        found = (0 == raw.names_.at(n).compare(key));
      }
      if (!found && OB_FAIL(raw.add_name(key))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    raw.col_cnt_ = raw.names_.count();
    for (int64_t r = 0; OB_SUCC(ret) && r < rows.count(); ++r) {
      ObFileSqlJsonRow *row = rows.at(r);
      for (int64_t n = 0; OB_SUCC(ret) && n < raw.names_.count(); ++n) {
        const char *val = NULL;
        int32_t val_len = 0;
        for (int64_t c = 0; c < row->keys_.count(); ++c) {
          if (0 == row->keys_.at(c).compare(raw.names_.at(n))) {
            val = row->vals_.at(c).ptr();
            val_len = row->vals_.at(c).length();
            break;
          }
        }
        if (OB_FAIL(raw.add_cell(NULL == val ? NULL : val, NULL == val ? 0 : val_len))) {
        }
      }
    }
    if (OB_SUCC(ret)) {
      raw.row_cnt_ = rows.count();
    }
  }
  return ret;
}

} // namespace

// ---------------------------------------------------------------- public API

bool ObFileSqlReader::is_valid_csv_format(const common::ObString &format)
{
  return format.empty() || str_eq_ci(format.ptr(), format.length(), "csv") ||
         str_eq_ci(format.ptr(), format.length(), "auto");
}

bool ObFileSqlReader::is_valid_json_format(const common::ObString &format)
{
  return format.empty() || str_eq_ci(format.ptr(), format.length(), "json") ||
         str_eq_ci(format.ptr(), format.length(), "ndjson") ||
         str_eq_ci(format.ptr(), format.length(), "newline_delimited") ||
         str_eq_ci(format.ptr(), format.length(), "auto");
}

int ObFileSqlReader::read_csv(const ObFileSqlOptions &opts, ObFileSqlTableData &table)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  int64_t len = 0;
  ObFileSqlRawData raw;
  if (OB_FAIL(read_whole_file(table.alloc_, opts.path_, buf, len))) {
  } else if (OB_FAIL(parse_csv_content(table.alloc_, buf, len, opts.header_, raw))) {
  } else if (OB_FAIL(materialize(table.alloc_, raw, table))) {
  }
  return ret;
}

int ObFileSqlReader::read_json(const ObFileSqlOptions &opts, ObFileSqlTableData &table)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  int64_t len = 0;
  ObFileSqlRawData raw;
  if (OB_FAIL(read_whole_file(table.alloc_, opts.path_, buf, len))) {
  } else if (OB_FAIL(parse_json_lines(table.alloc_, buf, len, raw))) {
  } else if (OB_FAIL(materialize(table.alloc_, raw, table))) {
  }
  return ret;
}

int ObFileSqlReader::read_glob(const ObFileSqlOptions &opts, ObFileSqlTableData &table)
{
  int ret = OB_SUCCESS;
#if defined(__linux__) || defined(__APPLE__)
  char *pattern = NULL;
  if (OB_FAIL(str_dup(table.alloc_, opts.path_.ptr(), opts.path_.length(), pattern))) {
  } else {
    glob_t glob_buf;
    MEMSET(&glob_buf, 0, sizeof(glob_buf));
    const int rc = ::glob(pattern, 0, NULL, &glob_buf);
    if (0 != rc && GLOB_NOMATCH != rc) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to glob", K(opts.path_), K(rc), K(ret));
    } else {
      const int64_t name_len = static_cast<int64_t>(STRLEN("file"));
      char *name_buf = NULL;
      if (OB_FAIL(str_dup(table.alloc_, "file", name_len, name_buf))) {
      } else {
        ObFileSqlColumnMeta meta;
        meta.name_.assign_ptr(name_buf, static_cast<int32_t>(name_len));
        meta.type_ = common::ObVarcharType;
        meta.kind_ = FILE_SQL_COL_VARCHAR;
        meta.cs_type_ = common::CS_TYPE_UTF8MB4_GENERAL_CI;
        meta.nullable_ = false;
        if (OB_FAIL(table.cols_.push_back(meta))) {
        } else {
          for (size_t i = 0; OB_SUCC(ret) && i < glob_buf.gl_pathc; ++i) {
            const char *matched = glob_buf.gl_pathv[i];
            const int64_t matched_len = NULL == matched ? 0 : static_cast<int64_t>(STRLEN(matched));
            char *path = NULL;
            common::ObObj obj;
            if (OB_ISNULL(matched)) {
            } else if (OB_FAIL(str_dup(table.alloc_, matched, matched_len, path))) {
            } else {
              obj.set_varchar(common::ObString(static_cast<int32_t>(matched_len), path));
              obj.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
              obj.set_collation_level(common::CS_LEVEL_IMPLICIT);
              if (OB_FAIL(table.cells_.push_back(obj))) {
              }
            }
          }
          if (OB_SUCC(ret)) {
            table.row_cnt_ = static_cast<int64_t>(glob_buf.gl_pathc);
          }
        }
      }
    }
    ::globfree(&glob_buf);
  }
#else
  UNUSED(opts);
  UNUSED(table);
  ret = OB_NOT_SUPPORTED;
#endif
  return ret;
}

// ---------------------------------------------------------------- parquet
//
// A minimal parquet reader: PLAIN and dictionary encoded values, optional
// (definition level 1) columns, flat schemas, uncompressed or snappy pages.

namespace {

const int32_t PQ_TYPE_BOOLEAN = 0;
const int32_t PQ_TYPE_INT32 = 1;
const int32_t PQ_TYPE_INT64 = 2;
const int32_t PQ_TYPE_INT96 = 3;
const int32_t PQ_TYPE_FLOAT = 4;
const int32_t PQ_TYPE_DOUBLE = 5;
const int32_t PQ_TYPE_BYTE_ARRAY = 6;
const int32_t PQ_TYPE_FIXED_LEN = 7;

const int32_t PQ_CODEC_UNCOMPRESSED = 0;
const int32_t PQ_CODEC_SNAPPY = 1;

const int32_t PQ_ENC_PLAIN = 0;
const int32_t PQ_ENC_PLAIN_DICTIONARY = 2;
const int32_t PQ_ENC_RLE_DICTIONARY = 8;

const int32_t PQ_PAGE_DATA = 0;
const int32_t PQ_PAGE_DICTIONARY = 2;
const int32_t PQ_PAGE_DATA_V2 = 3;

const int32_t PQ_REP_OPTIONAL = 1;

const int32_t PQ_CONVERTED_DATE = 6;

// thrift compact protocol type ids
const int32_t TC_TRUE = 1;
const int32_t TC_FALSE = 2;
const int32_t TC_BYTE = 3;
const int32_t TC_I16 = 4;
const int32_t TC_I32 = 5;
const int32_t TC_I64 = 6;
const int32_t TC_DOUBLE = 7;
const int32_t TC_BINARY = 8;
const int32_t TC_LIST = 9;
const int32_t TC_SET = 10;
const int32_t TC_MAP = 11;
const int32_t TC_STRUCT = 12;

class ObPqBuf
{
public:
  ObPqBuf(const char *buf, const int64_t len) : buf_(buf), len_(len), pos_(0) {}
  int read_byte(uint8_t &v)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(pos_ + 1 > len_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      v = static_cast<uint8_t>(buf_[pos_++]);
    }
    return ret;
  }
  int read_bytes(const char *&p, const int64_t n)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(n < 0) || OB_UNLIKELY(pos_ + n > len_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      p = buf_ + pos_;
      pos_ += n;
    }
    return ret;
  }
  int skip(const int64_t n)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(n < 0) || OB_UNLIKELY(pos_ + n > len_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      pos_ += n;
    }
    return ret;
  }
  int read_uvarint(uint64_t &v)
  {
    int ret = OB_SUCCESS;
    v = 0;
    int32_t shift = 0;
    bool done = false;
    while (OB_SUCC(ret) && !done) {
      uint8_t b = 0;
      if (OB_FAIL(read_byte(b))) {
      } else if (OB_UNLIKELY(shift > 63)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        v |= (static_cast<uint64_t>(b & 0x7f) << shift);
        shift += 7;
        done = (0 == (b & 0x80));
      }
    }
    return ret;
  }
  int read_zigzag(int64_t &v)
  {
    int ret = OB_SUCCESS;
    uint64_t u = 0;
    if (OB_FAIL(read_uvarint(u))) {
    } else {
      v = static_cast<int64_t>(u >> 1) ^ (-static_cast<int64_t>(u & 1));
    }
    return ret;
  }
  int64_t pos() const { return pos_; }
private:
  const char *buf_;
  int64_t len_;
  int64_t pos_;
};

struct ObPqField
{
  int32_t id_;
  int32_t type_;
};

int pq_read_field_header(ObPqBuf &buf, int32_t &last_id, ObPqField &field, bool &stop)
{
  int ret = OB_SUCCESS;
  uint8_t b = 0;
  stop = false;
  if (OB_FAIL(buf.read_byte(b))) {
  } else if (0 == b) {
    stop = true;
  } else {
    const int32_t delta = (b >> 4) & 0x0f;
    field.type_ = b & 0x0f;
    if (0 == delta) {
      int64_t zz = 0;
      if (OB_FAIL(buf.read_zigzag(zz))) {
      } else {
        field.id_ = static_cast<int32_t>(zz);
      }
    } else {
      field.id_ = last_id + delta;
    }
    if (OB_SUCC(ret)) {
      last_id = field.id_;
    }
  }
  return ret;
}

int pq_skip_value(ObPqBuf &buf, const int32_t type);

int pq_skip_struct(ObPqBuf &buf)
{
  int ret = OB_SUCCESS;
  int32_t last_id = 0;
  bool stop = false;
  while (OB_SUCC(ret) && !stop) {
    ObPqField field;
    if (OB_FAIL(pq_read_field_header(buf, last_id, field, stop))) {
    } else if (stop) {
    } else if (OB_FAIL(pq_skip_value(buf, field.type_))) {
    }
  }
  return ret;
}

int pq_skip_value(ObPqBuf &buf, const int32_t type)
{
  int ret = OB_SUCCESS;
  if (TC_TRUE == type || TC_FALSE == type) {
    // the value is encoded in the field header itself
  } else if (TC_BYTE == type) {
    ret = buf.skip(1);
  } else if (TC_I16 == type || TC_I32 == type || TC_I64 == type) {
    int64_t zz = 0;
    ret = buf.read_zigzag(zz);
  } else if (TC_DOUBLE == type) {
    ret = buf.skip(8);
  } else if (TC_BINARY == type) {
    uint64_t n = 0;
    if (OB_FAIL(buf.read_uvarint(n))) {
    } else if (OB_UNLIKELY(n > static_cast<uint64_t>(INT32_MAX))) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      ret = buf.skip(static_cast<int64_t>(n));
    }
  } else if (TC_LIST == type || TC_SET == type) {
    uint8_t b = 0;
    if (OB_FAIL(buf.read_byte(b))) {
    } else {
      int64_t n = (b >> 4) & 0x0f;
      const int32_t elem_type = b & 0x0f;
      if (15 == n) {
        uint64_t un = 0;
        if (OB_FAIL(buf.read_uvarint(un))) {
        } else {
          n = static_cast<int64_t>(un);
        }
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < n; ++i) {
        if (OB_FAIL(pq_skip_value(buf, elem_type))) {
        }
      }
    }
  } else if (TC_STRUCT == type) {
    ret = pq_skip_struct(buf);
  } else if (TC_MAP == type) {
    ret = OB_NOT_SUPPORTED;
  } else {
    ret = OB_ERR_UNEXPECTED;
  }
  return ret;
}

int pq_read_i32(ObPqBuf &buf, int32_t &v)
{
  int ret = OB_SUCCESS;
  int64_t zz = 0;
  if (OB_FAIL(buf.read_zigzag(zz))) {
  } else {
    v = static_cast<int32_t>(zz);
  }
  return ret;
}

int pq_read_i64(ObPqBuf &buf, int64_t &v)
{
  return buf.read_zigzag(v);
}

int pq_read_binary(ObPqBuf &buf, const char *&p, int32_t &len)
{
  int ret = OB_SUCCESS;
  uint64_t n = 0;
  if (OB_FAIL(buf.read_uvarint(n))) {
  } else if (OB_UNLIKELY(n > static_cast<uint64_t>(INT32_MAX))) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(buf.read_bytes(p, static_cast<int64_t>(n)))) {
  } else {
    len = static_cast<int32_t>(n);
  }
  return ret;
}

int pq_read_list_header(ObPqBuf &buf, int64_t &n, int32_t &elem_type)
{
  int ret = OB_SUCCESS;
  uint8_t b = 0;
  if (OB_FAIL(buf.read_byte(b))) {
  } else {
    n = (b >> 4) & 0x0f;
    elem_type = b & 0x0f;
    if (15 == n) {
      uint64_t un = 0;
      if (OB_FAIL(buf.read_uvarint(un))) {
      } else {
        n = static_cast<int64_t>(un);
      }
    }
  }
  return ret;
}

struct ObPqSchemaElem
{
  ObPqSchemaElem()
    : type_(PQ_TYPE_INT32), repetition_(0), converted_(-1), name_(), has_type_(false)
  {}
  int32_t type_;
  int32_t repetition_;
  int32_t converted_;
  ObString name_;
  bool has_type_;
};

int pq_parse_schema_elem(ObPqBuf &buf, ObPqSchemaElem &elem)
{
  int ret = OB_SUCCESS;
  int32_t last_id = 0;
  bool stop = false;
  while (OB_SUCC(ret) && !stop) {
    ObPqField field;
    if (OB_FAIL(pq_read_field_header(buf, last_id, field, stop))) {
    } else if (stop) {
    } else if (1 == field.id_) {
      if (OB_FAIL(pq_read_i32(buf, elem.type_))) {
      } else {
        elem.has_type_ = true;
      }
    } else if (3 == field.id_) {
      ret = pq_read_i32(buf, elem.repetition_);
    } else if (4 == field.id_) {
      const char *p = NULL;
      int32_t len = 0;
      if (OB_FAIL(pq_read_binary(buf, p, len))) {
      } else {
        elem.name_.assign_ptr(p, len);
      }
    } else if (6 == field.id_) {
      ret = pq_read_i32(buf, elem.converted_);
    } else if (OB_FAIL(pq_skip_value(buf, field.type_))) {
    }
  }
  return ret;
}

struct ObPqColMeta
{
  ObPqColMeta()
    : type_(PQ_TYPE_INT32), codec_(PQ_CODEC_UNCOMPRESSED), num_values_(0),
      data_page_offset_(-1), dict_page_offset_(-1)
  {}
  int32_t type_;
  int32_t codec_;
  int64_t num_values_;
  int64_t data_page_offset_;
  int64_t dict_page_offset_;
};

int pq_parse_col_meta(ObPqBuf &buf, ObPqColMeta &meta)
{
  int ret = OB_SUCCESS;
  int32_t last_id = 0;
  bool stop = false;
  while (OB_SUCC(ret) && !stop) {
    ObPqField field;
    if (OB_FAIL(pq_read_field_header(buf, last_id, field, stop))) {
    } else if (stop) {
    } else if (1 == field.id_) {
      ret = pq_read_i32(buf, meta.type_);
    } else if (4 == field.id_) {
      ret = pq_read_i32(buf, meta.codec_);
    } else if (5 == field.id_) {
      ret = pq_read_i64(buf, meta.num_values_);
    } else if (9 == field.id_) {
      ret = pq_read_i64(buf, meta.data_page_offset_);
    } else if (11 == field.id_) {
      ret = pq_read_i64(buf, meta.dict_page_offset_);
    } else if (OB_FAIL(pq_skip_value(buf, field.type_))) {
    }
  }
  return ret;
}

struct ObPqColChunk
{
  ObPqColChunk() : meta_() {}
  ObPqColMeta meta_;
};

int pq_parse_col_chunk(ObPqBuf &buf, ObPqColChunk &chunk)
{
  int ret = OB_SUCCESS;
  int32_t last_id = 0;
  bool stop = false;
  while (OB_SUCC(ret) && !stop) {
    ObPqField field;
    if (OB_FAIL(pq_read_field_header(buf, last_id, field, stop))) {
    } else if (stop) {
    } else if (3 == field.id_) {
      ret = pq_parse_col_meta(buf, chunk.meta_);
    } else if (OB_FAIL(pq_skip_value(buf, field.type_))) {
    }
  }
  return ret;
}

struct ObPqRowGroup
{
  ObPqRowGroup() : num_rows_(0) {}
  common::ObSEArray<ObPqColChunk, 8> chunks_;
  int64_t num_rows_;
};

int pq_parse_row_group(ObPqBuf &buf, ObPqRowGroup &group)
{
  int ret = OB_SUCCESS;
  int32_t last_id = 0;
  bool stop = false;
  while (OB_SUCC(ret) && !stop) {
    ObPqField field;
    if (OB_FAIL(pq_read_field_header(buf, last_id, field, stop))) {
    } else if (stop) {
    } else if (1 == field.id_) {
      int64_t n = 0;
      int32_t elem_type = 0;
      if (OB_FAIL(pq_read_list_header(buf, n, elem_type))) {
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < n; ++i) {
        ObPqColChunk chunk;
        if (OB_FAIL(pq_parse_col_chunk(buf, chunk))) {
        } else if (OB_FAIL(group.chunks_.push_back(chunk))) {
        }
      }
    } else if (3 == field.id_) {
      ret = pq_read_i64(buf, group.num_rows_);
    } else if (OB_FAIL(pq_skip_value(buf, field.type_))) {
    }
  }
  return ret;
}

struct ObPqPageHeader
{
  ObPqPageHeader()
    : type_(PQ_PAGE_DATA), uncompressed_size_(0), compressed_size_(0),
      num_values_(0), encoding_(PQ_ENC_PLAIN), def_levels_len_(0),
      rep_levels_len_(0), v2_is_compressed_(true)
  {}
  int32_t type_;
  int32_t uncompressed_size_;
  int32_t compressed_size_;
  int64_t num_values_;
  int32_t encoding_;
  int32_t def_levels_len_;
  int32_t rep_levels_len_;
  bool v2_is_compressed_;
};

int pq_parse_page_header(ObPqBuf &buf, ObPqPageHeader &header)
{
  int ret = OB_SUCCESS;
  int32_t last_id = 0;
  bool stop = false;
  while (OB_SUCC(ret) && !stop) {
    ObPqField field;
    if (OB_FAIL(pq_read_field_header(buf, last_id, field, stop))) {
    } else if (stop) {
    } else if (1 == field.id_) {
      ret = pq_read_i32(buf, header.type_);
    } else if (2 == field.id_) {
      ret = pq_read_i32(buf, header.uncompressed_size_);
    } else if (3 == field.id_) {
      ret = pq_read_i32(buf, header.compressed_size_);
    } else if (5 == field.id_ || 8 == field.id_) {
      const bool is_v2 = (8 == field.id_);
      int32_t inner_last_id = 0;
      bool inner_stop = false;
      while (OB_SUCC(ret) && !inner_stop) {
        ObPqField inner;
        if (OB_FAIL(pq_read_field_header(buf, inner_last_id, inner, inner_stop))) {
        } else if (inner_stop) {
        } else if (1 == inner.id_ && !is_v2) {
          int32_t v = 0;
          if (OB_FAIL(pq_read_i32(buf, v))) {
          } else {
            header.num_values_ = v;
          }
        } else if (2 == inner.id_ && !is_v2) {
          ret = pq_read_i32(buf, header.encoding_);
        } else if (is_v2 && 1 == inner.id_) {
          int32_t v = 0;
          if (OB_FAIL(pq_read_i32(buf, v))) {
          } else {
            header.num_values_ = v;
          }
        } else if (is_v2 && 4 == inner.id_) {
          ret = pq_read_i32(buf, header.encoding_);
        } else if (is_v2 && 5 == inner.id_) {
          ret = pq_read_i32(buf, header.def_levels_len_);
        } else if (is_v2 && 6 == inner.id_) {
          ret = pq_read_i32(buf, header.rep_levels_len_);
        } else if (is_v2 && 7 == inner.id_) {
          if (TC_TRUE == inner.type_) {
            header.v2_is_compressed_ = true;
          } else if (TC_FALSE == inner.type_) {
            header.v2_is_compressed_ = false;
          } else {
            int32_t v = 0;
            if (OB_FAIL(pq_read_i32(buf, v))) {
            } else {
              header.v2_is_compressed_ = (0 != v);
            }
          }
        } else if (OB_FAIL(pq_skip_value(buf, inner.type_))) {
        }
      }
    } else if (7 == field.id_) {
      int32_t inner_last_id = 0;
      bool inner_stop = false;
      while (OB_SUCC(ret) && !inner_stop) {
        ObPqField inner;
        if (OB_FAIL(pq_read_field_header(buf, inner_last_id, inner, inner_stop))) {
        } else if (inner_stop) {
        } else if (1 == inner.id_) {
          int32_t v = 0;
          if (OB_FAIL(pq_read_i32(buf, v))) {
          } else {
            header.num_values_ = v;
          }
        } else if (OB_FAIL(pq_skip_value(buf, inner.type_))) {
        }
      }
    } else if (OB_FAIL(pq_skip_value(buf, field.type_))) {
    }
  }
  return ret;
}

int pq_snappy_decompress(const char *in, const int64_t in_len,
                         char *out, const int64_t out_len, int64_t &written)
{
  int ret = OB_SUCCESS;
  written = 0;
  int64_t ip = 0;
  int64_t op = 0;
  uint64_t ulen = 0;
  int32_t shift = 0;
  bool got_len = false;
  while (!got_len && ip < in_len) {
    const uint8_t b = static_cast<uint8_t>(in[ip++]);
    ulen |= (static_cast<uint64_t>(b & 0x7f) << shift);
    shift += 7;
    got_len = (0 == (b & 0x80));
  }
  if (OB_UNLIKELY(!got_len) || OB_UNLIKELY(ulen != static_cast<uint64_t>(out_len))) {
    ret = OB_ERR_UNEXPECTED;
  }
  while (OB_SUCC(ret) && ip < in_len) {
    const uint8_t tag = static_cast<uint8_t>(in[ip++]);
    const int32_t kind = tag & 3;
    if (0 == kind) {
      int64_t n = (tag >> 2);
      if (n >= 60) {
        const int32_t extra = static_cast<int32_t>(n - 59);
        int64_t v = 0;
        for (int32_t i = 0; i < extra; ++i) {
          v |= (static_cast<int64_t>(static_cast<uint8_t>(in[ip++])) << (8 * i));
        }
        n = v;
      }
      ++n;
      if (OB_UNLIKELY(ip + n > in_len) || OB_UNLIKELY(op + n > out_len)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        MEMCPY(out + op, in + ip, n);
        ip += n;
        op += n;
      }
    } else {
      int64_t n = 0;
      int64_t offset = 0;
      if (1 == kind) {
        n = 4 + ((tag >> 2) & 0x07);
        if (OB_UNLIKELY(ip >= in_len)) {
          ret = OB_ERR_UNEXPECTED;
          break;
        }
        offset = ((static_cast<int64_t>(tag & 0xe0)) << 3)
                 | static_cast<uint8_t>(in[ip++]);
      } else if (2 == kind) {
        n = 1 + (tag >> 2);
        if (OB_UNLIKELY(ip + 2 > in_len)) {
          ret = OB_ERR_UNEXPECTED;
          break;
        }
        offset = static_cast<uint8_t>(in[ip])
                 | (static_cast<int64_t>(static_cast<uint8_t>(in[ip + 1])) << 8);
        ip += 2;
      } else {
        n = 1 + (tag >> 2);
        if (OB_UNLIKELY(ip + 4 > in_len)) {
          ret = OB_ERR_UNEXPECTED;
          break;
        }
        offset = static_cast<int64_t>(static_cast<uint8_t>(in[ip]))
                 | (static_cast<int64_t>(static_cast<uint8_t>(in[ip + 1])) << 8)
                 | (static_cast<int64_t>(static_cast<uint8_t>(in[ip + 2])) << 16)
                 | (static_cast<int64_t>(static_cast<uint8_t>(in[ip + 3])) << 24);
        ip += 4;
      }
      if (OB_UNLIKELY(offset <= 0) || OB_UNLIKELY(offset > op)
          || OB_UNLIKELY(op + n > out_len)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        int64_t src = op - offset;
        for (int64_t i = 0; i < n; ++i) {
          out[op + i] = out[src + i];
        }
        op += n;
      }
    }
  }
  if (OB_SUCC(ret)) {
    written = op;
  }
  return ret;
}

// RLE / bit-packed hybrid decoding, see the parquet specification
int pq_decode_levels(const char *buf, const int64_t buf_len, const int32_t bit_width,
                     const int64_t count, common::ObIArray<uint32_t> &out)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  while (OB_SUCC(ret) && pos < buf_len && out.count() < count) {
    uint64_t header = 0;
    int32_t shift = 0;
    bool got = false;
    while (!got && pos < buf_len) {
      const uint8_t b = static_cast<uint8_t>(buf[pos++]);
      header |= (static_cast<uint64_t>(b & 0x7f) << shift);
      shift += 7;
      got = (0 == (b & 0x80));
    }
    if (OB_UNLIKELY(!got)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (0 == bit_width) {
      const int64_t run = (header & 1) ? (static_cast<int64_t>(header >> 1) * 8)
                                       : static_cast<int64_t>(header >> 1);
      for (int64_t i = 0; OB_SUCC(ret) && i < run && out.count() < count; ++i) {
        if (OB_FAIL(out.push_back(0))) {
        }
      }
    } else if (header & 1) {
      const int64_t groups = static_cast<int64_t>(header >> 1);
      const int64_t nbytes = groups * bit_width;
      if (OB_UNLIKELY(pos + nbytes > buf_len)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        const int64_t total = groups * 8;
        for (int64_t i = 0; OB_SUCC(ret) && i < total && out.count() < count; ++i) {
          uint32_t v = 0;
          for (int32_t b = 0; b < bit_width; ++b) {
            const int64_t bit = i * bit_width + b;
            const uint32_t byte = static_cast<uint8_t>(buf[pos + (bit >> 3)]);
            v |= ((byte >> (bit & 7)) & 1u) << b;
          }
          if (OB_FAIL(out.push_back(v))) {
          }
        }
        pos += nbytes;
      }
    } else {
      const int64_t run = static_cast<int64_t>(header >> 1);
      const int32_t nbytes = (bit_width + 7) / 8;
      if (OB_UNLIKELY(pos + nbytes > buf_len)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        uint32_t v = 0;
        for (int32_t b = 0; b < nbytes; ++b) {
          v |= (static_cast<uint32_t>(static_cast<uint8_t>(buf[pos + b])) << (8 * b));
        }
        pos += nbytes;
        for (int64_t i = 0; OB_SUCC(ret) && i < run && out.count() < count; ++i) {
          if (OB_FAIL(out.push_back(v))) {
          }
        }
      }
    }
  }
  return ret;
}

int32_t pq_type_size(const int32_t type, const int32_t fixed_len)
{
  int32_t size = -1;
  switch (type) {
    case PQ_TYPE_BOOLEAN: size = 0; break;
    case PQ_TYPE_INT32: size = 4; break;
    case PQ_TYPE_INT64: size = 8; break;
    case PQ_TYPE_INT96: size = 12; break;
    case PQ_TYPE_FLOAT: size = 4; break;
    case PQ_TYPE_DOUBLE: size = 8; break;
    case PQ_TYPE_FIXED_LEN: size = fixed_len; break;
    default: size = -1; break;
  }
  return size;
}

int pq_push_value(common::ObIAllocator &alloc, const int32_t type, const int32_t converted,
                  const char *p, const int32_t len,
                  common::ObIArray<const char *> &ptrs, common::ObIArray<int32_t> &lens)
{
  int ret = OB_SUCCESS;
  char tmp[128];
  int32_t out_len = -1;
  if (PQ_TYPE_INT32 == type && PQ_CONVERTED_DATE == converted) {
    int32_t days = 0;
    MEMCPY(&days, p, 4);
    int64_t z = static_cast<int64_t>(days) + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = yoe + era * 400;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    const int64_t d = doy - (153 * mp + 2) / 5 + 1;
    const int64_t m = (mp < 10 ? mp + 3 : mp - 9);
    if (m <= 2) {
      ++y;
    }
    out_len = static_cast<int32_t>(snprintf(tmp, sizeof(tmp), "%04ld-%02ld-%02ld",
                                            static_cast<long>(y), static_cast<long>(m),
                                            static_cast<long>(d)));
  } else if (PQ_TYPE_INT32 == type) {
    int32_t v = 0;
    MEMCPY(&v, p, 4);
    out_len = static_cast<int32_t>(snprintf(tmp, sizeof(tmp), "%d", v));
  } else if (PQ_TYPE_INT64 == type) {
    int64_t v = 0;
    MEMCPY(&v, p, 8);
    out_len = static_cast<int32_t>(snprintf(tmp, sizeof(tmp), "%ld", static_cast<long>(v)));
  } else if (PQ_TYPE_FLOAT == type) {
    float v = 0;
    MEMCPY(&v, p, 4);
    out_len = static_cast<int32_t>(snprintf(tmp, sizeof(tmp), "%.9g", static_cast<double>(v)));
  } else if (PQ_TYPE_DOUBLE == type) {
    double v = 0;
    MEMCPY(&v, p, 8);
    out_len = static_cast<int32_t>(snprintf(tmp, sizeof(tmp), "%.17g", v));
  } else if (PQ_TYPE_BOOLEAN == type) {
    const bool v = (0 != (*reinterpret_cast<const uint8_t *>(p)));
    out_len = static_cast<int32_t>(snprintf(tmp, sizeof(tmp), "%s", v ? "true" : "false"));
  } else if (PQ_TYPE_INT96 == type) {
    out_len = 0;
  } else {
    out_len = len;
  }
  const char *src = (out_len >= 0 && PQ_TYPE_BYTE_ARRAY != type
                     && PQ_TYPE_FIXED_LEN != type) ? tmp : p;
  const int32_t copy_len = (PQ_TYPE_BYTE_ARRAY == type || PQ_TYPE_FIXED_LEN == type)
                            ? len : out_len;
  char *dst = static_cast<char *>(alloc.alloc(copy_len > 0 ? copy_len : 1));
  if (OB_ISNULL(dst)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    if (copy_len > 0) {
      MEMCPY(dst, src, copy_len);
    }
    if (OB_FAIL(ptrs.push_back(dst))) {
    } else if (OB_FAIL(lens.push_back(copy_len))) {
    }
  }
  return ret;
}

int pq_decode_values(common::ObIAllocator &alloc,
                     const int32_t type, const int32_t converted,
                     const bool dict_encoded,
                     const common::ObIArray<ObString> &dictionary,
                     const char *values, const int64_t values_size,
                     const common::ObIArray<uint32_t> &def_levels,
                     const int32_t max_def_level, const int64_t num_values,
                     common::ObIArray<const char *> &ptrs,
                     common::ObIArray<int32_t> &lens)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  int32_t present_cnt = static_cast<int32_t>(num_values);
  if (max_def_level > 0) {
    present_cnt = 0;
    for (int64_t i = 0; i < def_levels.count(); ++i) {
      if (static_cast<int32_t>(def_levels.at(i)) == max_def_level) {
        ++present_cnt;
      }
    }
  }
  common::ObSEArray<uint32_t, 64> indices;
  if (dict_encoded) {
    if (OB_UNLIKELY(values_size < 1)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const int32_t bit_width = static_cast<uint8_t>(values[pos]);
      ++pos;
      if (OB_FAIL(pq_decode_levels(values + pos, values_size - pos, bit_width,
                                   present_cnt, indices))) {
        LOG_WARN("failed to decode parquet dictionary indices", K(ret));
      }
    }
  }
  const int32_t type_size = pq_type_size(type, 0);
  const int64_t bool_base = pos;
  int32_t value_idx = 0;
  for (int64_t i = 0; OB_SUCC(ret) && i < num_values; ++i) {
    const bool is_null = (max_def_level > 0)
                         && (i >= def_levels.count()
                             || static_cast<int32_t>(def_levels.at(i)) != max_def_level);
    if (is_null) {
      if (OB_FAIL(ptrs.push_back(NULL))) {
      } else if (OB_FAIL(lens.push_back(0))) {
      }
    } else {
      const char *vp = NULL;
      int32_t vlen = 0;
      if (dict_encoded) {
        if (OB_UNLIKELY(value_idx >= indices.count())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          const uint32_t idx = indices.at(value_idx);
          if (OB_UNLIKELY(idx >= dictionary.count())) {
            ret = OB_ERR_UNEXPECTED;
          } else {
            vp = dictionary.at(idx).ptr();
            vlen = dictionary.at(idx).length();
          }
        }
      } else if (PQ_TYPE_BYTE_ARRAY == type) {
        if (OB_UNLIKELY(pos + 4 > values_size)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          int32_t len = 0;
          MEMCPY(&len, values + pos, 4);
          pos += 4;
          if (OB_UNLIKELY(len < 0) || OB_UNLIKELY(pos + len > values_size)) {
            ret = OB_ERR_UNEXPECTED;
          } else {
            vp = values + pos;
            vlen = len;
            pos += len;
          }
        }
      } else if (PQ_TYPE_BOOLEAN == type) {
        const int64_t byte_idx = bool_base + (value_idx >> 3);
        if (OB_UNLIKELY(byte_idx >= values_size)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          const uint8_t byte = static_cast<uint8_t>(values[byte_idx]);
          const bool bv = 0 != ((byte >> (value_idx & 7)) & 1u);
          char *dst = static_cast<char *>(alloc.alloc(8));
          if (OB_ISNULL(dst)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            const int32_t n = static_cast<int32_t>(snprintf(dst, 8, "%s",
                                                            bv ? "true" : "false"));
            if (OB_FAIL(ptrs.push_back(dst))) {
            } else if (OB_FAIL(lens.push_back(n))) {
            }
          }
        }
      } else if (type_size > 0) {
        if (OB_UNLIKELY(pos + type_size > values_size)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          vp = values + pos;
          vlen = type_size;
          pos += type_size;
        }
      } else {
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("unsupported parquet plain type", K(type), K(ret));
      }
      if (OB_SUCC(ret) && PQ_TYPE_BOOLEAN != type) {
        if (OB_FAIL(pq_push_value(alloc, type, converted, vp, vlen, ptrs, lens))) {
        }
      }
      ++value_idx;
    }
  }
  return ret;
}

int pq_decode_column(common::ObIAllocator &alloc,
                     const char *file_buf, const int64_t file_len,
                     const ObPqColMeta &meta, const int32_t max_def_level,
                     const int32_t fixed_len, const int32_t converted,
                     common::ObIArray<const char *> &ptrs,
                     common::ObIArray<int32_t> &lens)
{
  int ret = OB_SUCCESS;
  common::ObSEArray<ObString, 32> dictionary;
  const int64_t begin = (meta.dict_page_offset_ >= 0 && meta.dict_page_offset_ < meta.data_page_offset_)
                        ? meta.dict_page_offset_ : meta.data_page_offset_;
  if (OB_UNLIKELY(begin < 0) || OB_UNLIKELY(begin >= file_len)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid parquet page offset", K(begin), K(ret));
  } else {
    int64_t offset = begin;
    int64_t remaining = meta.num_values_;
    const int32_t type_size = pq_type_size(meta.type_, fixed_len);
    const bool can_compress = (PQ_CODEC_SNAPPY == meta.codec_);
    if (OB_UNLIKELY(!can_compress && PQ_CODEC_UNCOMPRESSED != meta.codec_)) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("unsupported parquet codec", K(meta.codec_), K(ret));
    }
    while (OB_SUCC(ret) && remaining > 0 && offset < file_len) {
      ObPqBuf header_buf(file_buf + offset, file_len - offset);
      ObPqPageHeader header;
      if (OB_FAIL(pq_parse_page_header(header_buf, header))) {
        LOG_WARN("failed to parse parquet page header", K(ret), K(offset));
      } else {
        const int64_t data_offset = offset + header_buf.pos();
        const int64_t data_size = header.compressed_size_;
        if (OB_UNLIKELY(data_size < 0) || OB_UNLIKELY(data_offset + data_size > file_len)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("parquet page is out of the file", K(data_offset), K(data_size), K(ret));
        } else {
          const char *page = file_buf + data_offset;
          char *decompressed = NULL;
          const char *plain = page;
          int64_t plain_size = data_size;
          const bool need_decompress = (PQ_CODEC_SNAPPY == meta.codec_)
                                       && ((PQ_PAGE_DATA_V2 == header.type_)
                                             ? header.v2_is_compressed_ : true);
          if (need_decompress) {
            const int64_t usize = header.uncompressed_size_ > 0
                                  ? header.uncompressed_size_ : data_size;
            decompressed = static_cast<char *>(alloc.alloc(usize > 0 ? usize : 1));
            if (OB_ISNULL(decompressed)) {
              ret = OB_ALLOCATE_MEMORY_FAILED;
            } else {
              int64_t written = 0;
              if (PQ_PAGE_DATA_V2 == header.type_) {
                const int64_t level_bytes = header.rep_levels_len_ + header.def_levels_len_;
                if (OB_UNLIKELY(level_bytes > data_size) || OB_UNLIKELY(level_bytes > usize)) {
                  ret = OB_ERR_UNEXPECTED;
                } else if (OB_FAIL(pq_snappy_decompress(page + level_bytes,
                                                        data_size - level_bytes,
                                                        decompressed + level_bytes,
                                                        usize - level_bytes, written))) {
                } else if (OB_UNLIKELY(written != usize - level_bytes)) {
                  ret = OB_ERR_UNEXPECTED;
                } else {
                  MEMCPY(decompressed, page, level_bytes);
                  plain = decompressed;
                  plain_size = usize;
                }
              } else if (OB_FAIL(pq_snappy_decompress(page, data_size, decompressed,
                                                      usize, written))) {
              } else if (OB_UNLIKELY(written != usize)) {
                ret = OB_ERR_UNEXPECTED;
              } else {
                plain = decompressed;
                plain_size = usize;
              }
              if (OB_FAIL(ret)) {
                LOG_WARN("failed to decompress parquet page", K(ret), K(header.type_));
              }
            }
          }
          if (OB_SUCC(ret) && PQ_PAGE_DICTIONARY == header.type_) {
            const int64_t count = header.num_values_;
            int64_t pos = 0;
            for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
              const char *vp = NULL;
              int32_t vlen = 0;
              if (PQ_TYPE_BYTE_ARRAY == meta.type_) {
                if (OB_UNLIKELY(pos + 4 > plain_size)) {
                  ret = OB_ERR_UNEXPECTED;
                } else {
                  int32_t len = 0;
                  MEMCPY(&len, plain + pos, 4);
                  pos += 4;
                  if (OB_UNLIKELY(len < 0) || OB_UNLIKELY(pos + len > plain_size)) {
                    ret = OB_ERR_UNEXPECTED;
                  } else {
                    vp = plain + pos;
                    vlen = len;
                    pos += len;
                  }
                }
              } else if (type_size > 0) {
                if (OB_UNLIKELY(pos + type_size > plain_size)) {
                  ret = OB_ERR_UNEXPECTED;
                } else {
                  vp = plain + pos;
                  vlen = type_size;
                  pos += type_size;
                }
              } else {
                ret = OB_NOT_SUPPORTED;
              }
              if (OB_SUCC(ret)) {
                ObString dict_string;
                dict_string.assign_ptr(vp, vlen);
                if (OB_FAIL(dictionary.push_back(dict_string))) {
                }
              }
            }
          } else if (OB_SUCC(ret)
                     && (PQ_PAGE_DATA == header.type_ || PQ_PAGE_DATA_V2 == header.type_)) {
            common::ObSEArray<uint32_t, 64> def_levels;
            const char *values = NULL;
            int64_t values_size = 0;
            if (PQ_PAGE_DATA_V2 == header.type_) {
              const int64_t level_bytes = header.rep_levels_len_ + header.def_levels_len_;
              if (OB_UNLIKELY(level_bytes > plain_size)) {
                ret = OB_ERR_UNEXPECTED;
              } else {
                if (max_def_level > 0 && header.def_levels_len_ > 0) {
                  if (OB_FAIL(pq_decode_levels(plain + header.rep_levels_len_,
                                               header.def_levels_len_, 1,
                                               header.num_values_, def_levels))) {
                  }
                }
                values = plain + level_bytes;
                values_size = plain_size - level_bytes;
              }
            } else if (max_def_level > 0) {
              int64_t pos = 0;
              if (OB_UNLIKELY(pos + 4 > plain_size)) {
                ret = OB_ERR_UNEXPECTED;
              } else {
                int32_t level_len = 0;
                MEMCPY(&level_len, plain + pos, 4);
                pos += 4;
                if (OB_UNLIKELY(level_len < 0) || OB_UNLIKELY(pos + level_len > plain_size)) {
                  ret = OB_ERR_UNEXPECTED;
                } else if (OB_FAIL(pq_decode_levels(plain + pos, level_len, 1,
                                                    header.num_values_, def_levels))) {
                } else {
                  values = plain + pos + level_len;
                  values_size = plain_size - pos - level_len;
                }
              }
            } else {
              values = plain;
              values_size = plain_size;
            }
            if (OB_SUCC(ret)) {
              const bool dict_encoded = (PQ_ENC_RLE_DICTIONARY == header.encoding_)
                                        || (PQ_ENC_PLAIN_DICTIONARY == header.encoding_);
              if (OB_UNLIKELY(!dict_encoded && PQ_ENC_PLAIN != header.encoding_)) {
                ret = OB_NOT_SUPPORTED;
                LOG_WARN("unsupported parquet encoding", K(header.encoding_), K(ret));
              } else if (OB_UNLIKELY(dict_encoded && dictionary.empty())) {
                ret = OB_ERR_UNEXPECTED;
                LOG_WARN("dictionary encoded page without a dictionary", K(ret));
              } else if (OB_FAIL(pq_decode_values(alloc, meta.type_, converted, dict_encoded,
                                                  dictionary, values, values_size,
                                                  def_levels, max_def_level,
                                                  header.num_values_, ptrs, lens))) {
                LOG_WARN("failed to decode parquet values", K(ret));
              }
            }
          }
        }
        offset = data_offset + data_size;
        remaining -= header.num_values_;
      }
    }
  }
  return ret;
}

} // anonymous namespace

int ObFileSqlReader::read_parquet(const ObFileSqlOptions &opts, ObFileSqlTableData &table)
{
  int ret = OB_SUCCESS;
  file_source::FileParquetReader reader;
  file_source::FileReadOptions read_options;
  file_source::FileSchema schema;
  file_source::FileSourceError error;
  const std::string path(opts.path_.ptr(), static_cast<size_t>(opts.path_.length()));
  read_options.format = file_source::FileFormat::PARQUET;
  if (OB_UNLIKELY(opts.path_.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!(error = reader.open(path, read_options)).is_ok()) {
    ret = OB_FILE_NOT_EXIST;
    LOG_WARN("failed to open parquet file", K(ret), K(opts.path_), "error", error.to_string().c_str());
  } else if (!(error = reader.infer_schema(schema)).is_ok()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to infer parquet schema", K(ret), K(opts.path_), "error", error.to_string().c_str());
  } else if (OB_UNLIKELY(schema.columns.empty())) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ObFileSqlRawData raw;
    for (int64_t j = 0; OB_SUCC(ret) && j < static_cast<int64_t>(schema.columns.size()); ++j) {
      const std::string &name = schema.columns[static_cast<size_t>(j)].name;
      common::ObString column_name(static_cast<int32_t>(name.size()), const_cast<char *>(name.data()));
      if (OB_FAIL(raw.add_name(column_name))) {
      }
    }
    file_source::FileRow row;
    bool has_row = false;
    while (OB_SUCC(ret) && (has_row = reader.next_row(row, error))) {
      if (OB_UNLIKELY(row.values.size() != schema.columns.size()
                      || row.nulls.size() != schema.columns.size())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("parquet row shape mismatch", K(ret), K(row.values.size()),
                 K(row.nulls.size()), K(schema.columns.size()));
      }
      for (int64_t j = 0; OB_SUCC(ret) && j < static_cast<int64_t>(row.values.size()); ++j) {
        if (row.nulls[static_cast<size_t>(j)]) {
          if (OB_FAIL(raw.add_cell(NULL, 0))) {
          }
        } else {
          const std::string &value = row.values[static_cast<size_t>(j)];
          const int32_t len = static_cast<int32_t>(value.size());
          char *copy = static_cast<char *>(table.alloc_.alloc(len > 0 ? len : 1));
          if (OB_ISNULL(copy)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            if (len > 0) {
              MEMCPY(copy, value.data(), len);
            }
            if (OB_FAIL(raw.add_cell(copy, len))) {
            }
          }
        }
      }
    }
    if (OB_SUCC(ret) && !error.is_ok()) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to read parquet rows", K(ret), K(opts.path_), "error", error.to_string().c_str());
    }
    if (OB_SUCC(ret)) {
      raw.col_cnt_ = static_cast<int64_t>(schema.columns.size());
      raw.row_cnt_ = raw.col_cnt_ > 0 ? raw.cells_.count() / raw.col_cnt_ : 0;
      if (OB_FAIL(materialize(table.alloc_, raw, table))) {
      }
    }
  }
  return ret;
}
int ObFileSqlReader::read_table(const ObFileSqlFuncKind kind,
                                const ObFileSqlOptions &opts,
                                ObFileSqlTableData &table)
{
  int ret = OB_SUCCESS;
  switch (kind) {
    case FILE_SQL_FUNC_READ_CSV:
      ret = read_csv(opts, table);
      break;
    case FILE_SQL_FUNC_READ_JSON:
      ret = read_json(opts, table);
      break;
    case FILE_SQL_FUNC_READ_PARQUET:
      ret = read_parquet(opts, table);
      break;
    case FILE_SQL_FUNC_GLOB:
      ret = read_glob(opts, table);
      break;
    default:
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("file sql function is not supported yet", K(kind), K(ret));
      break;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
