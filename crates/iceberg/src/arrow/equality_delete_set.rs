// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Equality deletes as a set of deleted keys.
//!
//! A data row is deleted when its key is in the set, so applying a delete file
//! costs one lookup per row whatever the number of keys the file names. As a
//! predicate with one term per deleted key, it cost one comparison per row per
//! key.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_cast::cast;
use arrow_row::{RowConverter, SortField};
use arrow_schema::{ArrowError, DataType};
use parquet::arrow::arrow_reader::ArrowPredicate;
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::schema::types::SchemaDescriptor;

use crate::{Error, ErrorKind, Result};

/// The keys one equality delete file names.
pub(crate) struct EqualityDeleteSet {
    /// Ascending. The key columns are encoded in this order, so two files with
    /// the same equality ids produce comparable keys.
    field_ids: Vec<i32>,
    data_types: Vec<DataType>,
    converter: RowConverter,
    keys: HashSet<Box<[u8]>>,
}

impl fmt::Debug for EqualityDeleteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EqualityDeleteSet")
            .field("field_ids", &self.field_ids)
            .field("data_types", &self.data_types)
            .field("keys", &self.keys.len())
            .finish()
    }
}

impl EqualityDeleteSet {
    /// A set that deletes nothing.
    pub(crate) fn empty() -> Self {
        Self {
            field_ids: Vec::new(),
            data_types: Vec::new(),
            converter: RowConverter::new(Vec::new()).expect("a converter of no columns"),
            keys: HashSet::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    /// Adds every row of a delete file batch.
    ///
    /// `equality_ids` are looked up among the batch's top-level columns by
    /// field id. An id that is not a top-level column is refused: the reader
    /// cannot address a nested column in a data file's row filter either.
    pub(crate) fn insert_batch(
        &mut self,
        batch: &RecordBatch,
        equality_ids: &HashSet<i32>,
    ) -> Result<()> {
        if batch.num_rows() == 0 || batch.num_columns() == 0 {
            return Ok(());
        }
        let columns = key_columns(batch, equality_ids)?;

        if self.field_ids.is_empty() {
            self.field_ids = columns.iter().map(|(id, _)| *id).collect();
            self.data_types = columns
                .iter()
                .map(|(_, array)| array.data_type().clone())
                .collect();
            self.converter = RowConverter::new(
                self.data_types
                    .iter()
                    .cloned()
                    .map(SortField::new)
                    .collect(),
            )
            .map_err(arrow_error)?;
        }

        let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, array)| array).collect();
        let rows = self.converter.convert_columns(&arrays).map_err(arrow_error)?;
        self.keys
            .extend(rows.iter().map(|row| Box::from(row.as_ref())));
        Ok(())
    }

    fn same_key_layout(&self, other: &Self) -> bool {
        self.field_ids == other.field_ids && self.data_types == other.data_types
    }
}

/// The equality columns of a delete file batch, ascending by field id.
fn key_columns(
    batch: &RecordBatch,
    equality_ids: &HashSet<i32>,
) -> Result<Vec<(i32, ArrayRef)>> {
    let schema = batch.schema();
    let mut columns = Vec::with_capacity(equality_ids.len());
    for (field, array) in schema.fields().iter().zip(batch.columns()) {
        let id = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id| id.parse::<i32>().ok());
        if let Some(id) = id.filter(|id| equality_ids.contains(id)) {
            if field.data_type().is_nested() {
                return Err(nested_key(id));
            }
            columns.push((id, Arc::clone(array)));
        }
    }
    if columns.len() != equality_ids.len() {
        let found: HashSet<i32> = columns.iter().map(|(id, _)| *id).collect();
        let mut missing: Vec<i32> = equality_ids.difference(&found).copied().collect();
        missing.sort_unstable();
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "equality ids {missing:?} are not top-level columns of the equality delete file"
            ),
        ));
    }
    columns.sort_by_key(|(id, _)| *id);
    Ok(columns)
}

fn nested_key(field_id: i32) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!("equality delete on field {field_id}, which is not a primitive top-level column"),
    )
}

fn arrow_error(err: ArrowError) -> Error {
    Error::new(ErrorKind::Unexpected, "encoding equality delete keys").with_source(err)
}

/// Delete sets that read their keys from the same columns, so a batch is
/// encoded once for all of them.
struct Layout {
    /// Per key column: its place in the projected batch, or `None` when the
    /// data file was written before the column existed.
    columns: Vec<Option<usize>>,
    sets: Vec<Arc<EqualityDeleteSet>>,
}

/// The row-filter stage that drops rows named by equality deletes.
pub(crate) struct EqualityDeletePredicate {
    projection: ProjectionMask,
    layouts: Vec<Layout>,
    rows_checked: Arc<AtomicU64>,
}

impl EqualityDeletePredicate {
    /// `field_id_map` is field id to leaf column index in the data file.
    pub(crate) fn try_new(
        sets: Vec<Arc<EqualityDeleteSet>>,
        parquet_schema: &SchemaDescriptor,
        field_id_map: &HashMap<i32, usize>,
        rows_checked: Arc<AtomicU64>,
    ) -> Result<Self> {
        let mut leaves: Vec<usize> = Vec::new();
        for set in &sets {
            for field_id in &set.field_ids {
                let Some(leaf) = field_id_map.get(field_id) else {
                    continue;
                };
                if parquet_schema.get_column_root(*leaf).is_group() {
                    return Err(nested_key(*field_id));
                }
                leaves.push(*leaf);
            }
        }
        leaves.sort_unstable();
        leaves.dedup();

        let mut layouts: Vec<Layout> = Vec::new();
        for set in sets {
            if let Some(layout) = layouts
                .iter_mut()
                .find(|layout| layout.sets[0].same_key_layout(&set))
            {
                layout.sets.push(set);
                continue;
            }
            let columns = set
                .field_ids
                .iter()
                .map(|field_id| {
                    field_id_map
                        .get(field_id)
                        .and_then(|leaf| leaves.binary_search(leaf).ok())
                })
                .collect();
            layouts.push(Layout {
                columns,
                sets: vec![set],
            });
        }

        Ok(Self {
            projection: ProjectionMask::leaves(parquet_schema, leaves),
            layouts,
            rows_checked,
        })
    }
}

impl ArrowPredicate for EqualityDeletePredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection
    }

    fn evaluate(&mut self, batch: RecordBatch) -> std::result::Result<BooleanArray, ArrowError> {
        let num_rows = batch.num_rows();
        self.rows_checked
            .fetch_add(num_rows as u64, Ordering::Relaxed);

        let mut keep = vec![true; num_rows];
        for layout in &self.layouts {
            let first = &layout.sets[0];
            let mut arrays = Vec::with_capacity(layout.columns.len());
            for (position, data_type) in layout.columns.iter().zip(&first.data_types) {
                arrays.push(match position {
                    // A data file keeps the type it was written with; the
                    // delete file's keys are in the table's current type.
                    Some(position) if batch.column(*position).data_type() != data_type => {
                        cast(batch.column(*position), data_type)?
                    }
                    Some(position) => Arc::clone(batch.column(*position)),
                    // A column added after the data file was written reads as
                    // null in it, and a null key deletes a null row.
                    None => new_null_array(data_type, num_rows),
                });
            }
            let rows = first.converter.convert_columns(&arrays)?;
            for (row, keep) in rows.iter().zip(keep.iter_mut()) {
                if *keep && layout.sets.iter().any(|set| set.keys.contains(row.as_ref())) {
                    *keep = false;
                }
            }
        }
        Ok(BooleanArray::from(keep))
    }
}
