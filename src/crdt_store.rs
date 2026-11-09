//! CRDT state kept in ALICE-DB (feature `crdt-db`): a value is stored under a key in its
//! [`crate::persist`] form (stamped with the state identifier) and read back through the same
//! decode, so a restart reads exactly what was written and an unknown identifier is refused.

use alice_db::blob::BlobStorage;

use crate::persist::{decode_state, encode_state, Decoded, Persist, PersistError};

/// Why a stored value could not be written or read
#[derive(Debug)]
pub enum StoreError {
    /// The store failed
    Io(std::io::Error),
    /// The stored bytes are not a value of this type and this build's state identifier
    Persist(PersistError),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "the store failed: {e}"),
            Self::Persist(e) => write!(f, "the stored value cannot be read: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

/// Write `value` under `key` and flush, so it is there after a restart
///
/// # Errors
///
/// [`StoreError::Io`] when the store cannot be written or flushed
pub fn store_state<T: Persist>(db: &BlobStorage, key: &[u8], value: &T) -> Result<(), StoreError> {
    db.put(key, &encode_state(value)).map_err(StoreError::Io)?;
    db.flush().map_err(StoreError::Io)
}

/// Read the value stored under `key` (`None` when nothing is)
///
/// # Errors
///
/// [`StoreError::Io`] when the store cannot be read, [`StoreError::Persist`] when the bytes are
/// not a value of `T` stamped with an identifier this build reads
pub fn load_state<T: Persist>(
    db: &BlobStorage,
    key: &[u8],
) -> Result<Option<Decoded<T>>, StoreError> {
    match db.get(key).map_err(StoreError::Io)? {
        None => Ok(None),
        Some(bytes) => decode_state(&bytes).map(Some).map_err(StoreError::Persist),
    }
}
