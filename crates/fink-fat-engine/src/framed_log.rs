//! Append-only logs of length-prefixed `rkyv` frames.
//!
//! # Format
//!
//! A log is a plain sequence of frames, each one
//!
//! ```text
//! [ u64 little-endian payload length ][ rkyv-serialized payload ]
//! ```
//!
//! with no header and no trailer, so appending a frame is a single
//! `write_all` pair at the end of the file and never rewrites what is
//! already there.
//!
//! # Crash tolerance
//!
//! A process killed while writing (OOM kill, power loss) can only leave a
//! *truncated last frame*: a cut length prefix, or a payload shorter than its
//! prefix announces. [`read_frames`] returns every complete frame before it
//! and reports where the valid prefix of the file ends
//! ([`FramedRead::valid_len`]), so a writer that wants to keep appending can
//! first cut the torn tail off — otherwise the stale prefix would swallow the
//! next frames.
//!
//! # Users
//!
//! * the archive of closed trajectories
//!   ([`write_archived_batch`](crate::topocentric_kf::branching::write_archived_batch) /
//!   [`read_archived_log`](crate::topocentric_kf::branching::read_archived_log)),
//!   one frame per night;
//! * `fink-fat-stream`'s night journal, one frame per batch of alerts.

use std::io::{self, Read, Write};

use camino::Utf8Path;
use rkyv::{
    Archive, Deserialize, Serialize,
    api::high::{HighDeserializer, HighSerializer, HighValidator},
    bytecheck::CheckBytes,
    rancor,
    ser::allocator::ArenaHandle,
    util::AlignedVec,
};

use crate::error::{EngineError, FinkFatError};

/// Byte length of a frame's length prefix.
pub const FRAME_PREFIX_LEN: u64 = 8;

/// Append `value` to `writer` as one frame (see the [module doc](self)).
///
/// The caller owns buffering and durability: wrap the file in a
/// `BufWriter` and flush / fsync after the frames that must survive a crash.
///
/// # Errors
/// [`FinkFatError::Message`] when `value` fails to serialize,
/// [`FinkFatError::Io`] when the write fails.
pub fn write_frame<T, W>(writer: &mut W, value: &T) -> Result<(), EngineError>
where
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
    W: Write,
{
    let bytes =
        rkyv::to_bytes::<rancor::Error>(value).map_err(|e| FinkFatError::Message(e.to_string()))?;
    writer
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .map_err(FinkFatError::Io)?;
    writer.write_all(&bytes).map_err(FinkFatError::Io)?;
    Ok(())
}

/// Everything [`read_frames`] recovered from a log.
#[derive(Debug, Clone, PartialEq)]
pub struct FramedRead<T> {
    /// Every complete frame, in the order they were appended.
    pub frames: Vec<T>,
    /// Byte length of the valid prefix of the file: the end of the last
    /// complete frame. Equal to the file length unless [`Self::torn_tail`].
    pub valid_len: u64,
    /// Whether the file ends with a truncated frame (ignored by the read).
    pub torn_tail: bool,
}

impl<T> Default for FramedRead<T> {
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            valid_len: 0,
            torn_tail: false,
        }
    }
}

/// Read every frame of the log at `path` (see the [module doc](self)).
///
/// A missing file is an empty log (`Ok` with no frame): a run that has not
/// written anything yet simply has no file. A truncated last frame is
/// tolerated and reported through [`FramedRead::torn_tail`].
///
/// The whole log is materialized in memory: logs are read offline or once at
/// start-up, never on the hot path.
///
/// # Errors
/// [`FinkFatError::Io`] when the file cannot be read, [`FinkFatError::Message`]
/// when a *complete* frame fails `rkyv` validation (a corrupt file, as
/// opposed to a torn tail).
pub fn read_frames<T>(path: &Utf8Path) -> Result<FramedRead<T>, EngineError>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, rancor::Error>>
        + Deserialize<T, HighDeserializer<rancor::Error>>,
{
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(FramedRead::default()),
        Err(e) => return Err(FinkFatError::Io(e).into()),
    };
    let file_len = file.metadata().map_err(FinkFatError::Io)?.len();
    let mut reader = io::BufReader::new(file);

    let mut out = FramedRead::default();
    let mut len_buf = [0u8; FRAME_PREFIX_LEN as usize];
    loop {
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                out.torn_tail = out.valid_len < file_len;
                break;
            }
            Err(e) => return Err(FinkFatError::Io(e).into()),
        }
        let len = u64::from_le_bytes(len_buf);

        // A payload announced past the end of the file is a torn tail —
        // checked before allocating, so a garbage prefix never triggers a
        // huge allocation.
        let frame_end = out.valid_len + FRAME_PREFIX_LEN;
        if len > file_len.saturating_sub(frame_end) {
            out.torn_tail = true;
            break;
        }

        // `AlignedVec`, not `Vec<u8>`: rkyv validation needs the payload at
        // its archive alignment, which a byte vector does not guarantee.
        let mut payload = AlignedVec::<16>::new();
        payload.resize(len as usize, 0);
        reader
            .read_exact(&mut payload[..])
            .map_err(FinkFatError::Io)?;

        let frame = rkyv::from_bytes::<T, rancor::Error>(&payload)
            .map_err(|e| FinkFatError::Message(e.to_string()))?;
        out.frames.push(frame);
        out.valid_len = frame_end + len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf;
    use tempfile::TempDir;

    fn log_path(dir: &TempDir) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(dir.path().join("log.rkyvlog")).expect("utf8 tempdir")
    }

    fn write_all_frames(path: &Utf8Path, frames: &[Vec<u64>]) {
        let mut file = std::fs::File::create(path).expect("create log");
        for frame in frames {
            write_frame(&mut file, frame).expect("write frame");
        }
        file.flush().expect("flush");
    }

    #[test]
    fn missing_file_is_an_empty_log() {
        let dir = TempDir::new().expect("tempdir");
        let read = read_frames::<Vec<u64>>(&log_path(&dir)).expect("read");
        assert_eq!(read, FramedRead::default());
    }

    #[test]
    fn frames_round_trip_in_order() {
        let dir = TempDir::new().expect("tempdir");
        let path = log_path(&dir);
        let frames = vec![vec![1, 2, 3], vec![], vec![42]];
        write_all_frames(&path, &frames);

        let read = read_frames::<Vec<u64>>(&path).expect("read");
        assert_eq!(read.frames, frames);
        assert!(!read.torn_tail);
        assert_eq!(read.valid_len, std::fs::metadata(&path).unwrap().len());
    }

    #[test]
    fn a_torn_tail_is_dropped_and_reported() {
        let dir = TempDir::new().expect("tempdir");
        let path = log_path(&dir);
        write_all_frames(&path, &[vec![7, 8]]);
        let first_len = std::fs::metadata(&path).unwrap().len();
        write_all_frames(&path, &[vec![7, 8], vec![9; 32]]);
        let full_len = std::fs::metadata(&path).unwrap().len();
        let first_only = vec![vec![7, 8]];

        // Cut inside the second frame's length prefix, then inside its payload.
        for cut in [first_len + 4, full_len - 3] {
            let bytes = std::fs::read(&path).unwrap();
            let torn_path = Utf8PathBuf::from(format!("{path}.{cut}"));
            std::fs::write(&torn_path, &bytes[..cut as usize]).unwrap();

            let read = read_frames::<Vec<u64>>(&torn_path).expect("read torn log");
            assert_eq!(read.frames, first_only, "cut at {cut}");
            assert!(read.torn_tail, "cut at {cut}");
            assert_eq!(read.valid_len, first_len, "cut at {cut}");
        }
    }

    #[test]
    fn a_complete_but_corrupt_frame_is_an_error() {
        let dir = TempDir::new().expect("tempdir");
        let path = log_path(&dir);
        let mut bytes = 4u64.to_le_bytes().to_vec();
        bytes.extend([0xFF; 4]);
        std::fs::write(&path, bytes).unwrap();
        assert!(read_frames::<Vec<u64>>(&path).is_err());
    }
}
