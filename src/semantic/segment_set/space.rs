//! How much a directory's filesystem can still take.
//!
//! The standard library cannot say, and the crate takes no dependency to ask, so this is
//! the one system call on each platform, declared here: `statvfs` on Linux, Android and
//! Apple, `GetDiskFreeSpaceExW` on Windows. Elsewhere — and wherever the call fails — the
//! answer is `None`, and an operation finds out instead: a write that fails with the
//! filesystem full is reported as
//! [`ArtifactError::InsufficientSpace`](crate::errors::ArtifactError::InsufficientSpace)
//! all the same.

use std::path::Path;

/// Bytes an unprivileged writer can still put on the filesystem holding `path`.
pub(crate) fn available(path: &Path) -> Option<u64> {
    imp::available(path)
}

#[cfg(any(
    all(
        any(target_os = "linux", target_os = "android"),
        target_pointer_width = "64"
    ),
    target_os = "macos",
    target_os = "ios"
))]
mod imp {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    extern "C" {
        fn statvfs(path: *const c_char, buf: *mut u8) -> c_int;
    }

    pub(super) fn available(path: &Path) -> Option<u64> {
        let path = CString::new(path.as_os_str().as_bytes()).ok()?;
        // Larger than any platform's `struct statvfs` (112 bytes on 64-bit Linux, 64 on
        // Apple) and aligned for its widest field; only the fields read below are used.
        let mut buf = [0u64; 64];
        // SAFETY: `path` is a NUL-terminated string that outlives the call, and `buf` is
        // writable, aligned to 8 and larger than the structure the call fills.
        let result = unsafe { statvfs(path.as_ptr(), buf.as_mut_ptr().cast::<u8>()) };
        if result != 0 {
            return None;
        }
        let (fragment, available) = fields(&buf);
        fragment.checked_mul(available)
    }

    /// `f_frsize` and `f_bavail`. Both layouts start with two `unsigned long`s, `f_bsize`
    /// and `f_frsize`, then `fsblkcnt_t f_blocks, f_bfree, f_bavail` — 64 bits on Linux and
    /// Android, 32 on Apple.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn fields(buf: &[u64; 64]) -> (u64, u64) {
        (buf[1], buf[4])
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn fields(buf: &[u64; 64]) -> (u64, u64) {
        // f_blocks and f_bfree share word 2; f_bavail is the low half of word 3.
        (buf[1], buf[3] & 0xFFFF_FFFF)
    }
}

#[cfg(windows)]
mod imp {
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            free_to_caller: *mut u64,
            total: *mut u64,
            free: *mut u64,
        ) -> i32;
    }

    pub(super) fn available(path: &Path) -> Option<u64> {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let (mut free_to_caller, mut total, mut free) = (0u64, 0u64, 0u64);
        // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call, and the
        // three out-pointers are valid, aligned `u64`s — `ULARGE_INTEGER` is a `u64`.
        let ok = unsafe {
            GetDiskFreeSpaceExW(wide.as_ptr(), &mut free_to_caller, &mut total, &mut free)
        };
        (ok != 0).then_some(free_to_caller)
    }
}

#[cfg(not(any(
    all(
        any(target_os = "linux", target_os = "android"),
        target_pointer_width = "64"
    ),
    target_os = "macos",
    target_os = "ios",
    windows
)))]
mod imp {
    use std::path::Path;

    pub(super) fn available(_path: &Path) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    /// On the platforms CI runs — Linux, macOS, Windows — the call answers, and with
    /// something a filesystem could hold.
    #[test]
    fn the_free_space_of_the_temporary_directory_is_known() {
        let free = super::available(&std::env::temp_dir());
        if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            let free = free.expect("the platform answers");
            assert!(free > 0 && free < 1 << 60, "{free}");
        }
    }
}
